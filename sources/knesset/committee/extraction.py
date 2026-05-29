"""Protocol extraction for Knesset committee sessions.

A committee "protocol" on S3 is a ``.doc`` file that is actually a
``.docx`` (a ZIP archive).  This module:

  1. Unzips the downloaded protocol archive into a temporary directory.
  2. Parses ``word/document.xml`` in a single walk, producing:
     * full verbatim text
     * clean text (no frontmatter, no speaker tags, normalised newlines)
     * timestamp map — list of ``(ts_ms, char_pos_in_clean_text)``
     * speaker segments — list of ``(speaker_id, start_char, end_char)``
     * speaker id -> name map
  3. Writes those artifacts into the session output directory.

The parsing logic mirrors the reference implementation in
``knesset_committee_data/parse_session_protocol.py``.  Only the I/O
boundary is changed: inputs are local paths under the output directory
(no S3 or manifest access), and outputs are written directly into the
caller-provided ``session_output_dir``.
"""

import logging
import pathlib
import re
import xml.etree.ElementTree as ET
import zipfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML namespace
# ---------------------------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


# ---------------------------------------------------------------------------
# Speaker / timestamp patterns
# ---------------------------------------------------------------------------

# Numeric-ID bookmark forms:
#   ET_yor_4996_3, ET_speakercontinue_845719_6,
#   ET_guest_845719_4, ET_speaker_5084_11
# Hebrew-name bookmark form:
#   ET_speaker_שגית_אפיק_9  (trailing integer is a counter)
SPEAKER_BM_NUMERIC_RE = re.compile(r"^ET_(yor|speakercontinue|guest|speaker)_(\d+)_(\d+)$")
SPEAKER_BM_NAME_RE = re.compile(r"^ET_(speaker)_(.+)_(\d+)$")
TIMESTAMP_RE = re.compile(r"^_ETM_Q1_(\d+)$")

# SDT <w:tag> inner-XML extraction (val is already entity-decoded by ET).
SDT_ID_RE = re.compile(r"<ID>(-?\d+)</ID>")
SDT_DATA_RE = re.compile(r"<Data>([^<]*)</Data>")
SDT_NAME_RE = re.compile(r"<Name>([^<]*)</Name>")

# Strip << … >> role markers, trailing colon.
_TAG_MARKER_RE = re.compile(r"<<[^>]*>>")

# Common formatting usage in the transcripts which should not be included in the captions
translate_formatting_to_replace_with_space = {
    # en-dash characters
    0x2013: " ",
    0x2014: " ",
}

# SDT alias values that identify a speaker block.
_SDT_SPEAKER_ALIASES = {"יור", "דובר", "דובר_המשך", "אורח", "קריאה", "interruption"}

# Subject / annotation paragraph detection.
_SUBJECT_TEXT_RE = re.compile(r"<<\s*נושא\s*>>")
_ALL_MARKERS_RE = re.compile(r"^(\s*<<[^>]*>>\s*)+$")

# Text-heuristic blacklist: Hebrew patterns that look like "Name:" but
# are ordinary prose / list introductions.
_HEURISTIC_BLACKLIST_RE = re.compile(
    r"(^אני )"
    r"|((אלה|אלו|יבוא|מאלה|ייאמר"
    r"|אומר|אומרת|נאמר|כך"
    r"|הבאים|הבאות):?\s*$)"
    r"|(\(.\))"
    r"|(\(\d+\))"
    r"|(\d\.)"
)

# Frontmatter staff-section keywords.  The *last* paragraph whose text
# starts with one of these AND ends with a colon marks the tail of the
# frontmatter block.
_STAFF_KEYWORDS_RE = re.compile(
    r"^("
    r"ייעוץ|יעוץ|יועץ|יועץ/ת|יועצת"
    r"|רישום"
    r"|רשמים|רשמות|רשמו|רשם|רשמת|רשמה"
    r"|קצרנים|קצרניות|קצרן|קצרנית"
    r"|מנהל/ת הו?ועדה|מנהלת הו?ועדה|מנהל הו?ועדה"
    r")\b.*:\s*$",
    re.UNICODE,
)

_SYNTH_ID_START = 10_000_000


# ---------------------------------------------------------------------------
# Helper primitives
# ---------------------------------------------------------------------------


def _clean_speaker_name(raw: str) -> str:
    name = _TAG_MARKER_RE.sub("", raw)
    return name.strip().rstrip(":").strip()


def _check_speaker_heuristic(para_text: str) -> str | None:
    """Return the cleaned speaker name if *para_text* looks like a
    ``Name:`` identification line, otherwise ``None``."""
    text = para_text.strip()
    if not text or ":" not in text:
        return None

    cleaned = _TAG_MARKER_RE.sub("", text).strip()
    if ":" not in cleaned:
        return None

    prefix, suffix = cleaned.split(":", 1)
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix:
        return None
    if len(prefix) < 5 or len(prefix) > 40:
        return None
    if len(prefix.split()) >= 6:
        return None
    if any(c.isdigit() for c in prefix):
        return None
    if any(c in prefix for c in ".?!;,"):
        return None
    if _HEURISTIC_BLACKLIST_RE.search(prefix):
        return None
    if len(suffix) > 30:
        return None
    return prefix


def _find_frontmatter_end(root) -> int:
    """Return the 0-based paragraph index of the last staff/admin keyword
    paragraph, or -1 if none is found.

    The actual frontmatter ends one paragraph *after* this (the staff
    member's name), so the text heuristic may activate from
    ``result + 2``.
    """
    last_staff_idx = -1
    for idx, p_elem in enumerate(root.iter(w("p"))):
        texts = [t.text for t in p_elem.iter(w("t")) if t.text]
        text = "".join(texts).strip()
        if text and _STAFF_KEYWORDS_RE.match(text):
            last_staff_idx = idx
    return last_staff_idx


def _walk(elem):
    """Yield ``('enter', elem)`` / ``('leave', elem)`` in document order."""
    yield ("enter", elem)
    for child in elem:
        yield from _walk(child)
    yield ("leave", elem)


def normalize_text_for_audio_tasks(text: str) -> str:
    """Apply text normalization suitable for audio/NLP tasks.
    
    This function:
    - Removes square-bracketed text
    - Replaces en-dashes used as sentence punctuation with commas
    - Replaces semicolons with commas (for NLP compatibility)
    - Removes other formatting characters
    - Deduplicates whitespace
    
    Args:
        text: The input text to normalize
        
    Returns:
        Normalized text string
    """
    # remove square bracketed text
    text = re.sub(r"\[.*?\]", "", text)

    # Replace en-dashes which stand for an end of sentence punctuation.
    # A semantic formatting that looks like "word – word" closely means "word. word" or even closer "word, word"
    # and this is much more familiar to downstream NLP tasks. We will use that more common formatting in the subtitles
    # output.
    text = re.sub(r"(\w)(\s[\u2013\u2014])", r"\g<1>,", text)

    # The ";" symbol marks a punctuation stronger than a "," but weaker than
    # a "." but is not as commonly supported in downstream NLP tasks. We choose to replace that
    # with a "," to simplify the output.
    text = re.sub(r"(\w);", r"\g<1>,", text)

    # Remove formatting other than the above
    text = text.translate(translate_formatting_to_replace_with_space)

    # Deduplicate whitespaces
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_document_xml(source):
    """Parse ``word/document.xml`` in a single walk.

    ``source`` is anything :func:`xml.etree.ElementTree.parse` accepts — a
    filesystem path or a binary file-like object (e.g. the stream returned
    by :meth:`zipfile.ZipFile.open`).

    Speaker identification is resolved, per paragraph, by three mechanisms
    (in priority order):

    1. Bookmark — ``ET_yor``, ``ET_speaker``, ``ET_speakercontinue``,
       ``ET_guest`` bookmark names.
    2. SDT — ``<w:sdt>`` blocks whose ``<w:tag>`` carries ``<ID>…</ID>``.
    3. Text heuristic — ``Name:`` pattern with Hebrew-aware validation
       (fallback for untagged protocols).

    Returns ``(full_text, clean_text, ts_map, speaker_segments, speaker_names)``.
    """
    tree = ET.parse(source)
    root = tree.getroot()

    # Full text — accumulated verbatim.
    full_parts: list[str] = []
    full_pos = 0

    # Clean text — committed at paragraph boundaries.
    clean_parts: list[str] = []
    clean_pos = 0
    first_speaker_seen = False
    # "A separator space is needed before the next content paragraph."  This
    # collapses any number of skipped / empty paragraphs.
    clean_needs_sep = False

    # Output maps (char offsets are into clean_text).
    ts_map: list[tuple[int, int]] = []
    speaker_segments: list[tuple[int, int, int]] = []
    speaker_names: dict[int, str] = {}

    # Current speaker accumulating spoken text.
    cur_spk_id: int | None = None
    cur_spk_start: int | None = None

    # Frontmatter boundary.  Two triggers end the frontmatter (first wins):
    #   1. We passed the last staff-keyword paragraph + 1 (name line).
    #   2. An XML-level speaker (bookmark / SDT) is encountered.
    _fm_last_staff_para = _find_frontmatter_end(root)
    _fm_heuristic_start = _fm_last_staff_para + 2 if _fm_last_staff_para >= 0 else -1
    _para_idx = -1
    frontmatter_ended = False

    # Name-based dedup for text-heuristic / name-bookmark speakers.
    _name_to_id: dict[str, int] = {}
    _next_synth = _SYNTH_ID_START

    def _id_for_name(name: str, preferred_id: int | None = None) -> int:
        nonlocal _next_synth
        if name in _name_to_id:
            return _name_to_id[name]
        for sid, sname in speaker_names.items():
            if sname == name:
                _name_to_id[name] = sid
                return sid
        if preferred_id is not None and preferred_id > 0:
            sid = preferred_id
        else:
            sid = _next_synth
            _next_synth += 1
        _name_to_id[name] = sid
        return sid

    def _switch_speaker(speaker_id: int | None, name: str | None) -> None:
        nonlocal cur_spk_id, cur_spk_start, first_speaker_seen
        if cur_spk_id is not None and cur_spk_start is not None:
            if clean_pos > cur_spk_start:
                speaker_segments.append((cur_spk_id, cur_spk_start, clean_pos))
        first_speaker_seen = True
        cur_spk_id = speaker_id
        cur_spk_start = clean_pos
        if speaker_id is not None and name:
            speaker_names[speaker_id] = name

    # Bookmark state.
    active_bm: dict[str, dict] = {}

    # SDT state.
    in_sdt_speaker = False
    sdt_speaker_id: int | None = None
    sdt_speaker_name: str | None = None

    # Per-paragraph buffer.
    seen_any_para = False
    in_para = False
    p_clean: list[str] = []
    p_clean_len = 0
    p_ts: list[tuple[int, int]] = []
    p_xml_spk = False
    p_xml_spk_id: int | None = None
    p_xml_spk_name: str | None = None
    p_is_subject = False

    def _in_spk_name_region() -> bool:
        return bool(active_bm) or in_sdt_speaker

    def _commit_para() -> None:
        nonlocal in_para, clean_pos, clean_needs_sep, p_clean_len
        if not in_para:
            return

        para_text = "".join(p_clean)
        is_subject = (
            p_is_subject
            or bool(_SUBJECT_TEXT_RE.search(para_text))
            or bool(_ALL_MARKERS_RE.match(para_text))
        )
        if is_subject:
            for ts_ms, _ in p_ts:
                ts_map.append((ts_ms, clean_pos))
            in_para = False
            p_clean.clear()
            p_clean_len = 0
            p_ts.clear()
            return

        is_spk = p_xml_spk
        spk_id = p_xml_spk_id
        spk_name = p_xml_spk_name

        if not is_spk and frontmatter_ended:
            detected = _check_speaker_heuristic(para_text)
            if detected:
                is_spk = True
                spk_name = detected
                spk_id = _id_for_name(detected)

        if is_spk:
            _switch_speaker(spk_id, spk_name)
            for ts_ms, _ in p_ts:
                ts_map.append((ts_ms, clean_pos))
        elif first_speaker_seen and p_clean_len > 0:
            if clean_needs_sep:
                clean_parts.append(" ")
                clean_pos += 1
            para_start = clean_pos
            for chunk in p_clean:
                clean_parts.append(chunk)
                clean_pos += len(chunk)
            clean_needs_sep = True
            for ts_ms, offset in p_ts:
                ts_map.append((ts_ms, para_start + offset))
        else:
            for ts_ms, _ in p_ts:
                ts_map.append((ts_ms, clean_pos))

        in_para = False
        p_clean.clear()
        p_clean_len = 0
        p_ts.clear()

    def _start_para() -> None:
        nonlocal seen_any_para, in_para, full_pos, _para_idx, frontmatter_ended
        nonlocal p_xml_spk, p_xml_spk_id, p_xml_spk_name, p_clean_len, p_is_subject
        # Force-close any speaker bookmarks that were opened in the current paragraph
        # but whose bookmarkEnd never arrived (malformed/cross-paragraph bookmark).
        # This mirrors the normal bookmarkEnd handling so the name is still extracted.
        for bm_id in list(active_bm.keys()):
            info = active_bm.pop(bm_id)
            spk_id = info["speaker_id"]
            raw = "".join(info["name_parts"]).strip()
            cleaned = _clean_speaker_name(raw)
            if cleaned:
                p_xml_spk_name = cleaned
                speaker_names[spk_id] = cleaned
        if in_para:
            _commit_para()
        if seen_any_para:
            full_parts.append("\n")
            full_pos += 1
        seen_any_para = True
        in_para = True
        _para_idx += 1
        if (
            not frontmatter_ended
            and _fm_heuristic_start >= 0
            and _para_idx >= _fm_heuristic_start
        ):
            frontmatter_ended = True
        p_xml_spk = False
        p_xml_spk_id = None
        p_xml_spk_name = None
        p_clean_len = 0
        p_is_subject = False

    # ---- Walk the XML tree ----
    for ev, elem in _walk(root):
        tag = elem.tag

        if ev == "enter":
            if tag == w("sdt"):
                sdt_pr = elem.find(w("sdtPr"))
                if sdt_pr is not None:
                    alias_el = sdt_pr.find(w("alias"))
                    alias = alias_el.get(w("val"), "") if alias_el is not None else ""
                    if alias == "נושא":
                        p_is_subject = True
                    tag_el = sdt_pr.find(w("tag"))
                    if tag_el is not None and alias in _SDT_SPEAKER_ALIASES:
                        val = tag_el.get(w("val"), "")
                        id_m = SDT_ID_RE.search(val)
                        if id_m:
                            in_sdt_speaker = True
                            raw_sdt_id = int(id_m.group(1))
                            data_m = SDT_DATA_RE.search(val)
                            name_m = SDT_NAME_RE.search(val)
                            raw = (
                                data_m.group(1)
                                if data_m
                                else name_m.group(1)
                                if name_m
                                else ""
                            )
                            sdt_speaker_name = _clean_speaker_name(raw)
                            sdt_speaker_id = (
                                _id_for_name(sdt_speaker_name, raw_sdt_id)
                                if sdt_speaker_name
                                else raw_sdt_id
                            )
                continue

            if tag == w("p"):
                _start_para()
                if in_sdt_speaker:
                    p_xml_spk = True
                    p_xml_spk_id = sdt_speaker_id
                    p_xml_spk_name = sdt_speaker_name
                    frontmatter_ended = True
                continue

            if tag == w("bookmarkStart"):
                bm_id = elem.get(w("id"))
                bm_name = elem.get(w("name"))
                if not bm_name:
                    continue

                ts_m = TIMESTAMP_RE.match(bm_name)
                if ts_m:
                    p_ts.append((int(ts_m.group(1)), p_clean_len))
                    continue

                if bm_name.startswith("ET_subject_"):
                    p_is_subject = True
                    continue

                spk_m = SPEAKER_BM_NUMERIC_RE.match(bm_name)
                if spk_m:
                    spk_id = int(spk_m.group(2))
                    active_bm[bm_id] = {"speaker_id": spk_id, "name_parts": []}
                    p_xml_spk = True
                    p_xml_spk_id = spk_id
                    frontmatter_ended = True
                    continue

                spk_m2 = SPEAKER_BM_NAME_RE.match(bm_name)
                if spk_m2 and not spk_m2.group(2).isdigit():
                    raw_name = spk_m2.group(2).replace("_", " ")
                    synth_id = _id_for_name(raw_name)
                    active_bm[bm_id] = {"speaker_id": synth_id, "name_parts": []}
                    p_xml_spk = True
                    p_xml_spk_id = synth_id
                    frontmatter_ended = True
                    speaker_names[synth_id] = raw_name
                    continue

                continue

            if tag == w("bookmarkEnd"):
                bm_id = elem.get(w("id"))
                if bm_id in active_bm:
                    info = active_bm.pop(bm_id)
                    spk_id = info["speaker_id"]
                    raw = "".join(info["name_parts"]).strip()
                    cleaned = _clean_speaker_name(raw)
                    if cleaned:
                        p_xml_spk_name = cleaned
                        speaker_names[spk_id] = cleaned
                continue

            if tag == w("t"):
                text = elem.text or ""
                if not text:
                    continue

                full_parts.append(text)
                full_pos += len(text)

                if active_bm:
                    for bm_info in active_bm.values():
                        bm_info["name_parts"].append(text)

                if not _in_spk_name_region():
                    clean_chunk = _TAG_MARKER_RE.sub("", text)
                    clean_chunk = normalize_text_for_audio_tasks(clean_chunk)
                    if clean_chunk:
                        p_clean.append(clean_chunk)
                        p_clean_len += len(clean_chunk)

                continue

        else:
            if tag == w("sdt") and in_sdt_speaker:
                in_sdt_speaker = False
                sdt_speaker_id = None
                sdt_speaker_name = None
                continue

    _commit_para()

    if cur_spk_id is not None and cur_spk_start is not None:
        if clean_pos > cur_spk_start:
            speaker_segments.append((cur_spk_id, cur_spk_start, clean_pos))

    full_text = "".join(full_parts)
    clean_text = "".join(clean_parts)

    assert len(full_text) == full_pos
    assert len(clean_text) == clean_pos

    ts_map.sort(key=lambda x: x[0])

    logger.info(
        "Parsed protocol: full=%d clean=%d | %d ts, %d seg, %d spk",
        full_pos,
        clean_pos,
        len(ts_map),
        len(speaker_segments),
        len(speaker_names),
    )
    return full_text, clean_text, ts_map, speaker_segments, speaker_names


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


RAW_FULL_PROTOCOL_FILENAME = "raw.full.protocol.txt"
RAW_PROTOCOL_FILENAME = "raw.protocol.txt"
SPEAKERS_FILENAME = "speakers.txt"
SPEAKERS_SEGMENTS_FILENAME = "speakers.segments.txt"


def write_session_outputs(
    session_output_dir: pathlib.Path,
    full_text: str,
    clean_text: str,
    speaker_segments: list[tuple[int, int, int]],
    speaker_names: dict[int, str],
) -> None:
    session_output_dir.mkdir(parents=True, exist_ok=True)

    (session_output_dir / RAW_FULL_PROTOCOL_FILENAME).write_text(
        full_text, encoding="utf-8"
    )
    (session_output_dir / RAW_PROTOCOL_FILENAME).write_text(
        clean_text, encoding="utf-8"
    )

    with open(
        session_output_dir / SPEAKERS_SEGMENTS_FILENAME, "w", encoding="utf-8"
    ) as f:
        for spk_id, start, end in speaker_segments:
            f.write(f"{spk_id}\t{start}\t{end}\n")

    with open(session_output_dir / SPEAKERS_FILENAME, "w", encoding="utf-8") as f:
        for spk_id in sorted(speaker_names):
            f.write(f"{spk_id}\t{speaker_names[spk_id]}\n")


# ---------------------------------------------------------------------------
# End-to-end extract
# ---------------------------------------------------------------------------


def is_extracted(session_output_dir: pathlib.Path) -> bool:
    """Return True iff the clean-text output file already exists."""
    return (session_output_dir / RAW_PROTOCOL_FILENAME).exists()


DOCX_DOCUMENT_XML = "word/document.xml"


def extract_protocol(
    protocol_zip_path: pathlib.Path,
    session_output_dir: pathlib.Path,
) -> None:
    """Parse ``word/document.xml`` straight out of the docx zip and write
    the extracted outputs.  No files are unpacked to disk — we stream the
    single member we need from the archive and parse it in memory.  The
    XML-level timestamp map produced by the parser is discarded; we rely
    on the pre-align stage to produce per-segment timings instead."""
    session_output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(protocol_zip_path, "r") as zf:
        try:
            info = zf.getinfo(DOCX_DOCUMENT_XML)
        except KeyError as exc:
            raise FileNotFoundError(
                f"{DOCX_DOCUMENT_XML} not found inside {protocol_zip_path}"
            ) from exc
        with zf.open(info) as stream:
            (
                full_text,
                clean_text,
                _ts_map,  # discarded — superseded by pre-align outputs
                speaker_segments,
                speaker_names,
            ) = parse_document_xml(stream)
    write_session_outputs(
        session_output_dir,
        full_text,
        clean_text,
        speaker_segments,
        speaker_names,
    )


def process_protocol(
    protocol_zip_path: pathlib.Path,
    session_output_dir: pathlib.Path,
    force_reprocess: bool = False,
) -> bool:
    """Extract the session protocol, skipping work when outputs already exist.

    Returns True on success, False on failure.
    """
    session_id = session_output_dir.name
    if not force_reprocess and is_extracted(session_output_dir):
        logger.info("Session %s already extracted; skipping.", session_id)
        return True
    try:
        extract_protocol(protocol_zip_path, session_output_dir)
        return True
    except Exception as exc:
        logger.error("Extraction failed for session %s: %s", session_id, exc)
        return False
