"""Create maps stage for Knesset committee sessions.

After the normalize (alignment) stage produces ``transcript.aligned.json``,
this stage maps each aligned segment back to ``raw.protocol.txt`` and the
speaker segments extracted during protocol parsing.

The aligned transcript segments carry text that originates from
``raw.protocol.txt`` and appears **in order** (though alignment may have
slightly altered segment boundaries).  We exploit this sequential property
with an overlap sweep: for each segment we search forward from the last
match position in ``raw.protocol.txt`` to find the matching character span.

Produces:

* ``transcript.aligned.map.json`` — a JSON array parallel to the segments
  in ``transcript.aligned.json``.  Each entry is::

      {"start_char": <int>, "end_char": <int>, "speaker_ids": [<int>, ...]}

  ``start_char`` / ``end_char`` index into ``raw.protocol.txt``.
  ``speaker_ids`` lists the speakers whose char span in
  ``speakers.segments.txt`` overlaps with this segment.

Public entry points:

* :func:`add_create_maps_args` — adds CLI flags to an ``argparse`` parser.
* :func:`create_maps_sessions` — batch entry point for all sessions under
  an output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

RAW_PROTOCOL_FILENAME = "raw.protocol.txt"
ALIGNED_TRANSCRIPT_FILENAME = "transcript.aligned.json"
SPEAKERS_SEGMENTS_FILENAME = "speakers.segments.txt"
ALIGNED_MAP_FILENAME = "transcript.aligned.map.json"


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def add_create_maps_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-create-maps",
        action="store_true",
        help="Skip the create-maps stage.",
    )
    parser.add_argument(
        "--force-create-maps",
        action="store_true",
        help="Force re-creation of the aligned map even if it already exists.",
    )


# ---------------------------------------------------------------------------
# Text matching helpers
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs into a single space for fuzzy matching."""
    return " ".join(text.split())


def _find_segment_span(
    protocol_text: str,
    segment_text: str,
    search_from: int,
) -> tuple[int, int] | None:
    """Find the char span of *segment_text* in *protocol_text* starting from
    *search_from*, using an overlap / subsequence approach.

    The segment text should appear in the protocol in order.  We normalise
    whitespace on both sides and do a simple substring search.  If an exact
    (whitespace-normalised) match is found we map it back to the original
    protocol char offsets.

    Returns ``(start_char, end_char)`` into *protocol_text*, or ``None``.
    """
    seg_norm = _normalize_whitespace(segment_text)
    if not seg_norm:
        return None

    # Build a whitespace-normalised view of the remaining protocol text,
    # keeping a mapping from normalised-index -> original-index.
    #
    # To avoid re-scanning the entire protocol for every segment, we only
    # process from ``search_from`` onward (with a small look-back to handle
    # boundary overlap).
    look_back = min(search_from, len(seg_norm) + 50)
    scan_start = search_from - look_back

    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    prev_was_space = True  # suppress leading space
    for orig_idx in range(scan_start, len(protocol_text)):
        ch = protocol_text[orig_idx]
        if ch in (" ", "\t", "\n", "\r"):
            if not prev_was_space and norm_chars:
                norm_chars.append(" ")
                norm_to_orig.append(orig_idx)
                prev_was_space = True
        else:
            norm_chars.append(ch)
            norm_to_orig.append(orig_idx)
            prev_was_space = False

    norm_view = "".join(norm_chars)

    # Find the segment in the normalised view.
    pos = norm_view.find(seg_norm)
    if pos < 0:
        return None

    # Map back to original indices.
    start_orig = norm_to_orig[pos]
    # end_orig: the original index just past the last matched char.
    end_norm_idx = pos + len(seg_norm) - 1
    end_orig = norm_to_orig[end_norm_idx] + 1

    return start_orig, end_orig


# ---------------------------------------------------------------------------
# Speaker overlap
# ---------------------------------------------------------------------------


def _load_speaker_segments(
    session_dir: pathlib.Path,
) -> list[tuple[int, int, int]]:
    """Load ``speakers.segments.txt`` — each line: ``speaker_id\\tstart\\tend``."""
    spk_path = session_dir / SPEAKERS_SEGMENTS_FILENAME
    if not spk_path.exists():
        return []
    segments: list[tuple[int, int, int]] = []
    for line in spk_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        segments.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return segments


def _assign_speakers(
    char_spans: list[tuple[int, int]],
    spk_segments: list[tuple[int, int, int]],
) -> list[list[int]]:
    """Two-pointer sweep to assign speaker IDs to each char span.

    Both ``char_spans`` and ``spk_segments`` are sorted by start position.
    Returns a list parallel to *char_spans* with overlapping speaker IDs.
    """
    result: list[list[int]] = []
    j = 0  # low-water mark into spk_segments
    n_spk = len(spk_segments)

    for seg_start, seg_end in char_spans:
        ids: list[int] = []
        k = j
        while k < n_spk:
            spk_id, spk_start, spk_end = spk_segments[k]
            if spk_start >= seg_end:
                break
            if spk_end > seg_start:
                ids.append(spk_id)
            k += 1

        # Advance low-water mark.
        while j < n_spk and spk_segments[j][2] <= seg_start:
            j += 1

        result.append(ids)

    return result


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def create_maps_for_session(
    session_dir: pathlib.Path,
    force: bool = False,
) -> bool:
    """Create ``transcript.aligned.map.json`` for one session.

    Returns True on success, False on failure / skip.
    """
    session_id = session_dir.name
    map_path = session_dir / ALIGNED_MAP_FILENAME

    if map_path.exists() and not force:
        logger.info("Session %s: aligned map already exists; skipping.", session_id)
        return True

    protocol_path = session_dir / RAW_PROTOCOL_FILENAME
    aligned_path = session_dir / ALIGNED_TRANSCRIPT_FILENAME

    if not protocol_path.exists():
        logger.warning("Session %s: %s not found; skipping maps.", session_id, RAW_PROTOCOL_FILENAME)
        return False
    if not aligned_path.exists():
        logger.warning("Session %s: %s not found; skipping maps.", session_id, ALIGNED_TRANSCRIPT_FILENAME)
        return False

    protocol_text = protocol_path.read_text(encoding="utf-8")
    aligned_data = json.loads(aligned_path.read_text(encoding="utf-8"))
    segments = aligned_data.get("segments", [])

    # Sequential sweep: find each segment's text in the protocol.
    search_from = 0
    char_spans: list[tuple[int, int]] = []
    map_entries: list[dict] = []

    for idx, seg in enumerate(segments):
        seg_text = seg.get("text", "")
        span = _find_segment_span(protocol_text, seg_text, search_from)
        if span is not None:
            start_char, end_char = span
            char_spans.append((start_char, end_char))
            map_entries.append({"start_char": start_char, "end_char": end_char, "speaker_ids": []})
            # Advance search_from to just past this match to maintain
            # sequential order, but allow a little overlap for boundary
            # fuzziness.
            search_from = max(search_from, start_char + 1)
        else:
            logger.warning(
                "Session %s: could not locate segment %d in protocol text (len=%d, text=%.60s...).",
                session_id, idx, len(seg_text), seg_text[:60],
            )
            # Insert a placeholder with -1 to signal unmapped.
            char_spans.append((-1, -1))
            map_entries.append({"start_char": -1, "end_char": -1, "speaker_ids": []})

    # Assign speakers.
    spk_segments = _load_speaker_segments(session_dir)
    if spk_segments:
        # Only pass valid spans for speaker assignment; keep -1 entries as-is.
        valid_indices = [i for i, (s, e) in enumerate(char_spans) if s >= 0]
        valid_spans = [char_spans[i] for i in valid_indices]
        if valid_spans:
            speaker_lists = _assign_speakers(valid_spans, spk_segments)
            for vi, si in enumerate(valid_indices):
                map_entries[si]["speaker_ids"] = speaker_lists[vi]

    map_path.write_text(
        json.dumps(map_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Session %s: wrote %s with %d entries.", session_id, ALIGNED_MAP_FILENAME, len(map_entries))
    return True


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def create_maps_sessions(
    input_folder: pathlib.Path,
    force: bool = False,
    session_ids: Optional[list[str]] = None,
    abort_on_error: bool = False,
) -> None:
    """Run the create-maps stage for all sessions under *input_folder*.

    Iterates over subdirectories of *input_folder* that contain an aligned
    transcript and produces ``transcript.aligned.map.json`` for each.
    """
    if not input_folder.is_dir():
        logger.warning("Input folder %s does not exist.", input_folder)
        return

    session_dirs = sorted(
        d for d in input_folder.iterdir()
        if d.is_dir() and (d / ALIGNED_TRANSCRIPT_FILENAME).exists()
    )

    if session_ids:
        wanted = set(session_ids)
        session_dirs = [d for d in session_dirs if d.name in wanted]

    if not session_dirs:
        logger.info("No sessions with aligned transcripts found for map creation.")
        return

    print(f"Creating maps for {len(session_dirs)} session(s)...")
    for session_dir in tqdm(session_dirs, desc="Creating maps"):
        try:
            ok = create_maps_for_session(session_dir, force=force)
            if not ok:
                tqdm.write(f" - WARNING: map creation skipped/failed for {session_dir.name}")
                if abort_on_error:
                    raise RuntimeError(f"Map creation failed for {session_dir.name}")
        except Exception as e:
            msg = f" - ERROR: map creation failed for {session_dir.name}: {e}"
            tqdm.write(msg)
            logger.error(msg)
            if abort_on_error:
                raise
