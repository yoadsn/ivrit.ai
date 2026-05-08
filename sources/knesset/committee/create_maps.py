"""Create maps stage for Knesset committee sessions.

After the normalize (alignment) stage produces ``transcript.aligned.json``
and optionally after the refine-segments stage produces
``transcript.refined.json``, this stage maps each segment back to
``raw.protocol.txt`` and the speaker segments extracted during protocol
parsing.

Produces one map file for each transcript that exists:

* ``transcript.aligned.map.json`` — parallel to ``transcript.aligned.json``
* ``transcript.refined.map.json``  — parallel to ``transcript.refined.json``
  (only written when ``transcript.refined.json`` is present)

Each map entry is::

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

from sources.common.definitions import SKIPPED_FLAG_FILENAME

logger = logging.getLogger(__name__)

RAW_PROTOCOL_FILENAME = "raw.protocol.txt"
ALIGNED_TRANSCRIPT_FILENAME = "transcript.aligned.json"
REFINED_TRANSCRIPT_FILENAME = "transcript.refined.json"
SPEAKERS_SEGMENTS_FILENAME = "speakers.segments.txt"
ALIGNED_MAP_FILENAME = "transcript.aligned.map.json"
REFINED_MAP_FILENAME = "transcript.refined.map.json"

# Maps each transcript filename to its corresponding map output filename.
TRANSCRIPT_MAP_PAIRS: list[tuple[str, str]] = [
    (ALIGNED_TRANSCRIPT_FILENAME, ALIGNED_MAP_FILENAME),
    (REFINED_TRANSCRIPT_FILENAME, REFINED_MAP_FILENAME),
]


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
        help="Force re-creation of the transcript maps even if they already exist.",
    )
    parser.add_argument(
        "--create-maps-workers",
        type=int,
        default=4,
        help="Number of parallel workers for the create-maps stage (default: 4).",
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
# Per-transcript processing
# ---------------------------------------------------------------------------


def _create_map_for_transcript(
    session_dir: pathlib.Path,
    transcript_filename: str,
    map_filename: str,
    protocol_text: str,
    spk_segments: list[tuple[int, int, int]],
    force: bool,
) -> bool:
    """Build one map file from a single transcript.

    Returns ``True`` on success (including skip), ``False`` on failure.
    """
    session_id = session_dir.name
    transcript_path = session_dir / transcript_filename
    map_path = session_dir / map_filename

    if not transcript_path.exists():
        # Not present — nothing to do (not an error).
        return True

    if map_path.exists() and not force:
        logger.info(
            "Session %s: %s already exists; skipping.",
            session_id, map_filename,
        )
        return True

    aligned_data = json.loads(transcript_path.read_text(encoding="utf-8"))
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
                "Session %s: could not locate segment %d in protocol text "
                "(transcript=%s, len=%d, text=%.60s...).",
                session_id, idx, transcript_filename, len(seg_text), seg_text[:60],
            )
            # Insert a placeholder with -1 to signal unmapped.
            char_spans.append((-1, -1))
            map_entries.append({"start_char": -1, "end_char": -1, "speaker_ids": []})

    # Assign speakers.
    if spk_segments:
        valid_indices = [i for i, (s, e) in enumerate(char_spans) if s >= 0]
        valid_spans = [char_spans[i] for i in valid_indices]
        if valid_spans:
            speaker_lists = _assign_speakers(valid_spans, spk_segments)
            for vi, si in enumerate(valid_indices):
                map_entries[si]["speaker_ids"] = speaker_lists[vi]

    map_path.write_text(
        json.dumps(map_entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Session %s: wrote %s with %d entries.",
        session_id, map_filename, len(map_entries),
    )
    return True


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def create_maps_for_session(
    session_dir: pathlib.Path,
    force: bool = False,
) -> bool:
    """Create map files for all transcripts present in *session_dir*.

    Processes ``transcript.aligned.json`` → ``transcript.aligned.map.json``
    and, if it exists, ``transcript.refined.json`` → ``transcript.refined.map.json``.

    Returns True if all attempted maps succeeded, False otherwise.
    """
    session_id = session_dir.name
    protocol_path = session_dir / RAW_PROTOCOL_FILENAME

    if not protocol_path.exists():
        logger.warning("Session %s: %s not found; skipping maps.", session_id, RAW_PROTOCOL_FILENAME)
        return False

    # Check that at least the base aligned transcript exists.
    if not (session_dir / ALIGNED_TRANSCRIPT_FILENAME).exists():
        logger.warning(
            "Session %s: %s not found; skipping maps.",
            session_id, ALIGNED_TRANSCRIPT_FILENAME,
        )
        return False

    protocol_text = protocol_path.read_text(encoding="utf-8")
    spk_segments = _load_speaker_segments(session_dir)

    all_ok = True
    for transcript_filename, map_filename in TRANSCRIPT_MAP_PAIRS:
        ok = _create_map_for_transcript(
            session_dir,
            transcript_filename,
            map_filename,
            protocol_text,
            spk_segments,
            force=force,
        )
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def create_maps_sessions(
    input_folder: pathlib.Path,
    force: bool = False,
    session_ids: Optional[list[str]] = None,
    abort_on_error: bool = False,
    workers: int = 4,
) -> None:
    """Run the create-maps stage for all sessions under *input_folder*.

    For each session that has ``transcript.aligned.json`` this produces
    ``transcript.aligned.map.json``.  If ``transcript.refined.json`` also
    exists, ``transcript.refined.map.json`` is produced as well.

    Parameters
    ----------
    workers:
        Number of parallel threads to use for processing sessions.
    """
    if not input_folder.is_dir():
        logger.warning("Input folder %s does not exist.", input_folder)
        return

    session_dirs = sorted(
        d for d in input_folder.iterdir()
        if d.is_dir()
        and not (d / SKIPPED_FLAG_FILENAME).exists()
        and (d / ALIGNED_TRANSCRIPT_FILENAME).exists()
    )

    if session_ids:
        wanted = set(session_ids)
        session_dirs = [d for d in session_dirs if d.name in wanted]

    if not session_dirs:
        logger.info("No sessions with aligned transcripts found for map creation.")
        return

    print(f"Creating maps for {len(session_dirs)} session(s) with {workers} worker(s)...")

    errors: list[Exception] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_dir = {
            executor.submit(create_maps_for_session, session_dir, force): session_dir
            for session_dir in session_dirs
        }
        with tqdm(total=len(session_dirs), desc="Creating maps") as pbar:
            for future in as_completed(future_to_dir):
                session_dir = future_to_dir[future]
                try:
                    ok = future.result()
                    if not ok:
                        tqdm.write(f" - WARNING: map creation skipped/failed for {session_dir.name}")
                        if abort_on_error:
                            raise RuntimeError(f"Map creation failed for {session_dir.name}")
                except Exception as e:
                    msg = f" - ERROR: map creation failed for {session_dir.name}: {e}"
                    tqdm.write(msg)
                    logger.error(msg)
                    errors.append(e)
                finally:
                    pbar.update(1)

    if errors and abort_on_error:
        raise errors[0]
