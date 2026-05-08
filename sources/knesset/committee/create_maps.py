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
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# ---------------------------------------------------------------------------
# Optimised implementation — pre-builds normalised view once per protocol.
# ---------------------------------------------------------------------------


def _build_normalized_protocol(protocol_text: str) -> tuple[str, list[int]]:
    """Build a whitespace-normalised view of *protocol_text* once.

    Returns ``(norm_text, norm_to_orig)`` where ``norm_to_orig[i]`` is the
    index into the original *protocol_text* for normalised character ``i``.
    """
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    prev_was_space = True  # suppress leading space
    for orig_idx, ch in enumerate(protocol_text):
        if ch in (" ", "\t", "\n", "\r"):
            if not prev_was_space and norm_chars:
                norm_chars.append(" ")
                norm_to_orig.append(orig_idx)
                prev_was_space = True
        else:
            norm_chars.append(ch)
            norm_to_orig.append(orig_idx)
            prev_was_space = False

    return "".join(norm_chars), norm_to_orig


def _find_segment_span(
    norm_text: str,
    norm_to_orig: list[int],
    segment_text: str,
    search_from_norm: int,
) -> tuple[int, int, int] | None:
    """Find the char span of *segment_text* using the pre-built normalised view.

    *search_from_norm* is the offset into *norm_text* to start searching from.

    Returns ``(start_orig, end_orig, end_norm_idx + 1)`` — the original char
    span plus the normalised position just past the match (to advance the
    caller's cursor), or ``None``.
    """
    seg_norm = _normalize_whitespace(segment_text)
    if not seg_norm:
        return None

    pos = norm_text.find(seg_norm, search_from_norm)
    if pos < 0:
        return None

    start_orig = norm_to_orig[pos]
    end_norm_idx = pos + len(seg_norm) - 1
    end_orig = norm_to_orig[end_norm_idx] + 1

    return start_orig, end_orig, pos + 1


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


def _build_orig_to_norm(norm_to_orig: list[int], protocol_len: int) -> list[int]:
    """Build a reverse mapping: original-index -> normalised-index.

    For original positions that don't map directly to a normalised character
    (i.e. whitespace that was collapsed), we map to the *next* valid
    normalised position so that ``search_from`` in original space translates
    to a safe (possibly slightly early) position in normalised space.
    """
    orig_to_norm = [len(norm_to_orig)] * protocol_len  # default: past end
    for norm_idx, orig_idx in enumerate(norm_to_orig):
        # Only store the first norm_idx that maps to each orig_idx.
        if orig_to_norm[orig_idx] > norm_idx:
            orig_to_norm[orig_idx] = norm_idx
    # Fill gaps: for positions not directly in the mapping (collapsed
    # whitespace), propagate backward so they point to the next valid
    # normalised position.
    next_norm = len(norm_to_orig)
    for i in range(protocol_len - 1, -1, -1):
        if orig_to_norm[i] <= next_norm:
            next_norm = orig_to_norm[i]
        else:
            orig_to_norm[i] = next_norm
    return orig_to_norm


def _create_map_for_transcript(
    session_dir: pathlib.Path,
    transcript_filename: str,
    map_filename: str,
    norm_text: str,
    norm_to_orig: list[int],
    orig_to_norm: list[int],
    spk_segments: list[tuple[int, int, int]],
    force: bool,
) -> tuple[bool, list[str]]:
    """Build one map file from a single transcript (optimised path).

    Returns ``(ok, messages)`` — *ok* is True on success (including skip),
    False on failure.  *messages* is a list of ``(level, msg)`` strings to
    be logged by the caller in the parent process.
    """
    session_id = session_dir.name
    transcript_path = session_dir / transcript_filename
    map_path = session_dir / map_filename
    messages: list[str] = []

    if not transcript_path.exists():
        return True, messages

    if map_path.exists() and not force:
        messages.append(
            f"INFO: Session {session_id}: {map_filename} already exists; skipping."
        )
        return True, messages

    aligned_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = aligned_data.get("segments", [])

    # Sequential sweep using pre-built normalised view.
    search_from_norm = 0
    char_spans: list[tuple[int, int]] = []
    map_entries: list[dict] = []

    for idx, seg in enumerate(segments):
        seg_text = seg.get("text", "")
        # Apply the same look-back as the old code: the old code used
        # look_back = min(search_from, len(seg_norm) + 50) in original space.
        # We replicate this by backing up in normalised space.
        seg_norm = _normalize_whitespace(seg_text)
        look_back = min(search_from_norm, len(seg_norm) + 50)
        effective_from = search_from_norm - look_back

        span = _find_segment_span(
            norm_text, norm_to_orig, seg_text, effective_from,
        )
        if span is not None:
            start_char, end_char, next_norm_pos = span
            char_spans.append((start_char, end_char))
            map_entries.append({"start_char": start_char, "end_char": end_char, "speaker_ids": []})
            # Advance: replicate old behaviour of
            # search_from = max(search_from, start_char + 1)
            # but in normalised space.
            new_norm = orig_to_norm[start_char + 1] if start_char + 1 < len(orig_to_norm) else len(norm_text)
            search_from_norm = max(search_from_norm, new_norm)
        else:
            messages.append(
                f"WARNING: Session {session_id}: could not locate segment {idx} "
                f"in protocol text (transcript={transcript_filename}, "
                f"len={len(seg_text)}, text={seg_text[:60]!s}...)."
            )
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
    messages.append(
        f"INFO: Session {session_id}: wrote {map_filename} with {len(map_entries)} entries."
    )
    return True, messages


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def create_maps_for_session(
    session_dir: pathlib.Path,
    force: bool = False,
) -> tuple[bool, list[str]]:
    """Create map files for all transcripts present in *session_dir*.

    Processes ``transcript.aligned.json`` → ``transcript.aligned.map.json``
    and, if it exists, ``transcript.refined.json`` → ``transcript.refined.map.json``.

    Returns ``(ok, messages)`` — *ok* is True if all attempted maps succeeded,
    *messages* is a list of log strings to be emitted by the parent process.
    """
    session_id = session_dir.name
    protocol_path = session_dir / RAW_PROTOCOL_FILENAME
    messages: list[str] = []

    if not protocol_path.exists():
        messages.append(
            f"WARNING: Session {session_id}: {RAW_PROTOCOL_FILENAME} not found; skipping maps."
        )
        return False, messages

    # Check that at least the base aligned transcript exists.
    if not (session_dir / ALIGNED_TRANSCRIPT_FILENAME).exists():
        messages.append(
            f"WARNING: Session {session_id}: {ALIGNED_TRANSCRIPT_FILENAME} not found; skipping maps."
        )
        return False, messages

    protocol_text = protocol_path.read_text(encoding="utf-8")
    spk_segments = _load_speaker_segments(session_dir)

    # Pre-build normalised protocol view once for the session.
    norm_text, norm_to_orig = _build_normalized_protocol(protocol_text)
    orig_to_norm = _build_orig_to_norm(norm_to_orig, len(protocol_text))

    all_ok = True
    for transcript_filename, map_filename in TRANSCRIPT_MAP_PAIRS:
        ok, transcript_msgs = _create_map_for_transcript(
            session_dir,
            transcript_filename,
            map_filename,
            norm_text,
            norm_to_orig,
            orig_to_norm,
            spk_segments,
            force=force,
        )
        messages.extend(transcript_msgs)
        if not ok:
            all_ok = False

    return all_ok, messages


def _worker_init() -> None:
    """Initializer for child processes in the ProcessPoolExecutor.

    Removes file-based handlers inherited from the parent process to avoid
    multiple processes racing on RotatingFileHandler log rotation.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)


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
        Number of parallel processes to use for processing sessions.
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

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        future_to_dir = {
            executor.submit(create_maps_for_session, session_dir, force): session_dir
            for session_dir in session_dirs
        }
        with tqdm(total=len(session_dirs), desc="Creating maps") as pbar:
            for future in as_completed(future_to_dir):
                session_dir = future_to_dir[future]
                try:
                    ok, messages = future.result()
                    # Re-emit worker messages through the parent's logger.
                    for msg in messages:
                        if msg.startswith("WARNING:"):
                            logger.warning(msg[len("WARNING: "):])
                        elif msg.startswith("INFO:"):
                            logger.info(msg[len("INFO: "):])
                        else:
                            logger.info(msg)
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
