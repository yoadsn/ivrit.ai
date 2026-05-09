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

Because the extraction stage (``extraction.py``) now produces text that is
character-identical to what the aligner outputs (modulo a leading space
tokenizer artifact), segment char offsets can be computed by simply
accumulating segment text lengths — no fuzzy text matching is needed.

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
    protocol_len: int,
    spk_segments: list[tuple[int, int, int]],
    force: bool,
) -> tuple[bool, list[str]]:
    """Build one map file from a single transcript.

    Segment char offsets are computed directly from cumulative segment text
    lengths — the aligned text is character-identical to ``raw.protocol.txt``
    after stripping the leading-space tokenizer artifact.

    Returns ``(ok, messages)``.
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

    # Verify that concatenated segment text length matches raw.protocol.txt.
    # If they diverge the direct offset computation would produce garbage.
    aligned_total = sum(len(seg.get("text", "")) for seg in segments)
    if aligned_total != protocol_len:
        messages.append(
            f"WARNING: Session {session_id}: text length mismatch for "
            f"{transcript_filename} (aligned={aligned_total}, "
            f"protocol={protocol_len}); skipping {map_filename}."
        )
        return False, messages

    # Compute char spans by accumulating segment text lengths.
    char_spans: list[tuple[int, int]] = []
    map_entries: list[dict] = []
    pos = 0

    for seg in segments:
        seg_text = seg.get("text", "")
        start_char = pos
        end_char = pos + len(seg_text)
        char_spans.append((start_char, end_char))
        map_entries.append({"start_char": start_char, "end_char": end_char, "speaker_ids": []})
        pos = end_char

    # Assign speakers.
    if spk_segments and char_spans:
        speaker_lists = _assign_speakers(char_spans, spk_segments)
        for i, spk_ids in enumerate(speaker_lists):
            map_entries[i]["speaker_ids"] = spk_ids

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

    Processes ``transcript.aligned.json`` -> ``transcript.aligned.map.json``
    and, if it exists, ``transcript.refined.json`` -> ``transcript.refined.map.json``.

    Returns ``(ok, messages)`` — *ok* is True if all attempted maps succeeded,
    *messages* is a list of log strings to be emitted by the parent process.
    """
    session_id = session_dir.name
    messages: list[str] = []

    # Check that at least the base aligned transcript exists.
    if not (session_dir / ALIGNED_TRANSCRIPT_FILENAME).exists():
        messages.append(
            f"WARNING: Session {session_id}: {ALIGNED_TRANSCRIPT_FILENAME} not found; skipping maps."
        )
        return False, messages

    protocol_path = session_dir / RAW_PROTOCOL_FILENAME
    if not protocol_path.exists():
        messages.append(
            f"WARNING: Session {session_id}: {RAW_PROTOCOL_FILENAME} not found; skipping maps."
        )
        return False, messages

    protocol_len = len(protocol_path.read_text(encoding="utf-8"))
    spk_segments = _load_speaker_segments(session_dir)

    all_ok = True
    for transcript_filename, map_filename in TRANSCRIPT_MAP_PAIRS:
        ok, transcript_msgs = _create_map_for_transcript(
            session_dir,
            transcript_filename,
            map_filename,
            protocol_len,
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
