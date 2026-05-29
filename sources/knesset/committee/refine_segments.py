"""Refine-segments stage for Knesset committee sessions.

After the VAD stage produces frame-level speech probabilities, this step
adjusts segment start/end times in ``transcript.aligned.json`` so that
boundaries fall in silence regions rather than in the middle of speech.

For each consecutive pair of segments (curr, next) whose gap exceeds
``min_gap_to_adjust`` seconds the gap boundaries are adjusted using the
VAD signal and the result is written to ``transcript.refined.json``.

The input ``transcript.aligned.json`` is **never modified**; it always
reflects the raw alignment output.  ``transcript.refined.json`` is the
downstream-preferred version when it exists.

Public entry points:

* :func:`add_refine_segments_args` — adds CLI flags to an ``argparse``
  parser.
* :func:`refine_segments_sessions` — batch entry point for all sessions
  under an output directory.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from stable_whisper import WhisperResult
from tqdm import tqdm

from sources.common.definitions import SKIPPED_FLAG_FILENAME
from vad.definitions import SPEECH_PROB_FRAME_DURATION, VAD_SPEECH_PROBS_FILENAME
from vad.vad_io import load_frame_vad_probs

logger = logging.getLogger(__name__)

ALIGNED_TRANSCRIPT_FILENAME = "transcript.aligned.json"
REFINED_TRANSCRIPT_FILENAME = "transcript.refined.json"

# Default tunables
DEFAULT_MIN_GAP_TO_ADJUST = 0.5  # seconds
DEFAULT_MAX_WORDS_TO_MERGE = 25


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def add_refine_segments_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-refine-segments",
        action="store_true",
        help="Skip the refine-segments stage.",
    )
    parser.add_argument(
        "--force-refine-segments",
        action="store_true",
        help="Force re-run of the refine-segments stage even if outputs exist.",
    )
    parser.add_argument(
        "--refine-segments-min-gap",
        type=float,
        default=DEFAULT_MIN_GAP_TO_ADJUST,
        help=(
            "Minimum gap (seconds) between consecutive segments before "
            "attempting boundary adjustment. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--refine-segments-workers",
        type=int,
        default=4,
        help="Number of threads for the refine-segments stage. Default: %(default)s",
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def smooth_speech_probs(
    speech_probs: np.ndarray,
    median_kernel: int = 5,
) -> np.ndarray:
    """Return a median-filtered copy of *speech_probs*.

    Pre-compute this once per session and pass the result to
    :func:`adjust_gap` to avoid redundant work across segment pairs.
    """
    from scipy.signal import medfilt

    k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
    k = min(k, len(speech_probs))
    return medfilt(speech_probs.astype(float), kernel_size=k) if k >= 2 else speech_probs.astype(float)


def adjust_gap(
    curr_end: float,
    next_start: float,
    smoothed_probs: np.ndarray,
    silence_threshold: float = 0.6,
    max_adjust_sec: float = 0.2,
    min_drop_for_max_adjust: float = 0.3,
) -> tuple[float, float]:
    """Adjust the boundary between two segments using VAD speech probabilities.

    Parameters
    ----------
    curr_end:
        Current end time (seconds) of the preceding segment.
    next_start:
        Current start time (seconds) of the following segment.
    smoothed_probs:
        Pre-smoothed array of per-frame speech probabilities (output of
        :func:`smooth_speech_probs`).  Frame *i* corresponds to time
        ``i * SPEECH_PROB_FRAME_DURATION``.
    silence_threshold:
        Speech-probability value below which a frame is considered silence.
        If no frame in the gap falls below this, no adjustment is made.
    max_adjust_sec:
        Base maximum distance (seconds) a boundary may move when a
        sufficient drop (≥ *min_drop_for_max_adjust*) is found quickly.
        The search extends up to ``2 * max_adjust_sec`` if needed.
    min_drop_for_max_adjust:
        Once the narrow search (within *max_adjust_sec*) has been walked
        past, stop extending as soon as the cumulative drop in speech
        probability reaches this value.  The wide limit
        (``2 * max_adjust_sec``) is always the hard stop.

    Returns
    -------
    (new_curr_end, new_next_start)

    Algorithm
    ---------
    1. Locate the gap ``[curr_end, next_start]`` as frame indices.
    2. If no frame in the gap is below *silence_threshold* → no-op.
    3. Otherwise for each boundary scan outward into the gap:
       - Keep going until either the wide limit or a sufficient drop is
         found after passing the narrow limit.
       - Pick the frame with the lowest probability encountered.
    4. If the adjusted ``curr_end`` would meet or exceed ``next_start``,
       snap both to the midpoint (segments touch but do not overlap).
    """
    # --- locate the gap as absolute frame indices -------------------------
    gap_frame_start = round(curr_end / SPEECH_PROB_FRAME_DURATION)
    gap_frame_end = round(next_start / SPEECH_PROB_FRAME_DURATION)

    gap_frame_start = max(0, min(gap_frame_start, len(smoothed_probs)))
    gap_frame_end = max(0, min(gap_frame_end, len(smoothed_probs)))

    if gap_frame_start >= gap_frame_end:
        return curr_end, next_start

    # --- contraindication: no silence in gap ------------------------------
    if smoothed_probs[gap_frame_start:gap_frame_end].min() >= silence_threshold:
        return curr_end, next_start

    max_adjust_frames = round(max_adjust_sec / SPEECH_PROB_FRAME_DURATION)
    max_adjust_frames_wide = round(2 * max_adjust_sec / SPEECH_PROB_FRAME_DURATION)

    def _scan_forward(origin_frame: int) -> int:
        """Scan forward from *origin_frame*, stop early when drop is enough."""
        origin_prob = smoothed_probs[origin_frame]
        best_frame, best_prob = origin_frame, origin_prob
        hard_limit = min(origin_frame + max_adjust_frames_wide, len(smoothed_probs))
        soft_limit = min(origin_frame + max_adjust_frames, len(smoothed_probs))
        for f in range(origin_frame, hard_limit):
            prob = smoothed_probs[f]
            if prob < best_prob:
                best_prob, best_frame = prob, f
            if f >= soft_limit and (origin_prob - best_prob) >= min_drop_for_max_adjust:
                break
        return best_frame

    def _scan_backward(origin_frame: int) -> int:
        """Scan backward from *origin_frame*, stop early when drop is enough."""
        origin_prob = smoothed_probs[origin_frame]
        best_frame, best_prob = origin_frame, origin_prob
        hard_limit = max(origin_frame - max_adjust_frames_wide, 0)
        soft_limit = max(origin_frame - max_adjust_frames, 0)
        for f in range(origin_frame, hard_limit - 1, -1):
            prob = smoothed_probs[f]
            if prob < best_prob:
                best_prob, best_frame = prob, f
            if f <= soft_limit and (origin_prob - best_prob) >= min_drop_for_max_adjust:
                break
        return best_frame

    # --- push curr_end forward --------------------------------------------
    new_curr_end   = _scan_forward(gap_frame_start)  * SPEECH_PROB_FRAME_DURATION

    # --- pull next_start backward -----------------------------------------
    new_next_start = _scan_backward(gap_frame_end - 1) * SPEECH_PROB_FRAME_DURATION

    # --- clamp against original boundaries --------------------------------
    # The frame-index round-trip can introduce sub-frame floating-point error
    # that moves a boundary in the wrong direction (e.g. new_curr_end slightly
    # less than curr_end, which would violate curr.start <= curr.end when the
    # segment is very short).  Clamp here to guarantee monotonicity:
    #   • curr_end may only move forward (increase)
    #   • next_start may only move backward (decrease)
    new_curr_end   = max(new_curr_end,   curr_end)
    new_next_start = min(new_next_start, next_start)

    # --- contraindication: overlap → snap to midpoint ---------------------
    if new_curr_end >= new_next_start:
        mid = (new_curr_end + new_next_start) / 2.0
        new_curr_end, new_next_start = mid, mid

    return new_curr_end, new_next_start


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------


def refine_segments_for_session(
    session_dir: pathlib.Path,
    *,
    min_gap_to_adjust: float = DEFAULT_MIN_GAP_TO_ADJUST,
    max_words_to_merge: int = DEFAULT_MAX_WORDS_TO_MERGE,
    force: bool = False,
) -> bool:
    """Refine segment boundaries for one session.

    Reads ``transcript.aligned.json`` (which is never modified) and
    ``speech_probs.frame``, applies boundary adjustments, and writes the
    result to ``transcript.refined.json``.

    Returns ``True`` on success, ``False`` on skip / failure.
    """
    session_id = session_dir.name
    transcript_path = session_dir / ALIGNED_TRANSCRIPT_FILENAME
    refined_path = session_dir / REFINED_TRANSCRIPT_FILENAME
    vad_path = session_dir / VAD_SPEECH_PROBS_FILENAME

    # Skip if already done (unless forced).
    if refined_path.exists() and not force:
        logger.info("Session %s: refined transcript already exists; skipping.", session_id)
        return True

    if not transcript_path.exists():
        logger.warning(
            "Session %s: %s not found; skipping refine-segments.",
            session_id,
            ALIGNED_TRANSCRIPT_FILENAME,
        )
        return False

    if not vad_path.exists():
        logger.warning(
            "Session %s: %s not found; skipping refine-segments.",
            session_id,
            VAD_SPEECH_PROBS_FILENAME,
        )
        return False

    # Load inputs.
    result = WhisperResult(str(transcript_path)).merge_by_gap(min_gap_to_adjust, max_words=max_words_to_merge)
    speech_probs = load_frame_vad_probs(str(vad_path))
    smoothed_probs = smooth_speech_probs(speech_probs)

    segments = list(result.segments)
    adjustments = 0

    for i in range(len(segments) - 1):
        curr = segments[i]
        nxt = segments[i + 1]

        gap = nxt.start - curr.end
        if gap <= min_gap_to_adjust:
            continue

        new_curr_end, new_next_start = adjust_gap(
            curr.end,
            nxt.start,
            smoothed_probs,
        )

        if new_curr_end != curr.end or new_next_start != nxt.start:
            curr.end = new_curr_end
            nxt.start = new_next_start
            adjustments += 1

    result.save_as_json(str(refined_path))

    logger.info(
        "Session %s: refine-segments complete — %d adjustment(s) applied.",
        session_id,
        adjustments,
    )
    return True


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def refine_segments_sessions(
    input_folder: pathlib.Path,
    *,
    force: bool = False,
    session_ids: Optional[list[str]] = None,
    abort_on_error: bool = False,
    min_gap_to_adjust: float = DEFAULT_MIN_GAP_TO_ADJUST,
    workers: int = 4,
) -> None:
    """Run the refine-segments stage for all sessions under *input_folder*.

    This stage is CPU/storage-bound, so it uses a thread pool to parallelise
    across sessions.
    """
    if not input_folder.is_dir():
        logger.warning("Input folder %s does not exist.", input_folder)
        return

    # A session is eligible when it has transcript.aligned.json (the input
    # that refine-segments reads from — never modified) and the VAD output.
    # The completion indicator is transcript.refined.json; whether it already
    # exists is checked per-session inside refine_segments_for_session.
    # Sessions with a skipped.flag are excluded entirely.
    session_dirs = sorted(
        d
        for d in input_folder.iterdir()
        if d.is_dir()
        and not (d / SKIPPED_FLAG_FILENAME).exists()
        and (d / ALIGNED_TRANSCRIPT_FILENAME).exists()
        and (d / VAD_SPEECH_PROBS_FILENAME).exists()
    )

    if session_ids:
        wanted = set(session_ids)
        session_dirs = [d for d in session_dirs if d.name in wanted]

    if not session_dirs:
        logger.info("No sessions eligible for refine-segments.")
        return

    print(f"Refining segment boundaries for {len(session_dirs)} session(s) " f"with {workers} worker(s)...")

    progress = tqdm(total=len(session_dirs), desc="Refining segments")

    def _process(sd: pathlib.Path) -> tuple[pathlib.Path, bool, Optional[str]]:
        try:
            ok = refine_segments_for_session(
                sd,
                min_gap_to_adjust=min_gap_to_adjust,
                force=force,
            )
            return sd, ok, None
        except Exception as e:
            return sd, False, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, sd): sd for sd in session_dirs}
        for future in as_completed(futures):
            sd, ok, err = future.result()
            progress.update(1)
            if err is not None:
                msg = f" - ERROR: refine-segments failed for {sd.name}: {err}"
                tqdm.write(msg)
                logger.error(msg)
                if abort_on_error:
                    # Cancel remaining futures and raise.
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(msg)
            elif not ok:
                tqdm.write(f" - WARNING: refine-segments skipped/failed for {sd.name}")
                if abort_on_error:
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(f"refine-segments failed for {sd.name}")

    progress.close()


__all__ = [
    "add_refine_segments_args",
    "refine_segments_sessions",
    "smooth_speech_probs",
    "adjust_gap",
    "DEFAULT_MIN_GAP_TO_ADJUST",
]
