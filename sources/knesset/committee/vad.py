"""VAD (Voice Activity Detection) step for Knesset committee sessions.

After the normalize stage, this step generates frame-level VAD predictions
for each session's audio file.  The predictions are stored as
``speech_probs.frame`` directly inside the session directory, i.e. as a
sibling of the audio file (``<output_dir>/<session_id>/speech_probs.frame``).

These VAD predictions will later be consumed by the segment-adjustment step
to refine segment start/end times (and possibly merge/split segments).

Public entry points:

* :func:`add_vad_args`       -- adds CLI flags to an ``argparse`` parser.
* :func:`vad_sessions`       -- batch entry point for all sessions under
  an output directory.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from tqdm import tqdm

from vad.definitions import VAD_SPEECH_PROBS_FILENAME
from vad.frame_vad_infer import generate_frame_vad_predictions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def add_vad_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-vad",
        action="store_true",
        help="Skip the VAD (voice activity detection) stage.",
    )
    parser.add_argument(
        "--force-vad",
        action="store_true",
        help="Force re-generation of VAD predictions even if they already exist.",
    )
    parser.add_argument(
        "--vad-pretranscode-workers",
        type=int,
        default=1,
        help="Number of CPU threads for pre-transcoding audio to mono 16 kHz WAV before VAD.",
    )
    parser.add_argument(
        "--vad-presplit-workers",
        type=int,
        default=1,
        help="Number of CPU workers for pre-splitting long audio files before VAD.",
    )
    parser.add_argument(
        "--vad-presplit-max-duration",
        type=int,
        default=400,
        help="Max audio duration (seconds) before splitting into chunks for VAD processing.",
    )
    parser.add_argument(
        "--vad-chunk-size",
        type=int,
        default=200,
        help="Number of audio files to send to the VAD model per chunk.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_audio_file(session_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the audio file inside *session_dir*, or ``None``."""
    return next(session_dir.glob("audio.*"), None)


def _has_vad_output(session_dir: pathlib.Path) -> bool:
    """Check whether a VAD prediction file already exists for this session.

    With sibling_mode the file is placed directly in the session directory
    alongside the audio file.
    """
    return (session_dir / VAD_SPEECH_PROBS_FILENAME).exists()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def vad_sessions(
    output_dir: pathlib.Path,
    *,
    force: bool = False,
    session_ids: list[str] | None = None,
    abort_on_error: bool = False,
    pretranscode_workers: int = 1,
    presplit_workers: int = 1,
    presplit_max_duration: int = 400,
    chunk_size: int = 200,
) -> None:
    """Generate frame-level VAD predictions for committee sessions.

    Parameters
    ----------
    output_dir:
        Root directory that contains one sub-folder per session.
    force:
        When ``True``, regenerate VAD even if the output file exists.
    session_ids:
        If given, restrict processing to these session IDs.
    abort_on_error:
        Raise on the first error instead of skipping.
    pretranscode_workers:
        CPU threads for ffmpeg pre-transcoding.
    presplit_workers:
        CPU workers for audio pre-splitting.
    presplit_max_duration:
        Max seconds per audio chunk sent to the model.
    chunk_size:
        How many files to batch into a single VAD model invocation.
    """

    # Discover sessions that have audio and (optionally) filter by ID.
    session_dirs: list[pathlib.Path] = sorted(
        d for d in output_dir.iterdir() if d.is_dir()
    )
    if session_ids:
        wanted = set(session_ids)
        session_dirs = [d for d in session_dirs if d.name in wanted]

    # Collect audio files that need VAD processing.
    audio_files: list[str] = []
    for sd in session_dirs:
        audio = _find_audio_file(sd)
        if audio is None:
            continue
        if not force and _has_vad_output(sd):
            tqdm.write(f" - Skipping VAD for session {sd.name}: already exists")
            continue
        audio_files.append(str(audio))

    if not audio_files:
        print("No audio files require VAD processing.")
        return

    print(f"Running VAD on {len(audio_files)} audio file(s)...")

    config = {
        "force_reprocess": force,
        "nemo_vad_pretranscode_workers": pretranscode_workers,
        "nemo_vad_presplit_workers": presplit_workers,
        "nemo_vad_presplit_duration": presplit_max_duration,
    }

    # Process in chunks to limit temp storage and ease recovery.
    for i in range(0, len(audio_files), chunk_size):
        chunk = audio_files[i : i + chunk_size]
        print(f"Processing VAD chunk: {len(chunk)} file(s) (starting at index {i})")
        try:
            generate_frame_vad_predictions(chunk, str(output_dir), config, sibling_mode=True)
        except Exception as e:
            msg = f"VAD processing failed for chunk starting at index {i}: {e}"
            logger.error(msg)
            tqdm.write(f" - ERROR: {msg}")
            if abort_on_error:
                raise
