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
import multiprocessing
import os
import pathlib

from tqdm import tqdm

from vad.definitions import VAD_SPEECH_PROBS_FILENAME
from vad.vad_io import clear_vad_worker_dirs

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
    parser.add_argument(
        "--vad-devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help=(
            "CUDA device IDs to use for VAD (e.g. cuda:0 cuda:1). "
            "Each device runs in a separate process. "
            "Defaults to a single process using whatever device the VAD library selects."
        ),
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
# Process-parallel worker (one process per GPU)
# ---------------------------------------------------------------------------


def _vad_worker(
    audio_files: list[str],
    output_dir: str,
    config: dict,
    device: str | None,
    worker_index: int = 0,
) -> None:
    """Run VAD for a subset of *audio_files* inside a worker process.

    If *device* is given (e.g. ``"cuda:1"``), the CUDA device index is
    extracted and applied via ``CUDA_VISIBLE_DEVICES`` **before** importing
    any CUDA-aware library, so the module-level ``device`` in
    ``vad/frame_vad_infer.py`` naturally picks the correct GPU.

    *worker_index* is appended to the temporary processing directory name so
    that concurrent worker processes do not share ``vad_temp_processing/`` and
    race on manifests, transcoded audio files, or NeMo intermediate outputs.
    Note: with ``sibling_mode=True`` the final ``speech_probs.frame`` files are
    written next to each audio file, so the per-worker ``output_dir`` only
    affects the temp directory location -- not the final outputs.
    """
    if device is not None:
        # ``cuda:N`` -> restrict this process to GPU N.
        # For plain ``"cuda"`` or ``"cpu"`` leave the env var alone.
        if ":" in device:
            gpu_index = device.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

    # Give each worker its own temp subdirectory so concurrent processes don't
    # collide on vad_temp_processing/ (manifests, transcoded WAVs, NeMo
    # intermediate files, frame predictions, and the final rmtree).
    worker_output_dir = os.path.join(output_dir, f"_vad_worker_{worker_index}")

    # Import *after* setting CUDA_VISIBLE_DEVICES so the VAD library sees it.
    from vad.frame_vad_infer import generate_frame_vad_predictions  # noqa: PLC0415

    chunk_size = config.get("_chunk_size", 200)
    for i in range(0, len(audio_files), chunk_size):
        chunk = audio_files[i : i + chunk_size]
        print(
            f"[{device or 'default'}] Processing VAD chunk: "
            f"{len(chunk)} file(s) (starting at index {i})"
        )
        generate_frame_vad_predictions(chunk, worker_output_dir, config, sibling_mode=True)


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
    devices: list[str] | None = None,
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
    devices:
        CUDA device IDs (e.g. ``["cuda:0", "cuda:1"]``).  When more than one
        device is supplied, audio files are split evenly across processes --
        one process per device -- each with ``CUDA_VISIBLE_DEVICES`` set so
        the VAD library targets the right GPU.  ``None`` / single-device
        falls back to single-process behaviour.
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

    logger.info(f"Running VAD on {len(audio_files)} audio file(s)...")

    config = {
        "force_reprocess": force,
        "nemo_vad_pretranscode_workers": pretranscode_workers,
        "nemo_vad_presplit_workers": presplit_workers,
        "nemo_vad_presplit_duration": presplit_max_duration,
        # Pass chunk_size into workers so they can iterate correctly.
        "_chunk_size": chunk_size,
    }

    # ------------------------------------------------------------------
    # Single-device path (no parallelism needed).
    # ------------------------------------------------------------------
    if not devices or len(devices) <= 1:
        device = devices[0] if devices else None
        try:
            _vad_worker(audio_files, str(output_dir), config, device, worker_index=0)
        except Exception as e:
            msg = f"VAD processing failed: {e}"
            logger.error(msg)
            tqdm.write(f" - ERROR: {msg}")
            if abort_on_error:
                raise
        finally:
            clear_vad_worker_dirs(str(output_dir))
        return

    # ------------------------------------------------------------------
    # Multi-device path: one process per device.
    # ------------------------------------------------------------------
    num_devices = len(devices)
    # Partition files as evenly as possible across devices.
    partitions: list[list[str]] = [[] for _ in range(num_devices)]
    for idx, f in enumerate(audio_files):
        partitions[idx % num_devices].append(f)

    logger.info(
        f"Distributing VAD across {num_devices} device(s): "
        + ", ".join(
            f"{dev} ({len(p)} file(s))" for dev, p in zip(devices, partitions)
        )
    )

    ctx = multiprocessing.get_context("spawn")
    processes: list[multiprocessing.Process] = []
    for worker_index, (device, partition) in enumerate(zip(devices, partitions)):
        if not partition:
            continue
        p = ctx.Process(
            target=_vad_worker,
            args=(partition, str(output_dir), config, device, worker_index),
            daemon=False,
        )
        p.start()
        processes.append(p)

    errors: list[str] = []
    for p in processes:
        p.join()
        if p.exitcode != 0:
            msg = f"VAD worker for process PID {p.pid} exited with code {p.exitcode}"
            logger.error(msg)
            errors.append(msg)

    clear_vad_worker_dirs(str(output_dir))

    if errors:
        combined = "; ".join(errors)
        tqdm.write(f" - ERROR: {combined}")
        if abort_on_error:
            raise RuntimeError(combined)
