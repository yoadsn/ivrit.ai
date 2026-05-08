import argparse
import csv
import logging
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler

from tqdm import tqdm

from sources.common.pre_align import (
    add_prealign_args,
)
from sources.common.pre_align import (
    pre_align_sessions as _common_pre_align_sessions,
)
from sources.knesset.committee.create_maps import (
    add_create_maps_args,
    create_maps_sessions,
)
from sources.knesset.committee.extraction import is_extracted, process_protocol
from sources.knesset.committee.manifest import build_manifest
from sources.knesset.committee.metadata import (
    CommitteeMetadata,
    committee_source_id,
    source_type,
)
from sources.knesset.committee.normalize import add_normalize_args, normalize_sessions
from sources.knesset.committee.refine_segments import (
    add_refine_segments_args,
    refine_segments_sessions,
)

RAW_PROTOCOL_FILENAME = "raw.protocol.txt"
from sources.common.definitions import SKIPPED_FLAG_FILENAME
from sources.knesset.committee.s3 import make_s3_client, s3_download, s3_uri_filename
from sources.knesset.committee.vad import add_vad_args, vad_sessions
from utils.audio import get_audio_info
from vad.definitions import VAD_SPEECH_PROBS_FILENAME
from vad.vad_io import is_empty_audio


def _download_to(
    s3_client,
    s3_uri: str,
    dest: pathlib.Path,
    force: bool = False,
) -> pathlib.Path:
    """Download an S3 object to ``dest`` unless already present."""
    if dest.exists() and not force:
        logging.info("Already downloaded: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3_download(s3_client, s3_uri, dest)
    return dest


def ensure_protocol_downloaded(
    s3_client,
    s3_uri: str,
    session_dir: pathlib.Path,
    force: bool = False,
) -> pathlib.Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / s3_uri_filename(s3_uri)
    return _download_to(s3_client, s3_uri, dest, force=force)


def ensure_audio_downloaded(
    s3_client,
    s3_uri: str,
    session_dir: pathlib.Path,
    force: bool = False,
) -> pathlib.Path:
    """Download the audio as ``audio.<ext>`` to align with other sources."""
    session_dir.mkdir(parents=True, exist_ok=True)
    ext = pathlib.Path(s3_uri_filename(s3_uri)).suffix or ".bin"
    dest = session_dir / f"audio{ext}"

    existing = next(session_dir.glob("audio.*"), None)
    if existing and not force:
        logging.info("Audio already present: %s", existing)
        return existing

    # force=True: clear out any older audio with a different extension.
    if force:
        for prev in session_dir.glob("audio.*"):
            if prev != dest:
                prev.unlink()

    return _download_to(s3_client, s3_uri, dest, force=force)


def get_audio_duration(session_dir: pathlib.Path) -> float | None:
    """Return the duration (seconds) of the audio file in ``session_dir``, or None."""
    audio_file = next(session_dir.glob("audio.*"), None)
    if audio_file is None:
        return None
    info = get_audio_info(str(audio_file))
    return info.duration if info is not None else None


def _committee_accurate_text_resolver(session_dir: pathlib.Path) -> pathlib.Path:
    return session_dir / RAW_PROTOCOL_FILENAME


def _is_empty_audio(session_dir: pathlib.Path) -> bool:
    """Return True if the VAD output indicates the audio is effectively silent.

    Returns False when no VAD output exists yet or the file cannot be read.
    """
    vad_file = session_dir / VAD_SPEECH_PROBS_FILENAME
    if not vad_file.exists():
        return False
    try:
        return is_empty_audio(str(vad_file))
    except Exception as exc:
        logging.warning("Could not read VAD output for %s: %s", session_dir.name, exc)
        return False


def _detect_and_flag_empty_audio_sessions(
    output_dir: pathlib.Path,
    session_ids: list[str],
) -> set[str]:
    """Inspect VAD output for every session and write ``skipped.flag`` for
    sessions that contain no meaningful speech.

    Returns the set of session IDs that were flagged so the caller can remove
    them from any further processing lists.
    """
    session_dirs: list[pathlib.Path] = sorted(d for d in output_dir.iterdir() if d.is_dir())
    if session_ids:
        wanted = set(session_ids)
        session_dirs = [d for d in session_dirs if d.name in wanted]

    flagged: set[str] = set()
    for sd in session_dirs:
        if not _is_empty_audio(sd):
            continue
        flag_file = sd / SKIPPED_FLAG_FILENAME
        flag_file.touch()
        flagged.add(sd.name)
        tqdm.write(f" - Empty audio detected, flagging session: {sd.name}")
        logging.info("Empty audio session flagged: %s", sd.name)

    if flagged:
        print(f"Empty audio sessions flagged ({len(flagged)}): {', '.join(sorted(flagged))}")
    return flagged


def pre_align_sessions(session_dirs, **kwargs):
    return _common_pre_align_sessions(
        session_dirs=session_dirs,
        accurate_text_resolver=_committee_accurate_text_resolver,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract Knesset committee session data."
    )
    parser.add_argument(
        "--input-manifest-file",
        type=str,
        required=True,
        help=(
            "Path to the input manifest CSV with columns: "
            "session_id, start_date, audio_file_path, protocol_file_path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory where session folders will be written.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of audio and protocol even if they exist locally.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of protocol artifacts even if outputs exist.",
    )
    parser.add_argument(
        "--force-pre-align",
        action="store_true",
        help="Force re-run of the pre-align stage (transcribe + time accurate text) even if outputs exist.",
    )
    parser.add_argument(
        "--abort-on-error",
        action="store_true",
        help="Abort the run on the first processing error instead of skipping.",
    )
    parser.add_argument(
        "--session-ids",
        type=str,
        nargs="+",
        default=[],
        help="Process only the specified session ids.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Maximum number of sessions to process in this run.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio download (useful when only the protocol text is needed).",
    )
    parser.add_argument(
        "--logs-folder",
        type=str,
        help="Folder to store log files. If not specified, logging is disabled.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Number of parallel workers for downloading and extracting sessions (default: 4).",
    )

    # AWS credential overrides.  When omitted, boto3 falls back to the
    # standard credential chain (env vars, ~/.aws/credentials, IAM role).
    parser.add_argument(
        "--aws-access-key-id",
        type=str,
        default=None,
        help="AWS access key ID (falls back to env / credentials file).",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        type=str,
        default=None,
        help="AWS secret access key (falls back to env / credentials file).",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help="AWS region (falls back to env / config).",
    )

    # Pre-align, normalization, VAD, and create-maps tunables.
    add_prealign_args(parser)
    add_normalize_args(parser)
    add_vad_args(parser)
    add_refine_segments_args(parser)
    add_create_maps_args(parser)
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip the normalize (alignment + quality scoring) stage.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip generating the output manifest CSV.",
    )
    parser.add_argument(
        "--metadata-manifest-file",
        type=str,
        default=None,
        help=(
            "Path to a CSV file (e.g. manifest_metadata.csv) whose 'session_id' column "
            "is used to look up supplemental metadata fields. All columns in this file "
            "are embedded in each session's metadata.json. If not provided, metadata "
            "enrichment is skipped."
        ),
    )

    args = parser.parse_args()

    input_manifest_file = pathlib.Path(args.input_manifest_file)
    if not input_manifest_file.exists() or not input_manifest_file.is_file():
        print(f"Input manifest file '{input_manifest_file}' does not exist or is not a file.")
        return

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging.
    logging.basicConfig(level=logging.CRITICAL + 1)
    if args.logs_folder:
        logs_folder = pathlib.Path(args.logs_folder)
        logs_folder.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        file_handler = RotatingFileHandler(
            logs_folder / "download_log", maxBytes=5 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)
        logging.info("Starting Knesset committee download into %s", output_dir)

    # Parse the manifest.
    expected_columns = {"session_id", "start_date", "audio_file_path", "protocol_file_path"}
    with open(input_manifest_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        if not expected_columns.issubset(header):
            print(
                f"Input manifest file '{input_manifest_file}' is missing required columns. "
                f"Expected: {sorted(expected_columns)}. Found: {sorted(header)}"
            )
            return
        manifest_entries = [row for row in reader]

    # Load supplemental metadata keyed by session_id (optional).
    # All columns from the CSV are stored verbatim; "NULL" and blank values become None.
    session_metadata_lookup: dict[str, dict] = {}
    if args.metadata_manifest_file:
        _meta_path = pathlib.Path(args.metadata_manifest_file)
        if not _meta_path.exists():
            print(f"Metadata manifest file '{_meta_path}' does not exist.")
            return
        with open(_meta_path, newline="", encoding="utf-8") as _f:
            _reader = csv.DictReader(_f)
            for _row in _reader:
                _sid = _row.get("session_id", "")
                if _sid:
                    session_metadata_lookup[_sid] = {
                        k: v.strip() if (v := (_row.get(k) or "").strip()) and v != "NULL" else None
                        for k in (_reader.fieldnames or [])
                        if k != "session_id"
                    }

    if args.session_ids:
        wanted = set(args.session_ids)
        manifest_entries = [e for e in manifest_entries if e["session_id"] in wanted]

    if args.max_sessions is not None:
        manifest_entries = manifest_entries[: args.max_sessions]

    if not manifest_entries:
        logging.info("No manifest entries to process.")
        return

    logging.info("Processing %d sessions.", len(manifest_entries))

    # Each worker thread gets its own S3 client to avoid boto3 thread-safety issues.
    _thread_local = threading.local()

    def _get_s3():
        if not hasattr(_thread_local, "client"):
            _thread_local.client = make_s3_client(
                aws_access_key_id=args.aws_access_key_id,
                aws_secret_access_key=args.aws_secret_access_key,
                aws_region=args.aws_region,
            )
        return _thread_local.client

    def _process_session(entry: dict) -> pathlib.Path | None:
        """Download + extract one session. Returns session dir on success, None on skip/error."""
        session_id = entry["session_id"]
        session_output_dir = output_dir / session_id
        s3 = _get_s3()

        session_output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Download protocol archive ---
        tqdm.write(f" - Downloading protocol for session {session_id}...")
        protocol_path = ensure_protocol_downloaded(
            s3,
            entry["protocol_file_path"],
            session_output_dir,
            force=args.force_download,
        )

        # --- 2. Download audio (optional) ---
        if not args.skip_audio:
            tqdm.write(f" - Downloading audio for session {session_id}...")
            ensure_audio_downloaded(
                s3,
                entry["audio_file_path"],
                session_output_dir,
                force=args.force_download,
            )

        # --- 3. Extract protocol artifacts ---
        tqdm.write(f" - Extracting protocol for session {session_id}...")
        extract_ok = process_protocol(
            protocol_path,
            session_output_dir,
            force_reprocess=args.force_extract,
        )
        if not extract_ok:
            msg = f" - ERROR: extraction failed for session {session_id}."
            tqdm.write(msg)
            logging.warning(msg)
            return None

        if not is_extracted(session_output_dir):
            msg = f" - ERROR: extracted outputs missing for session {session_id}."
            tqdm.write(msg)
            logging.warning(msg)
            return None

        # --- 4. Write session metadata ---
        duration = get_audio_duration(session_output_dir) if not args.skip_audio else None
        _extra = session_metadata_lookup.get(session_id, {})
        session_metadata = CommitteeMetadata(
            source_type=source_type,
            source_id=committee_source_id,
            source_entry_id=session_id,
            session_id=session_id,
            session_date=entry.get("start_date") or None,
            language="he",
            duration=duration,
            **_extra,
        )
        metadata_file = session_output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(session_metadata.model_dump_json(indent=2))

        tqdm.write(f" - Successfully processed session {session_id}")
        return session_output_dir

    ready_session_dirs: list[pathlib.Path] = []

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        future_to_entry = {executor.submit(_process_session, entry): entry for entry in manifest_entries}
        with tqdm(total=len(manifest_entries), desc="Processing sessions") as pbar:
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                session_id = entry["session_id"]
                try:
                    result = future.result()
                    if result is not None:
                        ready_session_dirs.append(result)
                    elif args.abort_on_error:
                        raise RuntimeError(f"Processing failed for session {session_id}")
                except Exception as e:
                    msg = f" - ERROR: Unexpected error processing session {session_id}: {e}"
                    tqdm.write(msg)
                    logging.warning(msg)
                    if args.abort_on_error:
                        raise
                    tqdm.write(" - Skipping to next session")
                finally:
                    pbar.update(1)

    # --- VAD stage (frame-level voice activity detection) ---
    if not args.skip_vad and not args.skip_audio:
        print("Running VAD predictions...")
        vad_sessions(
            output_dir,
            force=args.force_vad or args.force_normalize_reprocess or args.force_pre_align,
            session_ids=args.session_ids,
            abort_on_error=args.abort_on_error,
            pretranscode_workers=args.vad_pretranscode_workers,
            presplit_workers=args.vad_presplit_workers,
            presplit_max_duration=args.vad_presplit_max_duration,
            chunk_size=args.vad_chunk_size,
            devices=args.vad_devices,
        )

        # --- Empty-audio detection (runs right after VAD) ---
        print("Checking for empty/silent audio sessions...")
        flagged_ids = _detect_and_flag_empty_audio_sessions(output_dir, args.session_ids)
        if flagged_ids:
            ready_session_dirs = [d for d in ready_session_dirs if d.name not in flagged_ids]

    # --- Pre-align stage (batch, one worker per device) ---
    if ready_session_dirs and not args.skip_pre_align and not args.skip_audio:
        print(f"Pre-aligning {len(ready_session_dirs)} session(s)...")
        pre_align_sessions(
            ready_session_dirs,
            devices=args.pre_align_devices,
            model_name=args.pre_align_model_name,
            compute_type=args.pre_align_compute_type,
            language="he",
            force=args.force_pre_align,
            abort_on_error=args.abort_on_error,
        )

    # --- Normalize stage (alignment + quality scoring) ---
    if not args.skip_normalize:
        print("Starting normalization process...")
        normalize_sessions(
            output_dir,
            align_model=args.align_model,
            align_devices=args.align_devices or [],
            align_device_density=args.align_device_density,
            force_normalize_reprocess=args.force_normalize_reprocess
            or args.force_pre_align,
            force_rescore=args.force_rescore,
            failure_threshold=args.failure_threshold,
            plenum_ids=args.session_ids,
            abort_on_error=args.abort_on_error,
        )

    # --- Refine segments stage (adjust segment boundaries using VAD) ---
    if not args.skip_refine_segments and not args.skip_audio:
        print("Refining segment boundaries...")
        refine_segments_sessions(
            output_dir,
            force=args.force_refine_segments or args.force_vad or args.force_normalize_reprocess or args.force_pre_align,
            session_ids=args.session_ids,
            abort_on_error=args.abort_on_error,
            min_gap_to_adjust=args.refine_segments_min_gap,
            workers=args.refine_segments_workers,
        )

    # --- Create maps stage (char offsets + speaker IDs for aligned/refined segments) ---
    if not args.skip_create_maps:
        print("Creating transcript maps...")
        create_maps_sessions(
            output_dir,
            force=args.force_create_maps or args.force_refine_segments or args.force_normalize_reprocess or args.force_pre_align,
            session_ids=args.session_ids,
            abort_on_error=args.abort_on_error,
            workers=args.create_maps_workers,
        )

    # --- Build manifest stage ---
    if not args.skip_manifest:
        print("Generating manifest CSV...")
        build_manifest(str(output_dir))


if __name__ == "__main__":
    import sys

    print(
        "This module is not intended to be executed directly. "
        "Please use the top-level download.py.",
        file=sys.stderr,
    )
