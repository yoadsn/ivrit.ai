import json
import logging
import pathlib
import time
from typing import List, Optional

import torch

from stable_whisper.result import WhisperResult
from alignment.align import align_transcript_to_audio
from alignment.utils import get_breakable_align_model
from sources.common.normalize import (
    DEFAULT_ALIGN_DEVICE_DENSITY,
    DEFAULT_ALIGN_MODEL,
    DEFAULT_FAILURE_THRESHOLD,
    BaseNormalizer,
    add_common_normalize_args as add_normalize_args,
    normalize_entries,
)
from sources.generic.metadata import GenericMetadata
from utils.vtt import vtt_to_whisper_result

logger = logging.getLogger(__name__)


class GenericNormalizer(BaseNormalizer):
    """Normalizer for crowd recital entries."""

    def get_entry_id(self, entry_dir: pathlib.Path) -> str:
        """Get the session ID from the directory name."""
        return entry_dir.name

    def get_audio_file(self, entry_dir: pathlib.Path) -> pathlib.Path:
        """Get the audio file path."""
        # Find the audio file in the entry folder.
        # It starts with "audio" and the extension can be anything
        audio_file = next(entry_dir.glob("audio*"), None)
        if not audio_file:
            raise FileNotFoundError(f"No audio file found in {entry_dir}")
        return audio_file

    def get_input_transcript_file(self, entry_dir: pathlib.Path) -> pathlib.Path:
        # Find the transcript file in the entry folder.
        # It starts with "transcript" and the extension can be anything
        transcript_files = list(entry_dir.glob("transcript*"))
        # ignore any transcript files that are "aligned"
        aligned_transcript_files = [f for f in transcript_files if "aligned" not in f.name]

        transcript_file = aligned_transcript_files[0] if aligned_transcript_files else None
        if not transcript_file:
            raise FileNotFoundError(f"No transcript file found in {entry_dir}")
        return transcript_file

    def read_transcript_file_as_whisper_result(self, transcript_file: pathlib.Path):
        raise NotImplementedError("read_transcript_file_as_whisper_result is not implemented for this normalizer")

    def read_transcript_text_as_whisper_result(self, transcript_file: pathlib.Path, duration: float):
        assert transcript_file.suffix.lower() == ".txt"

        text = transcript_file.read_text()

        return WhisperResult({"segments": [{"start": 0, "end": duration, "text": text}]})

    def get_language(self, metadata: GenericMetadata) -> str:
        """Get the language for the session."""
        doc_lang = metadata.language.lower()
        if doc_lang not in ["he", "yi"]:
            raise ValueError(f"Unsupported language '{doc_lang}'. Only 'he', 'yi' are supported.")
        return doc_lang

    def get_duration(self, metadata: GenericMetadata) -> float:
        """Get the duration from metadata."""
        return metadata.duration

    def load_metadata(self, meta_file: pathlib.Path) -> GenericMetadata:
        """Load session metadata from file."""
        with open(meta_file, "r", encoding="utf-8") as f:
            return GenericMetadata(**json.load(f))

    def save_metadata(self, meta_file: pathlib.Path, metadata: GenericMetadata) -> None:
        """Save session metadata to file."""
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

    def load_model(self):
        """Get the alignment model using get_breakable_align_model."""
        device = self.align_device
        if device == "auto":
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if self.model is None:
            self.model = get_breakable_align_model(
                self.align_model, self.align_device, "int8"  # Using int8 as the default compute type
            )
        return self.model

    def normalize_entry(
        self,
        entry_dir: pathlib.Path,
        force_reprocess: bool = False,
        force_rescore: bool = False,
        abort_on_error: bool = False,
    ) -> bool:
        """
        Override the base normalize_entry method to use align_transcript_to_audio instead of stable_ts directly.

        Args:
            entry_dir: Directory containing the entry files
            force_reprocess: Whether to force reprocessing even if aligned transcript exists
            force_rescore: Whether to force recalculation of quality score

        Returns:
            True if processing was successful, False otherwise
        """
        entry_id = self.get_entry_id(entry_dir)
        from tqdm import tqdm

        # Start timing the normalization process
        start_time = time.time()

        meta_file = entry_dir / "metadata.json"
        aligned_transcript_file = entry_dir / self.aligned_transcript_filename

        need_align = self.should_normalize(meta_file, force_reprocess)
        need_rescore = force_rescore or need_align

        if not (need_align or need_rescore):
            tqdm.write(f" - Skipping entry {entry_id}: already processed")
            return True

        try:
            metadata = self.load_metadata(meta_file)
        except Exception as e:
            tqdm.write(f" - Failed to read metadata.json for entry {entry_id}: {e}")
            return False

        if need_align:

            logger.info(f"Starting normalization of entry {entry_id}")
            tqdm.write(f"Processing entry: {entry_id}")

            # Get language
            language = self.get_language(metadata)

            # Get audio file
            audio_file = self.get_audio_file(entry_dir)
            transcript_file = self.get_input_transcript_file(entry_dir)

            if not audio_file.exists():
                tqdm.write(f" - Skipping entry {entry_id} because required audio file is missing.")
                return False
            if not transcript_file.exists():
                tqdm.write(f" - Skipping entry {entry_id} because required transcript file is missing.")
                return False

            try:
                if transcript_file.suffix.lower() == ".txt":
                    whisper_result = self.read_transcript_text_as_whisper_result(transcript_file, metadata.duration)
                else:
                    whisper_result = self.read_transcript_file_as_whisper_result(transcript_file)
            except Exception as e:
                tqdm.write(f" - Error processing transcript for entry {entry_id}: {e}")
                return False

            try:
                align_result = align_transcript_to_audio(
                    audio_file=audio_file,
                    transcript=whisper_result,
                    model=self.model,
                    language=language,
                    zero_duration_segments_failure_ratio=self.failure_threshold,
                    entry_id=entry_id,
                )
            except Exception as e:
                tqdm.write(f" - Alignment failed for entry {entry_id}: {e}")
                if abort_on_error:
                    raise e
                return False
        else:
            try:
                import stable_whisper

                align_result = stable_whisper.WhisperResult(str(aligned_transcript_file))
            except Exception as e:
                tqdm.write(f" - Failed to load aligned transcript for entry {entry_id}: {e}")
                return False

        # Calculate quality score
        from sources.common.normalize import calculate_quality_score

        quality_score, per_segment_scores = calculate_quality_score(align_result, entry_id)

        # Update metadata with statistics
        self.update_metadata_with_stats(metadata, align_result, quality_score, per_segment_scores)

        # Save updated metadata
        try:
            self.save_metadata(meta_file, metadata)
        except Exception as e:
            tqdm.write(f" - Failed to update metadata.json for entry {entry_id}: {e}")
            return False

        # Save aligned transcript if needed
        if need_align:
            try:
                align_result.save_as_json(str(aligned_transcript_file))
            except Exception as e:
                tqdm.write(f" - Failed to save aligned transcript for entry {entry_id}: {e}")
                return False

        # Calculate processing time and log completion
        end_time = time.time()
        processing_time = end_time - start_time
        entry_duration = self.get_duration(metadata)
        processing_ratio = entry_duration / processing_time if processing_time > 0 else 0

        logger.info(
            f"Entry {entry_id} (duration {entry_duration:.2f} seconds) normalization done. "
            f"Took: {processing_time:.2f} seconds ({processing_ratio:.2f} s/sec). "
            f"Quality Score: {quality_score:.4f}"
        )

        tqdm.write(f" - Processed entry {entry_id}: quality score = {quality_score}")
        return True


def normalize_generic_entries(
    input_folder: pathlib.Path,
    align_model: str = DEFAULT_ALIGN_MODEL,
    align_devices: List[str] = [],
    align_device_density: int = DEFAULT_ALIGN_DEVICE_DENSITY,
    force_normalize_reprocess: bool = False,
    force_rescore: bool = False,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    entry_ids: Optional[List[str]] = None,
    abort_on_error: bool = False,
) -> None:
    """
    Normalize generic entries.

    Args:
        input_folder: Path to the folder containing entry directories
        align_model: Model to use for alignment
        align_devices: List of devices to use for alignment (e.g., ["cuda:0", "cuda:1"]) - this also defines the number of workers
        align_device_density: Number of workers per device
        force_normalize_reprocess: Whether to force reprocessing even if aligned transcript exists
        force_rescore: Whether to force recalculation of quality score
        failure_threshold: Threshold for alignment failure
        entry_ids: Optional list of entry IDs to process (if None, process all)
        abort_on_error: Whether to abort on error
    """

    # Normalize entries
    normalize_entries(
        input_folder=input_folder,
        align_model=align_model,
        align_devices=align_devices,
        align_device_density=align_device_density,
        normalizer_class=GenericNormalizer,
        force_reprocess=force_normalize_reprocess,
        force_rescore=force_rescore,
        failure_threshold=failure_threshold,
        entry_ids=entry_ids,
        abort_on_error=abort_on_error,
    )


__all__ = ["normalize_generic_entries", "add_normalize_args"]
