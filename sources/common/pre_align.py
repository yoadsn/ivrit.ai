"""Common pre-alignment logic shared across sources.

For each session/entry output directory, runs two sub-stages:

  1. **Transcribe** the audio with ``stable_whisper``
     (faster-whisper backend).  The resulting ``WhisperResult`` is saved as
     ``prealign.transcript.json`` -- native stable-ts schema, times in
     seconds.
  2. **Time the accurate text** by aligning it against the inaccurate
     transcription using a Text Fingerprint Vector (TFV) cosine-similarity
     sweep.  Produces:
        * ``transcript.json`` -- a canonical WhisperResult dump; segments
          carry only standard ``start / end / text`` fields.

All timestamps are in **seconds** (matching stable-ts native output).

Public entry points:

* :func:`add_prealign_args` -- adds CLI flags to an ``argparse`` parser.
* :func:`pre_align_sessions` -- batch orchestrator that spawns one worker
  process per device and drains a shared queue.
* Individual helpers for transcription, alignment, and segmentation.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

DEFAULT_PRE_ALIGN_DEVICES = "cuda:0"
DEFAULT_PRE_ALIGN_MODEL = "yoad/whisper-tiny-v2-ct2"
DEFAULT_COMPUTE_TYPE = "int8"


def add_prealign_args(parser: argparse.ArgumentParser) -> None:
    """Add common pre-align CLI flags to *parser*."""
    parser.add_argument(
        "--pre-align-devices",
        type=str,
        default=DEFAULT_PRE_ALIGN_DEVICES,
        help=(
            "Comma-separated list of torch devices for pre-align workers "
            f"(e.g. 'cuda:0,cuda:1'). Default: {DEFAULT_PRE_ALIGN_DEVICES}."
        ),
    )
    parser.add_argument(
        "--pre-align-model-name",
        type=str,
        default=DEFAULT_PRE_ALIGN_MODEL,
        help=("Whisper model name consumed by faster-whisper " f"(default: {DEFAULT_PRE_ALIGN_MODEL})."),
    )
    parser.add_argument(
        "--pre-align-compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        help=f"faster-whisper compute type (default: {DEFAULT_COMPUTE_TYPE}).",
    )
    parser.add_argument(
        "--skip-pre-align",
        action="store_true",
        help="Skip the pre-align stage.",
    )


# ---------------------------------------------------------------------------
# Hebrew TFV -- ported from knesset_committee_data/time_session_text.py
# ---------------------------------------------------------------------------

_BASE_LETTERS = "\u05d0\u05d1\u05d2\u05d3\u05d4\u05d5\u05d6\u05d7\u05d8\u05d9\u05db\u05dc\u05de\u05e0\u05e1\u05e2\u05e4\u05e6\u05e7\u05e8\u05e9\u05ea"
_FINAL_TO_BASE = {"\u05da": "\u05db", "\u05dd": "\u05de", "\u05df": "\u05e0", "\u05e3": "\u05e4", "\u05e5": "\u05e6"}
_DIGITS = "0123456789"
_PRIVATE_BINS = list(_BASE_LETTERS) + list(_DIGITS)
_HOMOPHONE_GROUPS: list[tuple[str, ...]] = [
    ("\u05d0", "\u05e2"),
    ("\u05d1", "\u05d5"),
    ("\u05d8", "\u05ea"),
    ("\u05db", "\u05d7"),
    ("\u05db", "\u05e7"),
    ("\u05e1", "\u05e9"),
]
_GROUP_BIN_LABELS = ["+".join(g) for g in _HOMOPHONE_GROUPS]
_ALL_BIN_LABELS = _PRIVATE_BINS + _GROUP_BIN_LABELS
_NUM_BINS = len(_ALL_BIN_LABELS)

_CHAR_TO_BIN_INDICES: dict[str, list[int]] = {}
for _i, _ch in enumerate(_PRIVATE_BINS):
    _CHAR_TO_BIN_INDICES.setdefault(_ch, []).append(_i)
_GROUP_OFFSET = len(_PRIVATE_BINS)
for _gi, _group in enumerate(_HOMOPHONE_GROUPS):
    for _ch in _group:
        _CHAR_TO_BIN_INDICES.setdefault(_ch, []).append(_GROUP_OFFSET + _gi)


def _normalize_char(ch: str) -> Optional[str]:
    if ch in _FINAL_TO_BASE:
        return _FINAL_TO_BASE[ch]
    if ch in _CHAR_TO_BIN_INDICES:
        return ch
    return None


def compute_tfv(text: str) -> np.ndarray:
    counts = np.zeros(_NUM_BINS, dtype=np.float64)
    for ch in text:
        base = _normalize_char(ch)
        if base is None:
            continue
        for idx in _CHAR_TO_BIN_INDICES[base]:
            counts[idx] += 1.0
    norm = np.linalg.norm(counts)
    if norm < 1e-12:
        return counts
    return counts / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Inaccurate-text loading (stable-ts WhisperResult JSON)
# ---------------------------------------------------------------------------


@dataclass
class InaccurateText:
    """Concatenated inaccurate text with per-char timestamps in seconds."""

    full_text: str
    char_ts: np.ndarray  # shape (len(full_text),), dtype float64, units = seconds


def _segment_char_timestamps(text: str, start: float, end: float) -> tuple[str, np.ndarray]:
    n = len(text)
    if n == 1 or end <= start:
        ts = np.repeat(0.5 * (start + end), n)
    else:
        ts = np.linspace(start, end, n)
    return text, ts


def load_inaccurate_text_from_stable_ts(trans_json_path: Path) -> InaccurateText:
    """Read a stable-ts WhisperResult JSON and build per-char timestamps.

    Uses word-level timestamps when available; falls back to segment-level
    linear interpolation otherwise.  Stable-ts word entries normally
    include a leading space as part of ``word`` -- we preserve it so the
    concatenated string matches spoken spacing.
    """
    data = json.loads(trans_json_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    ts_parts: list[np.ndarray] = []

    for seg in data.get("segments", []):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        words = seg.get("words") or []

        if words:
            for word in words:
                word_text = word.get("word", word.get("text", "")) or ""
                if not word_text:
                    continue
                w_start = float(word.get("start", seg_start))
                w_end = float(word.get("end", seg_end))
                text, ts = _segment_char_timestamps(word_text, w_start, w_end)
                parts.append(text)
                ts_parts.append(ts)
        else:
            seg_text = seg.get("text", "")
            if not seg_text:
                continue
            text, ts = _segment_char_timestamps(seg_text, seg_start, seg_end)
            parts.append(text)
            ts_parts.append(ts)

    full_text = "".join(parts)
    char_ts = np.concatenate(ts_parts) if ts_parts else np.array([], dtype=np.float64)
    return InaccurateText(full_text=full_text, char_ts=char_ts)


# ---------------------------------------------------------------------------
# Dynamic threshold calibration
# ---------------------------------------------------------------------------

_PROBE_NUM_PAIRS = 50
_PROBE_SIGMA_MULT = 2.0
_MIN_THRESHOLD = 0.7


def calibrate_threshold(accurate_text: str, inaccurate: InaccurateText, window_size: int) -> float:
    """Estimate a similarity threshold by sampling random non-corresponding
    window pairs.  Returns ``max(mean + 2*std, 0.7)``."""
    acc_len = len(accurate_text)
    inacc_len = len(inaccurate.full_text)
    w = min(window_size, acc_len, inacc_len)
    if acc_len < w or inacc_len < w:
        return _MIN_THRESHOLD

    ratio = inacc_len / acc_len
    half_inacc = inacc_len // 2
    max_inacc_start = inacc_len - w

    rng = random.Random(42)
    sims: list[float] = []
    for _ in range(_PROBE_NUM_PAIRS):
        acc_pos = rng.randint(0, acc_len - w)
        tfv_acc = compute_tfv(accurate_text[acc_pos : acc_pos + w])
        prop_pos = int(acc_pos * ratio)
        offset = rng.randint(half_inacc, inacc_len - 1)
        inacc_pos = (prop_pos + offset) % (max_inacc_start + 1)
        tfv_inacc = compute_tfv(inaccurate.full_text[inacc_pos : inacc_pos + w])
        sims.append(cosine_similarity(tfv_acc, tfv_inacc))

    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims))
    return max(mean_sim + _PROBE_SIGMA_MULT * std_sim, _MIN_THRESHOLD)


# ---------------------------------------------------------------------------
# Alignment engine
# ---------------------------------------------------------------------------

WINDOW_SIZE = 100
REFINE_WINDOW_SIZE = 40
COARSE_STRIDE_DIV = 5
FINE_STRIDE = 5
REFINE_STRIDE = 3
STEP_RATIO = 0.5
BACKTRACK_FRAC = 0.3


@dataclass
class AnchorPoint:
    char_index: int
    timestamp_s: float
    similarity: float


def _find_best_match(
    target_tfv: np.ndarray,
    inaccurate: InaccurateText,
    window_size: int,
    search_start: int,
    search_end: int,
) -> tuple[int, float]:
    itext = inaccurate.full_text
    n = len(itext)
    search_start = max(0, search_start)
    search_end = min(n - window_size, search_end)
    if search_start > search_end:
        search_start = 0
        search_end = n - window_size
    if search_end < 0:
        return 0, 0.0

    coarse_stride = max(1, window_size // COARSE_STRIDE_DIV)
    best_pos = search_start
    best_sim = -1.0

    pos = search_start
    while pos <= search_end:
        sim = cosine_similarity(target_tfv, compute_tfv(itext[pos : pos + window_size]))
        if sim > best_sim:
            best_sim = sim
            best_pos = pos
        pos += coarse_stride

    fine_start = max(search_start, best_pos - window_size)
    fine_end = min(search_end, best_pos + window_size)
    pos = fine_start
    while pos <= fine_end:
        sim = cosine_similarity(target_tfv, compute_tfv(itext[pos : pos + window_size]))
        if sim > best_sim:
            best_sim = sim
            best_pos = pos
        pos += FINE_STRIDE

    return best_pos, best_sim


def _refine_match(
    accurate_text: str,
    acc_center: int,
    inaccurate: InaccurateText,
    coarse_inacc_pos: int,
    coarse_window: int,
    refine_window: int = REFINE_WINDOW_SIZE,
) -> tuple[int, float]:
    itext = inaccurate.full_text
    acc_len = len(accurate_text)
    inacc_len = len(itext)
    rw = min(refine_window, acc_len, inacc_len)

    acc_start = max(0, acc_center - rw // 2)
    acc_start = min(acc_start, acc_len - rw)
    if acc_start < 0:
        acc_start = 0
    snippet = accurate_text[acc_start : acc_start + rw]
    if len(snippet) < rw:
        coarse_center = coarse_inacc_pos + coarse_window // 2
        return coarse_center, 0.0
    target_tfv = compute_tfv(snippet)

    coarse_center = coarse_inacc_pos + coarse_window // 2
    search_start = max(0, coarse_center - coarse_window)
    search_end = min(inacc_len - rw, coarse_center + coarse_window)
    if search_start > search_end:
        return coarse_center, 0.0

    best_pos = search_start
    best_sim = -1.0
    pos = search_start
    while pos <= search_end:
        sim = cosine_similarity(target_tfv, compute_tfv(itext[pos : pos + rw]))
        if sim > best_sim:
            best_sim = sim
            best_pos = pos
        pos += REFINE_STRIDE

    return best_pos + rw // 2, best_sim


def prealign_texts(
    accurate_text: str,
    inaccurate: InaccurateText,
    window_size: int = WINDOW_SIZE,
    threshold: float = _MIN_THRESHOLD,
) -> list[AnchorPoint]:
    anchors: list[AnchorPoint] = []
    acc_len = len(accurate_text)
    inacc_len = len(inaccurate.full_text)
    if acc_len == 0 or inacc_len == 0:
        return anchors

    w = min(window_size, acc_len, inacc_len)
    step = max(1, int(w * STEP_RATIO))
    backtrack = int(w * BACKTRACK_FRAC)
    ratio = inacc_len / acc_len if acc_len > 0 else 1.0

    last_inacc_center = 0
    acc_pos = 0
    while acc_pos + w <= acc_len:
        target_tfv = compute_tfv(accurate_text[acc_pos : acc_pos + w])
        estimated_inacc_pos = int(acc_pos * ratio)
        mono_start = last_inacc_center - backtrack
        prop_start = estimated_inacc_pos - 2 * w
        search_start = min(mono_start, prop_start)
        search_end = max(estimated_inacc_pos + 2 * w, last_inacc_center + 4 * w)

        coarse_pos, sim = _find_best_match(target_tfv, inaccurate, w, search_start, search_end)
        if sim < threshold:
            acc_pos += step
            continue

        acc_center = acc_pos + w // 2
        refined_center, _ = _refine_match(accurate_text, acc_center, inaccurate, coarse_pos, w)
        center_clamped = max(0, min(refined_center, len(inaccurate.char_ts) - 1))
        ts_s = float(inaccurate.char_ts[center_clamped])
        anchors.append(AnchorPoint(char_index=acc_center, timestamp_s=ts_s, similarity=sim))

        coarse_center = coarse_pos + w // 2
        if coarse_center > last_inacc_center - backtrack:
            last_inacc_center = max(last_inacc_center, coarse_center)
        acc_pos += step

    return anchors


def interpolate_timestamp(anchors: list[AnchorPoint], char_index: int) -> float:
    if not anchors:
        return 0.0
    if char_index <= anchors[0].char_index:
        return anchors[0].timestamp_s
    if char_index >= anchors[-1].char_index:
        return anchors[-1].timestamp_s
    lo, hi = 0, len(anchors) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if anchors[mid].char_index <= char_index:
            lo = mid
        else:
            hi = mid
    a, b = anchors[lo], anchors[hi]
    if a.char_index == b.char_index:
        return a.timestamp_s
    frac = (char_index - a.char_index) / (b.char_index - a.char_index)
    return a.timestamp_s + frac * (b.timestamp_s - a.timestamp_s)


def interpolate_timestamp_inverse(anchors: list[AnchorPoint], ts_s: float) -> float:
    if not anchors:
        return 0.0
    if ts_s <= anchors[0].timestamp_s:
        return float(anchors[0].char_index)
    if ts_s >= anchors[-1].timestamp_s:
        return float(anchors[-1].char_index)
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        if a.timestamp_s <= ts_s <= b.timestamp_s:
            dt = b.timestamp_s - a.timestamp_s
            if dt < 1e-6:
                return float(a.char_index)
            frac = (ts_s - a.timestamp_s) / dt
            return a.char_index + frac * (b.char_index - a.char_index)
    return float(anchors[-1].char_index)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

_SNAP_RADIUS = 60
_BOUNDARY_CHARS_PRIMARY = set(".\n?!")
_BOUNDARY_CHARS_SECONDARY = set(",;:")
_MIN_SEGMENT_CHARS = 20

# Minimum number of anchors required per this many characters of accurate text.
# Below this density the alignment is considered unreliable and the entire
# accurate text is emitted as a single zero-duration segment rather than being
# split on degenerate boundaries.
_MIN_ANCHORS_PER_CHARS = 1000

# Acceptable ratio of inaccurate-text length to accurate-text length.
# Outside [_MIN_LENGTH_RATIO, _MAX_LENGTH_RATIO] the inaccurate transcription
# is structurally incompatible with the accurate text (e.g. pure hallucination
# or wholesale silence) and alignment is skipped immediately.
_MIN_LENGTH_RATIO = 0.5
_MAX_LENGTH_RATIO = 2.0


@dataclass
class TextSegment:
    start_char: int
    end_char: int
    start_s: float
    end_s: float
    text: str


def _find_nearest_boundary(text: str, target: int, radius: int = _SNAP_RADIUS) -> Optional[int]:
    lo = max(0, target - radius)
    hi = min(len(text), target + radius)
    best_primary: Optional[int] = None
    best_primary_dist = radius + 1
    best_secondary: Optional[int] = None
    best_secondary_dist = radius + 1
    for i in range(lo, hi):
        ch = text[i]
        dist = abs(i - target)
        if ch in _BOUNDARY_CHARS_PRIMARY:
            if dist < best_primary_dist:
                best_primary = i + 1
                best_primary_dist = dist
        elif ch in _BOUNDARY_CHARS_SECONDARY:
            if dist < best_secondary_dist:
                best_secondary = i + 1
                best_secondary_dist = dist
    return best_primary if best_primary is not None else best_secondary


def _anchors_sufficient(accurate_text: str, anchors: list[AnchorPoint]) -> bool:
    """Return True when anchor density meets the minimum requirement.

    Requires at least one anchor per :data:`_MIN_ANCHORS_PER_CHARS` characters
    of accurate text.  Below this density the alignment is too coarse to
    produce meaningful segment boundaries.
    """
    acc_len = len(accurate_text)
    if acc_len == 0:
        return True
    required = max(1, acc_len // _MIN_ANCHORS_PER_CHARS)
    return len(anchors) >= required


def _single_segment_fallback(
    accurate_text: str,
    anchors: list[AnchorPoint],
) -> list[TextSegment]:
    """Emit the entire accurate text as one segment.

    Used when anchor density is too low to trust boundary placement.  The
    timestamp is taken from the first anchor when available, otherwise 0.0.
    """
    if not accurate_text.strip():
        return []
    ts = anchors[0].timestamp_s if anchors else 0.0
    return [TextSegment(0, len(accurate_text), ts, ts, accurate_text)]


def segment_text(
    accurate_text: str,
    anchors: list[AnchorPoint],
    trans_json_path: Path,
    inaccurate: Optional["InaccurateText"] = None,
) -> list[TextSegment]:
    # Guard 1: length-ratio sanity check.  If the inaccurate text is wildly
    # shorter or longer than the accurate text the transcription is structurally
    # incompatible and alignment is hopeless regardless of anchor count.
    if inaccurate is not None:
        acc_len = len(accurate_text)
        inacc_len = len(inaccurate.full_text)
        if acc_len > 0:
            ratio = inacc_len / acc_len
            if not (_MIN_LENGTH_RATIO <= ratio <= _MAX_LENGTH_RATIO):
                logger.warning(
                    "Inaccurate/accurate text length ratio %.3f is outside "
                    "[%.1f, %.1f]; emitting entire accurate text as a single segment.",
                    ratio, _MIN_LENGTH_RATIO, _MAX_LENGTH_RATIO,
                )
                return _single_segment_fallback(accurate_text, anchors)

    # Guard 2: if we don't have enough anchors to reliably place boundaries,
    # emit the whole text as one segment rather than splitting at spurious
    # positions (which would duplicate content across segments).
    if not _anchors_sufficient(accurate_text, anchors):
        logger.warning(
            "Anchor density too low (%d anchor(s) for %d chars); "
            "emitting entire accurate text as a single segment.",
            len(anchors),
            len(accurate_text),
        )
        return _single_segment_fallback(accurate_text, anchors)

    data = json.loads(trans_json_path.read_text(encoding="utf-8"))
    whisper_segs = data.get("segments", [])

    boundary_ts_set: set[float] = set()
    for seg in whisper_segs:
        boundary_ts_set.add(float(seg.get("start", 0.0)))
        boundary_ts_set.add(float(seg.get("end", 0.0)))
    boundary_ts_list = sorted(boundary_ts_set)

    raw_char_positions = [int(round(interpolate_timestamp_inverse(anchors, ts))) for ts in boundary_ts_list]

    acc_len = len(accurate_text)
    snapped: list[int] = [0]
    seen: set[int] = {0}
    for raw_pos in raw_char_positions:
        if raw_pos <= 0 or raw_pos >= acc_len:
            continue
        boundary = _find_nearest_boundary(accurate_text, raw_pos) or raw_pos
        boundary = max(0, min(boundary, acc_len))
        if boundary not in seen and 0 < boundary < acc_len:
            snapped.append(boundary)
            seen.add(boundary)
    snapped.append(acc_len)
    snapped.sort()

    merged: list[int] = [snapped[0]]
    for pos in snapped[1:]:
        if pos - merged[-1] >= _MIN_SEGMENT_CHARS:
            merged.append(pos)
    if merged[-1] != acc_len:
        merged.append(acc_len)

    segments: list[TextSegment] = []
    for i in range(len(merged) - 1):
        s_char = merged[i]
        e_char = merged[i + 1]
        s_s = interpolate_timestamp(anchors, s_char)
        e_s = interpolate_timestamp(anchors, e_char)
        seg_text = accurate_text[s_char:e_char]
        if not seg_text.strip():
            continue
        segments.append(TextSegment(s_char, e_char, s_s, e_s, seg_text))
    return segments


# ---------------------------------------------------------------------------
# Per-session/entry orchestration
# ---------------------------------------------------------------------------

PREALIGN_TRANSCRIPT_FILENAME = "prealign.transcript.json"
TRANSCRIPT_FILENAME = "transcript.json"


def is_pre_aligned(session_dir: Path) -> bool:
    return (session_dir / TRANSCRIPT_FILENAME).exists()


def find_audio(session_dir: Path) -> Path:
    """Locate the ``audio.*`` file inside *session_dir*."""
    audio_file = next(session_dir.glob("audio.*"), None)
    if audio_file is None:
        raise FileNotFoundError(f"No audio.* file found in {session_dir}")
    return audio_file


def transcribe_session(session_dir: Path, model, language: str) -> Path:
    """Transcribe the session audio via stable_ts and save the raw
    ``WhisperResult`` to ``prealign.transcript.json``."""
    out_path = session_dir / PREALIGN_TRANSCRIPT_FILENAME
    audio_path = find_audio(session_dir)
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        verbose=None,  # suppress per-session tqdm bar; overall progress shown by caller
    )
    result.save_as_json(str(out_path))
    return out_path


def prealign_session(
    session_dir: Path,
    accurate_text_path: Path,
    language: str = "he",
) -> None:
    """Time the accurate text and write ``transcript.json``.

    Parameters
    ----------
    session_dir:
        Directory containing ``prealign.transcript.json`` and audio.
    accurate_text_path:
        Path to the plain-text file with the accurate transcript.
    language:
        Language code for the output WhisperResult.
    """
    trans_path = session_dir / PREALIGN_TRANSCRIPT_FILENAME
    if not accurate_text_path.exists():
        raise FileNotFoundError(f"Accurate text not found: {accurate_text_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Pre-align transcription not found: {trans_path}")

    accurate_text = accurate_text_path.read_text(encoding="utf-8")
    inaccurate = load_inaccurate_text_from_stable_ts(trans_path)
    threshold = calibrate_threshold(accurate_text, inaccurate, WINDOW_SIZE)
    anchors = prealign_texts(accurate_text, inaccurate, threshold=threshold)

    # Drop non-monotonic anchors rather than plateauing them.
    if anchors:
        clean = [anchors[0]]
        for a in anchors[1:]:
            if a.timestamp_s >= clean[-1].timestamp_s:
                clean.append(a)
        anchors = clean

    segments = segment_text(accurate_text, anchors, trans_path, inaccurate=inaccurate)

    # Serialize via stable_whisper so transcript.json is a canonical
    # WhisperResult dump with only standard fields.
    import stable_whisper

    base_data = {
        "language": language,
        "text": accurate_text,
        "segments": [
            {
                "start": round(s.start_s, 4),
                "end": round(s.end_s, 4),
                "text": s.text,
            }
            for s in segments
        ],
    }
    result = stable_whisper.WhisperResult(base_data)
    out_path = session_dir / TRANSCRIPT_FILENAME
    result.save_as_json(str(out_path))


def process_session(
    session_dir: Path,
    model,
    accurate_text_path: Path,
    language: str,
    force: bool = False,
) -> bool:
    """Run the full pre-align pipeline for a single session/entry.

    Parameters
    ----------
    session_dir:
        Directory containing audio and where outputs are written.
    model:
        A loaded stable_whisper model (faster-whisper backend).
    accurate_text_path:
        Path to the plain-text file with the accurate transcript.
    language:
        Language code (e.g. ``"he"``).
    force:
        If ``True``, re-run even if outputs already exist.
    """
    session_id = session_dir.name
    prealign_path = session_dir / PREALIGN_TRANSCRIPT_FILENAME
    transcript_path = session_dir / TRANSCRIPT_FILENAME

    if transcript_path.exists() and not force:
        logger.info("Session %s already pre-aligned; skipping.", session_id)
        return True

    try:
        if not prealign_path.exists() or force:
            logger.info("Session %s: transcribing audio...", session_id)
            transcribe_session(session_dir, model, language=language)
        logger.info("Session %s: timing accurate text...", session_id)
        prealign_session(session_dir, accurate_text_path, language=language)
        return True
    except Exception as exc:
        logger.error("Pre-align failed for session %s: %s", session_id, exc)
        return False


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _device_type(dev: str) -> str:
    return dev.split(":", 1)[0]


def _device_index(dev: str) -> int:
    if ":" in dev:
        try:
            return int(dev.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _worker_main(
    device: str,
    model_name: str,
    compute_type: str,
    language: str,
    force: bool,
    accurate_text_resolver,
    task_queue,
    results_queue,
) -> None:
    """Worker process that loads one model and drains *task_queue*.

    Parameters
    ----------
    accurate_text_resolver:
        A callable ``(session_dir: Path) -> Path`` that returns the path to
        the accurate plain-text transcript for that session directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    worker_logger = logging.getLogger(f"pre_align.{device}")

    import stable_whisper  # imported inside worker so CUDA init is per-process

    init_kwargs = {"device": _device_type(device), "compute_type": compute_type}
    if _device_type(device) == "cuda":
        init_kwargs["device_index"] = _device_index(device)

    try:
        worker_logger.info("Loading model %s on %s (%s)", model_name, device, compute_type)
        model = stable_whisper.load_faster_whisper(model_name, **init_kwargs)
    except Exception as exc:
        worker_logger.error("Failed to load model on %s: %s", device, exc)
        # Drain so the producer doesn't block forever.
        while True:
            item = task_queue.get()
            if item is None:
                break
            results_queue.put((item, False, f"model load error: {exc}"))
        return

    while True:
        item = task_queue.get()
        if item is None:
            break
        session_dir = Path(item)
        try:
            accurate_text_path = accurate_text_resolver(session_dir)
            ok = process_session(
                session_dir,
                model,
                accurate_text_path=accurate_text_path,
                language=language,
                force=force,
            )
            results_queue.put((item, ok, None if ok else "process_session returned False"))
        except Exception as exc:
            worker_logger.exception("Session %s crashed", session_dir.name)
            results_queue.put((item, False, str(exc)))


# ---------------------------------------------------------------------------
# Public batch entry point
# ---------------------------------------------------------------------------


def pre_align_sessions(
    session_dirs: Iterable[Path],
    accurate_text_resolver,
    devices: str = DEFAULT_PRE_ALIGN_DEVICES,
    model_name: str = DEFAULT_PRE_ALIGN_MODEL,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str = "he",
    force: bool = False,
    abort_on_error: bool = False,
) -> dict[str, tuple[bool, Optional[str]]]:
    """Run the pre-align stage in parallel over a batch of session dirs.

    One worker process is spawned per device parsed from the comma-separated
    ``devices`` string.  Workers share a queue of session directory paths.

    Parameters
    ----------
    session_dirs:
        Session directories to process.
    accurate_text_resolver:
        A callable ``(session_dir: Path) -> Path`` that returns the path to
        the accurate plain-text transcript for that session directory.
    devices:
        Comma-separated device list (e.g. ``"cuda:0,cuda:1"``).
    model_name:
        Whisper model name for faster-whisper.
    compute_type:
        faster-whisper compute type.
    language:
        Language code.
    force:
        Re-run even if outputs exist.
    abort_on_error:
        Raise ``RuntimeError`` if any session fails.

    Returns
    -------
    dict[str, tuple[bool, Optional[str]]]
        Mapping ``{str(session_dir): (ok, error_message_or_None)}``.
    """
    device_list = [d.strip() for d in devices.split(",") if d.strip()]
    if not device_list:
        raise ValueError("pre_align_sessions requires at least one device.")

    all_dirs = [Path(d) for d in session_dirs]
    results: dict[str, tuple[bool, Optional[str]]] = {}
    pending: list[Path] = []
    for d in all_dirs:
        if not force and is_pre_aligned(d):
            logger.info("Session %s already pre-aligned; skipping.", d.name)
            results[str(d)] = (True, None)
        else:
            pending.append(d)

    if not pending:
        logger.info("No sessions pending pre-align.")
        return results

    logger.info(
        "Pre-aligning %d session(s) across %d worker(s): %s",
        len(pending),
        len(device_list),
        ",".join(device_list),
    )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    results_queue = ctx.Queue()

    for d in pending:
        task_queue.put(str(d))
    for _ in device_list:
        task_queue.put(None)

    workers = []
    for dev in device_list:
        p = ctx.Process(
            target=_worker_main,
            args=(
                dev,
                model_name,
                compute_type,
                language,
                force,
                accurate_text_resolver,
                task_queue,
                results_queue,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    first_error: Optional[str] = None
    failed = 0
    with tqdm(total=len(pending), unit="session", desc="Pre-aligning") as pbar:
        for _ in range(len(pending)):
            session_dir_str, ok, err = results_queue.get()
            results[session_dir_str] = (ok, err)
            if not ok:
                failed += 1
                tqdm.write(f" - Pre-align failed for {Path(session_dir_str).name}: {err}")
                if first_error is None and abort_on_error:
                    first_error = f"Pre-align failed for {session_dir_str}: {err}"
            pbar.set_postfix(failed=failed)
            pbar.update(1)

    for p in workers:
        p.join()

    if first_error:
        raise RuntimeError(first_error)

    return results
