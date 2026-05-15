#!/usr/bin/env python3
"""Benchmark the CPU-bound pre-alignment pipeline.

Profiles each stage of the alignment pipeline against real session data
and reports wall-clock timings.

Usage
-----
    # Run on all available sessions:
    uv run bench_prealign.py

    # Run on a specific session:
    uv run bench_prealign.py --session-dir data/knesset/committee/2076436
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sources.common.pre_align import (
    WINDOW_SIZE,
    AnchorPoint,
    InaccurateText,
    TextSegment,
    calibrate_threshold,
    detect_preamble_end,
    load_inaccurate_text_from_stable_ts,
    prealign_texts,
    segment_text,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PREALIGN_TRANSCRIPT = "prealign.transcript.json"
RAW_PROTOCOL = "raw.protocol.txt"

DEFAULT_DATA_ROOT = Path("data/knesset/committee")


def discover_sessions(root: Path) -> list[Path]:
    """Return session dirs that have both inputs required for benchmarking."""
    sessions = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / PREALIGN_TRANSCRIPT).exists() and (d / RAW_PROTOCOL).exists():
            sessions.append(d)
    return sessions


def load_inputs(session_dir: Path) -> tuple[str, InaccurateText, Path]:
    """Load the accurate text and inaccurate transcript for a session."""
    accurate_text = (session_dir / RAW_PROTOCOL).read_text(encoding="utf-8")
    trans_path = session_dir / PREALIGN_TRANSCRIPT
    inaccurate = load_inaccurate_text_from_stable_ts(trans_path)
    return accurate_text, inaccurate, trans_path


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Captures both outputs and timings from one pipeline run."""

    anchors: list[AnchorPoint]
    segments: list[TextSegment]
    t_detect_preamble: float
    t_calibrate: float
    t_prealign: float
    t_monotonic_filter: float
    t_segment: float

    @property
    def t_total(self) -> float:
        return (
            self.t_detect_preamble
            + self.t_calibrate
            + self.t_prealign
            + self.t_monotonic_filter
            + self.t_segment
        )


def run_pipeline(
    accurate_text: str,
    inaccurate: InaccurateText,
    trans_path: Path,
) -> PipelineResult:
    """Run the CPU-bound alignment pipeline and capture timings."""

    t0 = time.perf_counter()
    inacc_start_offset = detect_preamble_end(accurate_text, inaccurate)
    t_detect_preamble = time.perf_counter() - t0

    t0 = time.perf_counter()
    threshold = calibrate_threshold(accurate_text, inaccurate, WINDOW_SIZE)
    t_calibrate = time.perf_counter() - t0

    t0 = time.perf_counter()
    anchors = prealign_texts(
        accurate_text,
        inaccurate,
        threshold=threshold,
        inacc_start_offset=inacc_start_offset,
    )
    t_prealign = time.perf_counter() - t0

    t0 = time.perf_counter()
    if anchors:
        clean = [anchors[0]]
        for a in anchors[1:]:
            if a.timestamp_s >= clean[-1].timestamp_s:
                clean.append(a)
        anchors = clean
    t_monotonic = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments = segment_text(accurate_text, anchors, trans_path, inaccurate=inaccurate)
    t_segment = time.perf_counter() - t0

    return PipelineResult(
        anchors=anchors,
        segments=segments,
        t_detect_preamble=t_detect_preamble,
        t_calibrate=t_calibrate,
        t_prealign=t_prealign,
        t_monotonic_filter=t_monotonic,
        t_segment=t_segment,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def print_session_report(
    session_name: str,
    accurate_len: int,
    inaccurate_len: int,
    result: PipelineResult,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Session: {session_name}")
    print(f"  Accurate text:   {accurate_len:,} chars")
    print(f"  Inaccurate text: {inaccurate_len:,} chars")
    print(f"  Ratio:           {inaccurate_len / accurate_len:.2f}" if accurate_len else "  Ratio: N/A")
    print(f"  Anchors:         {len(result.anchors)}")
    print(f"  Segments:        {len(result.segments)}")

    print(f"\n  Timing breakdown:")
    print(f"  {'Function':<25} {'Time':>10}")
    print(f"  {'-' * 37}")

    rows = [
        ("detect_preamble_end", "t_detect_preamble"),
        ("calibrate_threshold", "t_calibrate"),
        ("prealign_texts", "t_prealign"),
        ("monotonic_filter", "t_monotonic_filter"),
        ("segment_text", "t_segment"),
        ("TOTAL", "t_total"),
    ]
    for label, attr in rows:
        val = getattr(result, attr)
        print(f"  {label:<25} {_fmt_time(val):>10}")


def print_summary(results: list[tuple[str, PipelineResult]]) -> None:
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    total = sum(r[1].t_total for r in results)
    print(f"  Sessions: {len(results)}")
    print(f"  Total:    {_fmt_time(total)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the pre-alignment CPU pipeline."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Run on a specific session directory. Default: run on all discovered sessions.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root directory to discover sessions (default: {DEFAULT_DATA_ROOT}).",
    )
    args = parser.parse_args()

    if args.session_dir:
        sessions = [args.session_dir]
    else:
        sessions = discover_sessions(args.data_root)
        if not sessions:
            print(f"No sessions found under {args.data_root}")
            sys.exit(1)

    print(f"Sessions to benchmark: {len(sessions)}")

    all_results: list[tuple[str, PipelineResult]] = []

    for session_dir in sessions:
        session_name = session_dir.name
        print(f"\nLoading {session_name}...")
        accurate_text, inaccurate, trans_path = load_inputs(session_dir)
        print(f"  accurate={len(accurate_text):,} chars, inaccurate={len(inaccurate.full_text):,} chars")

        print(f"  Running...", end="", flush=True)
        result = run_pipeline(accurate_text, inaccurate, trans_path)
        print(f" done ({_fmt_time(result.t_total)})")

        print_session_report(
            session_name,
            len(accurate_text),
            len(inaccurate.full_text),
            result,
        )
        all_results.append((session_name, result))

    print_summary(all_results)


if __name__ == "__main__":
    main()
