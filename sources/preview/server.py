# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi[standard]",
# ]
# ///
"""
Knesset Committee Data Preview Server

Read-only preview of the artifacts produced by the
``sources/knesset/committee/`` pipeline.  Each subdirectory under the
data directory is treated as a session.

Usage:
    uv run sources/preview/server.py [--data-dir DIR] [--audio-filename NAME]
                                     [--host HOST] [--port PORT]
"""

import argparse
import json
import logging
import struct
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preview")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preview-server",
        description="Knesset Committee Data Preview Server",
        add_help=False,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "knesset" / "committee"),
        help="Root data directory (default: data/knesset/committee)",
    )
    parser.add_argument(
        "--audio-filename",
        type=str,
        default="audio.m4a",
        help="Audio filename inside each session dir (default: audio.m4a)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-h", "--help", action="store_true")
    ns, _ = parser.parse_known_args()
    if ns.help:
        parser.print_help()
        sys.exit(0)
    return ns


_cli = _parse_cli_args()

DATA_DIR = Path(_cli.data_dir).resolve()
AUDIO_FILENAME = _cli.audio_filename

if not DATA_DIR.is_dir():
    raise SystemExit(f"Data directory does not exist: {DATA_DIR}")

log.info("Data directory : %s", DATA_DIR)
log.info("Audio filename : %s", AUDIO_FILENAME)

# ---------------------------------------------------------------------------
# Transcript fallback chain
# ---------------------------------------------------------------------------
_TRANSCRIPT_CANDIDATES = [
    "transcript.aligned.json",
    "transcript.json",
    "prealign.transcript.json",
]


def _find_transcript(session_dir: Path) -> Path | None:
    for name in _TRANSCRIPT_CANDIDATES:
        p = session_dir / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------
sessions: list[dict] = []


def _read_metadata(session_dir: Path) -> dict:
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_sessions() -> None:
    for entry in sorted(DATA_DIR.iterdir()):
        if not entry.is_dir():
            continue
        transcript_path = _find_transcript(entry)
        meta = _read_metadata(entry)
        # Parse date to ISO YYYY-MM-DD
        raw_date = meta.get("session_date", "")
        session_date = raw_date[:10] if raw_date else ""
        # Duration in seconds -> minutes (rounded)
        raw_duration = meta.get("duration")
        duration_min = round(raw_duration / 60) if raw_duration else None
        sessions.append(
            {
                "session_id": entry.name,
                "session_date": session_date,
                "duration_minutes": duration_min,
                "has_audio": (entry / AUDIO_FILENAME).exists(),
                "has_transcript": transcript_path is not None,
                "transcript_file": transcript_path.name if transcript_path else None,
                "has_speakers": (entry / "speakers.txt").exists(),
                "has_map": (entry / "transcript.aligned.map.json").exists(),
            }
        )
    log.info("Discovered %d sessions in %s", len(sessions), DATA_DIR)


def _get_session_dir(session_id: str) -> Path:
    d = DATA_DIR / session_id
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return d


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_sessions()
    yield


app = FastAPI(title="Knesset Committee Preview", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
_AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}


@app.get("/audio/{session_id}")
def serve_audio(session_id: str):
    d = _get_session_dir(session_id)
    path = d / AUDIO_FILENAME
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = _AUDIO_MEDIA_TYPES.get(
        path.suffix.lower(), "application/octet-stream"
    )
    return FileResponse(path, media_type=media_type)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/sessions")
def list_sessions():
    return sessions


@app.get("/api/session/{session_id}/transcript")
def get_transcript(session_id: str):
    """Return transcript segments with timing and optional speaker info.

    Resolves the transcript using the fallback chain:
      transcript.aligned.json -> transcript.json -> prealign.transcript.json

    If transcript.aligned.map.json and speakers.txt exist, each segment
    is enriched with ``speaker_ids`` and ``speaker_names``.
    """
    d = _get_session_dir(session_id)
    transcript_path = _find_transcript(d)
    if transcript_path is None:
        raise HTTPException(status_code=404, detail="No transcript file found")

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    # Try to load speaker map
    map_path = d / "transcript.aligned.map.json"
    speakers_path = d / "speakers.txt"
    speaker_map: dict[str, str] = {}
    segment_maps: list[dict] | None = None

    if speakers_path.exists():
        for line in speakers_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                speaker_map[parts[0]] = parts[1]

    if map_path.exists() and transcript_path.name == "transcript.aligned.json":
        segment_maps = json.loads(map_path.read_text(encoding="utf-8"))

    entries = []
    for i, seg in enumerate(segments):
        entry = {
            "index": i,
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "text": seg.get("text", ""),
        }
        # Include word-level timing when available
        raw_words = seg.get("words")
        if raw_words:
            words_out = []
            for w in raw_words:
                wd = {
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
                if "probability" in w:
                    wd["probability"] = float(w["probability"])
                words_out.append(wd)
            entry["words"] = words_out
        # Attach speaker info from map if available
        if segment_maps is not None and i < len(segment_maps):
            m = segment_maps[i]
            sids = m.get("speaker_ids", [])
            entry["speaker_ids"] = sids
            entry["speaker_names"] = [
                speaker_map.get(str(sid), f"Unknown ({sid})") for sid in sids
            ]
        entries.append(entry)

    return {
        "source_file": transcript_path.name,
        "segment_count": len(entries),
        "speakers": speaker_map,
        "segments": entries,
    }


@app.get("/api/session/{session_id}/speakers")
def get_speakers(session_id: str):
    """Return speaker id -> name map."""
    d = _get_session_dir(session_id)
    path = d / "speakers.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="speakers.txt not found")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


# ---------------------------------------------------------------------------
# Peaks – waveform data for the frontend
# ---------------------------------------------------------------------------
_peaks_cache: dict[str, dict] = {}

# Target number of peaks across the full audio
_PEAKS_TARGET = 4000


def _generate_peaks(audio_path: Path, num_peaks: int = _PEAKS_TARGET) -> dict:
    """Use ffmpeg to decode audio to mono 8kHz s16le, then downsample to *num_peaks* min/max pairs."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    fmt = json.loads(result.stdout).get("format", {})
    duration = float(fmt.get("duration", 0))
    if duration <= 0:
        raise HTTPException(status_code=500, detail="Cannot determine audio duration")

    sample_rate = 8000
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v", "quiet",
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-",
        ],
        capture_output=True,
    )
    raw = proc.stdout
    n_samples = len(raw) // 2
    if n_samples == 0:
        raise HTTPException(status_code=500, detail="Failed to decode audio")

    samples = struct.unpack(f"<{n_samples}h", raw)
    chunk = max(1, n_samples // num_peaks)
    peaks = []
    for i in range(0, n_samples, chunk):
        block = samples[i : i + chunk]
        peaks.append(round(max(block) / 32768, 4))
        peaks.append(round(min(block) / 32768, 4))

    return {"peaks": peaks, "duration": duration}


def _generate_zoomed_peaks(
    audio_path: Path, start: float, end: float, num_peaks: int = 800
) -> dict:
    """Generate high-resolution peaks for a specific time window using ffmpeg -ss/-t."""
    window_dur = end - start
    if window_dur <= 0:
        return {"peaks": [], "duration": 0, "start": start, "end": end}

    sample_rate = 16000  # higher rate for zoomed view
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v", "quiet",
            "-ss", str(start),
            "-t", str(window_dur),
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-",
        ],
        capture_output=True,
    )
    raw = proc.stdout
    n_samples = len(raw) // 2
    if n_samples == 0:
        return {"peaks": [], "duration": window_dur, "start": start, "end": end}

    samples = struct.unpack(f"<{n_samples}h", raw)
    chunk = max(1, n_samples // num_peaks)
    peaks = []
    for i in range(0, n_samples, chunk):
        block = samples[i : i + chunk]
        peaks.append(round(max(block) / 32768, 4))
        peaks.append(round(min(block) / 32768, 4))

    return {"peaks": peaks, "duration": window_dur, "start": start, "end": end}


@app.get("/api/session/{session_id}/peaks")
def get_peaks(session_id: str):
    """Return pre-computed waveform peaks for WaveSurfer rendering."""
    if session_id in _peaks_cache:
        return _peaks_cache[session_id]

    d = _get_session_dir(session_id)
    path = d / AUDIO_FILENAME
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    log.info("Generating peaks for %s …", session_id)
    data = _generate_peaks(path)
    _peaks_cache[session_id] = data
    log.info("Generated %d peak values for %s (%.1fs)",
             len(data["peaks"]), session_id, data["duration"])
    return data


@app.get("/api/session/{session_id}/peaks/zoom")
def get_zoomed_peaks(session_id: str, start: float, end: float):
    """Return high-resolution peaks for a time window (for segment zoom)."""
    d = _get_session_dir(session_id)
    path = d / AUDIO_FILENAME
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return _generate_zoomed_peaks(path, start, end)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return HTMLResponse(_INDEX_HTML)


_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knesset Committee Preview</title>
<script src="https://unpkg.com/wavesurfer.js@7"></script>
<script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"></script>
<style>
  :root {
    --bg: #f8f9fa;
    --surface: #fff;
    --border: #dee2e6;
    --text: #212529;
    --text-muted: #6c757d;
    --primary: #0d6efd;
    --primary-light: #e7f1ff;
    --highlight: #fff3cd;
    --speaker-bg: #e9ecef;
    --word-highlight: #ffc107;
    --card-hover: #f0f4ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }

  /* ── Screen management ── */
  .screen { display: none; }
  .screen.active { display: flex; flex-direction: column; height: 100vh; }

  /* ── Session grid screen ── */
  #grid-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    flex-shrink: 0;
  }
  #grid-header h1 { font-size: 18px; font-weight: 600; }
  #grid-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
  }
  .session-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px;
    max-width: 1200px;
    margin: 0 auto;
  }
  .session-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.12s, box-shadow 0.12s;
    overflow: hidden;
  }
  .session-card:hover {
    background: var(--card-hover);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px 8px;
    border-bottom: 1px solid var(--border);
    gap: 8px;
  }
  .card-session-id {
    font-size: 14px;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .card-date {
    font-size: 12px;
    color: var(--text-muted);
    white-space: nowrap;
    font-family: monospace;
  }
  .card-bottom {
    padding: 8px 14px;
    font-size: 12px;
    color: var(--text-muted);
  }

  /* ── Session detail screen ── */
  #detail-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }
  #detail-title {
    font-size: 18px;
    font-weight: 600;
    flex: 1;
  }
  #btn-back {
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 14px;
    cursor: pointer;
    color: var(--text);
    flex-shrink: 0;
  }
  #btn-back:hover { background: var(--bg); }
  #detail-status { font-size: 13px; color: var(--text-muted); }

  /* ── Waveform player bar ── */
  .player-bar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 8px 24px 4px;
    flex-shrink: 0;
  }
  #waveform-container {
    width: 100%;
    cursor: pointer;
  }
  .player-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 4px;
    padding-bottom: 4px;
  }
  .player-controls button {
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 13px;
    cursor: pointer;
    color: var(--text);
  }
  .player-controls button:hover { background: var(--bg); }
  .player-controls button.active {
    background: var(--primary-light);
    border-color: var(--primary);
    color: var(--primary);
  }
  .player-controls .time-display {
    font-size: 13px;
    font-family: monospace;
    color: var(--text-muted);
  }
  .player-controls .info {
    font-size: 12px;
    color: var(--text-muted);
    margin-right: auto;
  }
  .player-controls .zoom-label {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--speaker-bg);
    border-radius: 3px;
    padding: 1px 6px;
  }

  /* ── Segment list ── */
  #scroll-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }
  #virtual-list {
    position: relative;
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
  }
  .segment {
    position: absolute;
    left: 0;
    right: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    cursor: pointer;
    transition: background 0.1s;
  }
  .segment:hover { background: var(--primary-light); }
  .segment.active { background: var(--highlight); border-color: #ffc107; }
  .segment.zoomed { border-color: var(--primary); border-width: 2px; }
  .segment .time {
    font-size: 12px;
    font-family: monospace;
    color: var(--text-muted);
    white-space: nowrap;
    min-width: 100px;
    flex-shrink: 0;
    padding-top: 2px;
  }
  .segment .speaker {
    font-size: 12px;
    background: var(--speaker-bg);
    border-radius: 4px;
    padding: 1px 6px;
    white-space: nowrap;
    min-width: 100px;
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 0;
  }
  .segment .text {
    flex: 1;
    font-size: 14px;
    position: relative;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .segment .text .word {
    cursor: pointer;
    border-radius: 2px;
    border-bottom: 2px solid transparent;
  }
  .segment .text .word:hover { background: var(--primary-light); }
  .segment .text .word.word-active {
    text-decoration: underline;
    text-decoration-color: var(--word-highlight);
    text-decoration-thickness: 3px;
    text-underline-offset: 2px;
  }
  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
  }
</style>
</head>
<body>

<!-- ── Screen 1: session grid ── -->
<div id="screen-grid" class="screen active">
  <div id="grid-header"><h1>Knesset Committee Preview</h1></div>
  <div id="grid-body">
    <div class="session-grid" id="session-grid"></div>
  </div>
</div>

<!-- ── Screen 2: session detail ── -->
<div id="screen-detail" class="screen">
  <div id="detail-header">
    <button id="btn-back">&#x2715;</button>
    <span id="detail-title"></span>
    <span id="detail-status"></span>
  </div>
  <div class="player-bar" id="player-bar">
    <audio id="audio" preload="metadata" style="display:none"></audio>
    <div id="waveform-container"></div>
    <div class="player-controls">
      <button id="btn-play">Play</button>
      <span class="time-display" id="time-display">00:00.0 / 00:00.0</span>
      <span class="info" id="player-info"></span>
      <span class="zoom-label" id="zoom-label">Full view</span>
      <button id="btn-zoom-out" style="display:none">Zoom out</button>
    </div>
  </div>
  <div id="scroll-container">
    <div id="virtual-list"></div>
  </div>
</div>

<script>
// ── DOM refs ──
const $screenGrid      = document.getElementById('screen-grid');
const $screenDetail    = document.getElementById('screen-detail');
const $sessionGrid     = document.getElementById('session-grid');
const $detailTitle     = document.getElementById('detail-title');
const $detailStatus    = document.getElementById('detail-status');
const $audio           = document.getElementById('audio');
const $waveContainer   = document.getElementById('waveform-container');
const $btnPlay         = document.getElementById('btn-play');
const $timeDisplay     = document.getElementById('time-display');
const $playerInfo      = document.getElementById('player-info');
const $zoomLabel       = document.getElementById('zoom-label');
const $btnZoomOut      = document.getElementById('btn-zoom-out');
const $scrollContainer = document.getElementById('scroll-container');
const $virtualList     = document.getElementById('virtual-list');

function showScreen(name) {
  $screenGrid.classList.toggle('active', name === 'grid');
  $screenDetail.classList.toggle('active', name === 'detail');
}

// ── State ──
let segments = [];
let segmentHeights = [];
let segmentTops = [];
let totalHeight = 0;
let activeSegIdx = -1;
let activeWordIdx = -1;
let renderedRange = { start: -1, end: -1 };
let segmentElements = new Map();
let currentSessionId = null;

// WaveSurfer state
let ws = null;
let wsRegions = null;
let peaksData = null;   // { peaks: [...], duration: N }
let zoomedSegIdx = -1;  // which segment is zoomed, -1 = full view

const BUFFER = 10;
const DEFAULT_HEIGHT = 54;
const GAP = 4;
const ZOOM_PAD = 3; // seconds of padding around segment

// ── Load session list ──
fetch('/api/sessions').then(r => r.json()).then(data => {
  data.forEach(s => {
    const card = document.createElement('div');
    card.className = 'session-card';

    const top = document.createElement('div');
    top.className = 'card-top';

    const idEl = document.createElement('div');
    idEl.className = 'card-session-id';
    idEl.textContent = s.session_id;
    idEl.title = s.session_id;

    const dateEl = document.createElement('div');
    dateEl.className = 'card-date';
    dateEl.textContent = s.session_date || '\\u2014';

    top.appendChild(idEl);
    top.appendChild(dateEl);

    const bottom = document.createElement('div');
    bottom.className = 'card-bottom';
    const dur = s.duration_minutes != null ? s.duration_minutes + ' min' : 'unknown duration';
    bottom.textContent = dur;

    card.appendChild(top);
    card.appendChild(bottom);

    card.addEventListener('click', () => openSession(s.session_id));
    $sessionGrid.appendChild(card);
  });
});

// ── Format helpers ──
function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 10);
  return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0') + '.' + ms;
}
function probColor(p) {
  const r = Math.round(0xFD + (0x3C - 0xFD) * p);
  const g = Math.round(0x1D + (0xB4 - 0x1D) * p);
  const b = Math.round(0x1D + (0x3A - 0x1D) * p);
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

// ── WaveSurfer lifecycle ──
// WaveSurfer is used purely for visual rendering (peaks + regions).
// A separate <audio> element handles all playback.
// viewOffset tracks the time offset when zoomed (0 in full view).
let viewOffset = 0;
let viewDuration = 0;
let _regionStopHandler = null;

function destroyWaveSurfer() {
  if (ws) {
    ws.destroy();
    ws = null;
    wsRegions = null;
  }
  zoomedSegIdx = -1;
  viewOffset = 0;
}

function _createWS(peaks, dur) {
  // Destroy previous instance
  if (ws) { ws.destroy(); ws = null; wsRegions = null; }

  const regions = WaveSurfer.Regions.create();
  wsRegions = regions;

  ws = WaveSurfer.create({
    container: $waveContainer,
    waveColor: '#b0c4de',
    progressColor: '#4682b4',
    height: 90,
    cursorColor: '#333',
    cursorWidth: 1,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    interact: true,
    hideScrollbar: true,
    peaks: [peaks],
    duration: dur,
    plugins: [regions],
  });

  viewDuration = dur;

  // Click waveform -> seek $audio and play
  ws.on('interaction', (localTime) => {
    const realTime = viewOffset + localTime;
    $audio.currentTime = realTime;
    $audio.play();
    $btnPlay.textContent = 'Pause';
  });

  // Region click -> play just that word's time range
  regions.on('region-clicked', (region, e) => {
    e.stopPropagation();
    const realStart = viewOffset + region.start;
    const realEnd = viewOffset + region.end;
    $audio.currentTime = realStart;
    $audio.play();
    $btnPlay.textContent = 'Pause';
    // Stop at region end
    if (_regionStopHandler) {
      $audio.removeEventListener('timeupdate', _regionStopHandler);
    }
    _regionStopHandler = () => {
      if ($audio.currentTime >= realEnd) {
        $audio.pause();
        $btnPlay.textContent = 'Play';
        $audio.removeEventListener('timeupdate', _regionStopHandler);
        _regionStopHandler = null;
      }
    };
    $audio.addEventListener('timeupdate', _regionStopHandler);
  });

  return ws;
}

function createFullViewWS() {
  if (!peaksData) return;
  viewOffset = 0;
  _createWS(peaksData.peaks, peaksData.duration);
}

async function zoomToSegment(segIdx) {
  if (!peaksData) return;
  const seg = segments[segIdx];
  if (!seg) return;

  const duration = peaksData.duration;
  const viewStart = Math.max(0, seg.start - ZOOM_PAD);
  const viewEnd = Math.min(duration, seg.end + ZOOM_PAD);
  const windowDur = viewEnd - viewStart;

  // Fetch high-resolution peaks for this window from the server
  $zoomLabel.textContent = 'Loading...';
  let zoomedPeaks;
  try {
    const resp = await fetch(
      '/api/session/' + currentSessionId + '/peaks/zoom?start=' + viewStart + '&end=' + viewEnd
    );
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    zoomedPeaks = data.peaks;
  } catch (e) {
    console.error('Failed to load zoomed peaks:', e);
    // Fallback: slice from global peaks
    const allPeaks = peaksData.peaks;
    const pairsTotal = allPeaks.length / 2;
    const startPair = Math.floor((viewStart / duration) * pairsTotal);
    const endPair = Math.ceil((viewEnd / duration) * pairsTotal);
    zoomedPeaks = allPeaks.slice(startPair * 2, endPair * 2);
  }

  viewOffset = viewStart;
  _createWS(zoomedPeaks, windowDur);

  // Add word regions AFTER WaveSurfer is ready (regions plugin needs it)
  ws.once('ready', () => {
    if (seg.words && seg.words.length) {
      const colors = [
        'rgba(13, 110, 253, 0.15)',
        'rgba(13, 110, 253, 0.25)',
      ];
      seg.words.forEach((w, wi) => {
        const region = wsRegions.addRegion({
          start: w.start - viewStart,
          end: w.end - viewStart,
          color: colors[wi % 2],
          content: w.word,
          drag: false,
          resize: false,
        });
        if (region.element) {
          region.element.style.fontSize = '10px';
          region.element.style.overflow = 'hidden';
          region.element.style.cursor = 'pointer';
          region.element.title = w.word + ' [' + w.start.toFixed(3) + ' - ' + w.end.toFixed(3) + ']';
        }
      });
    }
  });

  zoomedSegIdx = segIdx;
  $zoomLabel.textContent = 'Segment ' + segIdx + ' (' + fmtTime(seg.start) + ' - ' + fmtTime(seg.end) + ')';
  $btnZoomOut.style.display = '';

  // Update segment highlight
  for (const [idx, el] of segmentElements) {
    el.classList.toggle('zoomed', idx === segIdx);
  }
}

function zoomOut() {
  if (!peaksData) return;

  createFullViewWS();

  zoomedSegIdx = -1;
  $zoomLabel.textContent = 'Full view';
  $btnZoomOut.style.display = 'none';

  for (const [idx, el] of segmentElements) {
    el.classList.remove('zoomed');
  }
}

// ── Open session ──
async function openSession(sid) {
  $detailTitle.textContent = sid;
  $detailStatus.textContent = 'Loading...';
  showScreen('detail');

  destroyWaveSurfer();
  currentSessionId = sid;
  segments = [];
  segmentHeights = [];
  segmentTops = [];
  totalHeight = 0;
  activeSegIdx = -1;
  activeWordIdx = -1;
  renderedRange = { start: -1, end: -1 };
  segmentElements.clear();
  $virtualList.innerHTML = '';
  $scrollContainer.scrollTop = 0;
  peaksData = null;
  $zoomLabel.textContent = 'Full view';
  $btnZoomOut.style.display = 'none';

  try {
    // Fetch transcript and peaks in parallel
    const [transcriptResp, peaksResp] = await Promise.all([
      fetch('/api/session/' + sid + '/transcript'),
      fetch('/api/session/' + sid + '/peaks'),
    ]);
    if (!transcriptResp.ok) throw new Error(await transcriptResp.text());
    if (!peaksResp.ok) throw new Error('Failed to load peaks: ' + await peaksResp.text());

    const data = await transcriptResp.json();
    peaksData = await peaksResp.json();

    segments = data.segments;
    $playerInfo.textContent =
      data.source_file + ' | ' + data.segment_count + ' segments' +
      (Object.keys(data.speakers).length ? ' | ' + Object.keys(data.speakers).length + ' speakers' : '');

    // Set up audio element and create WaveSurfer with pre-computed peaks
    $audio.src = '/audio/' + sid;
    $audio.load();
    createFullViewWS();

    initVirtualList();
    $detailStatus.textContent = '';
  } catch (e) {
    $detailStatus.textContent = 'Error: ' + e.message;
    $virtualList.innerHTML = '<div class="empty-state">Failed to load: ' + e.message + '</div>';
  }
}

// ── Back to grid ──
function goBack() {
  $audio.pause();
  $audio.removeAttribute('src');
  destroyWaveSurfer();
  showScreen('grid');
}

document.getElementById('btn-back').addEventListener('click', goBack);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && $screenDetail.classList.contains('active')) goBack();
});

// ── Player controls ──
$btnPlay.addEventListener('click', () => {
  if ($audio.paused) {
    $audio.play();
    $btnPlay.textContent = 'Pause';
  } else {
    $audio.pause();
    $btnPlay.textContent = 'Play';
  }
});
$btnZoomOut.addEventListener('click', zoomOut);

// ── Audio timeupdate: sync WaveSurfer cursor + time display + active segment ──
$audio.addEventListener('timeupdate', () => {
  const t = $audio.currentTime;
  const dur = peaksData ? peaksData.duration : ($audio.duration || 0);
  $timeDisplay.textContent = fmtTime(t) + ' / ' + fmtTime(dur);

  // Update WaveSurfer cursor position
  if (ws && viewDuration > 0) {
    const localTime = t - viewOffset;
    // Only update if within the visible window
    if (localTime >= 0 && localTime <= viewDuration) {
      ws.setTime(localTime);
    }
  }

  updateActiveSegment(t);
});
$audio.addEventListener('play', () => { $btnPlay.textContent = 'Pause'; });
$audio.addEventListener('pause', () => { $btnPlay.textContent = 'Play'; });

// ── Virtual list ──
function initVirtualList() {
  segmentHeights = segments.map(() => DEFAULT_HEIGHT);
  computeTops();
  $virtualList.style.height = totalHeight + 'px';
  renderVisible();
}

function computeTops() {
  segmentTops = [];
  let y = 0;
  for (let i = 0; i < segments.length; i++) {
    segmentTops.push(y);
    y += segmentHeights[i] + GAP;
  }
  totalHeight = y;
}

function getVisibleRange() {
  const scrollTop = $scrollContainer.scrollTop;
  const viewHeight = $scrollContainer.clientHeight;
  let lo = 0, hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (segmentTops[mid] + segmentHeights[mid] < scrollTop) lo = mid + 1;
    else hi = mid;
  }
  const start = Math.max(0, lo - BUFFER);
  lo = start; hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (segmentTops[mid] <= scrollTop + viewHeight) lo = mid;
    else hi = mid - 1;
  }
  return { start, end: Math.min(segments.length - 1, lo + BUFFER) };
}

function renderVisible() {
  const range = getVisibleRange();
  if (range.start === renderedRange.start && range.end === renderedRange.end) return;

  for (const [idx, el] of segmentElements) {
    if (idx < range.start || idx > range.end) {
      el.remove();
      segmentElements.delete(idx);
    }
  }
  for (let i = range.start; i <= range.end; i++) {
    if (!segmentElements.has(i)) {
      const el = createSegmentEl(i);
      $virtualList.appendChild(el);
      segmentElements.set(i, el);
      const h = el.offsetHeight;
      if (h !== segmentHeights[i]) segmentHeights[i] = h;
    }
  }
  computeTops();
  $virtualList.style.height = totalHeight + 'px';
  for (const [idx, el] of segmentElements) {
    el.style.top = segmentTops[idx] + 'px';
  }
  renderedRange = range;
}

function createSegmentEl(i) {
  const seg = segments[i];
  const el = document.createElement('div');
  el.className = 'segment' + (i === activeSegIdx ? ' active' : '') + (i === zoomedSegIdx ? ' zoomed' : '');
  el.dataset.idx = i;
  el.style.top = segmentTops[i] + 'px';

  const timeEl = document.createElement('div');
  timeEl.className = 'time';
  timeEl.textContent = fmtTime(seg.start) + ' - ' + fmtTime(seg.end);
  el.appendChild(timeEl);

  if (seg.speaker_names && seg.speaker_names.length) {
    const spkEl = document.createElement('div');
    spkEl.className = 'speaker';
    spkEl.textContent = seg.speaker_names.join(', ');
    spkEl.title = seg.speaker_names.join(', ');
    el.appendChild(spkEl);
  }

  const textEl = document.createElement('div');
  textEl.className = 'text';
  if (seg.words && seg.words.length) {
    seg.words.forEach((w, wi) => {
      const span = document.createElement('span');
      span.className = 'word';
      span.textContent = w.word;
      span.dataset.wi = wi;
      if (w.probability != null) span.style.borderBottomColor = probColor(w.probability);
      if (i === activeSegIdx && wi === activeWordIdx) span.classList.add('word-active');
      span.addEventListener('click', e => {
        e.stopPropagation();
        if (ws) { ws.setTime(w.start); ws.play(); }
        zoomToSegment(i);
      });
      textEl.appendChild(span);
    });
  } else {
    textEl.textContent = seg.text;
  }
  el.appendChild(textEl);

  // Click segment -> zoom waveform to it and play
  el.addEventListener('click', () => {
    if (ws) { ws.setTime(seg.start); ws.play(); }
    zoomToSegment(i);
  });

  return el;
}

$scrollContainer.addEventListener('scroll', () => requestAnimationFrame(renderVisible));

// ── Active segment tracking ──
function updateActiveSegment(t) {
  let newSegIdx = -1;
  for (let i = 0; i < segments.length; i++) {
    if (t >= segments[i].start && t < segments[i].end) { newSegIdx = i; break; }
  }
  if (newSegIdx === -1) {
    for (let i = 0; i < segments.length; i++) {
      if (segments[i].start > t) break;
      newSegIdx = i;
    }
  }

  let newWordIdx = -1;
  if (newSegIdx >= 0 && segments[newSegIdx].words) {
    const words = segments[newSegIdx].words;
    for (let wi = 0; wi < words.length; wi++) {
      if (t >= words[wi].start && t < words[wi].end) { newWordIdx = wi; break; }
    }
    if (newWordIdx === -1) {
      for (let wi = 0; wi < words.length; wi++) {
        if (words[wi].start > t) break;
        newWordIdx = wi;
      }
    }
  }

  if (newSegIdx !== activeSegIdx || newWordIdx !== activeWordIdx) {
    if (activeSegIdx >= 0 && segmentElements.has(activeSegIdx)) {
      const oldEl = segmentElements.get(activeSegIdx);
      oldEl.classList.remove('active');
      const oldWord = oldEl.querySelector('.word.word-active');
      if (oldWord) oldWord.classList.remove('word-active');
    }
    activeSegIdx = newSegIdx;
    activeWordIdx = newWordIdx;
    if (activeSegIdx >= 0 && segmentElements.has(activeSegIdx)) {
      const el = segmentElements.get(activeSegIdx);
      el.classList.add('active');
      if (activeWordIdx >= 0) {
        const wordEl = el.querySelector('.word[data-wi="' + activeWordIdx + '"]');
        if (wordEl) wordEl.classList.add('word-active');
      }
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (activeSegIdx >= 0) {
      $scrollContainer.scrollTop = segmentTops[activeSegIdx] - 100;
    }
  }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=_cli.host,
        port=_cli.port,
        reload=True,
    )
