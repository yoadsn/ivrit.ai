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


def _load_sessions() -> None:
    for entry in sorted(DATA_DIR.iterdir()):
        if not entry.is_dir():
            continue
        transcript_path = _find_transcript(entry)
        sessions.append(
            {
                "session_id": entry.name,
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
            entry["words"] = [
                {
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
                for w in raw_words
            ]
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
    --segment-height: 52px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  header select {
    font-size: 14px;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    min-width: 200px;
  }
  #status { font-size: 13px; color: var(--text-muted); }
  .player-bar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    position: fixed;
    top: 50px;
    left: 0;
    right: 0;
    z-index: 99;
    display: none;
  }
  .player-bar audio { width: 100%; max-width: 800px; }
  .player-bar .info { font-size: 13px; color: var(--text-muted); margin-top: 6px; }
  #scroll-container {
    position: fixed;
    top: 130px;
    bottom: 0;
    left: 0;
    right: 0;
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
  }
  .segment .text .word:hover {
    background: var(--primary-light);
  }
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
<header>
  <h1>Knesset Committee Preview</h1>
  <select id="session-select"><option value="">-- Select session --</option></select>
  <span id="status"></span>
</header>
<div class="player-bar" id="player-bar">
  <audio id="audio" controls preload="metadata"></audio>
  <div class="info" id="player-info"></div>
</div>
<div id="scroll-container">
  <div id="virtual-list"></div>
</div>
<script>
const $select = document.getElementById('session-select');
const $status = document.getElementById('status');
const $playerBar = document.getElementById('player-bar');
const $audio = document.getElementById('audio');
const $playerInfo = document.getElementById('player-info');
const $scrollContainer = document.getElementById('scroll-container');
const $virtualList = document.getElementById('virtual-list');

let segments = [];
let segmentHeights = [];
let segmentTops = [];
let totalHeight = 0;
let activeSegIdx = -1;
let activeWordIdx = -1;
let renderedRange = { start: -1, end: -1 };
let segmentElements = new Map();

const BUFFER = 10;
const DEFAULT_HEIGHT = 54;
const GAP = 4;

// -- Init --
fetch('/api/sessions').then(r => r.json()).then(data => {
  data.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.session_id;
    opt.textContent = s.session_id;
    if (!s.has_transcript) opt.textContent += ' (no transcript)';
    $select.appendChild(opt);
  });
});

$select.addEventListener('change', () => {
  const sid = $select.value;
  if (!sid) return;
  loadSession(sid);
});

async function loadSession(sid) {
  $status.textContent = 'Loading...';
  segments = [];
  segmentHeights = [];
  segmentTops = [];
  totalHeight = 0;
  activeSegIdx = -1;
  activeWordIdx = -1;
  renderedRange = { start: -1, end: -1 };
  segmentElements.clear();
  $virtualList.innerHTML = '';

  $audio.src = '/audio/' + sid;
  $audio.load();
  $playerBar.style.display = 'block';

  try {
    const resp = await fetch('/api/session/' + sid + '/transcript');
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    segments = data.segments;
    $playerInfo.textContent =
      data.source_file + ' | ' + data.segment_count + ' segments' +
      (Object.keys(data.speakers).length ? ' | ' + Object.keys(data.speakers).length + ' speakers' : '');
    initVirtualList();
    $status.textContent = '';
  } catch (e) {
    $status.textContent = 'Error: ' + e.message;
    $virtualList.innerHTML = '<div class="empty-state">Failed to load transcript</div>';
  }
}

function initVirtualList() {
  // Estimate heights (all same initially, refine on render)
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
  let start = 0, end = segments.length - 1;
  // Binary search for start
  let lo = 0, hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (segmentTops[mid] + segmentHeights[mid] < scrollTop) lo = mid + 1;
    else hi = mid;
  }
  start = Math.max(0, lo - BUFFER);
  // Find end
  lo = start; hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (segmentTops[mid] <= scrollTop + viewHeight) lo = mid;
    else hi = mid - 1;
  }
  end = Math.min(segments.length - 1, lo + BUFFER);
  return { start, end };
}

function renderVisible() {
  const range = getVisibleRange();
  if (range.start === renderedRange.start && range.end === renderedRange.end) return;

  // Remove out-of-range
  for (const [idx, el] of segmentElements) {
    if (idx < range.start || idx > range.end) {
      el.remove();
      segmentElements.delete(idx);
    }
  }

  // Add in-range
  for (let i = range.start; i <= range.end; i++) {
    if (!segmentElements.has(i)) {
      const el = createSegmentEl(i);
      $virtualList.appendChild(el);
      segmentElements.set(i, el);
      // Measure actual height
      const h = el.offsetHeight;
      if (h !== segmentHeights[i]) {
        segmentHeights[i] = h;
      }
    }
  }

  // Recompute tops if heights changed
  computeTops();
  $virtualList.style.height = totalHeight + 'px';

  // Reposition all rendered
  for (const [idx, el] of segmentElements) {
    el.style.top = segmentTops[idx] + 'px';
  }

  renderedRange = range;
}

function createSegmentEl(i) {
  const seg = segments[i];
  const el = document.createElement('div');
  el.className = 'segment' + (i === activeSegIdx ? ' active' : '');
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

  // If words exist, create spans for each word
  if (seg.words && seg.words.length) {
    seg.words.forEach((w, wi) => {
      const span = document.createElement('span');
      span.className = 'word';
      span.textContent = w.word;
      span.dataset.wi = wi;
      if (i === activeSegIdx && wi === activeWordIdx) {
        span.classList.add('word-active');
      }
      span.addEventListener('click', (e) => {
        e.stopPropagation();
        $audio.currentTime = w.start;
        $audio.play();
      });
      textEl.appendChild(span);
    });
  } else {
    textEl.textContent = seg.text;
  }
  el.appendChild(textEl);

  // Click on segment (outside words) seeks to segment start
  el.addEventListener('click', () => {
    $audio.currentTime = seg.start;
    $audio.play();
  });

  return el;
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 10);
  return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0') + '.' + ms;
}

$scrollContainer.addEventListener('scroll', () => {
  requestAnimationFrame(renderVisible);
});

// -- Audio time update: highlight segment and word --
$audio.addEventListener('timeupdate', () => {
  const t = $audio.currentTime;

  // Find active segment
  let newSegIdx = -1;
  for (let i = 0; i < segments.length; i++) {
    if (t >= segments[i].start && t < segments[i].end) {
      newSegIdx = i;
      break;
    }
  }
  // Fallback: find last segment where t >= start
  if (newSegIdx === -1) {
    for (let i = 0; i < segments.length; i++) {
      if (segments[i].start > t) break;
      newSegIdx = i;
    }
  }

  // Find active word within segment
  let newWordIdx = -1;
  if (newSegIdx >= 0 && segments[newSegIdx].words) {
    const words = segments[newSegIdx].words;
    for (let wi = 0; wi < words.length; wi++) {
      if (t >= words[wi].start && t < words[wi].end) {
        newWordIdx = wi;
        break;
      }
    }
    // Fallback: last word where t >= start
    if (newWordIdx === -1) {
      for (let wi = 0; wi < words.length; wi++) {
        if (words[wi].start > t) break;
        newWordIdx = wi;
      }
    }
  }

  const segChanged = newSegIdx !== activeSegIdx;
  const wordChanged = newWordIdx !== activeWordIdx;

  if (segChanged || wordChanged) {
    // Update old segment element
    if (activeSegIdx >= 0 && segmentElements.has(activeSegIdx)) {
      const oldEl = segmentElements.get(activeSegIdx);
      oldEl.classList.remove('active');
      const oldWord = oldEl.querySelector('.word.word-active');
      if (oldWord) oldWord.classList.remove('word-active');
    }

    activeSegIdx = newSegIdx;
    activeWordIdx = newWordIdx;

    // Update new segment element
    if (activeSegIdx >= 0 && segmentElements.has(activeSegIdx)) {
      const el = segmentElements.get(activeSegIdx);
      el.classList.add('active');
      if (activeWordIdx >= 0) {
        const wordEl = el.querySelector('.word[data-wi="' + activeWordIdx + '"]');
        if (wordEl) wordEl.classList.add('word-active');
      }
      // Scroll into view
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (activeSegIdx >= 0) {
      // Segment not rendered, scroll to bring it into view
      $scrollContainer.scrollTop = segmentTops[activeSegIdx] - 100;
    }
  }
});
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
        reload=False,
    )
