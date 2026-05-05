"""
Tiny FastAPI test server for the cat vocalization engine.

Run:
    uvicorn api:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from engine.audio_io import AudioDecodeError, load_audio, save_audio
from engine.curated_dataset import CuratedDataset
from engine.prosody import analyze
from engine.renderer import render


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "curated_cat_sounds"
STYLES = {"natural", "expressive", "subtle"}

app = FastAPI(title="Cat Vocalization Engine Test API")


def _dataset_path(dataset: str | None) -> Path:
    path = Path(dataset).expanduser() if dataset else DEFAULT_DATASET
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {path}")
    if not (path / "metadata.json").exists():
        raise HTTPException(status_code=400, detail=f"metadata.json not found in: {path}")
    return path


def _validate_style(style: str) -> str:
    if style not in STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style '{style}'. Use one of: {', '.join(sorted(STYLES))}",
        )
    return style


def _render_file(input_path: Path, output_path: Path, dataset: Path, style: str, sr: int) -> dict:
    try:
        y, sr = load_audio(str(input_path), sr=sr)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    segments = analyze(y, sr)
    if not segments:
        raise HTTPException(
            status_code=422,
            detail="No vocal segments found. Try a clearer or less silent input.",
        )

    clips = CuratedDataset(str(dataset), sr=sr)
    output = render(segments, clips, style=style, sr=sr, verbose=False)
    save_audio(str(output_path), output, sr)
    return {
        "input_seconds": round(len(y) / sr, 3),
        "output_seconds": round(len(output) / sr, 3),
        "segments": len(segments),
    }


def _write_demo_input(path: Path, sr: int = 22050) -> None:
    gap = np.zeros(int(0.35 * sr), dtype=np.float32)
    chunks = []
    for i, dur in enumerate([0.75, 0.42, 1.15, 0.35]):
        t = np.linspace(0.0, dur, int(dur * sr), endpoint=False)
        sweep = 360 + i * 60
        tone = np.sin(2 * np.pi * (sweep * t + 95 * t * t))
        env = np.minimum(1.0, np.linspace(0, 1, len(t)) * 8)
        env *= np.minimum(1.0, np.linspace(1, 0, len(t)) * 8)
        chunks.append((tone * env * 0.28).astype(np.float32))
        chunks.append(gap)
    sf.write(str(path), np.concatenate(chunks), sr)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "default_dataset": str(DEFAULT_DATASET),
        "dataset_exists": DEFAULT_DATASET.exists(),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Cat Vocalization Engine</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; line-height: 1.45; }
      label { display: block; margin: 16px 0 6px; font-weight: 650; }
      input, select, button { font: inherit; padding: 8px; }
      input[type="text"] { width: 100%; box-sizing: border-box; }
      button { margin-top: 18px; cursor: pointer; }
      .row { margin-top: 24px; }
    </style>
  </head>
  <body>
    <h1>Cat Vocalization Engine</h1>
    <form action="/render" method="post" enctype="multipart/form-data">
      <label>Input audio</label>
      <input name="file" type="file" accept="audio/*" required />

      <label>Style</label>
      <select name="style">
        <option value="natural">natural</option>
        <option value="expressive">expressive</option>
        <option value="subtle">subtle</option>
      </select>

      <label>Dataset path</label>
      <input name="dataset" type="text" value="" placeholder="Leave blank for curated_cat_sounds" />

      <button type="submit">Render WAV</button>
    </form>
    <div class="row">
      <a href="/render-demo?style=natural">Render built-in demo input</a>
    </div>
  </body>
</html>
"""


@app.post("/render")
async def render_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: str = Form("natural"),
    dataset: str | None = Form(None),
    sr: int = Form(22050),
) -> FileResponse:
    style = _validate_style(style)
    dataset_path = _dataset_path(dataset)

    work_dir = Path(tempfile.mkdtemp(prefix="cat_api_"))
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    input_path = work_dir / f"input{suffix}"
    output_path = work_dir / "cat_output.wav"

    try:
        input_path.write_bytes(await file.read())
        stats = _render_file(input_path, output_path, dataset_path, style, sr)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
    headers = {
        "X-Cat-Segments": str(stats["segments"]),
        "X-Input-Seconds": str(stats["input_seconds"]),
        "X-Output-Seconds": str(stats["output_seconds"]),
    }
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="cat_output.wav",
        headers=headers,
        background=background_tasks,
    )


@app.get("/render-demo")
def render_demo(
    background_tasks: BackgroundTasks,
    style: str = "natural",
    dataset: str | None = None,
    sr: int = 22050,
) -> FileResponse:
    style = _validate_style(style)
    dataset_path = _dataset_path(dataset)

    work_dir = Path(tempfile.mkdtemp(prefix="cat_api_demo_"))
    input_path = work_dir / "demo_input.wav"
    output_path = work_dir / "cat_demo_output.wav"

    try:
        _write_demo_input(input_path, sr=sr)
        stats = _render_file(input_path, output_path, dataset_path, style, sr)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
    headers = {
        "X-Cat-Segments": str(stats["segments"]),
        "X-Input-Seconds": str(stats["input_seconds"]),
        "X-Output-Seconds": str(stats["output_seconds"]),
    }
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="cat_demo_output.wav",
        headers=headers,
        background=background_tasks,
    )
