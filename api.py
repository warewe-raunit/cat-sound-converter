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

from engine.audio_io import AudioDecodeError, load_audio, save_audio, validate_sample_rate
from engine.curated_dataset import CuratedDataset
from engine.cat_interpreter import translate_cat_audio
from engine.prosody import analyze
from engine.renderer import render
from engine.tts import ALLOWED_VOICES, DEFAULT_VOICE, TextToSpeechError, speak_text


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


def _validate_sr(sr: int) -> int:
    try:
        return validate_sample_rate(sr)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_tts_rate(rate: int) -> int:
    try:
        rate = int(rate)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="TTS rate must be an integer.") from exc
    if rate < 80 or rate > 260:
        raise HTTPException(status_code=400, detail="TTS rate must be between 80 and 260.")
    return rate


def _render_file(input_path: Path, output_path: Path, dataset: Path, style: str, sr: int) -> dict:
    sr = _validate_sr(sr)
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

    try:
        clips = CuratedDataset(str(dataset), sr=sr)
        output = render(segments, clips, style=style, sr=sr, verbose=False)
        save_audio(str(output_path), output, sr)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "input_seconds": round(len(y) / sr, 3),
        "output_seconds": round(len(output) / sr, 3),
        "segments": len(segments),
    }


def _translate_file(input_path: Path, sr: int) -> dict:
    sr = _validate_sr(sr)
    try:
        y, sr = load_audio(str(input_path), sr=sr)
    except AudioDecodeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    try:
        result = translate_cat_audio(y, sr)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result["segments"] == 0:
        raise HTTPException(
            status_code=422,
            detail="No clear cat vocal segments found. Try a louder or cleaner recording.",
        )
    return result


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
        "features": ["audio_to_cat", "cat_audio_to_human", "cat_audio_to_human_speech"],
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
    <h2>Human audio to cat</h2>
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
    <h2>Cat audio to human (text)</h2>
    <form action="/translate-cat" method="post" enctype="multipart/form-data">
      <label>Cat audio</label>
      <input name="file" type="file" accept="audio/*" required />
      <button type="submit">Translate to Text</button>
    </form>
    <h2>Cat audio to human (voice)</h2>
    <form action="/translate-cat-speech" method="post" enctype="multipart/form-data">
      <label>Cat audio</label>
      <input name="file" type="file" accept="audio/*" required />
      <label>Voice</label>
      <select name="voice">
        <option value="en-US-GuyNeural">Guy</option>
        <option value="en-GB-SoniaNeural">Sonia</option>
      </select>
      <button type="submit">Translate to Voice</button>
    </form>
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
    sr = _validate_sr(sr)
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
    sr = _validate_sr(sr)
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


@app.post("/translate-cat")
async def translate_cat_upload(
    file: UploadFile = File(...),
    sr: int = Form(22050),
) -> dict:
    sr = _validate_sr(sr)
    work_dir = Path(tempfile.mkdtemp(prefix="cat_translate_"))
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    input_path = work_dir / f"input{suffix}"

    try:
        input_path.write_bytes(await file.read())
        return _translate_file(input_path, sr)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/translate-cat-speech")
async def translate_cat_speech_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sr: int = Form(22050),
    voice: str = Form(DEFAULT_VOICE),
    tts_rate: int = Form(150),
) -> FileResponse:
    """Translate cat audio and return the translation as spoken English WAV."""
    sr = _validate_sr(sr)
    tts_rate = _validate_tts_rate(tts_rate)
    if voice not in ALLOWED_VOICES:
        raise HTTPException(
            status_code=400,
            detail="Invalid voice. Use en-US-GuyNeural or en-GB-SoniaNeural.",
        )

    work_dir = Path(tempfile.mkdtemp(prefix="cat_speech_"))
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    input_path = work_dir / f"input{suffix}"
    speech_path = work_dir / "translation_speech.wav"

    try:
        input_path.write_bytes(await file.read())
        result = _translate_file(input_path, sr)
        speak_text(
            result["translation"],
            output_path=str(speech_path),
            rate=tts_rate,
            voice=voice,
        )
    except TextToSpeechError as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    background_tasks.add_task(shutil.rmtree, work_dir, ignore_errors=True)
    headers = {
        "X-Cat-Intent": result["intent"],
        "X-Cat-Confidence": str(result["confidence"]),
        "X-Cat-Translation": result["translation"],
    }
    return FileResponse(
        speech_path,
        media_type="audio/wav",
        filename="translation_speech.wav",
        headers=headers,
        background=background_tasks,
    )
