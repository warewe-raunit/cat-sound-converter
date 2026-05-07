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
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from engine.audio_io import AudioDecodeError, load_audio, save_audio, validate_sample_rate
from engine.curated_dataset import CuratedDataset
from engine.cat_interpreter import translate_cat_audio
from engine.prosody import analyze
from engine.renderer import render
from engine.tts import ALLOWED_VOICES, DEFAULT_VOICE, TextToSpeechError, speak_text


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "curated_cat_sounds"
STYLES = {"natural", "expressive", "subtle"}

STYLE_OPTIONS = sorted(STYLES)
STYLE_DESCRIPTION = (
    "Rendering style for generated cat vocalizations. "
    "'natural' is balanced, 'expressive' adds stronger pitch/timing movement, "
    "and 'subtle' keeps the output restrained."
)
DATASET_DESCRIPTION = (
    "Optional server-side dataset directory containing metadata.json and curated cat clips. "
    "Leave blank to use the bundled curated_cat_sounds dataset."
)
SAMPLE_RATE_DESCRIPTION = (
    "Target sample rate in Hz used while decoding and processing audio. "
    "Valid range: 8000 to 48000. Recommended value: 22050."
)
HUMAN_AUDIO_DESCRIPTION = (
    "Human speech or vocal audio to convert into a cat-like WAV. "
    "WAV, MP3, M4A, FLAC, and most ffmpeg-readable audio files are intended inputs."
)
CAT_AUDIO_DESCRIPTION = (
    "Cat vocalization audio to interpret. Use a short, clear recording with audible meows, "
    "chirps, trills, or purr-like sounds for best results."
)
VOICE_DESCRIPTION = (
    "English voice used for the spoken translation. "
    "Allowed values: en-US-GuyNeural or en-GB-SoniaNeural."
)
TTS_RATE_DESCRIPTION = (
    "Speech speed in words per minute for the returned translation audio. "
    "Valid range: 80 to 260. Recommended value: 150."
)

API_DESCRIPTION = """
Interactive API for the Cat Vocalization Engine.

Use the endpoints below to:

* Convert human audio into a cat-like WAV file.
* Render a built-in demo input without uploading a file.
* Interpret cat vocalizations as best-effort human-readable text.
* Return that cat-audio interpretation as spoken English WAV audio.

The translation is an acoustic interpretation, not a literal cat language decoder.
For best results, upload short and clear clips with audible vocal segments.
"""

OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "Deployment readiness and basic API capability checks.",
    },
    {
        "name": "Browser UI",
        "description": "Simple HTML upload form for manual testing outside Swagger.",
    },
    {
        "name": "Human to Cat Audio",
        "description": "Render uploaded or demo human audio into cat-like WAV output.",
    },
    {
        "name": "Cat to Human Translation",
        "description": "Interpret cat vocalizations as text or spoken English.",
    },
]


class ErrorResponse(BaseModel):
    detail: str = Field(
        ...,
        description="Human-readable reason the request could not be completed.",
        examples=["Sample rate must be between 8000 and 48000 Hz."],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": "No vocal segments found. Try a clearer or less silent input.",
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    ok: bool = Field(..., description="True when the API process is running.")
    features: list[str] = Field(
        ...,
        description="Feature flags exposed by this build.",
        examples=[["audio_to_cat", "cat_audio_to_human", "cat_audio_to_human_speech"]],
    )
    default_dataset: str = Field(
        ...,
        description="Absolute path to the dataset used when the dataset field is left blank.",
        examples=[str(DEFAULT_DATASET)],
    )
    dataset_exists: bool = Field(
        ...,
        description="True when the bundled dataset directory exists on the server.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ok": True,
                    "features": [
                        "audio_to_cat",
                        "cat_audio_to_human",
                        "cat_audio_to_human_speech",
                    ],
                    "default_dataset": str(DEFAULT_DATASET),
                    "dataset_exists": True,
                }
            ]
        }
    )


class AcousticFeatureResponse(BaseModel):
    intensity: float = Field(
        ...,
        description="Relative loudness of the segment, scaled by the analyzer.",
        examples=[0.642],
    )
    pitch_hz: float = Field(
        ...,
        description="Estimated average pitch for the segment in Hertz.",
        examples=[487.3],
    )
    pitch_direction: str = Field(
        ...,
        description="Pitch contour label such as rising, falling, flat, or varied.",
        examples=["rising"],
    )
    brightness: float = Field(
        ...,
        description="Average spectral centroid in Hertz. Higher values sound brighter.",
        examples=[2218.5],
    )
    roughness: float = Field(
        ...,
        description="Spectral flatness style roughness cue. Higher values sound noisier.",
        examples=[0.0831],
    )
    periodicity: float = Field(
        ...,
        description="Autocorrelation cue for purr-like periodic tone.",
        examples=[0.412],
    )


class SegmentInterpretationResponse(BaseModel):
    index: int = Field(..., description="One-based segment number in the input clip.", examples=[1])
    start_s: float = Field(..., description="Segment start time in seconds.", examples=[0.184])
    end_s: float = Field(..., description="Segment end time in seconds.", examples=[0.932])
    duration: float = Field(..., description="Segment duration in seconds.", examples=[0.748])
    intent: str = Field(
        ...,
        description="Machine intent key selected for this segment.",
        examples=["request"],
    )
    label: str = Field(
        ...,
        description="Human-readable intent label for this segment.",
        examples=["request"],
    )
    translation: str = Field(
        ...,
        description="Plain-English best-effort translation for this segment.",
        examples=["I need something. Please check food, water, the door, or the litter box."],
    )
    confidence: float = Field(
        ...,
        description="Confidence score from 0.35 to 0.90 for this segment interpretation.",
        examples=[0.72],
    )
    signals: list[str] = Field(
        ...,
        description="Acoustic cues that influenced the selected intent.",
        examples=[["moderate-high intensity", "rising pitch"]],
    )
    features: AcousticFeatureResponse = Field(
        ...,
        description="Measured acoustic features used by the interpreter.",
    )


class CatTranslationResponse(BaseModel):
    input_seconds: float = Field(
        ...,
        description="Decoded input duration in seconds.",
        examples=[2.847],
    )
    segments: int = Field(
        ...,
        description="Number of clear cat vocal segments interpreted.",
        examples=[2],
    )
    intent: str = Field(
        ...,
        description="Dominant machine intent key across all interpreted segments.",
        examples=["request"],
    )
    label: str = Field(
        ...,
        description="Human-readable label for the dominant intent.",
        examples=["request"],
    )
    translation: str = Field(
        ...,
        description="Overall best-effort English interpretation.",
        examples=["I need something. Please check food, water, the door, or the litter box."],
    )
    confidence: float = Field(
        ...,
        description="Weighted confidence score for the overall interpretation.",
        examples=[0.7],
    )
    segment_interpretations: list[SegmentInterpretationResponse] = Field(
        ...,
        description="Per-segment timing, intent, confidence, and acoustic cues.",
    )
    note: str = Field(
        ...,
        description="Reminder about interpretation limits.",
        examples=[
            "Best-effort acoustic interpretation. Cat meaning depends on context and body language."
        ],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "input_seconds": 2.847,
                    "segments": 2,
                    "intent": "request",
                    "label": "request",
                    "translation": (
                        "I need something. Please check food, water, the door, "
                        "or the litter box."
                    ),
                    "confidence": 0.7,
                    "segment_interpretations": [
                        {
                            "index": 1,
                            "start_s": 0.184,
                            "end_s": 0.932,
                            "duration": 0.748,
                            "intent": "request",
                            "label": "request",
                            "translation": (
                                "I need something. Please check food, water, "
                                "the door, or the litter box."
                            ),
                            "confidence": 0.72,
                            "signals": ["moderate-high intensity", "rising pitch"],
                            "features": {
                                "intensity": 0.642,
                                "pitch_hz": 487.3,
                                "pitch_direction": "rising",
                                "brightness": 2218.5,
                                "roughness": 0.0831,
                                "periodicity": 0.412,
                            },
                        }
                    ],
                    "note": (
                        "Best-effort acoustic interpretation. Cat meaning depends "
                        "on context and body language."
                    ),
                }
            ]
        }
    )


BAD_REQUEST_RESPONSE = {
    "model": ErrorResponse,
    "description": "Invalid parameter value, missing dataset, or unusable request option.",
    "content": {
        "application/json": {
            "examples": {
                "invalidSampleRate": {
                    "summary": "Sample rate outside the supported DSP range",
                    "value": {"detail": "Sample rate must be between 8000 and 48000 Hz."},
                },
                "missingDataset": {
                    "summary": "Dataset directory is unavailable on the server",
                    "value": {"detail": "metadata.json not found in: curated_cat_sounds"},
                },
            }
        }
    },
}
DECODE_ERROR_RESPONSE = {
    "model": ErrorResponse,
    "description": "Uploaded file could not be decoded as supported audio.",
    "content": {
        "application/json": {
            "examples": {
                "unsupportedAudio": {
                    "summary": "Unsupported or corrupt audio upload",
                    "value": {
                        "detail": "Could not decode audio file 'input.bin'. Try WAV, MP3, M4A, or FLAC."
                    },
                }
            }
        }
    },
}
NO_SEGMENTS_RESPONSE = {
    "model": ErrorResponse,
    "description": (
        "Audio decoded successfully, but no clear vocal segments were found. "
        "FastAPI can also return 422 when a required form field is missing."
    ),
    "content": {
        "application/json": {
            "examples": {
                "noVocalSegments": {
                    "summary": "No clear vocal audio detected",
                    "value": {
                        "detail": "No vocal segments found. Try a clearer or less silent input."
                    },
                }
            }
        }
    },
}
TTS_UNAVAILABLE_RESPONSE = {
    "model": ErrorResponse,
    "description": "Text-to-speech backend failed or timed out while creating the WAV file.",
    "content": {
        "application/json": {
            "examples": {
                "ttsUnavailable": {
                    "summary": "Speech synthesis unavailable",
                    "value": {
                        "detail": (
                            "Text-to-speech is unavailable. Install edge-tts/ffmpeg "
                            "or a working pyttsx3 voice backend."
                        )
                    },
                }
            }
        }
    },
}
RENDER_AUDIO_RESPONSE = {
    "description": "Generated cat-like WAV audio.",
    "content": {
        "audio/wav": {
            "schema": {
                "type": "string",
                "format": "binary",
                "description": "Binary WAV file containing the rendered cat-like audio.",
            }
        }
    },
    "headers": {
        "X-Cat-Segments": {
            "description": "Number of vocal segments detected in the input.",
            "schema": {"type": "integer", "example": 4},
        },
        "X-Input-Seconds": {
            "description": "Decoded input duration in seconds.",
            "schema": {"type": "number", "example": 2.87},
        },
        "X-Output-Seconds": {
            "description": "Generated output duration in seconds.",
            "schema": {"type": "number", "example": 2.41},
        },
    },
}
SPEECH_AUDIO_RESPONSE = {
    "description": "Spoken English translation as a WAV file.",
    "content": {
        "audio/wav": {
            "schema": {
                "type": "string",
                "format": "binary",
                "description": "Binary WAV file containing the spoken translation.",
            }
        }
    },
    "headers": {
        "X-Cat-Intent": {
            "description": "Dominant machine intent key selected by the interpreter.",
            "schema": {"type": "string", "example": "request"},
        },
        "X-Cat-Confidence": {
            "description": "Weighted confidence score for the interpretation.",
            "schema": {"type": "number", "example": 0.7},
        },
        "X-Cat-Translation": {
            "description": "Plain-English translation used to generate the spoken audio.",
            "schema": {
                "type": "string",
                "example": "I need something. Please check food, water, the door, or the litter box.",
            },
        },
    },
}

app = FastAPI(
    title="Cat Vocalization Engine API",
    summary="Render human audio into cat-like vocalizations and interpret cat audio.",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=OPENAPI_TAGS,
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "tryItOutEnabled": True,
        "defaultModelsExpandDepth": 2,
    },
)


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


@app.get(
    "/health",
    tags=["System"],
    summary="Check API health and dataset readiness",
    description=(
        "Use this endpoint for deployment health checks. It confirms the API process is "
        "running and shows whether the bundled default dataset is available."
    ),
    operation_id="checkHealth",
    response_model=HealthResponse,
    response_description="API capability and bundled dataset status.",
)
def health() -> dict:
    return {
        "ok": True,
        "features": ["audio_to_cat", "cat_audio_to_human", "cat_audio_to_human_speech"],
        "default_dataset": str(DEFAULT_DATASET),
        "dataset_exists": DEFAULT_DATASET.exists(),
    }


@app.get(
    "/",
    tags=["Browser UI"],
    summary="Open the manual upload page",
    description=(
        "Returns a small HTML page with forms for the main audio workflows. "
        "This is useful for quick manual testing when Swagger is not convenient."
    ),
    operation_id="openBrowserUploadPage",
    response_class=HTMLResponse,
    response_description="HTML page with upload forms for render and translation workflows.",
    responses={
        200: {
            "content": {
                "text/html": {
                    "schema": {
                        "type": "string",
                        "example": "<!doctype html><html><body><h1>Cat Vocalization Engine</h1></body></html>",
                    }
                }
            }
        }
    },
)
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


@app.post(
    "/render",
    tags=["Human to Cat Audio"],
    summary="Convert uploaded human audio into cat-like WAV",
    description=(
        "Upload a human voice or vocal audio clip and receive a generated cat-like WAV. "
        "The response body is the WAV file. Processing statistics are returned as "
        "X-Cat-Segments, X-Input-Seconds, and X-Output-Seconds response headers."
    ),
    operation_id="renderUploadedAudio",
    response_class=FileResponse,
    responses={
        200: RENDER_AUDIO_RESPONSE,
        400: BAD_REQUEST_RESPONSE,
        415: DECODE_ERROR_RESPONSE,
        422: NO_SEGMENTS_RESPONSE,
    },
)
async def render_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description=HUMAN_AUDIO_DESCRIPTION),
    style: str = Form(
        "natural",
        description=STYLE_DESCRIPTION,
        examples=["natural"],
        json_schema_extra={"enum": STYLE_OPTIONS},
    ),
    dataset: str | None = Form(
        None,
        description=DATASET_DESCRIPTION,
        examples=["curated_cat_sounds"],
    ),
    sr: int = Form(
        22050,
        description=SAMPLE_RATE_DESCRIPTION,
        examples=[22050],
    ),
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


@app.get(
    "/render-demo",
    tags=["Human to Cat Audio"],
    summary="Render a built-in demo clip into cat-like WAV",
    description=(
        "Generates a short synthetic demo input on the server and renders it into "
        "cat-like audio. Use this endpoint to confirm the renderer and dataset work "
        "without uploading an audio file."
    ),
    operation_id="renderDemoAudio",
    response_class=FileResponse,
    responses={
        200: RENDER_AUDIO_RESPONSE,
        400: BAD_REQUEST_RESPONSE,
        415: DECODE_ERROR_RESPONSE,
        422: NO_SEGMENTS_RESPONSE,
    },
)
def render_demo(
    background_tasks: BackgroundTasks,
    style: str = Query(
        "natural",
        description=STYLE_DESCRIPTION,
        examples=["natural"],
        json_schema_extra={"enum": STYLE_OPTIONS},
    ),
    dataset: str | None = Query(
        None,
        description=DATASET_DESCRIPTION,
        examples=["curated_cat_sounds"],
    ),
    sr: int = Query(
        22050,
        description=SAMPLE_RATE_DESCRIPTION,
        examples=[22050],
    ),
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


@app.post(
    "/translate-cat",
    tags=["Cat to Human Translation"],
    summary="Translate uploaded cat audio into human-readable text",
    description=(
        "Upload a cat vocalization recording and receive a JSON interpretation. "
        "The response includes the dominant intent, a plain-English translation, "
        "confidence, per-segment timing, and the acoustic cues used by the interpreter."
    ),
    operation_id="translateCatAudioToText",
    response_model=CatTranslationResponse,
    response_description="Best-effort text interpretation with per-segment acoustic details.",
    responses={
        400: BAD_REQUEST_RESPONSE,
        415: DECODE_ERROR_RESPONSE,
        422: NO_SEGMENTS_RESPONSE,
    },
)
async def translate_cat_upload(
    file: UploadFile = File(..., description=CAT_AUDIO_DESCRIPTION),
    sr: int = Form(
        22050,
        description=SAMPLE_RATE_DESCRIPTION,
        examples=[22050],
    ),
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


@app.post(
    "/translate-cat-speech",
    tags=["Cat to Human Translation"],
    summary="Translate uploaded cat audio into spoken English WAV",
    description=(
        "Upload a cat vocalization recording and receive a spoken English WAV file. "
        "The endpoint first creates the same interpretation as /translate-cat, then "
        "synthesizes the translation with the selected voice and speech rate. "
        "The chosen intent, confidence, and translation are also returned in response headers."
    ),
    operation_id="translateCatAudioToSpeech",
    response_class=FileResponse,
    responses={
        200: SPEECH_AUDIO_RESPONSE,
        400: BAD_REQUEST_RESPONSE,
        415: DECODE_ERROR_RESPONSE,
        422: NO_SEGMENTS_RESPONSE,
        503: TTS_UNAVAILABLE_RESPONSE,
    },
)
async def translate_cat_speech_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description=CAT_AUDIO_DESCRIPTION),
    sr: int = Form(
        22050,
        description=SAMPLE_RATE_DESCRIPTION,
        examples=[22050],
    ),
    voice: str = Form(
        DEFAULT_VOICE,
        description=VOICE_DESCRIPTION,
        examples=[DEFAULT_VOICE],
        json_schema_extra={"enum": sorted(ALLOWED_VOICES)},
    ),
    tts_rate: int = Form(
        150,
        description=TTS_RATE_DESCRIPTION,
        examples=[150],
    ),
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
