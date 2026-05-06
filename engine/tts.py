"""
Text-to-speech output for cat translation results.

Saved speech prefers edge-tts neural voices because pyttsx3 falls back to
eSpeak on Linux, which sounds noticeably robotic. pyttsx3 remains the offline
fallback so the API still works when the server has no network access.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


DEFAULT_VOICE = "en-US-GuyNeural"
ALLOWED_VOICES = {"en-US-GuyNeural", "en-GB-SoniaNeural"}
NEURAL_TTS_TIMEOUT_S = 30


class TextToSpeechError(RuntimeError):
    pass


def _edge_rate(rate: int) -> str:
    """Convert pyttsx-style words/minute to edge-tts percentage rate."""
    pct = int(round((rate - 150) / 150 * 100))
    pct = max(-35, min(30, pct))
    return f"{pct:+d}%"


def _run_async_blocking(coro_factory, timeout_s: int = NEURAL_TTS_TIMEOUT_S) -> None:
    """
    Run an async TTS job from either sync CLI code or an already-running
    FastAPI event loop.
    """
    async def with_timeout() -> None:
        await asyncio.wait_for(coro_factory(), timeout=timeout_s)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(with_timeout())
        return

    error: list[BaseException] = []

    def runner() -> None:
        try:
            asyncio.run(with_timeout())
        except BaseException as exc:  # pragma: no cover - defensive thread bridge
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout_s + 5)
    if thread.is_alive():
        raise TimeoutError("Neural TTS timed out.")
    if error:
        raise error[0]


async def _save_neural_voice(
    text: str,
    output_path: str,
    rate: int,
    voice: str,
) -> None:
    import edge_tts
    from imageio_ffmpeg import get_ffmpeg_exe

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cat_tts_") as tmp:
        mp3_path = Path(tmp) / "speech.mp3"
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=_edge_rate(rate),
        )
        await communicate.save(str(mp3_path))

        if out.suffix.lower() == ".mp3":
            shutil.copyfile(mp3_path, out)
            return

        cmd = [
            get_ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp3_path),
            "-ar",
            "22050",
            "-ac",
            "1",
            str(out),
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )


def _speak_with_pyttsx3(text: str, output_path: str | None, rate: int) -> None:
    """Offline fallback. On Linux this is functional but less natural."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", rate)

        if output_path:
            Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        else:
            engine.say(text)
            engine.runAndWait()
    except Exception as exc:
        raise TextToSpeechError(
            "Text-to-speech is unavailable. Install edge-tts/ffmpeg or a working pyttsx3 voice backend."
        ) from exc


def speak_text(
    text: str,
    output_path: str | None = None,
    rate: int = 150,
    voice: str = DEFAULT_VOICE,
) -> None:
    """Speak *text* aloud or save to *output_path* as a WAV file."""
    voice = voice if voice in ALLOWED_VOICES else DEFAULT_VOICE
    if output_path:
        try:
            _run_async_blocking(
                lambda: _save_neural_voice(
                    text=text,
                    output_path=output_path,
                    rate=rate,
                    voice=voice,
                )
            )
            return
        except Exception as exc:
            # Keep the endpoint usable when neural TTS cannot be reached.
            print(
                f"Neural TTS unavailable ({exc}); falling back to offline voice.",
                file=sys.stderr,
            )
            pass

    _speak_with_pyttsx3(text, output_path, rate)
