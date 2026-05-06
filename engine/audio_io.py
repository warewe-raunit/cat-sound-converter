import warnings
warnings.filterwarnings("ignore")
import subprocess
import tempfile
from pathlib import Path
import numpy as np


MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000


class AudioDecodeError(RuntimeError):
    pass


def validate_sample_rate(sr: int) -> int:
    """Keep caller-provided sample rates inside a sane DSP range."""
    try:
        sr = int(sr)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sample rate must be an integer.") from exc

    if sr < MIN_SAMPLE_RATE or sr > MAX_SAMPLE_RATE:
        raise ValueError(
            f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz."
        )
    return sr


def sanitize_audio(y: np.ndarray) -> np.ndarray:
    """Return mono float32 audio with non-finite samples replaced by silence."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_with_ffmpeg(path: str, sr: int) -> tuple[np.ndarray, int]:
    try:
        import imageio_ffmpeg
        import soundfile as sf
    except ImportError as exc:
        raise AudioDecodeError(
            "Could not decode this audio format. Install imageio-ffmpeg or upload a WAV file."
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "wav",
            str(tmp_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "ffmpeg failed to decode input"
            raise AudioDecodeError(detail)

        y, native_sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
        return sanitize_audio(y), int(native_sr)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    sr = validate_sample_rate(sr)
    try:
        import librosa

        y, _ = librosa.load(path, sr=sr, mono=True)
        return sanitize_audio(y), sr
    except Exception as exc:
        try:
            return _load_with_ffmpeg(path, sr)
        except AudioDecodeError:
            raise
        except Exception as fallback_exc:
            raise AudioDecodeError(
                f"Could not decode audio file '{path}'. Try WAV, MP3, M4A, or FLAC."
            ) from fallback_exc


def save_audio(path: str, y: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sr = validate_sample_rate(sr)
    sf.write(str(path), sanitize_audio(y), sr, subtype="PCM_16")
