import warnings
warnings.filterwarnings("ignore")
import subprocess
import tempfile
from pathlib import Path
import numpy as np


class AudioDecodeError(RuntimeError):
    pass


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
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return np.asarray(y, dtype=np.float32), int(native_sr)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    import librosa
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return np.asarray(y, dtype=np.float32), sr
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
    sf.write(str(path), y, sr, subtype="PCM_16")
