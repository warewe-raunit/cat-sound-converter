"""
DSP primitives for the cat vocalization engine.
All functions take/return float32 numpy arrays.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np

_MIN_PS_LEN = 2048   # minimum samples needed for pitch_shift
_MIN_TS_LEN = 2048   # minimum samples needed for time_stretch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_len(y: np.ndarray, min_len: int) -> tuple[np.ndarray, int]:
    y = np.asarray(y, dtype=np.float32)
    orig = len(y)
    if orig < min_len:
        y = np.pad(y, (0, min_len - orig))
    return y, orig


def _crossfade_concat(parts: list[np.ndarray], sr: int, fade_ms: float = 12) -> np.ndarray:
    """Concatenate parts with equal-power crossfades at joins."""
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    fade_n = int(fade_ms * sr / 1000)
    if fade_n <= 0:
        return np.concatenate(parts).astype(np.float32)
    result = parts[0].copy()
    for part in parts[1:]:
        if len(result) < fade_n or len(part) < fade_n:
            result = np.concatenate([result, part])
            continue
        t = np.linspace(0.0, 1.0, fade_n)
        fade_out = np.sqrt(1 - t)
        fade_in = np.sqrt(t)
        result[-fade_n:] *= fade_out
        overlap = result[-fade_n:] + part[:fade_n] * fade_in
        result = np.concatenate([result[:-fade_n], overlap, part[fade_n:]])
    return result


def crossfade_concat(parts: list[np.ndarray], sr: int, fade_ms: float = 12) -> np.ndarray:
    """Public wrapper for equal-power crossfade concatenation."""
    return _crossfade_concat(parts, sr, fade_ms=fade_ms).astype(np.float32)


# ---------------------------------------------------------------------------
# Core DSP operations
# ---------------------------------------------------------------------------

def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 0.05:
        return y
    import librosa
    y_pad, orig = _ensure_len(y, _MIN_PS_LEN)
    out = librosa.effects.pitch_shift(y_pad, sr=sr, n_steps=float(semitones))
    return out[:orig]


def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate > 1 = faster (shorter), < 1 = slower (longer)."""
    if abs(rate - 1.0) < 0.04:
        return y
    rate = float(np.clip(rate, 0.35, 3.0))
    import librosa
    y_pad, _ = _ensure_len(y, _MIN_TS_LEN)
    return librosa.effects.time_stretch(y_pad, rate=rate)


def apply_gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 0.05:
        return y
    return y * (10.0 ** (gain_db / 20.0))


def shape_attack(y: np.ndarray, sr: int, attack_scale: float) -> np.ndarray:
    """
    Reshape the transient.
    attack_scale > 1 → sharper/faster (convex ramp)
    attack_scale < 1 → softer/slower (concave ramp)
    """
    if abs(attack_scale - 1.0) < 0.05:
        return y
    n = min(int(0.06 * sr), len(y) // 4)
    if n < 4:
        return y
    t = np.linspace(0.0, 1.0, n)
    ramp = t ** (1.0 / max(attack_scale, 1e-3))
    out = y.copy()
    out[:n] *= ramp
    return out


def eq_brightness(y: np.ndarray, sr: int, gain_db: float, freq: float = 3000.0) -> np.ndarray:
    """High-shelf boost/cut via a 2-pole Butterworth high-pass mix."""
    if len(y) == 0 or abs(gain_db) < 0.05:
        return y
    from scipy import signal as sp
    nyq = sr / 2.0
    norm = min(freq / nyq, 0.97)
    b, a = sp.butter(2, norm, btype="high")
    high = sp.lfilter(b, a, y)
    linear_delta = 10.0 ** (gain_db / 20.0) - 1.0
    return y + high * linear_delta


def shape_pitch_contour(
    y: np.ndarray, sr: int, direction: str, amount_st: float = 3.0
) -> np.ndarray:
    """
    Simulate a pitch contour by applying interpolated pitch shifts across
    4 equal segments and crossfading the joins.
    direction: "rising" | "falling"
    """
    if direction not in ("rising", "falling") or len(y) < _MIN_PS_LEN * 2:
        return y

    n_segs = 4
    seg_len = len(y) // n_segs
    if seg_len < _MIN_PS_LEN:
        # Clip too short — apply average shift only
        mid_shift = amount_st / 2.0 * (1 if direction == "rising" else -1)
        return pitch_shift(y, sr, mid_shift)

    if direction == "rising":
        shifts = np.linspace(-amount_st / 2, amount_st / 2, n_segs)
    else:
        shifts = np.linspace(amount_st / 2, -amount_st / 2, n_segs)

    parts = []
    for i, sh in enumerate(shifts):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segs - 1 else len(y)
        parts.append(pitch_shift(y[start:end], sr, float(sh)))

    return _crossfade_concat(parts, sr, fade_ms=10)


def match_duration(
    y: np.ndarray, sr: int, target_dur: float, max_stretch: float = 1.6
) -> np.ndarray:
    """
    Time-stretch y to approximate target_dur, within ±max_stretch.
    Falls back to crop/pad when stretch would be too extreme.
    """
    target_n = max(0, int(target_dur * sr))
    if target_n == 0:
        return np.zeros(0, dtype=np.float32)
    if len(y) == 0:
        return np.zeros(target_n, dtype=np.float32)

    current = len(y) / sr
    if abs(current - target_dur) / (current + 1e-9) < 0.08:
        return y

    rate = current / (target_dur + 1e-9)   # > 1 = speed up
    clamped = float(np.clip(rate, 1.0 / max_stretch, max_stretch))
    y_s = time_stretch(y, sr, clamped)

    if len(y_s) > target_n:
        return y_s[:target_n]
    if len(y_s) < target_n:
        return np.pad(y_s, (0, target_n - len(y_s)))
    return y_s


def normalize(
    y: np.ndarray,
    target_db: float = -3.0,
    only_if_louder: bool = False,
) -> np.ndarray:
    if len(y) == 0:
        return y
    peak = np.max(np.abs(y))
    if peak < 1e-9:
        return y
    target_peak = 10.0 ** (target_db / 20.0)
    if only_if_louder and peak <= target_peak:
        return y
    return y * (target_peak / peak)


def soft_limit(y: np.ndarray, threshold: float = 0.96) -> np.ndarray:
    """Soft-knee limiter — prevents hard clipping."""
    if len(y) == 0:
        return y
    knee = threshold * 0.88
    over = np.abs(y) > knee
    sign = np.sign(y)
    excess = np.abs(y[over]) - knee
    headroom = 1.0 - knee + 1e-9
    y_out = y.copy()
    y_out[over] = sign[over] * (knee + excess / (1.0 + excess / headroom))
    return y_out
