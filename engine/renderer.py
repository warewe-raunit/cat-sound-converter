"""
Main render loop: prosody segments → cat vocalizations.

For each input segment:
  1. Infer target emotion from prosody
  2. Select source clip from dataset (cycling top-quality clips)
  3. Apply DSP chain (contour, pitch, duration, attack, EQ, gain, repeat)
  4. Place result at correct time position, preserving pause structure
  5. Final soft-limit + normalize
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from engine.dsp import (
    pitch_shift, apply_gain, shape_attack,
    eq_brightness, shape_pitch_contour, normalize,
    soft_limit, match_duration, crossfade_concat,
)
from engine.emotion_presets import PRESETS, STYLE_SCALES, infer_emotion
from engine.curated_dataset import CuratedDataset


def _silence(sr: int, dur_s: float) -> np.ndarray:
    n = max(0, int(dur_s * sr))
    return np.zeros(n, dtype=np.float32)


def _fit_exact(y: np.ndarray, sr: int, target_dur: float) -> np.ndarray:
    target_n = max(1, int(target_dur * sr))
    if len(y) > target_n:
        return y[:target_n]
    if len(y) < target_n:
        return np.pad(y, (0, target_n - len(y)))
    return y


def _render_one(
    clip: np.ndarray,
    sr: int,
    emotion: str,
    style: str,
    target_dur: float,
) -> np.ndarray:
    """Apply the full DSP chain for one vocal event."""
    p = PRESETS[emotion]
    s = STYLE_SCALES.get(style, STYLE_SCALES["natural"])

    y = clip.copy()
    repeat = max(1, int(p.repeat))
    repeat_gap_s = p.repeat_gap_ms / 1000.0

    # 1. Pitch contour (before overall shift so contour is relative to source)
    if p.contour and p.contour != "flat":
        y = shape_pitch_contour(y, sr, p.contour, p.contour_amount_st * s["pitch"])

    # 2. Overall pitch shift
    st = p.pitch_shift_st * s["pitch"]
    if abs(st) > 0.05:
        y = pitch_shift(y, sr, st)

    # 3. Duration match. Repeated phrases are fit inside the segment window
    # instead of stretching the segment 2-3x longer after repetition.
    if target_dur > 0:
        if repeat > 1:
            min_unit_dur = 0.08
            total_gap = repeat_gap_s * (repeat - 1)
            if target_dur < total_gap + min_unit_dur * repeat:
                repeat_gap_s = max(
                    0.0,
                    (target_dur - min_unit_dur * repeat) / max(1, repeat - 1),
                )
                total_gap = repeat_gap_s * (repeat - 1)
            unit_dur = max(0.05, (target_dur - total_gap) / repeat)
            y = match_duration(y, sr, unit_dur, max_stretch=2.2)
        else:
            y = match_duration(y, sr, target_dur)

    # 4. Attack shaping
    atk = 1.0 + (p.attack_scale - 1.0) * s["attack"]
    y = shape_attack(y, sr, atk)

    # 5. Brightness EQ
    bright = p.brightness_db * s["brightness"]
    if abs(bright) > 0.05:
        y = eq_brightness(y, sr, bright)

    # 6. Gain
    gain = p.gain_db * s["gain"]
    if abs(gain) > 0.05:
        y = apply_gain(y, gain)

    # 7. Repetition (playful / excited)
    if repeat > 1:
        gap = _silence(sr, repeat_gap_s)
        parts: list[np.ndarray] = []
        for i in range(repeat):
            if i > 0:
                parts.append(gap)
            parts.append(y.copy())
        y = crossfade_concat(parts, sr, fade_ms=4)
        if target_dur > 0:
            y = _fit_exact(y, sr, target_dur)

    return y.astype(np.float32)


def render(
    segments: list[dict],
    dataset: CuratedDataset,
    style: str = "natural",
    sr: int = 22050,
    verbose: bool = True,
) -> np.ndarray:
    """
    Render all prosody segments to cat audio.
    Pause durations from the input are preserved (scaled by tempo_scale).
    """
    if not segments:
        return _silence(sr, 1.0)

    parts: list[np.ndarray] = []
    history: list[dict] = []

    # Leading silence
    if segments[0]["pause_before_s"] > 0.05:
        parts.append(_silence(sr, segments[0]["pause_before_s"]))

    for i, seg in enumerate(segments):
        emotion = infer_emotion(seg, history)
        preset = PRESETS[emotion]
        clip, meta = dataset.get_clip(preset.source_character)

        cat_chunk = _render_one(
            clip=clip,
            sr=sr,
            emotion=emotion,
            style=style,
            target_dur=seg["duration"],
        )
        parts.append(cat_chunk)

        if verbose:
            src = meta["character"]
            print(
                f"  seg {i+1:02d}  [{emotion:12s}]  "
                f"dur={seg['duration']:.2f}s  "
                f"intensity={seg['intensity']:.2f}  "
                f"pitch={seg['pitch_direction']:7s}  "
                f"src={src}"
            )

        # Pause between this and the next segment
        if i + 1 < len(segments):
            raw_pause = segments[i + 1]["pause_before_s"]
            if raw_pause > 0.04:
                pause_dur = raw_pause * preset.tempo_scale
                parts.append(_silence(sr, pause_dur))

        history.append(seg)

    out = crossfade_concat(parts, sr, fade_ms=8)
    out = soft_limit(out)
    out = normalize(out, target_db=-3.0, only_if_louder=True)
    return out
