"""
Best-effort cat audio to human-readable intent.

This is not a literal language translation. It interprets acoustic cues
such as duration, intensity, pitch movement, roughness, and repeated bursts
into practical human-facing meanings.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from engine.audio_io import sanitize_audio, validate_sample_rate
from engine.prosody import analyze


INTENT_TRANSLATIONS: dict[str, str] = {
    "attention": "I want your attention.",
    "request": "I need something. Please check food, water, the door, or the litter box.",
    "urgent": "I need you now.",
    "annoyed": "I am irritated or overstimulated. Please give me space.",
    "lonely": "Where are you? I want company.",
    "calm": "I am relaxed and just checking in.",
    "playful": "Let's play.",
    "content": "I feel safe and content.",
    "unclear": "I made a sound, but it is not clear enough to interpret.",
}

INTENT_LABELS: dict[str, str] = {
    "attention": "attention seeking",
    "request": "request",
    "urgent": "urgent request",
    "annoyed": "annoyed or overstimulated",
    "lonely": "lonely or calling",
    "calm": "calm check-in",
    "playful": "playful",
    "content": "content or soothing",
    "unclear": "unclear",
}


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _pitch_features(y: np.ndarray, sr: int) -> dict:
    try:
        import librosa

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=60,
            fmax=8000,
            sr=sr,
            frame_length=1024,
        )
        if f0 is None or voiced_flag is None:
            return {"pitch_slope": 0.0, "voiced_ratio": 0.0, "pitch_variation": 0.0}
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    except Exception:
        voiced_f0 = np.array([])
        f0 = np.array([])

    if len(voiced_f0) < 4:
        return {
            "pitch_slope": 0.0,
            "voiced_ratio": 0.0,
            "pitch_variation": 0.0,
        }

    x = np.linspace(0.0, 1.0, len(voiced_f0))
    pitch_slope = float(np.polyfit(x, voiced_f0, 1)[0])
    pitch_variation = float(np.std(voiced_f0) / (_safe_mean(voiced_f0) + 1e-9))
    voiced_ratio = float(len(voiced_f0) / max(1, len(f0)))
    return {
        "pitch_slope": pitch_slope,
        "voiced_ratio": voiced_ratio,
        "pitch_variation": pitch_variation,
    }


def _periodicity(y: np.ndarray, sr: int) -> float:
    if len(y) < int(0.08 * sr):
        return 0.0

    window = y[: min(len(y), int(0.35 * sr))]
    if np.max(np.abs(window)) < 1e-7:
        return 0.0

    ac = np.correlate(window, window, mode="full")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-9)
    low = int(sr / 100)
    high = int(sr / 20)
    if high <= low or len(ac) <= low:
        return 0.0
    band = ac[low : min(high, len(ac))]
    return float(np.max(np.abs(band))) if len(band) else 0.0


def _acoustic_features(y: np.ndarray, sr: int) -> dict:
    if len(y) == 0:
        return {
            "brightness": 0.0,
            "roughness": 0.0,
            "periodicity": 0.0,
            "pitch_slope": 0.0,
            "voiced_ratio": 0.0,
            "pitch_variation": 0.0,
        }

    try:
        import librosa

        brightness = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=y,
                    sr=sr,
                    n_fft=1024,
                    hop_length=256,
                )[0]
            )
        )
        roughness = float(
            np.mean(
                librosa.feature.spectral_flatness(
                    y=y,
                    n_fft=1024,
                    hop_length=256,
                )[0]
            )
        )
    except Exception:
        brightness = 0.0
        roughness = 0.0

    features = {
        "brightness": brightness,
        "roughness": roughness,
        "periodicity": _periodicity(y, sr),
    }
    features.update(_pitch_features(y, sr))
    return features


def _segment_audio(y: np.ndarray, sr: int, segment: dict) -> np.ndarray:
    start = max(0, int(segment["start_s"] * sr))
    end = min(len(y), int(segment["end_s"] * sr))
    return y[start:end]


def _confidence(intent: str, segment: dict, features: dict, rapid_burst: bool) -> float:
    confidence = 0.48
    intensity = segment["intensity"]
    direction = segment["pitch_direction"]
    dur = segment["duration"]

    if intent == "content":
        confidence += min(0.25, features["periodicity"] * 0.35)
        if dur > 0.8:
            confidence += 0.08
    elif intent == "playful":
        confidence += 0.16 if dur < 0.55 else 0.06
        confidence += 0.12 if rapid_burst else 0.0
        confidence += min(0.08, features["pitch_variation"] * 0.25)
    elif intent in {"urgent", "request"}:
        confidence += min(0.2, intensity * 0.18)
        confidence += 0.12 if direction == "rising" else 0.0
    elif intent == "annoyed":
        confidence += min(0.16, intensity * 0.15)
        confidence += 0.08 if direction == "varied" else 0.0
        confidence += 0.08 if features["roughness"] > 0.08 else 0.0
    elif intent == "lonely":
        confidence += 0.12 if dur > 1.0 else 0.0
        confidence += 0.14 if direction == "falling" else 0.0
    elif intent == "calm":
        confidence += 0.12 if intensity < 0.35 else 0.0
        confidence += 0.06 if direction in {"flat", "falling"} else 0.0
    else:
        confidence += 0.08

    return round(float(np.clip(confidence, 0.35, 0.9)), 2)


def _interpret_segment(
    y: np.ndarray,
    sr: int,
    segment: dict,
    index: int,
    previous: dict | None,
) -> dict:
    chunk = _segment_audio(y, sr, segment)
    features = _acoustic_features(chunk, sr)

    dur = segment["duration"]
    intensity = segment["intensity"]
    direction = segment["pitch_direction"]
    pitch = segment["pitch_mean"]
    rapid_burst = bool(
        previous
        and dur < 0.65
        and previous["duration"] < 0.65
        and segment["pause_before_s"] < 0.35
    )

    signals: list[str] = []
    intent = "attention"

    purr_like = dur > 0.8 and (
        (0 < pitch < 220 and features["roughness"] < 0.18)
        or (
            features["periodicity"] > 0.45
            and features["brightness"] < 1400
            and pitch < 350
        )
    )

    if purr_like:
        intent = "content"
        signals.append("low periodic rumble")
    elif rapid_burst or (dur < 0.45 and pitch > 450):
        intent = "playful"
        signals.append("short repeated burst" if rapid_burst else "short chirp-like burst")
    elif intensity > 0.78 and direction == "rising":
        intent = "urgent"
        signals.extend(["high intensity", "rising pitch"])
    elif intensity > 0.52 and direction == "rising":
        intent = "request"
        signals.extend(["moderate-high intensity", "rising pitch"])
    elif intensity > 0.65 and (
        direction == "varied"
        or features["roughness"] > 0.08
        or features["brightness"] > 2800
    ):
        intent = "annoyed"
        signals.append("rough or bright vocal tone")
    elif dur > 1.0 and direction == "falling":
        intent = "lonely"
        signals.extend(["long vocalization", "falling pitch"])
    elif intensity < 0.32 and direction in {"flat", "falling"}:
        intent = "calm"
        signals.append("soft low-intensity vocalization")
    else:
        signals.append("general meow contour")

    confidence = _confidence(intent, segment, features, rapid_burst)

    return {
        "index": index,
        "start_s": _round(segment["start_s"]),
        "end_s": _round(segment["end_s"]),
        "duration": _round(dur),
        "intent": intent,
        "label": INTENT_LABELS[intent],
        "translation": INTENT_TRANSLATIONS[intent],
        "confidence": confidence,
        "signals": signals,
        "features": {
            "intensity": _round(intensity),
            "pitch_hz": _round(pitch, 1),
            "pitch_direction": direction,
            "brightness": _round(features["brightness"], 1),
            "roughness": _round(features["roughness"], 4),
            "periodicity": _round(features["periodicity"], 3),
        },
    }


def _overall_translation(segments: list[dict]) -> tuple[str, str, float]:
    if not segments:
        return INTENT_TRANSLATIONS["unclear"], "unclear", 0.0

    weighted: dict[str, float] = defaultdict(float)
    confidence_total = 0.0
    weight_total = 0.0

    for segment in segments:
        weight = max(0.2, segment["duration"]) * max(0.1, segment["confidence"])
        weighted[segment["intent"]] += weight
        confidence_total += segment["confidence"] * weight
        weight_total += weight

    dominant = max(weighted, key=weighted.get)
    confidence = round(confidence_total / max(weight_total, 1e-9), 2)
    primary = INTENT_TRANSLATIONS[dominant]

    secondary_lines = []
    for segment in segments:
        line = segment["translation"]
        if line != primary and line not in secondary_lines:
            secondary_lines.append(line)

    if secondary_lines:
        return f"{primary} Also: {secondary_lines[0]}", dominant, confidence
    return primary, dominant, confidence


def translate_cat_audio(y: np.ndarray, sr: int) -> dict:
    """
    Interpret cat vocalizations from an audio buffer.

    Returns a JSON-serializable dict with an overall translation plus
    per-segment interpretations and acoustic cues.
    """
    sr = validate_sample_rate(sr)
    y = sanitize_audio(y)
    segments = analyze(y, sr)
    interpreted: list[dict] = []

    previous = None
    for index, segment in enumerate(segments, start=1):
        result = _interpret_segment(y, sr, segment, index, previous)
        interpreted.append(result)
        previous = segment

    translation, intent, confidence = _overall_translation(interpreted)

    return {
        "input_seconds": _round(len(y) / sr),
        "segments": len(interpreted),
        "intent": intent,
        "label": INTENT_LABELS[intent],
        "translation": translation,
        "confidence": confidence,
        "segment_interpretations": interpreted,
        "note": "Best-effort acoustic interpretation. Cat meaning depends on context and body language.",
    }
