"""
Emotion presets and prosody-to-emotion inference.

Each preset defines a source character + DSP parameter offsets.
Style scales (natural/expressive/subtle) modulate how strongly
DSP is applied without changing the character mapping.
"""
from dataclasses import dataclass


@dataclass
class EmotionPreset:
    source_character: str
    pitch_shift_st: float = 0.0       # semitones added on top of source
    gain_db: float = 0.0
    brightness_db: float = 0.0        # high-shelf gain at ~3 kHz
    attack_scale: float = 1.0         # > 1 = sharper transient
    contour: str = None               # "rising" | "falling" | None
    contour_amount_st: float = 2.5    # sweep range in semitones
    tempo_scale: float = 1.0          # pause duration multiplier
    repeat: int = 1                   # clip repetitions (playful/excited)
    repeat_gap_ms: float = 110.0


PRESETS: dict[str, EmotionPreset] = {
    "calm": EmotionPreset(
        source_character="calm_soft",
        pitch_shift_st=0.0,
        gain_db=-1.5,
        brightness_db=-1.5,
        attack_scale=0.75,
        contour=None,
        tempo_scale=1.15,
    ),
    "lonely": EmotionPreset(
        source_character="lonely_falling",
        pitch_shift_st=-1.5,
        gain_db=-2.0,
        brightness_db=-2.0,
        attack_scale=0.7,
        contour="falling",
        contour_amount_st=3.0,
        tempo_scale=1.2,
    ),
    "attention": EmotionPreset(
        source_character="neutral_attention",
        pitch_shift_st=0.0,
        gain_db=0.0,
        brightness_db=0.0,
        attack_scale=1.0,
        contour=None,
        tempo_scale=1.0,
    ),
    "hungry": EmotionPreset(
        source_character="neutral_attention",
        pitch_shift_st=3.0,
        gain_db=3.0,
        brightness_db=2.0,
        attack_scale=1.35,
        contour="rising",
        contour_amount_st=2.5,
        tempo_scale=0.88,
    ),
    "urgent": EmotionPreset(
        source_character="neutral_attention",
        pitch_shift_st=5.0,
        gain_db=5.5,
        brightness_db=4.5,
        attack_scale=2.2,
        contour="rising",
        contour_amount_st=2.0,
        tempo_scale=0.65,
    ),
    "annoyed": EmotionPreset(
        source_character="neutral_attention",
        pitch_shift_st=2.0,
        gain_db=4.0,
        brightness_db=5.0,
        attack_scale=2.0,
        contour="falling",
        contour_amount_st=1.5,
        tempo_scale=0.8,
    ),
    "playful": EmotionPreset(
        source_character="playful_chirp_trill",
        pitch_shift_st=1.0,
        gain_db=1.0,
        brightness_db=1.0,
        attack_scale=1.2,
        contour=None,
        tempo_scale=0.9,
        repeat=2,
        repeat_gap_ms=105.0,
    ),
    "excited": EmotionPreset(
        source_character="playful_chirp_trill",
        pitch_shift_st=3.5,
        gain_db=3.5,
        brightness_db=3.0,
        attack_scale=1.6,
        contour="rising",
        contour_amount_st=2.0,
        tempo_scale=0.65,
        repeat=3,
        repeat_gap_ms=75.0,
    ),
}

# How strongly DSP is applied per style
STYLE_SCALES: dict[str, dict[str, float]] = {
    "natural":    {"pitch": 0.60, "gain": 0.65, "brightness": 0.55, "attack": 0.65},
    "expressive": {"pitch": 1.00, "gain": 1.00, "brightness": 1.00, "attack": 1.00},
    "subtle":     {"pitch": 0.30, "gain": 0.35, "brightness": 0.25, "attack": 0.40},
}


def infer_emotion(segment: dict, history: list[dict] | None = None) -> str:
    """
    Map a prosody segment to a target emotion name.
    history = list of previously processed segments (for context).
    """
    dur = segment["duration"]
    intensity = segment["intensity"]      # 0–1, normalised
    pitch_dir = segment["pitch_direction"]

    # Rapid short-burst context → playful / excited
    if history and len(history) >= 2:
        recent = history[-2:]
        avg_dur = sum(s["duration"] for s in recent) / len(recent)
        avg_pause = sum(s["pause_before_s"] for s in recent) / len(recent)
        if avg_dur < 0.5 and avg_pause < 0.35:
            return "excited" if intensity > 0.55 else "playful"

    if dur < 0.45:
        return "playful"

    # High intensity + rising → urgent / hungry
    if intensity > 0.78 and pitch_dir == "rising":
        return "urgent"
    if intensity > 0.50 and pitch_dir == "rising":
        return "hungry"

    # Loud + varied → annoyed
    if intensity > 0.65 and pitch_dir == "varied":
        return "annoyed"

    # Long + falling → lonely
    if dur > 1.1 and pitch_dir == "falling":
        return "lonely"

    # Quiet + non-rising → calm
    if intensity < 0.30 and pitch_dir in ("flat", "falling"):
        return "calm"

    return "attention"
