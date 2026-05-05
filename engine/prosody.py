"""
Analyze input audio into vocal segments with prosody features.
Each segment carries: start_s, end_s, duration, rms, intensity,
pitch_mean, pitch_direction, pause_before_s.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def _pitch_direction(f0: np.ndarray) -> str:
    if len(f0) < 4:
        return "flat"
    mean = np.mean(f0) + 1e-9
    std = np.std(f0)
    slope = np.polyfit(np.linspace(0, 1, len(f0)), f0, 1)[0]
    if slope > mean * 0.08:
        return "rising"
    if slope < -mean * 0.08:
        return "falling"
    if std / mean > 0.2:
        return "varied"
    return "flat"


def analyze(y: np.ndarray, sr: int) -> list[dict]:
    """
    Detect vocal events and extract per-segment prosody.
    Returns list sorted by start time.
    """
    import librosa

    hop = 256
    frame_len = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    if len(rms) == 0 or float(np.max(rms)) < 1e-7:
        return []

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    noise_floor = np.percentile(rms_db, 20)
    active_threshold = max(float(noise_floor + 10.0), -45.0)
    active_threshold = min(active_threshold, -6.0)
    active = rms_db > active_threshold

    if not np.any(active):
        fallback_threshold = min(float(np.percentile(rms_db, 70)), -3.0)
        active = rms_db >= fallback_threshold

    # Collect contiguous active regions
    regions: list[tuple[int, int]] = []
    in_seg = False
    start_f = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg, start_f = True, i
        elif not a and in_seg:
            in_seg = False
            regions.append((start_f, i))
    if in_seg:
        regions.append((start_f, len(active)))

    if not regions:
        return []

    # Merge gaps shorter than 150 ms
    merge_gap = int(0.15 * sr / hop)
    merged = [list(regions[0])]
    for s, e in regions[1:]:
        if s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Drop regions shorter than 150 ms
    min_frames = int(0.15 * sr / hop)
    merged = [r for r in merged if r[1] - r[0] >= min_frames]
    if not merged:
        return []

    # Normalize intensity across segments
    seg_rms_vals = [
        float(np.sqrt(np.mean(y[r[0] * hop: r[1] * hop] ** 2)))
        for r in merged
    ]
    max_rms = max(seg_rms_vals) + 1e-9

    segments: list[dict] = []
    for i, (sf_f, ef_f) in enumerate(merged):
        start_s = float(sf_f * hop / sr)
        end_s = min(float(ef_f * hop / sr), len(y) / sr)
        chunk = y[sf_f * hop: ef_f * hop]

        seg_rms = seg_rms_vals[i]

        # Pitch via pyin
        try:
            f0, voiced_flag, _ = librosa.pyin(
                chunk, fmin=80, fmax=8000, sr=sr, frame_length=1024
            )
            if f0 is not None and voiced_flag is not None:
                voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
            else:
                voiced_f0 = np.array([])
        except Exception:
            voiced_f0 = np.array([])

        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 2 else 200.0
        direction = _pitch_direction(voiced_f0) if len(voiced_f0) > 2 else "flat"

        prev_end = segments[-1]["end_s"] if segments else 0.0
        pause_before = max(0.0, start_s - prev_end)

        segments.append({
            "start_s": start_s,
            "end_s": end_s,
            "duration": end_s - start_s,
            "rms": seg_rms,
            "intensity": seg_rms / max_rms,
            "pitch_mean": pitch_mean,
            "pitch_direction": direction,
            "pause_before_s": pause_before,
        })

    return segments
