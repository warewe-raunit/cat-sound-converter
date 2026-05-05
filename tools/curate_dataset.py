"""
Cat audio curation pipeline.

Usage:
    python tools/curate_dataset.py --dataset <path> [--output curated_cat_sounds] [--target 30]

Stages:
  1. Load wav/mp3/m4a from dataset folder (recursive)
  2. Split each file into candidate vocal events via energy onset detection
  3. Reject bad candidates (too short/long, clipped, noisy, human voice, silence-heavy)
  4. Score remaining candidates on SNR, pitch clarity, envelope quality
  5. Cluster by voice similarity (MFCC-based) → pick largest cluster
  6. Classify surviving clips into vocal characters by acoustic features
  7. Save top 20-40 clips + metadata.json
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so missing ones give clear messages
# ---------------------------------------------------------------------------

def _require(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(f"Required: pip install {name}")


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 22050) -> Optional[tuple[np.ndarray, int]]:
    """Load audio file, return (mono float32 array, sr). None on failure."""
    librosa = _require("librosa")
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y, sr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Onset / event splitting
# ---------------------------------------------------------------------------

def split_vocal_events(
    y: np.ndarray,
    sr: int,
    min_dur: float = 0.25,
    max_dur: float = 3.0,
    pre_pad: float = 0.02,
    post_pad: float = 0.05,
) -> list[np.ndarray]:
    """
    Detect energy onsets, then extract non-overlapping segments around them.
    Returns list of audio chunks.
    """
    librosa = _require("librosa")

    # RMS energy envelope
    frame_len = 512
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Threshold: regions above noise floor
    noise_floor = np.percentile(rms_db, 20)
    active_threshold = noise_floor + 12  # 12 dB above noise floor

    active = rms_db > active_threshold

    # Convert frame mask to sample regions
    segments = []
    in_seg = False
    start_frame = 0

    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg = True
            start_frame = i
        elif not a and in_seg:
            in_seg = False
            end_frame = i
            start_sample = max(0, int((start_frame * hop) - pre_pad * sr))
            end_sample = min(len(y), int((end_frame * hop) + post_pad * sr))
            chunk = y[start_sample:end_sample]
            dur = len(chunk) / sr
            if min_dur <= dur <= max_dur:
                segments.append(chunk)

    if in_seg:
        end_sample = min(len(y), int((len(active) * hop) + post_pad * sr))
        start_sample = max(0, int((start_frame * hop) - pre_pad * sr))
        chunk = y[start_sample:end_sample]
        dur = len(chunk) / sr
        if min_dur <= dur <= max_dur:
            segments.append(chunk)

    return segments


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(y: np.ndarray, sr: int) -> dict:
    """Extract acoustic features for quality scoring and classification."""
    librosa = _require("librosa")

    dur = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))

    # Pitch via pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=100, fmax=8000, sr=sr, frame_length=1024
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
    except Exception:
        voiced_f0 = np.array([])

    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 3 else 0.0
    voiced_ratio = len(voiced_f0) / max(1, len(f0)) if f0 is not None else 0.0

    # Pitch contour slope (positive = rising, negative = falling)
    if len(voiced_f0) >= 4:
        x = np.linspace(0, 1, len(voiced_f0))
        pitch_slope = float(np.polyfit(x, voiced_f0, 1)[0])
        pitch_contour = voiced_f0.tolist()
    else:
        pitch_slope = 0.0
        pitch_contour = []

    # Spectral features
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    # Brightness: spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)[0]
    brightness = float(np.mean(centroid))

    # Roughness proxy: spectral flatness (high = noisy/rough)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=256)[0]
    roughness = float(np.mean(flatness))

    # Attack speed: time to reach 80% of peak RMS in first 200ms
    attack_window = min(int(0.2 * sr), len(y))
    attack_chunk = y[:attack_window]
    attack_rms = librosa.feature.rms(y=attack_chunk, frame_length=256, hop_length=64)[0]
    if len(attack_rms) > 1:
        peak_rms = np.max(attack_rms)
        threshold_80 = 0.8 * peak_rms
        above = np.where(attack_rms >= threshold_80)[0]
        attack_speed = float(above[0] / len(attack_rms)) if len(above) > 0 else 1.0
    else:
        attack_speed = 1.0
    # Invert: lower value = faster attack
    attack_speed = 1.0 - attack_speed

    # MFCCs for voice clustering
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
    mfcc_mean = mfcc.mean(axis=1).tolist()

    # SNR estimate: ratio of signal energy to background energy
    frame_rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    sorted_rms = np.sort(frame_rms)
    noise_est = float(np.mean(sorted_rms[:max(1, len(sorted_rms) // 5)]))
    signal_est = float(np.mean(sorted_rms[-max(1, len(sorted_rms) // 3):]))
    snr = float(20 * np.log10(signal_est / (noise_est + 1e-9) + 1e-9))

    # Silence ratio: fraction of frames below noise threshold
    rms_db = librosa.amplitude_to_db(frame_rms, ref=np.max)
    silence_ratio = float(np.mean(rms_db < -40))

    # Clipping check
    clip_ratio = float(np.mean(np.abs(y) > 0.98))

    # Periodicity (purr detection): autocorrelation at low lags
    # Purrs have strong periodicity at 20-100 Hz
    ac = np.correlate(y[:min(len(y), int(0.5 * sr))], y[:min(len(y), int(0.5 * sr))], mode="full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-9)
    lag_range = slice(int(sr / 100), int(sr / 20))  # 20–100 Hz
    periodicity = float(np.max(np.abs(ac[lag_range]))) if ac[lag_range].size > 0 else 0.0

    return {
        "duration": dur,
        "rms": rms,
        "peak": peak,
        "pitch_mean": pitch_mean,
        "pitch_contour": pitch_contour,
        "pitch_slope": pitch_slope,
        "voiced_ratio": voiced_ratio,
        "brightness": brightness,
        "roughness": roughness,
        "attack_speed": attack_speed,
        "snr": snr,
        "silence_ratio": silence_ratio,
        "clip_ratio": clip_ratio,
        "periodicity": periodicity,
        "mfcc_mean": mfcc_mean,
    }


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def passes_quality_gate(feat: dict) -> tuple[bool, str]:
    """Hard reject based on quality criteria. Returns (pass, reason)."""
    if feat["clip_ratio"] > 0.01:
        return False, f"clipped ({feat['clip_ratio']:.3f})"
    if feat["snr"] < 8.0:
        return False, f"low SNR ({feat['snr']:.1f} dB)"
    if feat["silence_ratio"] > 0.5:
        return False, f"too much silence ({feat['silence_ratio']:.2f})"
    if feat["rms"] < 0.002:
        return False, f"too quiet (rms={feat['rms']:.4f})"
    # Reject if pitch is unclear and it's not a purr candidate
    if feat["voiced_ratio"] < 0.2 and feat["periodicity"] < 0.3:
        return False, f"no clear pitch or periodicity"
    # Reject very high spectral flatness (white noise / background)
    if feat["roughness"] > 0.4:
        return False, f"too noisy/flat spectrum ({feat['roughness']:.3f})"
    # Human voice range check: reject if fundamental is clearly in speech range
    # with no cat-typical modulation (rough heuristic)
    if 80 < feat["pitch_mean"] < 300 and feat["brightness"] < 800:
        return False, f"possible human voice (low pitch+brightness)"
    return True, "ok"


# ---------------------------------------------------------------------------
# Quality score (for ranking within category)
# ---------------------------------------------------------------------------

def quality_score(feat: dict) -> float:
    """Higher = better. Range ~0-1."""
    snr_score = min(1.0, feat["snr"] / 30.0)
    voiced_score = feat["voiced_ratio"]
    silence_pen = 1.0 - feat["silence_ratio"]
    clip_pen = 1.0 - min(1.0, feat["clip_ratio"] * 100)
    flatness_pen = 1.0 - min(1.0, feat["roughness"] * 2)
    return float(np.mean([snr_score, voiced_score, silence_pen, clip_pen, flatness_pen]))


# ---------------------------------------------------------------------------
# Vocal character classification
# ---------------------------------------------------------------------------

CHARACTERS = [
    "neutral_attention",
    "calm_soft",
    "lonely_falling",
    "annoyed_urgent",
    "playful_chirp_trill",
    "purr",
]


def classify_character(feat: dict) -> str:
    """
    Rule-based classification from acoustic features.
    Priority order matters — more specific rules first.
    """
    dur = feat["duration"]
    rms = feat["rms"]
    pitch = feat["pitch_mean"]
    slope = feat["pitch_slope"]
    brightness = feat["brightness"]
    roughness = feat["roughness"]
    attack = feat["attack_speed"]
    periodicity = feat["periodicity"]

    # Purr: low periodic rumble, sustained. Plain meows are also periodic, so
    # do not classify high-pitch meow harmonics as purr just from autocorrelation.
    if dur > 0.8 and (
        (0 < pitch < 200 and roughness < 0.15)
        or (pitch == 0 and periodicity > 0.45 and brightness < 1200)
    ):
        return "purr"

    # Playful chirp/trill: very short, high pitch, modulated
    if dur < 0.4 and pitch > 700:
        return "playful_chirp_trill"
    if dur < 0.6 and pitch > 500 and len(feat["pitch_contour"]) > 4:
        contour = np.array(feat["pitch_contour"])
        modulation = float(np.std(contour) / (np.mean(contour) + 1e-9))
        if modulation > 0.15:
            return "playful_chirp_trill"

    # Annoyed/urgent: loud, fast attack, bright or rough
    if rms > 0.08 and attack > 0.6 and (brightness > 2500 or roughness > 0.1):
        return "annoyed_urgent"

    # Lonely/falling: longer, clearly falling pitch
    if dur > 0.8 and slope < -200:
        return "lonely_falling"

    # Calm/soft: quiet, smooth envelope (slow attack), not too long
    if rms < 0.03 and attack < 0.4:
        return "calm_soft"

    # Default: neutral attention
    return "neutral_attention"


# ---------------------------------------------------------------------------
# Voice clustering
# ---------------------------------------------------------------------------

def cluster_by_voice(candidates: list[dict], n_clusters: int = 5) -> list[dict]:
    """
    MFCC-based clustering. Returns candidates annotated with cluster_id.
    Falls back gracefully if sklearn unavailable or too few samples.
    """
    if len(candidates) < n_clusters * 2:
        for c in candidates:
            c["cluster_id"] = 0
        return candidates

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        for c in candidates:
            c["cluster_id"] = 0
        return candidates

    X = np.array([c["features"]["mfcc_mean"] for c in candidates])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    actual_k = min(n_clusters, len(candidates))
    km = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    for c, lbl in zip(candidates, labels):
        c["cluster_id"] = int(lbl)

    return candidates


def pick_dominant_cluster(candidates: list[dict]) -> list[dict]:
    """Keep only candidates from the largest cluster (most consistent voice)."""
    if not candidates:
        return candidates
    cluster_ids = [c.get("cluster_id", 0) for c in candidates]
    counts = {}
    for cid in cluster_ids:
        counts[cid] = counts.get(cid, 0) + 1
    dominant = max(counts, key=counts.__getitem__)
    kept = [c for c in candidates if c.get("cluster_id", 0) == dominant]
    # If dominant cluster is too small relative to total, keep all
    if len(kept) < 5 and len(candidates) >= 10:
        return candidates
    return kept


# ---------------------------------------------------------------------------
# Selection: pick top N per character
# ---------------------------------------------------------------------------

def select_top_clips(
    candidates: list[dict],
    target_total: int = 30,
    min_per_char: int = 2,
) -> list[dict]:
    """
    Distribute target_total across characters.
    Each character gets at least min_per_char if available.
    Remaining slots go to highest-quality clips overall.
    """
    by_char: dict[str, list[dict]] = {ch: [] for ch in CHARACTERS}
    for c in candidates:
        ch = c["character"]
        by_char[ch].append(c)

    # Sort each bucket by quality score descending
    for ch in CHARACTERS:
        by_char[ch].sort(key=lambda x: x["quality_score"], reverse=True)

    selected = []
    per_char = max(min_per_char, target_total // len(CHARACTERS))

    # First pass: take up to per_char from each
    for ch in CHARACTERS:
        selected.extend(by_char[ch][:per_char])

    # Second pass: fill up to target_total with remaining best
    already_selected = {id(c) for c in selected}
    remaining = [
        c for c in candidates
        if id(c) not in already_selected
    ]
    remaining.sort(key=lambda x: x["quality_score"], reverse=True)

    slots_left = target_total - len(selected)
    selected.extend(remaining[:max(0, slots_left)])

    return selected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_dir: str,
    output_dir: str = "curated_cat_sounds",
    target_total: int = 30,
    sr: int = 22050,
    verbose: bool = True,
) -> None:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)

    # Create output dirs
    for ch in CHARACTERS:
        (output_path / ch).mkdir(parents=True, exist_ok=True)

    # Collect audio files
    extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    audio_files = [
        f for f in dataset_path.rglob("*")
        if f.suffix.lower() in extensions
    ]

    if not audio_files:
        print(f"No audio files found in {dataset_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Stage 1: Split + feature extract
    candidates = []
    for audio_file in audio_files:
        result = load_audio(str(audio_file), sr=sr)
        if result is None:
            if verbose:
                print(f"  SKIP (load failed): {audio_file.name}")
            continue

        y, _ = result
        events = split_vocal_events(y, sr)

        if verbose:
            print(f"  {audio_file.name}: {len(events)} events")

        for i, chunk in enumerate(events):
            feat = extract_features(chunk, sr)
            ok, reason = passes_quality_gate(feat)
            if not ok:
                if verbose:
                    print(f"    event {i}: REJECT ({reason})")
                continue
            score = quality_score(feat)
            character = classify_character(feat)
            candidates.append({
                "source_file": str(audio_file),
                "event_index": i,
                "audio": chunk,
                "features": feat,
                "quality_score": score,
                "character": character,
                "cluster_id": 0,
            })

    print(f"\n{len(candidates)} candidates passed quality gate")

    if not candidates:
        print("No usable clips found. Try a different dataset or loosen thresholds.")
        return

    # Stage 2: Voice clustering + dominant cluster selection
    candidates = cluster_by_voice(candidates)
    before = len(candidates)
    candidates = pick_dominant_cluster(candidates)
    print(f"Voice clustering: {before} -> {len(candidates)} (dominant cluster)")

    # Stage 3: Select top clips
    selected = select_top_clips(candidates, target_total=target_total)
    print(f"Selected {len(selected)} clips")

    # Stage 4: Save clips + metadata
    metadata = []
    char_counters: dict[str, int] = {ch: 0 for ch in CHARACTERS}

    for clip in selected:
        ch = clip["character"]
        char_counters[ch] += 1
        out_name = f"{ch}_{char_counters[ch]:03d}.wav"
        out_path = output_path / ch / out_name

        # Save wav
        try:
            import soundfile as sf
            sf.write(str(out_path), clip["audio"], sr, subtype="PCM_16")
        except ImportError:
            # Fallback: use scipy
            from scipy.io import wavfile
            audio_int16 = (clip["audio"] * 32767).astype(np.int16)
            wavfile.write(str(out_path), sr, audio_int16)

        feat = clip["features"]
        metadata.append({
            "file": str(out_path.relative_to(output_path)),
            "source_file": clip["source_file"],
            "character": ch,
            "quality_score": round(clip["quality_score"], 4),
            "cluster_id": clip["cluster_id"],
            "duration": round(feat["duration"], 4),
            "rms": round(feat["rms"], 6),
            "peak": round(feat["peak"], 6),
            "pitch_mean": round(feat["pitch_mean"], 2),
            "pitch_contour": [round(p, 2) for p in feat["pitch_contour"][:50]],
            "pitch_slope": round(feat["pitch_slope"], 2),
            "attack_speed": round(feat["attack_speed"], 4),
            "brightness": round(feat["brightness"], 2),
            "roughness": round(feat["roughness"], 6),
            "snr_db": round(feat["snr"], 2),
            "voiced_ratio": round(feat["voiced_ratio"], 4),
            "periodicity": round(feat["periodicity"], 4),
        })
        if verbose:
            print(f"  SAVED {out_name}  q={clip['quality_score']:.3f}  [{ch}]")

    # Write metadata.json
    meta_path = output_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n=== Done ===")
    print(f"Output: {output_path.resolve()}")
    print(f"Metadata: {meta_path}")
    print("\nClips per character:")
    for ch in CHARACTERS:
        count = sum(1 for m in metadata if m["character"] == ch)
        print(f"  {ch:25s}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate raw cat audio dataset into clean meow atoms"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to raw dataset folder (searched recursively)"
    )
    parser.add_argument(
        "--output", default="curated_cat_sounds",
        help="Output directory (default: curated_cat_sounds)"
    )
    parser.add_argument(
        "--target", type=int, default=30,
        help="Target number of output clips (default: 30)"
    )
    parser.add_argument(
        "--sr", type=int, default=22050,
        help="Sample rate (default: 22050)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file logging"
    )
    args = parser.parse_args()

    run_pipeline(
        dataset_dir=args.dataset,
        output_dir=args.output,
        target_total=args.target,
        sr=args.sr,
        verbose=not args.quiet,
    )
