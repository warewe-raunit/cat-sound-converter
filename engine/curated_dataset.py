"""
Loads and indexes curated cat clips. Cycles through top clips per character
for subtle variety while staying consistent within a session.
"""
import json
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine.audio_io import validate_sample_rate

# Fallback priority when a requested character has no clips
_FALLBACK_ORDER = [
    "neutral_attention",
    "calm_soft",
    "lonely_falling",
    "playful_chirp_trill",
    "annoyed_urgent",
    "purr",
]


class CuratedDataset:
    def __init__(self, dataset_dir: str, sr: int = 22050):
        self.dir = Path(dataset_dir)
        self.sr = validate_sample_rate(sr)
        self._cache: dict[str, np.ndarray] = {}
        self._cursors: dict[str, int] = defaultdict(int)

        meta_path = self.dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {dataset_dir}")

        with open(meta_path) as f:
            metadata = json.load(f)

        self._by_char: dict[str, list[dict]] = defaultdict(list)
        skipped_missing: list[str] = []
        for entry in metadata:
            rel_path = entry.get("file")
            character = entry.get("character")
            if not rel_path or not character:
                continue
            if not (self.dir / rel_path).exists():
                skipped_missing.append(rel_path)
                continue
            self._by_char[entry["character"]].append(entry)

        if not self._by_char:
            raise RuntimeError(
                f"No usable clips found in {dataset_dir}: metadata entries point to missing files"
            )

        # Sort each bucket best-first
        for ch in self._by_char:
            self._by_char[ch].sort(
                key=lambda x: x.get("quality_score", 0), reverse=True
            )

        self._available = {ch for ch, clips in self._by_char.items() if clips}
        print(f"Dataset: {sum(len(v) for v in self._by_char.values())} clips total")
        for ch, clips in self._by_char.items():
            print(f"  {ch}: {len(clips)}")
        if skipped_missing:
            print(
                f"  (skipped {len(skipped_missing)} metadata row(s) with missing audio files)"
            )
        missing = [c for c in _FALLBACK_ORDER if c not in self._available]
        if missing:
            print(f"  (missing: {', '.join(missing)} - will use fallbacks)")

    # ------------------------------------------------------------------

    def _load(self, entry: dict) -> np.ndarray:
        key = entry["file"]
        if key not in self._cache:
            full = self.dir / entry["file"]
            if not full.exists():
                raise FileNotFoundError(f"Curated clip missing: {full}")
            from engine.audio_io import load_audio
            y, _ = load_audio(str(full), sr=self.sr)
            self._cache[key] = y
        return self._cache[key].copy()

    def _fallback(self, character: str) -> str:
        if character in self._available:
            return character
        for fb in _FALLBACK_ORDER:
            if fb in self._available:
                return fb
        raise RuntimeError("Dataset is empty — no clips available")

    def get_clip(self, character: str) -> tuple[np.ndarray, dict]:
        """
        Return (audio, metadata_entry) for the requested character.
        Cycles through top-5 clips to avoid always using the same one.
        Falls back gracefully when character has no clips.
        """
        char = self._fallback(character)
        clips = self._by_char[char]
        top_n = min(5, len(clips))
        idx = self._cursors[char] % top_n
        self._cursors[char] += 1
        entry = clips[idx]
        return self._load(entry), entry
