"""
Cat vocalization engine CLI.

Usage:
    python cli.py input.wav --dataset "C:/Users/aman/cat audio clean/curated_cat_sounds" --out output.wav
    python cli.py input.wav --dataset ./curated_cat_sounds --out output.wav --style expressive
"""
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audio to cat vocalizations using DSP engine"
    )
    parser.add_argument("input", help="Input audio file (wav / mp3 / m4a / flac)")
    parser.add_argument(
        "--dataset", required=True,
        help="Path to curated_cat_sounds directory (must contain metadata.json)"
    )
    parser.add_argument(
        "--out", default="output.wav",
        help="Output wav path (default: output.wav)"
    )
    parser.add_argument(
        "--style", default="natural",
        choices=["natural", "expressive", "subtle"],
        help="DSP intensity — natural (default), expressive, subtle"
    )
    parser.add_argument(
        "--sr", type=int, default=22050,
        help="Internal sample rate (default: 22050)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-segment log"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    dataset_path = Path(args.dataset)

    if not input_path.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    print(f"Input  : {input_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Style  : {args.style}")
    print(f"Output : {args.out}")
    print()

    from engine.audio_io import load_audio, save_audio
    from engine.prosody import analyze
    from engine.curated_dataset import CuratedDataset
    from engine.renderer import render

    print("Loading input...")
    y, sr = load_audio(str(input_path), sr=args.sr)
    print(f"  {len(y)/sr:.2f}s  @{sr} Hz\n")

    print("Analysing prosody...")
    segments = analyze(y, sr)
    print(f"  {len(segments)} vocal segment(s) detected\n")

    if not segments:
        print(
            "No vocal segments found. Input may be too quiet or already silent.\n"
            "Try a file with clearer speech or other audio events.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading dataset...")
    dataset = CuratedDataset(str(dataset_path), sr=sr)
    print()

    print(f"Rendering ({args.style})...")
    output = render(segments, dataset, style=args.style, sr=sr, verbose=not args.quiet)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(str(out_path), output, sr)

    dur_in = len(y) / sr
    dur_out = len(output) / sr
    print(f"\nDone.")
    print(f"  Input duration : {dur_in:.2f}s")
    print(f"  Output duration: {dur_out:.2f}s")
    print(f"  Saved          : {out_path.resolve()}")


if __name__ == "__main__":
    main()
