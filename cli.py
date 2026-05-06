"""
Cat vocalization engine CLI.

Usage:
    python cli.py input.wav --dataset "C:/Users/aman/cat audio clean/curated_cat_sounds" --out output.wav
    python cli.py input.wav --dataset ./curated_cat_sounds --out output.wav --style expressive
"""
import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


def _run() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audio to cat vocalizations, or interpret cat audio as human-readable intent"
    )
    parser.add_argument("input", help="Input audio file (wav / mp3 / m4a / flac)")
    parser.add_argument(
        "--dataset",
        help="Path to curated_cat_sounds directory (required unless --to-human is used)"
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
    parser.add_argument(
        "--to-human", action="store_true",
        help="Interpret cat audio as human-readable intent instead of rendering cat audio"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print JSON result when used with --to-human"
    )
    parser.add_argument(
        "--speak", action="store_true",
        help="Speak the translation aloud (requires --to-human)"
    )
    parser.add_argument(
        "--voice-out", metavar="PATH",
        help="Save spoken translation as WAV (requires --to-human)"
    )
    parser.add_argument(
        "--voice", default="en-US-GuyNeural",
        choices=["en-US-GuyNeural", "en-GB-SoniaNeural"],
        help="Neural TTS voice for --voice-out (default: en-US-GuyNeural)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if (args.speak or args.voice_out) and not args.to_human:
        print("Error: --speak and --voice-out require --to-human", file=sys.stderr)
        sys.exit(1)

    from engine.audio_io import load_audio, save_audio, validate_sample_rate

    try:
        args.sr = validate_sample_rate(args.sr)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.to_human:
        from engine.cat_interpreter import translate_cat_audio

        print("Loading cat audio...")
        y, sr = load_audio(str(input_path), sr=args.sr)
        result = translate_cat_audio(y, sr)

        if args.json:
            print(json.dumps(result, indent=2))
            return

        print(f"Input      : {input_path}")
        print(f"Duration   : {result['input_seconds']:.2f}s")
        print(f"Segments   : {result['segments']}")
        print(f"Intent     : {result['label']} ({result['confidence']:.2f})")
        print(f"Translation: {result['translation']}")
        print(f"Note       : {result['note']}")
        if result["segment_interpretations"]:
            print("\nSegments:")
            for segment in result["segment_interpretations"]:
                print(
                    f"  {segment['index']:02d}. {segment['start_s']:.2f}-{segment['end_s']:.2f}s "
                    f"{segment['label']} ({segment['confidence']:.2f}) - "
                    f"{segment['translation']}"
                )

        if args.speak or args.voice_out:
            from engine.tts import speak_text
            spoken_text = result["translation"]
            if args.voice_out:
                speak_text(spoken_text, output_path=args.voice_out, voice=args.voice)
                print(f"\nVoice saved: {args.voice_out}")
            if args.speak:
                speak_text(spoken_text)
        return

    if not args.dataset:
        print("Error: --dataset is required unless --to-human is used", file=sys.stderr)
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    print(f"Input  : {input_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Style  : {args.style}")
    print(f"Output : {args.out}")
    print()

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


def main() -> None:
    try:
        _run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
