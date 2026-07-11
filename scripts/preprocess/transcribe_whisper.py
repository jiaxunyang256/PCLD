"""Transcribe WAV files with Whisper and preserve the relative ID layout."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import whisper
    model = whisper.load_model(args.model, device=args.device)
    files = sorted(args.audio_dir.rglob("*.wav"))
    for audio in files:
        target = args.output_dir / audio.relative_to(args.audio_dir).with_suffix(".txt")
        if target.exists() and not args.overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        result = model.transcribe(str(audio), language=args.language, fp16=False if args.device == "cpu" else None)
        target.write_text(result["text"].strip(), encoding="utf-8")
    print(f"processed {len(files)} audio files")


if __name__ == "__main__":
    main()
