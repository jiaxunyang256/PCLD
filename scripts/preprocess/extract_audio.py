"""Extract mono WAV files from videos with FFmpeg."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("FFmpeg is required and must be available on PATH.")
    videos = sorted(p for p in args.video_dir.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"})
    for video in videos:
        relative = video.relative_to(args.video_dir).with_suffix(".wav")
        target = args.output_dir / relative
        if target.exists() and not args.overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", str(args.sample_rate), str(target)], check=True)
    print(f"processed {len(videos)} videos")


if __name__ == "__main__":
    main()
