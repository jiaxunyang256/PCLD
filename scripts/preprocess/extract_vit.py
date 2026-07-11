"""Sample video frames and extract ViT CLS embeddings."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def sample_frames(path: Path, count: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    frames = []
    for index in np.linspace(0, total - 1, count).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    videos = sorted(p for p in args.video_dir.rglob("*.mp4"))
    with torch.no_grad():
        for video in videos:
            target = args.output_dir / video.relative_to(args.video_dir).with_suffix(".p")
            if target.exists() and not args.overwrite:
                continue
            frames = sample_frames(video, args.frames)
            if not frames:
                continue
            inputs = processor(images=frames, return_tensors="pt")
            hidden = model(**{k: v.to(args.device) for k, v in inputs.items()}).last_hidden_state[:, 0]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                pickle.dump(np.asarray(hidden.cpu(), dtype=np.float32), handle)
    print(f"processed {len(videos)} videos")


if __name__ == "__main__":
    main()
