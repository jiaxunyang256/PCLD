"""Detect faces and optionally encode them with a TorchScript FER-VT model."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="TorchScript FER-VT feature extractor")
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    encoder = torch.jit.load(str(args.model), map_location=args.device).eval()
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    videos = sorted(args.video_dir.rglob("*.mp4"))
    for video in videos:
        target = args.output_dir / video.relative_to(args.video_dir).with_suffix(".p")
        if target.exists() and not args.overwrite:
            continue
        capture, features, frame_no = cv2.VideoCapture(str(video)), [], 0
        with torch.no_grad():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_no % args.stride == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for x, y, w, h in detector.detectMultiScale(gray, 1.1, 4):
                        face = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
                        tensor = torch.from_numpy(face[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(args.device)
                        value = encoder(tensor)
                        if isinstance(value, (tuple, list)):
                            value = value[0]
                        features.append(value.flatten().cpu().numpy())
                frame_no += 1
        capture.release()
        if features:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                pickle.dump(np.asarray(features, dtype=np.float32), handle)
    print(f"processed {len(videos)} videos")


if __name__ == "__main__":
    main()
