"""Extract 40-dimensional MFCC sequences from WAV files."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import librosa
    files = sorted(args.audio_dir.rglob("*.wav"))
    for source in files:
        target = args.output_dir / source.relative_to(args.audio_dir).with_suffix(".p")
        if target.exists() and not args.overwrite:
            continue
        audio, sample_rate = librosa.load(source, sr=16000, mono=True)
        feature = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=args.n_mfcc).T.astype(np.float32)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(feature, handle)
    print(f"processed {len(files)} audio files")


if __name__ == "__main__":
    main()
