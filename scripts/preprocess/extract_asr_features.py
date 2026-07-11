"""Encode ASR transcripts into fixed vectors with a Hugging Face encoder."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    files = sorted(args.text_dir.rglob("*.txt"))
    with torch.no_grad():
        for source in files:
            target = args.output_dir / source.relative_to(args.text_dir).with_suffix(".p")
            if target.exists() and not args.overwrite:
                continue
            tokens = tokenizer(source.read_text(encoding="utf-8"), truncation=True, max_length=args.max_length,
                               return_tensors="pt")
            output = model(**{k: v.to(args.device) for k, v in tokens.items()}).last_hidden_state[:, 0]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                pickle.dump(np.asarray(output[0].cpu(), dtype=np.float32), handle)
    print(f"processed {len(files)} transcripts")


if __name__ == "__main__":
    main()
