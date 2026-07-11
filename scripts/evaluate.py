from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PCLD checkpoint")
    parser.add_argument("--config", default="configs/pclmmplus.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.config, args.checkpoint, args.split), indent=2))


if __name__ == "__main__":
    main()
