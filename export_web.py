"""Export a NumPy checkpoint to demo/model.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("demo/model.json"))
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    with np.load(args.checkpoint) as data:
        payload = {"size": args.size, "channels": int(data["b2"].shape[0]), "hiddenChannels": int(data["b1"].shape[0]), "updateProbability": 0.5, "aliveThreshold": 0.1, "w1": data["w1"].astype(np.float32).ravel().tolist(), "b1": data["b1"].astype(np.float32).tolist(), "w2": data["w2"].astype(np.float32).ravel().tolist(), "b2": data["b2"].astype(np.float32).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"exported={args.output}")


if __name__ == "__main__":
    main()

