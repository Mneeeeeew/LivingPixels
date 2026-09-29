"""Export a NumPy checkpoint to demo/model.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from livingpixels.web import load_checkpoint_payload

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("demo/model.json"))
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    payload = load_checkpoint_payload(args.checkpoint, size=args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"exported={args.output}")


if __name__ == "__main__":
    main()
