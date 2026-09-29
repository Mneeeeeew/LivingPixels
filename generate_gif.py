"""Create a portfolio-friendly growth and regeneration GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from livingpixels.nca import NCA, NCAConfig


def to_image(state: np.ndarray, scale: int) -> Image.Image:
    rgba = np.clip(NCA.render(state)[0].transpose(1, 2, 0), 0.0, 1.0)
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
    image = Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def load_model(checkpoint: Path) -> NCA:
    with np.load(checkpoint) as data:
        channels = int(data["b2"].shape[0])
        hidden_channels = int(data["b1"].shape[0])
    model = NCA(NCAConfig(channels=channels, hidden_channels=hidden_channels), seed=7)
    model.load(checkpoint)
    model.rng = np.random.default_rng(123)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/livingpixels.gif"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--growth-steps", type=int, default=96)
    parser.add_argument("--repair-steps", type=int, default=64)
    parser.add_argument("--damage-fraction", type=float, default=0.35)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--duration", type=int, default=80)
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("frame-stride must be positive")
    model = load_model(args.checkpoint)
    state = model.seed_state(size=args.size)
    frames: list[Image.Image] = [to_image(state, args.scale)]

    for step in range(1, args.growth_steps + 1):
        state = model.step(state, training=True)
        if step % args.frame_stride == 0 or step == args.growth_steps:
            frames.append(to_image(state, args.scale))

    state = model.damage(state, fraction=args.damage_fraction, mode="circle")
    frames.append(to_image(state, args.scale))
    for step in range(1, args.repair_steps + 1):
        state = model.step(state, training=True)
        if step % args.frame_stride == 0 or step == args.repair_steps:
            frames.append(to_image(state, args.scale))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=args.duration, loop=0, optimize=False)
    print(f"frames={len(frames)}")
    print(f"gif={args.output}")


if __name__ == "__main__":
    main()

