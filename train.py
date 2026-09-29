"""Train a LivingPixels model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from livingpixels.nca import NCA, NCAConfig, mse_rgba
from livingpixels.targets import living_pixel_target
from livingpixels.training import PoolConfig, fit, fit_pool


def save_rgba(path: Path, image: np.ndarray, scale: int = 8) -> None:
    rgba = np.clip(image.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").resize((rgba.shape[1] * scale, rgba.shape[0] * scale), Image.Resampling.NEAREST).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--seed-fraction", type=float, default=0.25)
    parser.add_argument("--damage-probability", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/livingpixels.npz"))
    args = parser.parse_args()
    target = living_pixel_target(args.size)
    model = NCA(NCAConfig(), seed=7)
    print(f"parameters={model.parameter_count:,}")
    if args.pool:
        history = fit_pool(model, target, args.iterations, args.steps, PoolConfig(args.pool_size, args.batch_size, args.seed_fraction, args.damage_probability))
    else:
        history = fit(model, target, args.iterations, args.steps, args.batch_size)
    model.save(args.output)
    model.rng = np.random.default_rng(123)
    preview = model.rollout(model.seed_state(size=args.size), args.steps, training=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_rgba(args.output.with_name("preview.png"), model.render(preview)[0])
    save_rgba(args.output.with_name("target.png"), target)
    np.save(args.output.with_name("loss.npy"), np.asarray(history, dtype=np.float32))
    print(f"final_loss={mse_rgba(preview, target):.6f}")
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    main()

