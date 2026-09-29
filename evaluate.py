"""Run a damage sweep and save JSON, CSV, and SVG."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from livingpixels.evaluation import DamageResult, damage_sweep
from livingpixels.nca import NCA, NCAConfig
from livingpixels.targets import living_pixel_target


def write_svg(path: Path, results: list[DamageResult]) -> None:
    width, height, left, top, chart_width, chart_height = 760, 460, 70, 35, 640, 340
    fractions = sorted({r.fraction for r in results})
    modes = list(dict.fromkeys(r.mode for r in results))
    colors = {mode: color for mode, color in zip(modes, ("#58d5e6", "#ff8f70", "#b9e769"))}
    x = lambda value: left + (value - min(fractions)) / max(1e-9, max(fractions) - min(fractions)) * chart_width
    y = lambda value: top + (1 - value) * chart_height
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="#081018"/><text x="70" y="22" fill="#d8f4ff" font-family="monospace" font-size="16">LivingPixels damage recovery</text>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#5b7a86"/><line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#5b7a86"/>']
    for mode in modes:
        points = [r for r in results if r.mode == mode]
        lines.append(f'<polyline points="{" ".join(f"{x(r.fraction):.1f},{y(r.success_rate):.1f}" for r in points)}" fill="none" stroke="{colors[mode]}" stroke-width="3"/>')
    for index, mode in enumerate(modes):
        lines.append(f'<text x="{left + index * 135}" y="{height - 28}" fill="{colors[mode]}" font-family="monospace" font-size="13">● {mode}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--growth-steps", type=int, default=96)
    parser.add_argument("--repair-steps", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/damage-sweep"))
    args = parser.parse_args()
    with np.load(args.checkpoint) as data:
        config = NCAConfig(channels=int(data["b2"].shape[0]), hidden_channels=int(data["b1"].shape[0]))
    model = NCA(config, seed=7)
    model.load(args.checkpoint)
    results = damage_sweep(model, living_pixel_target(args.size), trials=args.trials, growth_steps=args.growth_steps, repair_steps=args.repair_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].as_dict()))
        writer.writeheader(); writer.writerows(r.as_dict() for r in results)
    write_svg(args.output_dir / "success-rate.svg", results)
    for r in results: print(f"{r.mode:6s} damage={r.fraction:.2f} success={r.success_rate:.1%} loss={r.mean_loss:.5f}")
    print(f"results={args.output_dir}")


if __name__ == "__main__":
    main()

