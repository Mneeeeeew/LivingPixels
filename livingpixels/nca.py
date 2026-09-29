"""NumPy implementation of a neural cellular automaton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class NCAConfig:
    channels: int = 16
    hidden_channels: int = 64
    update_probability: float = 0.5
    alive_threshold: float = 0.1


class NCA:
    """A shared local rule operating on [batch, channels, height, width]."""

    def __init__(self, config: NCAConfig | None = None, seed: int = 7):
        self.config = config or NCAConfig()
        if self.config.channels < 4:
            raise ValueError("channels must include RGBA")
        self.rng = np.random.default_rng(seed)
        width = self.config.channels * 3
        self.w1 = (self.rng.standard_normal((width, self.config.hidden_channels)) * np.sqrt(2 / width)).astype(np.float32)
        self.b1 = np.zeros(self.config.hidden_channels, dtype=np.float32)
        self.w2 = np.zeros((self.config.hidden_channels, self.config.channels), dtype=np.float32)
        self.b2 = np.zeros(self.config.channels, dtype=np.float32)
        self.kernels = np.asarray([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        ], dtype=np.float32) / 8.0

    @property
    def parameter_count(self) -> int:
        return sum(getattr(self, name).size for name in ("w1", "b1", "w2", "b2"))

    def state_dict(self) -> dict[str, Array]:
        return {name: getattr(self, name).copy() for name in ("w1", "b1", "w2", "b2")}

    def load_state_dict(self, state: dict[str, Array]) -> None:
        for name in ("w1", "b1", "w2", "b2"):
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != getattr(self, name).shape:
                raise ValueError(f"{name} has shape {value.shape}; expected {getattr(self, name).shape}")
            setattr(self, name, value.copy())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.state_dict())

    def load(self, path: str | Path) -> None:
        with np.load(path) as data:
            self.load_state_dict({name: data[name] for name in ("w1", "b1", "w2", "b2")})

    def seed_state(self, batch_size: int = 1, size: int = 32) -> Array:
        if batch_size < 1 or size < 3:
            raise ValueError("batch_size must be positive and size must be at least 3")
        state = np.zeros((batch_size, self.config.channels, size, size), dtype=np.float32)
        state[:, 3, size // 2, size // 2] = 1.0
        return state

    def perceive(self, state: Array) -> Array:
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 4 or state.shape[1] != self.config.channels:
            raise ValueError("state must have shape [B, channels, H, W]")
        padded = np.pad(state, ((0, 0), (0, 0), (1, 1), (1, 1)))
        features = [state]
        for kernel in self.kernels:
            response = np.zeros_like(state)
            for row in range(3):
                for col in range(3):
                    if kernel[row, col]:
                        response += kernel[row, col] * padded[:, :, row:row + state.shape[2], col:col + state.shape[3]]
            features.append(response)
        return np.concatenate(features, axis=1)

    def alive_mask(self, state: Array) -> Array:
        alpha = state[:, 3:4]
        padded = np.pad(alpha, ((0, 0), (0, 0), (1, 1), (1, 1)))
        neighborhood = np.maximum.reduce([padded[:, :, row:row + alpha.shape[2], col:col + alpha.shape[3]] for row in range(3) for col in range(3)])
        return (neighborhood > self.config.alive_threshold).astype(np.float32)

    def step(self, state: Array, update_mask: Array | None = None, training: bool = True) -> Array:
        features = self.perceive(state)
        batch, _, height, width = features.shape
        flat = features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1])
        hidden = np.maximum(flat @ self.w1 + self.b1, 0.0)
        delta = (hidden @ self.w2 + self.b2).reshape(batch, height, width, self.config.channels).transpose(0, 3, 1, 2)
        if update_mask is None:
            if training:
                update_mask = (self.rng.random((batch, 1, height, width)) < self.config.update_probability).astype(np.float32)
            else:
                update_mask = np.ones((batch, 1, height, width), dtype=np.float32)
        return state + delta * update_mask * self.alive_mask(state)

    def rollout(self, state: Array, steps: int, training: bool = False) -> Array:
        for _ in range(steps):
            state = self.step(state, training=training)
        return state

    @staticmethod
    def render(state: Array) -> Array:
        return np.clip(np.asarray(state)[:, :4], 0.0, 1.0)

    def damage(self, state: Array, fraction: float = 0.25, mode: str = "circle") -> Array:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        damaged = np.asarray(state, dtype=np.float32).copy()
        batch, _, height, width = damaged.shape
        if mode == "circle":
            radius = max(1, int(np.sqrt(fraction * height * width / np.pi)))
            yy, xx = np.ogrid[:height, :width]
            for item in range(batch):
                cy = int(self.rng.integers(radius, max(radius + 1, height - radius)))
                cx = int(self.rng.integers(radius, max(radius + 1, width - radius)))
                damaged[item, :, (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 0.0
        elif mode == "random":
            mask = self.rng.random((batch, 1, height, width)) < fraction
            damaged[mask.repeat(self.config.channels, axis=1)] = 0.0
        else:
            raise ValueError("mode must be 'circle' or 'random'")
        return damaged


def mse_rgba(state: Array, target: Array) -> float:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim == 3:
        target = target[None, ...]
    return float(np.mean((NCA.render(state) - target[:, :4]) ** 2))

