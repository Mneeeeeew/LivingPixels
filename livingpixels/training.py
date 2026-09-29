"""Manual backpropagation and persistent-pool training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nca import NCA, Array


@dataclass
class Adam:
    learning_rate: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.t = 0
        self.m: dict[str, Array] = {}
        self.v: dict[str, Array] = {}

    def step(self, model: NCA, gradients: dict[str, Array]) -> None:
        self.t += 1
        for name, gradient in gradients.items():
            parameter = getattr(model, name)
            self.m.setdefault(name, np.zeros_like(parameter))
            self.v.setdefault(name, np.zeros_like(parameter))
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * gradient
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * gradient * gradient
            mean = self.m[name] / (1 - self.beta1 ** self.t)
            variance = self.v[name] / (1 - self.beta2 ** self.t)
            parameter -= self.learning_rate * mean / (np.sqrt(variance) + self.epsilon)


def _backward_perception(model: NCA, dfeatures: Array) -> Array:
    batch, _, height, width = dfeatures.shape
    channels = model.config.channels
    dstate = dfeatures[:, :channels].copy()
    for filter_index, kernel in enumerate(model.kernels):
        gradient = dfeatures[:, (filter_index + 1) * channels:(filter_index + 2) * channels]
        padded = np.zeros((batch, channels, height + 2, width + 2), dtype=dfeatures.dtype)
        for row in range(3):
            for col in range(3):
                if kernel[row, col]:
                    padded[:, :, row:row + height, col:col + width] += kernel[row, col] * gradient
        dstate += padded[:, :, 1:-1, 1:-1]
    return dstate


def train_step(model: NCA, target: Array, steps: int = 48, batch_size: int = 8, initial_state: Array | None = None, return_state: bool = False):
    """Run one rollout and return loss, gradients, and optionally final state."""
    if initial_state is None:
        state = model.seed_state(batch_size=batch_size, size=target.shape[-1])
    else:
        state = np.asarray(initial_state, dtype=np.float32).copy()
        batch_size = state.shape[0]
        if state.ndim != 4 or state.shape[1] != model.config.channels or state.shape[-2:] != target.shape[-2:]:
            raise ValueError("initial_state and target have incompatible shapes")

    features_list, hidden_list, preactivation_list = [], [], []
    masks, alive_masks = [], []
    for _ in range(steps):
        features = model.perceive(state)
        batch, _, height, width = features.shape
        flat = features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1])
        preactivation = flat @ model.w1 + model.b1
        hidden = np.maximum(preactivation, 0.0)
        delta = (hidden @ model.w2 + model.b2).reshape(batch, height, width, model.config.channels).transpose(0, 3, 1, 2)
        update_mask = (model.rng.random((batch, 1, height, width)) < model.config.update_probability).astype(np.float32)
        alive = model.alive_mask(state)
        state = state + delta * update_mask * alive
        features_list.append(features)
        hidden_list.append(hidden)
        preactivation_list.append(preactivation)
        masks.append(update_mask)
        alive_masks.append(alive)

    target_batch = np.broadcast_to(target[None, :4], (batch_size, 4, target.shape[-2], target.shape[-1]))
    error = np.clip(state[:, :4], 0, 1) - target_batch
    loss = float(np.mean(error * error))
    dstate = np.zeros_like(state)
    dstate[:, :4] = 2 * error / error.size
    gradients = {name: np.zeros_like(getattr(model, name)) for name in ("w1", "b1", "w2", "b2")}

    for index in range(steps - 1, -1, -1):
        ddelta = dstate * masks[index] * alive_masks[index]
        ddelta_flat = ddelta.transpose(0, 2, 3, 1).reshape(-1, model.config.channels)
        dhidden = ddelta_flat @ model.w2.T
        dz = dhidden * (preactivation_list[index] > 0)
        flat_features = features_list[index].transpose(0, 2, 3, 1).reshape(-1, features_list[index].shape[1])
        gradients["w2"] += hidden_list[index].T @ ddelta_flat
        gradients["b2"] += ddelta.sum(axis=(0, 2, 3))
        gradients["w1"] += flat_features.T @ dz
        gradients["b1"] += dz.sum(axis=0)
        height, width = state.shape[-2:]
        dfeatures = (dz @ model.w1.T).reshape(batch_size, height, width, -1).transpose(0, 3, 1, 2)
        dstate += _backward_perception(model, dfeatures)

    for gradient in gradients.values():
        gradient /= batch_size
        np.clip(gradient, -1, 1, out=gradient)
    result = (loss, gradients, state) if return_state else (loss, gradients)
    return result


def fit(model: NCA, target: Array, iterations: int = 1000, steps: int = 48, batch_size: int = 8, log_every: int = 100) -> list[float]:
    optimizer = Adam()
    history = []
    for iteration in range(1, iterations + 1):
        loss, gradients = train_step(model, target, steps=steps, batch_size=batch_size)
        optimizer.step(model, gradients)
        history.append(loss)
        if log_every and (iteration == 1 or iteration % log_every == 0):
            print(f"iteration={iteration:5d} loss={loss:.6f}")
    return history


@dataclass(frozen=True)
class PoolConfig:
    pool_size: int = 512
    batch_size: int = 8
    seed_fraction: float = 0.25
    damage_probability: float = 0.25


def fit_pool(model: NCA, target: Array, iterations: int = 1000, steps: int = 48, pool_config: PoolConfig | None = None, log_every: int = 100) -> list[float]:
    settings = pool_config or PoolConfig()
    if settings.pool_size < 1 or not 1 <= settings.batch_size <= settings.pool_size:
        raise ValueError("batch_size must be in [1, pool_size]")
    size = target.shape[-1]
    pool = model.seed_state(settings.pool_size, size)
    optimizer = Adam()
    history = []
    seed_count = max(1, int(round(settings.batch_size * settings.seed_fraction)))
    for iteration in range(1, iterations + 1):
        indices = model.rng.choice(settings.pool_size, settings.batch_size, replace=False)
        batch = pool[indices].copy()
        batch[:seed_count] = model.seed_state(seed_count, size)
        if settings.damage_probability and settings.batch_size > seed_count and model.rng.random() < settings.damage_probability:
            damaged_count = max(1, int(round((settings.batch_size - seed_count) * 0.5)))
            damaged = model.rng.choice(np.arange(seed_count, settings.batch_size), damaged_count, replace=False)
            batch[damaged] = model.damage(batch[damaged], float(model.rng.uniform(0.1, 0.5)), "circle")
        loss, gradients, final_state = train_step(model, target, steps=steps, initial_state=batch, return_state=True)
        optimizer.step(model, gradients)
        pool[indices] = final_state
        history.append(loss)
        if log_every and (iteration == 1 or iteration % log_every == 0):
            print(f"pool_iteration={iteration:5d} loss={loss:.6f}")
    return history

