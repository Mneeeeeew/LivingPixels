"""Damage and regeneration evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .nca import NCA, Array, mse_rgba


@dataclass(frozen=True)
class DamageResult:
    mode: str
    fraction: float
    trials: int
    success_rate: float
    mean_loss: float
    median_loss: float
    mean_recovery_steps: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def is_recovered(state: Array, target: Array, loss_threshold: float = 0.04, min_alpha_ratio: float = 0.5) -> bool:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim == 3:
        target = target[None, ...]
    target_area = int(np.count_nonzero(target[0, 3] > 0.1))
    predicted_area = int(np.count_nonzero(NCA.render(state)[0, 3] > 0.1))
    return mse_rgba(state, target) <= loss_threshold and predicted_area >= int(target_area * min_alpha_ratio)


def damage_sweep(model: NCA, target: Array, fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.7), modes: tuple[str, ...] = ("circle", "random"), trials: int = 20, growth_steps: int = 96, repair_steps: int = 128, loss_threshold: float = 0.04) -> list[DamageResult]:
    if trials < 1 or repair_steps < 1:
        raise ValueError("trials must be positive and repair_steps must be positive")
    grown = model.rollout(model.seed_state(size=target.shape[-1]), growth_steps, training=True)
    results = []
    for mode in modes:
        for fraction in fractions:
            losses, recovery_times = [], []
            successes = 0
            for _ in range(trials):
                state = model.damage(grown, fraction=fraction, mode=mode)
                recovered_at = None
                for step in range(1, repair_steps + 1):
                    state = model.step(state, training=True)
                    if recovered_at is None and is_recovered(state, target, loss_threshold):
                        recovered_at = step
                losses.append(mse_rgba(state, target))
                if is_recovered(state, target, loss_threshold):
                    successes += 1
                if recovered_at is not None:
                    recovery_times.append(recovered_at)
            results.append(DamageResult(mode, fraction, trials, successes / trials, float(np.mean(losses)), float(np.median(losses)), float(np.mean(recovery_times)) if recovery_times else None))
    return results

