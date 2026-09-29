"""Serialization helpers shared by the checkpoint exporter and local demo API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .nca import NCA, NCAConfig


def model_payload(model: NCA, size: int = 32) -> dict[str, object]:
    return {
        "size": size,
        "channels": model.config.channels,
        "hiddenChannels": model.config.hidden_channels,
        "updateProbability": model.config.update_probability,
        "aliveThreshold": model.config.alive_threshold,
        "w1": model.w1.astype(np.float32).ravel().tolist(),
        "b1": model.b1.astype(np.float32).tolist(),
        "w2": model.w2.astype(np.float32).ravel().tolist(),
        "b2": model.b2.astype(np.float32).tolist(),
    }


def load_checkpoint_payload(checkpoint: str | Path, size: int = 32) -> dict[str, object]:
    checkpoint = Path(checkpoint)
    with np.load(checkpoint) as data:
        config = NCAConfig(channels=int(data["b2"].shape[0]), hidden_channels=int(data["b1"].shape[0]))
        model = NCA(config, seed=7)
        model.load_state_dict({name: data[name] for name in ("w1", "b1", "w2", "b2")})
    return model_payload(model, size=size)

