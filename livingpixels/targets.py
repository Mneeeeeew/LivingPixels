"""Deterministic target images."""

from __future__ import annotations

import numpy as np


def living_pixel_target(size: int = 32) -> np.ndarray:
    image = np.zeros((4, size, size), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    cx = cy = (size - 1) / 2
    body = ((xx - cx) / (size * 0.28)) ** 2 + ((yy - cy) / (size * 0.34)) ** 2 <= 1
    ring = ((xx - cx) / (size * 0.40)) ** 2 + ((yy - cy) / (size * 0.46)) ** 2 <= 1
    image[0, body], image[1, body], image[2, body], image[3, body] = 0.18, 0.85, 1.0, 1.0
    image[0, ring & ~body], image[1, ring & ~body], image[2, ring & ~body], image[3, ring & ~body] = 0.08, 0.28, 0.48, 0.30
    radius = max(1, size // 16)
    for eye_x in (int(size * 0.39), int(size * 0.61)):
        eye = (xx - eye_x) ** 2 + (yy - int(size * 0.45)) ** 2 <= radius ** 2
        image[:, eye] = 0
        image[0, eye], image[1, eye], image[3, eye] = 1.0, 0.35, 1.0
    return image

