"""Min/max normalization to [-1, 1] (replaces CMU's hard-coded norm.py arrays)."""

from __future__ import annotations

import numpy as np


def normalize_data(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    rng = np.where(stats["max"] - stats["min"] == 0, 1.0, stats["max"] - stats["min"])
    return 2.0 * (np.asarray(data, dtype=np.float64) - stats["min"]) / rng - 1.0


def unnormalize_data(ndata: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    rng = stats["max"] - stats["min"]
    return (np.asarray(ndata, dtype=np.float64) + 1.0) / 2.0 * rng + stats["min"]
