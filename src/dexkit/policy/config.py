"""Policy configuration and normalization bounds derived from hand.yaml / gantry.yaml."""

from __future__ import annotations

import numpy as np

from dexkit.config import GantryConfig, HandConfig, PolicyConfig, load_policy_config
from dexkit.hw.base import ACTION_DIM, GANTRY_SLICE, N_FINGERS, ROLL_INDEX

__all__ = ["PolicyConfig", "load_policy_config", "build_stats"]


def build_stats(hand_cfg: HandConfig, gantry_cfg: GantryConfig | None) -> dict[str, np.ndarray]:
    """Per-dimension min/max: fingers [0,1], roll +/- range_deg, gantry travel box (mm)."""
    lo = np.zeros(ACTION_DIM)
    hi = np.ones(ACTION_DIM)
    lo[:N_FINGERS], hi[:N_FINGERS] = 0.0, 1.0
    lo[ROLL_INDEX], hi[ROLL_INDEX] = -hand_cfg.roll.range_deg, hand_cfg.roll.range_deg
    if gantry_cfg is not None:
        lo[GANTRY_SLICE] = gantry_cfg.travel_min
        hi[GANTRY_SLICE] = gantry_cfg.travel_max
    else:
        lo[GANTRY_SLICE], hi[GANTRY_SLICE] = -1.0, 1.0
    return {"min": lo, "max": hi}
