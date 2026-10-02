"""DexKitEnv: step(action) / reset() / observe() over the 16-dim action space.

Mirrors the contract of CMU's foam_env.FoamEnv so the policy layer can swap it in.
step() order: EStop.check() -> clamp -> drivers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from dexkit.hw.base import (
    ACTION_DIM,
    N_FINGERS,
    GantryInterface,
    HandInterface,
    join_action,
    split_action,
)
from dexkit.hw.safety import ESTOP, EStop, SafetyTrip

log = logging.getLogger(__name__)


class DexKitEnv:
    def __init__(
        self,
        hand: HandInterface,
        gantry: GantryInterface | None = None,
        camera: Any | None = None,
        estop: EStop = ESTOP,
        gantry_epsilon_mm: float = 0.05,
        gantry_feed: float | None = None,
    ) -> None:
        self.hand = hand
        self.gantry = gantry
        self.camera = camera
        self.estop = estop
        self.gantry_epsilon_mm = gantry_epsilon_mm
        self.gantry_feed = gantry_feed
        self._last_gantry_target: np.ndarray | None = None
        self.last_action = np.zeros(ACTION_DIM)
        self.last_step_ms = 0.0

    @property
    def gantry_enabled(self) -> bool:
        return self.gantry is not None

    def _gantry_xyz(self) -> np.ndarray:
        if self.gantry is None:
            return np.zeros(3)
        return self.gantry.get_state().xyz

    def agent_pos(self) -> np.ndarray:
        hs = self.hand.get_state()
        return join_action(np.clip(hs.fingers, 0, 1), hs.roll_deg, self._gantry_xyz())

    def observe(self) -> dict[str, Any]:
        hs = self.hand.get_state()
        gs = self.gantry.get_state() if self.gantry is not None else None
        xyz = gs.xyz if gs is not None else np.zeros(3)
        obs: dict[str, Any] = {
            "agent_pos": join_action(np.clip(hs.fingers, 0, 1), hs.roll_deg, xyz),
            "hand_state": hs,
            "gantry_state": gs,
            "image": None,
        }
        if self.camera is not None:
            obs["image"] = self.camera.read()
        return obs

    def reset(self, duration_s: float = 1.0, rate_hz: float = 20.0) -> dict[str, Any]:
        """Open the hand smoothly; the gantry stays where it is."""
        self.estop.check()
        start, roll = self.hand.last_command
        n = max(1, int(round(duration_s * rate_hz)))
        for i in range(1, n + 1):
            self.estop.check()
            a = i / n
            self.hand.set_targets(start * (1 - a), roll * (1 - a))
            time.sleep(1.0 / rate_hz)
        self._last_gantry_target = None
        return self.observe()

    def step(self, action: np.ndarray, observe: bool = True) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.estop.check()
        fingers, roll, xyz = split_action(action)
        if not np.all(np.isfinite(action)):
            raise SafetyTrip(f"non-finite action {action}")
        sent_f, sent_r = self.hand.set_targets(np.clip(fingers, 0.0, 1.0), roll)
        sent_xyz = xyz
        if self.gantry is not None:
            if not self.gantry.frame_valid:
                raise SafetyTrip("gantry frame not valid; zero or home before stepping")
            last = self._last_gantry_target
            if last is None or np.max(np.abs(xyz - last)) > self.gantry_epsilon_mm:
                if self.gantry.move_to(*xyz, feed=self.gantry_feed, wait=False):
                    self._last_gantry_target = xyz.copy()
            sent_xyz = self._last_gantry_target if self._last_gantry_target is not None else xyz
        self.last_action = join_action(sent_f, sent_r, sent_xyz)
        out = self.observe() if observe else {}
        out["action_sent"] = self.last_action.copy()
        self.last_step_ms = (time.perf_counter() - t0) * 1000
        return out

    def hold_action(self) -> np.ndarray:
        """Current commanded action (useful as a starting point for teleop/replay)."""
        f, r = self.hand.last_command
        xyz = self._gantry_xyz() if self.gantry is not None and self.gantry.frame_valid else np.zeros(3)
        return join_action(f[:N_FINGERS], r, xyz)
