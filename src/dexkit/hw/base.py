"""Hardware abstractions and the fixed 16-dim action layout.

Action vector order (fixed): indices 0..11 finger servos in [0, 1],
12 forearm roll in degrees, 13..15 gantry X Y Z in mm (zeroed frame).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

N_FINGERS = 12
ROLL_INDEX = 12
GANTRY_SLICE = slice(13, 16)
ACTION_DIM = 16


def split_action(action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    a = np.asarray(action, dtype=float)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"action must have shape ({ACTION_DIM},), got {a.shape}")
    return a[:N_FINGERS].copy(), float(a[ROLL_INDEX]), a[GANTRY_SLICE].copy()


def join_action(fingers: np.ndarray, roll_deg: float, xyz: np.ndarray) -> np.ndarray:
    a = np.zeros(ACTION_DIM)
    a[:N_FINGERS] = fingers
    a[ROLL_INDEX] = roll_deg
    a[GANTRY_SLICE] = xyz
    return a


@dataclass
class HandState:
    fingers: np.ndarray                 # (12,) normalized [0,1]
    roll_deg: float
    ticks: dict[int, int]               # raw present position per servo id
    load: dict[int, int] = field(default_factory=dict)
    voltage: dict[int, float] = field(default_factory=dict)
    temperature: dict[int, float] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def min_voltage(self) -> float | None:
        return min(self.voltage.values()) if self.voltage else None


@dataclass
class GantryState:
    state: str                          # Idle, Run, Jog, Hold, Alarm, Door, Home, Check, Sleep, Unknown
    xyz: np.ndarray                     # (3,) work position in the zeroed frame, mm
    mpos: np.ndarray                    # (3,) machine position, mm
    feed: float = 0.0
    planner_free: int | None = None
    substate: str | None = None
    frame_valid: bool = False
    timestamp: float = 0.0

    @property
    def idle(self) -> bool:
        return self.state == "Idle"


class HandInterface(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def set_targets(self, fingers: np.ndarray, roll_deg: float) -> tuple[np.ndarray, float]:
        """Command normalized finger targets and roll; return what was actually sent."""

    @abc.abstractmethod
    def get_state(self) -> HandState: ...

    @abc.abstractmethod
    def read_voltages(self) -> dict[int, float]: ...

    @abc.abstractmethod
    def relax(self) -> None:
        """Torque off on every servo. Must never raise."""

    def rehome(self) -> None:
        """Re-base after the operator opened the hand with torque off: current pose becomes open."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def last_command(self) -> tuple[np.ndarray, float]: ...


class GantryInterface(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None: ...

    @property
    @abc.abstractmethod
    def frame_valid(self) -> bool: ...

    @abc.abstractmethod
    def set_zero(self) -> None: ...

    @abc.abstractmethod
    def home(self) -> None: ...

    @abc.abstractmethod
    def jog(self, dx: float, dy: float, dz: float, feed: float | None = None) -> bool: ...

    @abc.abstractmethod
    def jog_cancel(self) -> None: ...

    @abc.abstractmethod
    def move_to(self, x: float, y: float, z: float, feed: float | None = None, wait: bool = False) -> bool: ...

    @abc.abstractmethod
    def wait_idle(self, timeout: float = 30.0) -> GantryState: ...

    @abc.abstractmethod
    def get_state(self) -> GantryState: ...

    @abc.abstractmethod
    def feed_hold(self) -> None: ...

    @abc.abstractmethod
    def resume(self) -> None: ...

    @abc.abstractmethod
    def soft_reset(self, wait: bool = True) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...
