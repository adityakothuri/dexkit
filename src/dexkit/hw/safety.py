"""Safety layer: every hardware command passes through here first.

VoltageGate  - servo supply voltage window (guards against the Drok PSU left at 12 V)
TickClamp    - per-servo [lo, hi] tick limits plus max step per tick
LoadWatch    - back a finger off toward slack if it stalls against a jammed tendon
TempWatch    - trip above the configured temperature
TravelBox    - gantry soft limits in the zeroed frame
EStop        - one object that stops everything, in a fixed order, idempotently
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dexkit.config import data_dir

if TYPE_CHECKING:
    from dexkit.hw.base import GantryInterface, HandInterface

log = logging.getLogger(__name__)


class SafetyTrip(RuntimeError):
    """Raised when a safety check fails. Callers must stop commanding hardware."""


class EStopTripped(SafetyTrip):
    pass


# --------------------------------------------------------------------------- voltage


class VoltageGate:
    def __init__(self, window: tuple[float, float]) -> None:
        self.lo, self.hi = window

    def violations(self, voltages: dict[int, float]) -> dict[int, float]:
        return {sid: v for sid, v in voltages.items() if not self.lo <= v <= self.hi}

    def check(self, voltages: dict[int, float], expected_ids: Iterable[int] | None = None) -> None:
        if expected_ids is not None:
            missing = sorted(set(expected_ids) - set(voltages))
            if missing:
                raise SafetyTrip(f"no voltage reading from servos {missing}")
        if not voltages:
            raise SafetyTrip("no voltage readings at all")
        bad = self.violations(voltages)
        if bad:
            desc = ", ".join(f"id {k}: {v:.1f} V" for k, v in sorted(bad.items()))
            raise SafetyTrip(
                f"servo voltage outside {self.lo:.1f}-{self.hi:.1f} V ({desc}). "
                "Check the PSU setting before enabling torque."
            )


# --------------------------------------------------------------------------- clamps


class TickClamp:
    """Per-servo tick limits and per-tick slew limit, vectorized over servo order."""

    def __init__(self, lo: Iterable[int], hi: Iterable[int], max_delta: Iterable[int]) -> None:
        self.lo = np.asarray(list(lo), dtype=np.int64)
        self.hi = np.asarray(list(hi), dtype=np.int64)
        self.max_delta = np.asarray(list(max_delta), dtype=np.int64)
        if not (self.lo.shape == self.hi.shape == self.max_delta.shape):
            raise ValueError("lo, hi, max_delta must have the same length")
        if np.any(self.lo > self.hi):
            raise ValueError("lo must be <= hi for every servo")

    def clamp(self, target: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(target).astype(np.int64), self.lo, self.hi)

    def apply(self, target: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        """Clamp to limits, then limit the step from `previous` (last commanded ticks).

        The result always lies between `previous` and an in-range target, so a servo
        that starts outside its limits walks back into range instead of jumping.
        """
        t = self.clamp(target)
        if previous is not None:
            prev = np.asarray(previous, dtype=np.int64)
            t = np.clip(t, prev - self.max_delta, prev + self.max_delta)
        return t


class LoadWatch:
    """Tracks how long each finger has been above its stall load."""

    def __init__(self, stall_load: Iterable[int], stall_time_s: float = 0.5) -> None:
        self.stall_load = np.asarray(list(stall_load), dtype=float)
        self.stall_time_s = stall_time_s
        self._since: dict[int, float] = {}

    def update(self, loads: np.ndarray, now: float) -> list[int]:
        """Return indices that have been stalled for longer than stall_time_s."""
        stalled = []
        for i, load in enumerate(np.asarray(loads, dtype=float)):
            if load > self.stall_load[i]:
                start = self._since.setdefault(i, now)
                if now - start >= self.stall_time_s:
                    stalled.append(i)
                    self._since[i] = now  # re-arm so back-off steps are spaced
            else:
                self._since.pop(i, None)
        return stalled


class TempWatch:
    def __init__(self, max_c: float = 65.0) -> None:
        self.max_c = max_c

    def check(self, temps: dict[int, float]) -> None:
        hot = {k: v for k, v in temps.items() if v > self.max_c}
        if hot:
            desc = ", ".join(f"id {k}: {v:.0f} C" for k, v in sorted(hot.items()))
            raise SafetyTrip(f"servo over temperature ({desc} > {self.max_c:.0f} C)")


# --------------------------------------------------------------------------- gantry


class TravelBox:
    def __init__(self, min_xyz: Iterable[float], max_xyz: Iterable[float]) -> None:
        self.min = np.asarray(list(min_xyz), dtype=float)
        self.max = np.asarray(list(max_xyz), dtype=float)
        if self.min.shape != (3,) or self.max.shape != (3,) or np.any(self.min >= self.max):
            raise ValueError("travel box needs 3-vectors with min < max")

    def contains(self, xyz: Iterable[float], tol: float = 1e-9) -> bool:
        p = np.asarray(list(xyz), dtype=float)
        return bool(np.all(p >= self.min - tol) and np.all(p <= self.max + tol))

    def check(self, x: float, y: float, z: float) -> None:
        if not self.contains((x, y, z)):
            raise SafetyTrip(
                f"gantry target ({x:.2f}, {y:.2f}, {z:.2f}) outside travel box "
                f"min {self.min.tolist()} max {self.max.tolist()}"
            )

    def clamp_jog(self, current: Iterable[float], delta: Iterable[float]) -> np.ndarray:
        """Shrink a jog delta so current + delta stays inside the box."""
        cur = np.asarray(list(current), dtype=float)
        target = np.clip(cur + np.asarray(list(delta), dtype=float), self.min, self.max)
        d = target - cur
        d[np.abs(d) < 1e-6] = 0.0
        return d


# --------------------------------------------------------------------------- e-stop


def estop_flag_path() -> Path:
    """Written by `dexkit-estop` so every running control loop stops on its next tick."""
    return data_dir() / "state" / "ESTOP"


def set_estop_flag(reason: str) -> Path:
    p = estop_flag_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason}\n")
    return p


def clear_estop_flag() -> bool:
    p = estop_flag_path()
    if p.exists():
        p.unlink()
        return True
    return False


class EStop:
    """trip(reason): GRBL '!' -> 0x85 -> 0x18, then torque off, then set `tripped`.

    Each step is attempted even if an earlier one fails. Idempotent: a second trip
    re-sends the stop commands (cheap and safe) but keeps the first reason.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.tripped = False
        self.reason: str | None = None
        self._hands: list[HandInterface] = []
        self._gantries: list[GantryInterface] = []
        self.log: list[str] = []

    def register(self, hand: HandInterface | None = None, gantry: GantryInterface | None = None) -> None:
        if hand is not None and hand not in self._hands:
            self._hands.append(hand)
        if gantry is not None and gantry not in self._gantries:
            self._gantries.append(gantry)

    def _try(self, label: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
            self.log.append(label)
        except Exception as e:  # noqa: BLE001 - e-stop must continue no matter what
            self.log.append(f"{label}:failed")
            log.error("e-stop step %s failed: %s", label, e)

    def trip(self, reason: str) -> None:
        with self._lock:
            if not self.tripped:
                self.reason = reason
                log.critical("EMERGENCY STOP: %s", reason)
            self.tripped = True
            for g in self._gantries:
                self._try("gantry.feed_hold", g.feed_hold)
                self._try("gantry.jog_cancel", g.jog_cancel)
                self._try("gantry.soft_reset", lambda g=g: g.soft_reset(wait=False))
            for h in self._hands:
                self._try("hand.relax", h.relax)

    def check(self) -> None:
        if not self.tripped and estop_flag_path().exists():
            self.trip("external e-stop (dexkit-estop); clear with dexkit-estop --clear")
        if self.tripped:
            raise EStopTripped(f"e-stop tripped: {self.reason}")

    def reset(self) -> None:
        """Clear the flag. Only for tests or an explicit operator re-arm."""
        with self._lock:
            self.tripped = False
            self.reason = None


ESTOP = EStop()


def install_shutdown_handlers(hand: HandInterface | None, gantry: GantryInterface | None) -> None:
    """SIGINT, SIGTERM and atexit all route to hand.relax() and gantry feed hold."""

    def shutdown() -> None:
        if gantry is not None:
            try:
                gantry.feed_hold()
                gantry.jog_cancel()
            except Exception as e:  # noqa: BLE001
                log.debug("gantry shutdown: %s", e)
        if hand is not None:
            try:
                hand.relax()
            except Exception as e:  # noqa: BLE001
                log.debug("hand shutdown: %s", e)

    atexit.register(shutdown)

    def handler(signum: int, _frame: object) -> None:
        log.warning("signal %d: relaxing hand and holding gantry", signum)
        shutdown()
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
