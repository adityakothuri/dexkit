"""Record (t, action[16], state[16]) at rate_hz and replay through DexKitEnv."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dexkit.config import data_dir
from dexkit.env.dexkit_env import DexKitEnv
from dexkit.hw.base import ACTION_DIM, GANTRY_SLICE

log = logging.getLogger(__name__)


def recordings_dir() -> Path:
    return data_dir() / "recordings"


@dataclass
class Recording:
    t: np.ndarray        # (N,) seconds from start
    action: np.ndarray   # (N, 16)
    state: np.ndarray    # (N, 16)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, t=self.t, action=self.action, state=self.state,
                            meta=np.array([repr(self.meta)]))
        return p

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        p = resolve_recording(path)
        with np.load(p, allow_pickle=False) as z:
            rec = cls(t=z["t"], action=z["action"], state=z["state"])
        if rec.action.ndim != 2 or rec.action.shape[1] != ACTION_DIM:
            raise ValueError(f"{p}: action must be (N, {ACTION_DIM})")
        return rec


def resolve_recording(name: str | Path) -> Path:
    p = Path(name)
    for cand in (p, p.with_suffix(".npz"), recordings_dir() / p, recordings_dir() / p.with_suffix(".npz")):
        if cand.exists() and cand.is_file():
            return cand
    raise FileNotFoundError(f"recording '{name}' not found (looked in {recordings_dir()})")


class Recorder:
    def __init__(self) -> None:
        self.active = False
        self._t0 = 0.0
        self._t: list[float] = []
        self._a: list[np.ndarray] = []
        self._s: list[np.ndarray] = []

    def start(self) -> None:
        self.active = True
        self._t0 = time.monotonic()
        self._t, self._a, self._s = [], [], []
        log.info("recording started")

    def add(self, action: np.ndarray, state: np.ndarray, t: float | None = None) -> None:
        if not self.active:
            return
        self._t.append((time.monotonic() if t is None else t) - self._t0)
        self._a.append(np.asarray(action, dtype=float).copy())
        self._s.append(np.asarray(state, dtype=float).copy())

    def stop(self) -> Recording:
        self.active = False
        rec = Recording(
            t=np.asarray(self._t, dtype=float),
            action=np.asarray(self._a, dtype=float).reshape(-1, ACTION_DIM),
            state=np.asarray(self._s, dtype=float).reshape(-1, ACTION_DIM),
        )
        log.info("recording stopped: %d frames, %.1fs", len(rec), rec.duration)
        return rec

    def stop_and_save(self, name: str | None = None) -> Path | None:
        rec = self.stop()
        if len(rec) == 0:
            return None
        name = name or time.strftime("rec_%Y%m%d_%H%M%S")
        path = rec.save(recordings_dir() / name)
        log.info("saved %s", path)
        return path


def replay(env: DexKitEnv, rec: Recording, speed: float = 1.0, lead_in_s: float = 1.0,
           rate_hz: float = 20.0, use_gantry: bool = True) -> int:
    """Interpolate to the first frame over lead_in_s, then stream actions with recorded timing."""
    if len(rec) == 0:
        return 0
    if speed <= 0:
        raise ValueError("speed must be positive")
    actions = rec.action.copy()
    if not use_gantry or env.gantry is None:
        actions[:, GANTRY_SLICE] = env.hold_action()[GANTRY_SLICE]
    start = env.hold_action()
    if not use_gantry or env.gantry is None:
        start[GANTRY_SLICE] = actions[0, GANTRY_SLICE]
    n = max(1, int(round(lead_in_s * rate_hz)))
    for i in range(1, n + 1):
        a = i / n
        env.step(start + (actions[0] - start) * a, observe=False)
        time.sleep(1.0 / rate_hz)
    if env.gantry is not None and use_gantry:
        env.gantry.wait_idle(timeout=60.0)
    t0 = time.monotonic()
    for k in range(len(actions)):
        due = t0 + rec.t[k] / speed
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        env.step(actions[k], observe=False)
    return len(actions)


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import add_gantry_flags, common_parser, init, open_session

    p = common_parser("Replay a recorded teleop motion.")
    p.add_argument("name", help="recording name in data/recordings or a path to .npz")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--hand-only", action="store_true", help="ignore the gantry columns")
    add_gantry_flags(p)
    args = p.parse_args(argv)
    init(args)
    rec = Recording.load(args.name)
    print(f"recording {args.name}: {len(rec)} frames, {rec.duration:.1f}s")
    use_gantry = not (args.hand_only or args.no_gantry)
    with open_session(args, need_hand=True, want_gantry=use_gantry) as s:
        assert s.hand is not None and s.hand_cfg is not None
        gantry = s.gantry if (s.gantry is not None and s.gantry.frame_valid) else None
        if use_gantry and s.gantry is not None and gantry is None:
            print("gantry frame not valid: replaying hand only")
        env = DexKitEnv(s.hand, gantry, estop=s.estop)
        n = replay(env, rec, speed=args.speed, rate_hz=s.hand_cfg.rate_hz, use_gantry=gantry is not None)
        print(f"replayed {n} frames")


if __name__ == "__main__":
    main()
