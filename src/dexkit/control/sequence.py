"""YAML sequence runner: chains poses, roll moves, gantry moves and waits.

Every step is validated (pose names, travel box, feed) before anything moves.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexkit.config import load_yaml
from dexkit.control.poses import Pose, PoseLibrary, go_to_pose
from dexkit.hw.base import GantryInterface, HandInterface
from dexkit.hw.safety import ESTOP, EStop, TravelBox

log = logging.getLogger(__name__)

STEP_KINDS = ("pose", "gantry", "roll", "wait")


class SequenceError(ValueError):
    pass


@dataclass
class Step:
    kind: str
    pose: str | None = None
    xyz: tuple[float, float, float] | None = None
    roll: float | None = None
    duration: float = 1.0
    feed: float | None = None

    def describe(self) -> str:
        if self.kind == "pose":
            return f"pose {self.pose} over {self.duration:.2f}s"
        if self.kind == "gantry":
            assert self.xyz is not None
            return f"gantry to ({self.xyz[0]:.1f}, {self.xyz[1]:.1f}, {self.xyz[2]:.1f}) mm at F{self.feed:.0f}"
        if self.kind == "roll":
            return f"roll to {self.roll:+.1f} deg over {self.duration:.2f}s"
        return f"wait {self.duration:.2f}s"


@dataclass
class Sequence:
    name: str
    steps: list[Step]


def load_sequence(path: str | Path) -> dict[str, Any]:
    data = load_yaml(path)
    if "steps" not in data or not isinstance(data["steps"], list):
        raise SequenceError(f"{path}: needs a 'steps' list")
    return data


def validate(
    data: dict[str, Any],
    poses: PoseLibrary,
    box: TravelBox | None,
    start_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    feed_max: float = 1500.0,
    feed_default: float = 800.0,
    roll_range: float = 90.0,
) -> Sequence:
    """Resolve partial gantry targets and check everything. Raises SequenceError listing all problems."""
    errors: list[str] = []
    steps: list[Step] = []
    xyz = list(start_xyz)
    for i, raw in enumerate(data.get("steps", []), start=1):
        if not isinstance(raw, dict):
            errors.append(f"step {i}: must be a mapping")
            continue
        kinds = [k for k in STEP_KINDS if k in raw]
        if len(kinds) != 1:
            errors.append(f"step {i}: needs exactly one of {STEP_KINDS}, got {sorted(raw)}")
            continue
        kind = kinds[0]
        duration = float(raw.get("duration", 1.0))
        if duration < 0:
            errors.append(f"step {i}: negative duration")
        if kind == "pose":
            name = str(raw["pose"])
            if name not in poses:
                errors.append(f"step {i}: unknown pose '{name}'")
            steps.append(Step("pose", pose=name, duration=duration))
        elif kind == "roll":
            roll = float(raw["roll"])
            if abs(roll) > roll_range:
                errors.append(f"step {i}: roll {roll} outside +/-{roll_range} deg")
            steps.append(Step("roll", roll=roll, duration=duration))
        elif kind == "wait":
            steps.append(Step("wait", duration=float(raw["wait"])))
        elif kind == "gantry":
            g = raw["gantry"]
            if not isinstance(g, dict) or not set(g) <= {"x", "y", "z"} or not g:
                errors.append(f"step {i}: gantry needs a mapping with x/y/z")
                continue
            for j, ax in enumerate("xyz"):
                if ax in g:
                    xyz[j] = float(g[ax])
            feed = float(raw.get("feed", feed_default))
            if feed <= 0:
                errors.append(f"step {i}: feed must be positive")
            if feed > feed_max:
                errors.append(f"step {i}: feed {feed} exceeds feed_max {feed_max}")
            if box is None:
                errors.append(f"step {i}: gantry step but no gantry configured/connected")
            elif not box.contains(xyz):
                errors.append(f"step {i}: gantry target {tuple(round(v, 2) for v in xyz)} outside travel box "
                              f"{box.min.tolist()}..{box.max.tolist()}")
            steps.append(Step("gantry", xyz=(xyz[0], xyz[1], xyz[2]), feed=feed))
    if not steps and not errors:
        errors.append("sequence has no steps")
    if errors:
        raise SequenceError("sequence invalid:\n  " + "\n  ".join(errors))
    return Sequence(name=str(data.get("name", "sequence")), steps=steps)


def execute(
    seq: Sequence,
    hand: HandInterface,
    gantry: GantryInterface | None,
    poses: PoseLibrary,
    rate_hz: float = 20.0,
    estop: EStop = ESTOP,
) -> None:
    for i, step in enumerate(seq.steps, start=1):
        estop.check()
        log.info("[%d/%d] %s", i, len(seq.steps), step.describe())
        if step.kind == "pose":
            assert step.pose is not None
            go_to_pose(hand, poses[step.pose], step.duration, rate_hz, estop=estop)
        elif step.kind == "roll":
            f, _ = hand.last_command
            go_to_pose(hand, Pose(f, float(step.roll or 0.0)), step.duration, rate_hz, estop=estop)
        elif step.kind == "gantry":
            if gantry is None:
                raise SequenceError("gantry step but no gantry connected")
            assert step.xyz is not None
            gantry.move_to(*step.xyz, feed=step.feed, wait=False)
            deadline = time.monotonic() + 120.0
            while True:
                estop.check()
                st = gantry.get_state()
                if st.state == "Idle":
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("gantry move timed out")
                time.sleep(0.1)
        elif step.kind == "wait":
            end = time.monotonic() + step.duration
            while time.monotonic() < end:
                estop.check()
                time.sleep(min(0.05, max(0.0, end - time.monotonic())))


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import add_gantry_flags, common_parser, init, open_session
    from dexkit.config import load_gantry_config, load_hand_config

    p = common_parser("Run a YAML sequence of poses and gantry moves.")
    p.add_argument("sequence", help="path to sequence YAML")
    p.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    add_gantry_flags(p)
    args = p.parse_args(argv)
    init(args)

    hand_cfg = load_hand_config()
    poses = PoseLibrary.load(hand_cfg)
    data = load_sequence(args.sequence)
    has_gantry_steps = any(isinstance(s, dict) and "gantry" in s for s in data["steps"])
    gcfg = load_gantry_config() if has_gantry_steps and not args.no_gantry else None
    box = TravelBox(gcfg.travel_min, gcfg.travel_max) if gcfg else None
    kw = {"feed_max": gcfg.feed_max_mm_min, "feed_default": gcfg.feed_default_mm_min} if gcfg else {}

    # Validate first against the zeroed origin; re-validated below from the live position.
    seq = validate(data, poses, box, roll_range=hand_cfg.roll.range_deg, **kw)
    print(f"sequence '{seq.name}': {len(seq.steps)} steps")
    for i, st in enumerate(seq.steps, start=1):
        print(f"  {i:2d}. {st.describe()}")
    if args.dry_run:
        print("dry run: nothing moved")
        return

    with open_session(args, need_hand=True, want_gantry=has_gantry_steps) as s:
        assert s.hand is not None
        if has_gantry_steps:
            if s.gantry is None or not s.gantry.frame_valid:
                raise SystemExit("gantry frame not valid: zero it (dexkit-teleop, key z) or pass --restore-zero")
            start = tuple(float(v) for v in s.gantry.get_state().xyz)
            seq = validate(data, poses, box, start_xyz=start, roll_range=hand_cfg.roll.range_deg, **kw)
        t0 = time.monotonic()
        execute(seq, s.hand, s.gantry, poses, rate_hz=hand_cfg.rate_hz, estop=s.estop)
        print(f"sequence '{seq.name}' done in {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
