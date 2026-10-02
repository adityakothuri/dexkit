"""Pose library (12 finger values in [0,1] + roll degrees) and smooth transitions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexkit.config import FINGERS, ROLES, TENDONS, HandConfig, config_dir, dump_yaml, load_yaml
from dexkit.hw.base import N_FINGERS, GantryInterface, HandInterface
from dexkit.hw.safety import ESTOP, EStop, EStopTripped, SafetyTrip
from dexkit.util import Rate

log = logging.getLogger(__name__)


@dataclass
class Pose:
    fingers: np.ndarray
    roll: float = 0.0

    def __post_init__(self) -> None:
        self.fingers = np.clip(np.asarray(self.fingers, dtype=float), 0.0, 1.0)
        if self.fingers.shape != (N_FINGERS,):
            raise ValueError(f"pose needs {N_FINGERS} finger values")

    def to_yaml(self) -> dict[str, Any]:
        return {"fingers": [round(float(v), 4) for v in self.fingers], "roll": round(float(self.roll), 3)}


def resolve_fingers(
    spec: Any, names: list[str], groups: list[str], roles: list[str] | None = None,
) -> np.ndarray:
    """List of 12, or a mapping -> value. Keys: servo name > finger group > role > default.

    The canonical tendon vocabulary (thumb_flex, extend, pinky, ...) is always accepted,
    even for tendons no servo channel has been assigned to yet; those entries do nothing.
    """
    if isinstance(spec, (list, tuple)):
        return np.asarray(spec, dtype=float)
    if not isinstance(spec, dict):
        raise ValueError(f"fingers must be a list or mapping, got {type(spec).__name__}")
    roles = roles or [""] * len(names)
    valid = set(names) | set(groups) | set(roles) | set(FINGERS) | set(ROLES) | set(TENDONS) | {"default"}
    unknown = set(spec) - valid
    if unknown:
        raise ValueError(f"unknown finger keys {sorted(unknown)}; valid: {sorted(valid - {''})}")
    out = np.full(N_FINGERS, float(spec.get("default", 0.0)))
    for layer in (roles, groups, names):
        for i, key in enumerate(layer):
            if key and key in spec:
                out[i] = float(spec[key])
    return out


POSES_DIRNAME = "poses"
CUSTOM_FILE = "custom.yaml"  # where `save NAME` at the pose prompt writes


def pose_files(path: str | Path | None = None) -> list[Path]:
    """The pose library: every *.yaml in config/poses/ (sorted), plus a legacy config/poses.yaml.
    `path` may name one file or a directory instead."""
    if path is not None:
        p = Path(path)
        return sorted(p.glob("*.yaml")) if p.is_dir() else [p]
    cfg = config_dir()
    files = sorted((cfg / POSES_DIRNAME).glob("*.yaml")) if (cfg / POSES_DIRNAME).is_dir() else []
    legacy = cfg / "poses.yaml"
    if legacy.exists():
        files.append(legacy)
    return files


class PoseLibrary:
    def __init__(self, poses: dict[str, Pose], path: Path | None = None,
                 sources: dict[str, str] | None = None) -> None:
        self.poses = poses
        self.path = path  # where saves go
        self.sources = sources or {}  # pose name -> file stem ("basic", "digits", "generated", ...)

    @classmethod
    def load(cls, hand_cfg: HandConfig, path: str | Path | None = None) -> PoseLibrary:
        files = pose_files(path)
        names = hand_cfg.names
        groups = [s.finger for s in hand_cfg.servos]
        roles = [s.role for s in hand_cfg.servos]
        poses: dict[str, Pose] = {}
        sources: dict[str, str] = {}
        for p in files:
            data = load_yaml(p).get("poses") or {}
            for name, d in data.items():
                if name in poses:
                    raise ValueError(f"{p}: pose '{name}' is already defined in {sources[name]}.yaml")
                try:
                    poses[name] = Pose(resolve_fingers(d.get("fingers", {}), names, groups, roles),
                                       float(d.get("roll", 0.0)))
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{p}: pose '{name}': {e}") from e
                sources[name] = p.stem
        for i, s in enumerate(hand_cfg.servos):
            f = np.zeros(N_FINGERS)
            f[i] = 1.0
            for key in (f"finger_{i + 1}_curl", f"{s.name}_only" if s.name in TENDONS else None):
                if key and key not in poses:  # by tendon name, independent of channel wiring
                    poses[key] = Pose(f.copy(), 0.0)
                    sources[key] = "generated"
        if hand_cfg.unassigned:
            log.warning("servo channels %s are not assigned to a tendon yet; poses only drive the assigned ones",
                        hand_cfg.unassigned)
        if path is not None and not Path(path).is_dir():
            save_to = Path(path)
        else:
            save_to = (Path(path) if path is not None else config_dir() / POSES_DIRNAME) / CUSTOM_FILE
        return cls(poses, save_to, sources)

    def __contains__(self, name: str) -> bool:
        return name in self.poses

    def __getitem__(self, name: str) -> Pose:
        if name not in self.poses:
            raise KeyError(f"unknown pose '{name}'. Known: {', '.join(sorted(self.poses))}")
        return self.poses[name]

    def names(self) -> list[str]:
        return sorted(self.poses)

    def save_pose(self, name: str, pose: Pose, path: str | Path | None = None) -> None:
        """Write one pose into the custom file (config/poses/custom.yaml), keeping the others as written."""
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("no pose file to save into")
        if name in self.sources and self.sources[name] not in (p.stem, "generated"):
            raise ValueError(f"'{name}' is defined in {self.sources[name]}.yaml; pick another name or edit that file")
        data = load_yaml(p) if p.exists() else {}
        data.setdefault("poses", {})
        data["poses"][name] = pose.to_yaml()
        dump_yaml(data, p, header="# Poses saved from the dexkit-pose prompt (`save NAME`). Edit freely.\n")
        self.poses[name] = pose
        self.sources[name] = p.stem


def min_jerk(alpha: np.ndarray | float) -> np.ndarray | float:
    a = np.clip(alpha, 0.0, 1.0)
    return 10 * a**3 - 15 * a**4 + 6 * a**5


def interpolate(start: Pose, end: Pose, alpha: float, mode: str = "linear") -> Pose:
    s = float(min_jerk(alpha)) if mode == "minjerk" else float(np.clip(alpha, 0.0, 1.0))
    return Pose(start.fingers + (end.fingers - start.fingers) * s, start.roll + (end.roll - start.roll) * s)


def pose_trajectory(start: Pose, end: Pose, duration_s: float, rate_hz: float, mode: str = "linear") -> list[Pose]:
    """Waypoints after `start`, ending exactly at `end`; one per tick of rate_hz."""
    n = max(1, int(round(duration_s * rate_hz)))
    return [interpolate(start, end, i / n, mode) for i in range(1, n + 1)]


def go_to_pose(
    hand: HandInterface,
    pose: Pose,
    duration_s: float = 1.0,
    rate_hz: float = 20.0,
    mode: str = "linear",
    estop: EStop = ESTOP,
    on_tick: Callable[[Pose], None] | None = None,
    rate: Rate | None = None,
    max_settle_ticks: int = 100,
    arrive_timeout_s: float = 12.0,
    arrive_tol: float = 0.05,
) -> Pose:
    f, r = hand.last_command
    start = Pose(f, r)
    rate = rate or Rate(rate_hz)
    for wp in pose_trajectory(start, pose, duration_s, rate_hz, mode):
        estop.check()
        hand.set_targets(wp.fingers, wp.roll)
        if on_tick:
            on_tick(wp)
        rate.sleep()
    # The per-tick slew limit (max_delta_ticks) can lag a short trajectory; keep sending
    # the final target until the commanded position stops changing. (It may settle short
    # of the pose: the antagonist limit or a stall back-off can scale a target down.)
    prev_f, prev_r = hand.last_command
    for _ in range(max_settle_ticks):
        if np.max(np.abs(prev_f - pose.fingers)) < 1e-3 and abs(prev_r - pose.roll) < 0.1:
            break
        estop.check()
        f, r = hand.set_targets(pose.fingers, pose.roll)
        if np.max(np.abs(f - prev_f)) < 1e-6 and abs(r - prev_r) < 1e-6:
            break
        prev_f, prev_r = f, r
        rate.sleep()
    # The servos follow at their own speed register; wait until they have actually arrived
    # (within arrive_tol of what was sent) so the next step starts from a formed pose.
    if arrive_timeout_s > 0:
        expect = getattr(hand, "expected_fingers", lambda f: f)
        want = np.asarray(expect(prev_f), dtype=float)  # pay-out moves antagonists past slack
        deadline = time.monotonic() + arrive_timeout_s
        while True:
            estop.check()
            st = hand.get_state()
            err = float(np.max(np.abs(np.asarray(st.fingers, dtype=float) - want)))
            if err < arrive_tol and abs(st.roll_deg - prev_r) < 3.0:
                break
            if time.monotonic() > deadline:
                log.warning("pose not reached within %.0fs (max finger error %.2f); continuing",
                            arrive_timeout_s, err)
                break
            hand.set_targets(prev_f, prev_r)  # keep commanding; also keeps the safety checks running
            rate.sleep()
    return pose


def current_pose(hand: HandInterface) -> Pose:
    st = hand.get_state()
    return Pose(np.clip(st.fingers, 0, 1), st.roll_deg)


# --------------------------------------------------------------------------- CLI


INTERACTIVE_HELP = """commands:
  <pose name>          move to a pose (e.g. open, fist, peace, rock_on, pinky_flex_only)
  pose <name>          same, for a pose that shares a name with a command (e.g. `pose zero`)
  f <N> <value>        set finger servo N (1-12) to value 0..1
  roll <deg>           set forearm roll in degrees
  save <name>          save the current commanded pose to config/poses/custom.yaml
  list                 list poses
  state                print measured finger values, roll, voltage, load
  home                 torque off, you pull every finger fully open, Enter: that becomes 'open'
  relax                torque off (hand goes limp) and quit
  quit / q             relax and quit"""

GANTRY_HELP = """gantry commands (dexkit-pose --gantry):
  where                print gantry position (mm) and state
  jog <x|y|z> <mm>     move one axis by mm, e.g. jog x 10, jog z -5 (z negative = down)
  zero                 declare the current gantry position as 0,0,0
  goto <x> <y> <z>     move to an absolute position in mm (after zero)"""

AXES = {"x": 0, "y": 1, "z": 2}


def gantry_command(gantry: GantryInterface, parts: list[str], estop: EStop) -> bool:
    """Handle one gantry command line. Returns False if `parts` is not a gantry command."""
    cmd = parts[0].lower()
    if cmd == "where" and len(parts) == 1:
        st = gantry.get_state()
        print(f"gantry {np.round(st.xyz, 2).tolist()} mm, {st.state}"
              + ("" if st.frame_valid else " (not zeroed: jog to your corner, then type zero)"))
    elif cmd == "zero" and len(parts) == 1:
        gantry.set_zero()
        print("gantry zero set here (0, 0, 0)")
    elif cmd == "jog" and len(parts) == 3:
        if parts[1].lower() not in AXES:
            raise ValueError("axis must be x, y or z")
        delta = [0.0, 0.0, 0.0]
        delta[AXES[parts[1].lower()]] = float(parts[2])
        if not gantry.jog(*delta):
            print("jog blocked by the travel box")
        _wait_gantry(gantry, estop)
        print(f"gantry {np.round(gantry.get_state().xyz, 2).tolist()} mm")
    elif cmd == "goto" and len(parts) == 4:
        x, y, z = (float(v) for v in parts[1:])
        gantry.move_to(x, y, z, wait=False)
        _wait_gantry(gantry, estop)
        print(f"gantry {np.round(gantry.get_state().xyz, 2).tolist()} mm")
    else:
        return False
    return True


def _wait_gantry(gantry: GantryInterface, estop: EStop, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    time.sleep(0.1)
    while gantry.get_state().state != "Idle":
        estop.check()
        if time.monotonic() > deadline:
            raise TimeoutError("gantry move timed out")
        time.sleep(0.1)


def interactive(hand: HandInterface, lib: PoseLibrary, rate_hz: float, duration: float, mode: str,
                estop: EStop, read: Callable[[str], str] = input, gantry: GantryInterface | None = None) -> None:
    print(INTERACTIVE_HELP)
    if gantry is not None:
        print(GANTRY_HELP)
    while True:
        try:
            line = read("pose> ").strip()
        except EOFError:
            return
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd in ("q", "quit", "exit", "relax"):
                return
            if cmd in ("help", "?"):
                print(INTERACTIVE_HELP)
                if gantry is not None:
                    print(GANTRY_HELP)
            elif cmd == "list":
                print(", ".join(lib.names()))
            elif cmd == "state":
                st = hand.get_state()
                print(f"fingers {np.round(st.fingers, 2).tolist()} roll {st.roll_deg:+.1f} "
                      f"V {st.min_voltage} load {st.load}")
            elif cmd == "home":
                hand.relax()
                read("torque OFF. Pull every finger fully OPEN and the wrist to neutral, then press Enter ")
                hand.rehome()
                print("re-based: current position is now open / roll 0")
            elif cmd == "save" and len(parts) == 2:
                f, r = hand.last_command
                lib.save_pose(parts[1], Pose(f, r))
                print(f"saved '{parts[1]}'")
            elif cmd == "f" and len(parts) == 3:
                n = int(parts[1])
                if not 1 <= n <= N_FINGERS:
                    raise ValueError(f"finger servo must be 1..{N_FINGERS}")
                f, r = hand.last_command
                f[n - 1] = float(parts[2])
                go_to_pose(hand, Pose(f, r), duration, rate_hz, mode, estop=estop)
            elif cmd == "roll" and len(parts) == 2:
                f, _ = hand.last_command
                go_to_pose(hand, Pose(f, float(parts[1])), duration, rate_hz, mode, estop=estop)
            elif gantry is not None and gantry_command(gantry, parts, estop):
                pass  # gantry commands win over a pose of the same name (e.g. `zero`); use `pose zero`
            elif cmd in lib or (cmd == "pose" and len(parts) == 2 and parts[1] in lib):
                name = parts[1] if cmd == "pose" else cmd
                go_to_pose(hand, lib[name], duration, rate_hz, mode, estop=estop)
                f, r = hand.last_command
                print(f"-> {name}: commanded {np.round(f, 2).tolist()} roll {r:+.1f}")
            else:
                print(f"unknown command or pose '{line}' (type help, list or quit)")
        except (ValueError, IndexError) as e:
            print(f"bad input: {e}")
        except EStopTripped:
            raise
        except SafetyTrip as e:
            print(f"refused: {e}")
        except (OSError, TimeoutError) as e:
            print(f"gantry error: {e}")


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import common_parser, init, open_session

    p = common_parser("Move the hand to named poses. With no pose name: interactive prompt.")
    p.add_argument("name", nargs="?", help="pose name; omit for the interactive prompt")
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--minjerk", action="store_true", help="minimum-jerk instead of linear interpolation")
    p.add_argument("--list", action="store_true", help="list known poses and exit")
    p.add_argument("--hold", type=float, default=2.0,
                   help="seconds to hold the pose before relaxing and exiting (one-shot mode)")
    p.add_argument("--gantry", action="store_true",
                   help="interactive mode: also connect the gantry (where / jog / zero / goto)")
    p.add_argument("--restore-zero", action="store_true",
                   help="with --gantry: restore the last saved gantry zero without asking")
    args = p.parse_args(argv)
    init(args)

    from dexkit.config import load_hand_config

    lib = PoseLibrary.load(load_hand_config())
    if args.list:
        by_file: dict[str, list[str]] = {}
        for n in lib.names():
            by_file.setdefault(lib.sources.get(n, "?"), []).append(n)
        for src in sorted(by_file, key=lambda k: (k == "generated", k)):
            print(f"[{src}]")
            for n in by_file[src]:
                pose = lib[n]
                print(f"  {n:20s} roll {pose.roll:+6.1f}  {np.round(pose.fingers, 2).tolist()}")
        return
    if args.name and args.name not in lib:
        p.error(f"unknown pose '{args.name}'. Known: {', '.join(lib.names())}")
    mode = "minjerk" if args.minjerk else "linear"

    with open_session(args, need_hand=True, need_gantry=bool(args.gantry and not args.name)) as s:
        assert s.hand is not None and s.hand_cfg is not None
        if not args.name:
            interactive(s.hand, lib, s.hand_cfg.rate_hz, args.duration, mode, s.estop, gantry=s.gantry)
            return
        t0 = time.monotonic()
        go_to_pose(s.hand, lib[args.name], args.duration, s.hand_cfg.rate_hz, mode, estop=s.estop)
        time.sleep(args.hold)
        st = s.hand.get_state()
        print(f"pose '{args.name}' reached in {time.monotonic() - t0:.2f}s; "
              f"measured fingers {np.round(st.fingers, 2).tolist()} roll {st.roll_deg:+.1f}")
        print("relaxing (torque off). Use `dexkit-pose` with no name to stay connected between poses.")


if __name__ == "__main__":
    main()
