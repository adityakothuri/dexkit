"""Interactive per-servo calibration for the tendon-driven foam hand.

Foam fingers have no hard stops, so limits come from the tendons. The servos run in
multi-turn mode, so a tendon spool may need more than one revolution; positions are
15-bit and are allowed to pass 0 / 4095.
  slack  - torque off, operator pulls the finger fully open by hand, read ticks
  tight  - low torque limit, servo steps calib_step_ticks at a time while the operator
           watches; operator presses 't' at the desired tight pose ('r' reverses
           direction, 'u' undoes 5 steps, 'f' / 's' doubles / halves the speed,
           'x' aborts the servo)
  roll   - operator centres the forearm by hand; center ticks recorded

Writes config/hand.yaml, marking each captured servo `calibrated: true`. A plain run
does only the servos not yet marked; --force redoes all of them; --servo N redoes one
(a .bak copy is kept). --name/--finger relabel that servo once you see which finger
it moves (the shipped names are guesses).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import deque
from pathlib import Path

from dexkit.config import (
    MAX_TICKS,
    TENDONS,
    ConfigError,
    HandConfig,
    config_dir,
    data_dir,
    dump_yaml,
    hand_config_from_dict,
    load_hand_config,
    load_yaml,
)
from dexkit.hw.feetech_hand import FeetechDriver, open_serial
from dexkit.hw.feetech_protocol import FeetechBus
from dexkit.hw.safety import SafetyTrip, VoltageGate
from dexkit.util import setup_logging

HEADER = """# DexKit hand configuration (written by dexkit-calibrate-hand).
# Hand actions are floats in [0, 1] per finger servo (0 = slack, 1 = tight);
# roll is in degrees. Conversion to ticks happens only in hw/feetech_hand.py.
# HLS vs STS: address 44 is Goal Torque on HLS, Goal Time on STS.
"""


class Prompter:
    """Interactive operator I/O."""

    def say(self, msg: str) -> None:
        print(msg, flush=True)

    def wait_enter(self, msg: str) -> None:
        input(msg + " [Enter] ")

    def poll_key(self, timeout: float) -> str | None:
        import select

        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            line = sys.stdin.readline().strip().lower()
            return line[:1] or "\n"
        return None


class ScriptedPrompter(Prompter):
    """Deterministic answers for --mock runs: Enter for every wait, then for each
    servo step `tight_steps` times and press 't'."""

    def __init__(self, tight_steps: int = 60) -> None:
        self.tight_steps = tight_steps
        self._keys: deque[str | None] = deque()

    def wait_enter(self, msg: str) -> None:
        print(msg + " [scripted Enter]")
        self._keys = deque([None] * self.tight_steps + ["t"])

    def poll_key(self, timeout: float) -> str | None:
        return self._keys.popleft() if self._keys else "t"


def capture_servo(
    d: FeetechDriver, sid: int, name: str, cfg: HandConfig, ui: Prompter, step_delay: float = 0.08,
    direction: int = 1,
) -> dict | None:
    defaults = cfg.defaults
    ui.say(f"\n=== servo {sid} ({name}) ===")
    d.set_torque(sid, False)
    ui.wait_enter("  Torque OFF. Put this finger in its RELAXED, neutral position (not pulled either way) and hold it")
    slack = d.read_position(sid)
    if slack is None:
        ui.say("  no position reading; skipping")
        return None
    ui.say(f"  slack = {slack}")
    d.set_torque_limit(sid, defaults.calib_torque_limit)
    d.set_goal_torque(sid, defaults.calib_torque_limit)
    d.set_accel(sid, defaults.accel)
    d.set_speed(sid, defaults.speed)  # bench servos shipped with speed 100: they crawled
    d.write_position(sid, slack)
    d.set_torque(sid, True)
    ui.wait_enter("  Torque ON (low limit). Release the finger. Stepping will start; "
                  "press t+Enter at the fully PULLED pose: curled in for a flexor, bent back for an extensor, "
                  "over for an adduct (r reverse, u undo, f faster, s slower, x abort)")
    ui.say(f"  stepping toward {'HIGHER' if direction > 0 else 'LOWER'} counts; watch the spool, "
           "r+Enter if it unwinds the tendon")
    step = defaults.calib_step_ticks
    goal = slack
    history: list[int] = []
    while True:
        key = ui.poll_key(step_delay)
        if key == "t":
            break
        if key == "x":
            d.set_torque(sid, False)
            ui.say("  aborted")
            return None
        if key == "r":
            direction = -direction
            ui.say(f"  direction {'+' if direction > 0 else '-'}")
            continue
        if key in ("f", "s"):
            step = min(step * 2, 320) if key == "f" else max(step // 2, 2)
            ui.say(f"  speed: {step} ticks per step")
            continue
        if key == "u" and history:
            goal = history[max(0, len(history) - 5)]
            history = history[: max(0, len(history) - 5)]
            d.write_position(sid, goal)
            continue
        nxt = goal + direction * step
        if abs(nxt) > MAX_TICKS:
            ui.say(f"  reached the position limit (+/-{MAX_TICKS}) before 't'; stopping.")
            break
        history.append(goal)
        goal = nxt
        d.write_position(sid, goal)
        if abs(goal - slack) > defaults.calib_max_travel_ticks:
            ui.say(f"  moved {defaults.calib_max_travel_ticks} ticks ({defaults.calib_max_travel_ticks / 4096:.1f} turns) "
                   "without 't'; stopping for safety (calib_max_travel_ticks in hand.yaml)")
            break
    time.sleep(0.2)
    tight = d.read_position(sid)
    load = d.read_load(sid) or 0
    tight = goal if tight is None else tight
    # Leave the finger relaxed and restore the operating torque limit.
    d.write_position(sid, slack)
    time.sleep(0.3)
    d.set_torque(sid, False)
    d.set_torque_limit(sid, defaults.torque_limit)
    d.set_goal_torque(sid, defaults.goal_torque)
    inverted = tight < slack
    stall = max(int(load * defaults.stall_load_factor), defaults.stall_load_floor)
    ui.say(f"  tight = {tight} (load {load}), inverted = {inverted}, stall_load = {stall}")
    return {"slack": int(slack), "tight": int(tight), "inverted": bool(inverted), "stall_load": stall}


def apply_label(entry: dict, name: str, finger: str | None) -> None:
    entry["name"] = name
    if name in TENDONS:
        entry["finger"], entry["role"] = TENDONS[name]
    else:
        entry["finger"] = finger or entry.get("finger", "")
        entry["role"] = ""


def print_wiring(cfg: HandConfig) -> None:
    print("channel -> tendon")
    for s in cfg.servos:
        label = s.name if s.name in TENDONS else f"{s.name}  (not a tendon name yet)"
        print(f"  ch {s.id:2d} -> {label:34s} {'calibrated' if s.calibrated else 'NOT calibrated'}")
    print(f"  ch {cfg.roll.id:2d} -> forearm roll                       "
          f"{'calibrated' if cfg.roll.calibrated else 'NOT calibrated'}")
    missing = sorted(set(TENDONS) - {s.name for s in cfg.servos})
    if missing:
        print("  tendons with no channel yet:", ", ".join(missing))


def capture_roll(d: FeetechDriver, sid: int, ui: Prompter) -> int | None:
    ui.say(f"\n=== roll servo {sid} ===")
    d.set_torque(sid, False)
    ui.wait_enter("  Torque OFF. Rotate the forearm to its CENTER (palm neutral) by hand")
    center = d.read_position(sid)
    ui.say(f"  center = {center}")
    return center


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--servo", type=int, metavar="ID", help="redo one servo ID (or the roll servo ID)")
    p.add_argument("--force", action="store_true", help="overwrite an existing real calibration")
    p.add_argument("--out", help="output yaml (default config/hand.yaml; data/mock/hand.yaml with --mock)")
    p.add_argument("--range-deg", type=float, default=None, help="roll range +/- degrees (default from yaml)")
    p.add_argument("--name", help="with --servo: the tendon this channel drives, e.g. pinky_flex, index_extend, "
                   "thumb_adduct (canonical names set finger/role automatically)")
    p.add_argument("--finger", help="with --servo and a non-canonical --name: its finger group")
    p.add_argument("--label-only", action="store_true",
                   help="with --servo and --name: just write the label, do not touch the servo")
    p.add_argument("--wiring", action="store_true", help="print the channel -> tendon table and exit")
    p.add_argument("--reverse", action="store_true",
                   help="step the opposite way from calib_direction in hand.yaml")
    p.add_argument("--step-ticks", type=int, default=None,
                   help="starting speed: ticks per 0.08 s step (default calib_step_ticks in hand.yaml)")
    p.add_argument("--torque", type=int, default=None,
                   help="torque 0..1000 while finding 'tight' (default calib_torque_limit in hand.yaml)")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--scripted", action="store_true", help="scripted operator answers (for --mock)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)

    src = config_dir() / "hand.yaml"
    out = Path(args.out) if args.out else (data_dir() / "mock" / "hand.yaml" if args.mock else src)
    base_path = out if out.exists() else src
    raw = load_yaml(base_path)
    cfg = hand_config_from_dict(raw, source=base_path)
    if args.step_ticks is not None:
        cfg.defaults.calib_step_ticks = max(1, args.step_ticks)
    if args.torque is not None:
        cfg.defaults.calib_torque_limit = min(max(0, args.torque), 1000)
    if out.exists() and cfg.is_calibrated and args.servo is None and not args.force:
        print(f"{out} already holds a full calibration from {cfg.calibrated_at}; use --force or --servo N")
        sys.exit(2)
    if (args.name or args.finger or args.label_only) and args.servo is None:
        p.error("--name/--finger/--label-only need --servo N")
    if args.wiring:
        print_wiring(cfg)
        return
    if args.name and args.name not in TENDONS and not args.finger:
        print(f"'{args.name}' is not a canonical tendon name {sorted(TENDONS)}; pass --finger too, or use one of those")
        sys.exit(2)
    if args.label_only:
        if not args.name or args.servo == cfg.roll.id:
            p.error("--label-only needs --servo N (a finger channel) and --name")
        by_id = {s["id"]: s for s in raw["servos"]}
        if args.servo not in by_id:
            print(f"servo {args.servo} is not in hand.yaml (IDs {cfg.ids})")
            sys.exit(2)
        apply_label(by_id[args.servo], args.name, args.finger)
        try:
            hand_config_from_dict(raw)
        except ConfigError as e:
            print(f"refused: {e}")
            sys.exit(2)
        dump_yaml(raw, out, header=HEADER)
        print(f"servo {args.servo} -> {args.name}; wrote {out}")
        print_wiring(load_hand_config(out))
        return

    if args.mock:
        from dexkit.hw.mock import MockFeetechSerial, mock_servos_for

        transport = MockFeetechSerial(mock_servos_for(cfg), baudrate=cfg.baud)
    else:
        transport = open_serial(cfg.port, cfg.baud, cfg.timeout_s)
    d = FeetechDriver(FeetechBus(transport, timeout_s=cfg.timeout_s), cfg.register_map())
    ui: Prompter = ScriptedPrompter() if args.scripted else Prompter()
    step_delay = 0.0 if args.scripted else 0.08

    try:
        missing = [sid for sid in cfg.ids if d.ping(sid) is None]
        if args.servo is not None:
            if args.servo not in cfg.ids:
                print(f"servo {args.servo} is not in hand.yaml (IDs {cfg.ids})")
                sys.exit(2)
            targets = [args.servo]
        elif args.force:
            targets = cfg.ids
        else:
            targets = cfg.uncalibrated_ids
            done = [sid for sid in cfg.ids if sid not in targets]
            if done:
                print(f"already calibrated, skipping: {done} (use --force to redo them)")
            if not targets:
                print("nothing to calibrate: every servo is done (use --force or --servo N to redo one)")
                print_wiring(cfg)
                return
        if set(targets) & set(missing):
            print(f"servos not responding: {sorted(set(targets) & set(missing))}; run dexkit-scan")
            sys.exit(1)
        volts = {sid: v for sid in targets if (v := d.read_voltage(sid)) is not None}
        VoltageGate(cfg.voltage_window).check(volts, expected_ids=targets)
        print("voltages OK:", ", ".join(f"{k}:{v:.1f}V" for k, v in volts.items()))

        by_id = {s["id"]: s for s in raw["servos"]}
        if args.servo is not None and args.servo != cfg.roll.id and args.name:
            apply_label(by_id[args.servo], args.name, args.finger)
        direction = -cfg.defaults.calib_direction if args.reverse else cfg.defaults.calib_direction
        captured: list[int] = []
        for sid in targets:
            if sid == cfg.roll.id:
                center = capture_roll(d, sid, ui)
                if center is not None:
                    raw["roll"]["center"] = int(center)
                    raw["roll"]["calibrated"] = True
                    captured.append(sid)
                    if args.range_deg is not None:
                        raw["roll"]["range_deg"] = float(args.range_deg)
                continue
            name = by_id[sid]["name"]
            result = capture_servo(d, sid, name, cfg, ui, step_delay=step_delay, direction=direction)
            if result:
                by_id[sid].update(result, calibrated=True)
                captured.append(sid)
    except SafetyTrip as e:
        print(f"SAFETY: {e}")
        sys.exit(1)
    finally:
        try:
            d.set_torque_all(cfg.ids, False)
        finally:
            transport.close()

    raw["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    hand_config_from_dict(raw)  # validate before writing
    if out.exists() and args.servo is not None:
        shutil.copy(out, out.with_suffix(".yaml.bak"))
    dump_yaml(raw, out, header=HEADER)
    final = load_hand_config(out)  # re-read to prove the written file loads
    print(f"\nwrote {out}: captured {captured}")
    print_wiring(final)
    if final.uncalibrated_ids:
        print(f"still to do: {final.uncalibrated_ids}  (run dexkit-calibrate-hand again, or --servo N)")
    else:
        print("all servos calibrated; the hand is ready for dexkit-pose")

if __name__ == "__main__":
    main()
