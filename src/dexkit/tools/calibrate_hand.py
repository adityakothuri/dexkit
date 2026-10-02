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
    ServoConfig,
    config_dir,
    data_dir,
    dump_yaml,
    hand_config_from_dict,
    load_hand_config,
    load_yaml,
)
from dexkit.hw.feetech_hand import FeetechDriver, open_serial
from dexkit.hw.feetech_protocol import FeetechBus
from dexkit.hw.safety import EStopTripped, SafetyTrip, VoltageGate
from dexkit.util import setup_logging

HEADER = """# DexKit hand configuration (written by dexkit-calibrate-hand).
# Hand actions are floats in [0, 1] per finger servo (0 = slack, 1 = tight);
# roll is in degrees. Conversion to ticks happens only in hw/feetech_hand.py.
# HLS vs STS: address 44 is Goal Torque on HLS, Goal Time on STS.
"""


class Prompter:
    """Interactive operator I/O. While a servo is stepping, single keys work without Enter
    (cbreak mode); SPACE is the emergency stop."""

    def say(self, msg: str) -> None:
        print(msg, flush=True)

    def wait_enter(self, msg: str) -> None:
        input(msg + " [Enter] ")

    def begin_keys(self) -> None:
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def end_keys(self) -> None:
        import termios

        saved = getattr(self, "_saved", None)
        if saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, saved)
            self._saved = None

    def poll_key(self, timeout: float) -> str | None:
        import os
        import select

        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            ch = os.read(sys.stdin.fileno(), 1).decode("ascii", errors="replace")
            return " " if ch == " " else (ch.lower() if ch not in ("\r", "\n") else "\n")
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

    def begin_keys(self) -> None:
        pass

    def end_keys(self) -> None:
        pass

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
                  "press t at the fully PULLED pose: curled in for a flexor, bent back for an extensor, "
                  "over for an adduct (r reverse, u undo, f faster, s slower, x abort, SPACE = e-stop)")
    ui.say(f"  stepping toward {'HIGHER' if direction > 0 else 'LOWER'} counts; watch the spool, "
           "press r if it unwinds the tendon")
    goal = slack
    history: list[int] = []
    ui.begin_keys()
    try:
        return _step_to_tight(d, sid, slack, cfg, ui, step_delay, direction, goal, history)
    finally:
        ui.end_keys()


def capture_tight_from_open(
    d: FeetechDriver, sid: int, name: str, cfg: HandConfig, ui: Prompter, open_pos: int,
    step_delay: float = 0.08, direction: int = 1,
) -> dict | None:
    """Re-set only the curl limit: `open` is already known and trusted, so start there, wind
    slowly until the operator presses t, and return to open."""
    defaults = cfg.defaults
    ui.say(f"\n=== {name} (servo {sid}): curl limit from its open position {open_pos} ===")
    d.set_torque_limit(sid, defaults.calib_torque_limit)
    d.set_goal_torque(sid, defaults.calib_torque_limit)
    d.set_accel(sid, defaults.accel)
    d.set_speed(sid, defaults.speed)
    d.write_position(sid, open_pos)
    d.set_torque(sid, True)
    ui.wait_enter("  Winding will start slowly from open; press t at a FIRM, unstrained curl "
                  "(u back up, f faster, s slower, x skip, SPACE = e-stop)")
    ui.begin_keys()
    try:
        return _step_to_tight(d, sid, open_pos, cfg, ui, step_delay, direction, open_pos, [])
    finally:
        ui.end_keys()


AUTO_LOAD_STOP = 220       # present-load (0..1000) that means "the finger is resisting"
AUTO_LAG_STOP = 160        # goal - position gap that means "the servo cannot keep up" (stalled)
AUTO_BACKOFF = 0.08        # record the limit this fraction of the travel short of the stall point
AUTO_STEP_TICKS = 24
AUTO_STEP_S = 0.1
AUTO_MAX_FRACTION = 1.15   # never travel more than this times the previous span


def auto_tight_from_open(d: FeetechDriver, s: ServoConfig, cfg: HandConfig, open_pos: int, say) -> dict | None:
    """Find the curl limit without an operator: wind from open at calibration torque and stop where
    the load rises or the servo stalls. Checks the dexkit-estop flag every step."""
    from dexkit.hw.safety import estop_flag_path

    defaults = cfg.defaults
    sid, sign = s.id, (1 if s.span >= 0 else -1)
    cap = int(AUTO_MAX_FRACTION * max(abs(s.span), 600))
    d.set_goal_torque(sid, defaults.calib_torque_limit)
    d.set_torque_limit(sid, defaults.calib_torque_limit)
    d.set_accel(sid, defaults.accel)
    d.set_speed(sid, 400)
    d.write_position(sid, open_pos)
    d.set_torque(sid, True)
    time.sleep(0.6)
    goal, hits, log = open_pos, 0, []
    reason = "cap"
    while abs(goal - open_pos) < cap:
        if estop_flag_path().exists():
            d.set_torque_all(cfg.ids, False)
            raise EStopTripped("dexkit-estop flag set during auto calibration")
        goal += sign * AUTO_STEP_TICKS
        d.write_position(sid, goal)
        time.sleep(AUTO_STEP_S)
        pos = d.read_position(sid)
        load = d.read_load(sid) or 0
        if pos is None:
            continue
        lag = abs(goal - pos)
        log.append((abs(goal - open_pos), load, lag))
        hits = hits + 1 if load >= AUTO_LOAD_STOP else 0
        if hits >= 2:
            reason = f"load {load}"
            break
        if lag >= AUTO_LAG_STOP and abs(goal - open_pos) > 200:
            reason = f"stall (lag {lag})"
            break
    pos = d.read_position(sid) or goal
    travel = abs(pos - open_pos)
    tight = int(open_pos + sign * max(0, travel * (1 - AUTO_BACKOFF)))
    say(f"  {s.name}: stopped after {travel} ticks ({reason}); limit set at {abs(tight - open_pos)} "
        f"(was {abs(s.span)})  load trace: {[ld for _, ld, _ in log[::max(1, len(log) // 8)]]}")
    d.write_position(sid, open_pos)
    time.sleep(0.2 + travel / 400)
    d.set_torque(sid, False)
    d.set_goal_torque(sid, defaults.goal_torque)
    d.set_torque_limit(sid, defaults.torque_limit)
    if travel < 150:
        say(f"  {s.name}: barely moved before resisting; left unchanged (check the tendon)")
        return None
    return {"slack": int(open_pos), "tight": tight, "inverted": bool(tight < open_pos),
            "stall_load": defaults.stall_load_floor}


def _step_to_tight(d: FeetechDriver, sid: int, slack: int, cfg: HandConfig, ui: Prompter, step_delay: float,
                   direction: int, goal: int, history: list[int]) -> dict | None:
    defaults = cfg.defaults
    step = defaults.calib_step_ticks
    while True:
        key = ui.poll_key(step_delay)
        if key == " ":
            d.set_torque_all(cfg.ids, False)
            raise EStopTripped("operator pressed SPACE during calibration (all torque off)")
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


def tight_only(args: argparse.Namespace, cfg: HandConfig, raw: dict, out: Path, ui: Prompter,
               step_delay: float) -> None:
    """--tight-only: connect like a normal session (so open positions are restored), then re-capture
    each driven tendon's curl limit from that open and write hand.yaml + the position memory."""
    from dexkit.hw.feetech_hand import save_positions_state

    if args.mock:
        from dexkit.hw.mock import MockHand

        hand = MockHand(cfg)
    else:
        from dexkit.hw.feetech_hand import FeetechHand

        hand = FeetechHand(cfg)
    hand.connect()
    assert hand.driver is not None
    chosen = [s for s in cfg.servos if s.enabled and (not args.only or s.name in args.only.split(","))]
    if getattr(args, "tight_auto", False):
        print("AUTO curl limits: each tendon winds from open at calibration torque and stops on resistance.\n"
              "Stand by with `dexkit-estop` in another window or the PSU switch.")
    if not chosen:
        print(f"no driven tendon matches --only {args.only}")
        sys.exit(2)
    untrusted = [s.name for s in chosen if s.name in hand.restore_report.get("assumed", [])]
    if untrusted and not args.trust_current:
        hand.close()
        print(f"open position not trusted for {untrusted}: run `dexkit-relax --unwind` first, "
              "or pass --trust-current to take where they are now as open")
        sys.exit(2)
    by_id = {s["id"]: s for s in raw["servos"]}
    captured: list[int] = []
    try:
        for s in chosen:
            i = cfg.servos.index(s)
            open_pos = int(hand._slack[i])
            direction = 1 if s.span >= 0 else -1
            if getattr(args, "tight_auto", False):
                result = auto_tight_from_open(hand.driver, s, cfg, open_pos, ui.say)
            else:
                result = capture_tight_from_open(hand.driver, s.id, s.name, cfg, ui, open_pos,
                                                 step_delay=step_delay, direction=direction)
            if result:
                old = abs(s.span)
                by_id[s.id].update(result, calibrated=True)
                captured.append(s.id)
                print(f"  {s.name}: span {old} -> {abs(result['tight'] - result['slack'])} ticks")
    except EStopTripped as e:
        print(f"\nEMERGENCY STOP: {e}. Nothing written. Restart to continue.")
        sys.exit(3)
    except SafetyTrip as e:
        print(f"SAFETY: {e}")
        sys.exit(1)
    finally:
        hand.close()
    if not captured:
        print("nothing captured; hand.yaml unchanged")
        return
    raw["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    hand_config_from_dict(raw)
    shutil.copy(out, out.with_suffix(".yaml.bak")) if out.exists() else None
    dump_yaml(raw, out, header=HEADER)
    save_positions_state({sid: {"pos": by_id[sid]["slack"], "slack": by_id[sid]["slack"]} for sid in captured})
    final = load_hand_config(out)
    print(f"\nwrote {out}: curl limits re-set for {[by_id[s]['name'] for s in captured]}")
    print_wiring(final)


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
    p.add_argument("--tight-only", action="store_true",
                   help="re-set only each driven tendon's curl limit, starting from its trusted open "
                        "position (set by dexkit-relax --unwind); no 'relax the finger' step")
    p.add_argument("--tight-auto", action="store_true",
                   help="like --tight-only but automatic: stops where the load rises or the servo stalls "
                        "(calibration torque), backs off 8%%, records the limit. Stand by with dexkit-estop")
    p.add_argument("--only", help="with --tight-only/--tight-auto: comma-separated tendon names (default: all driven)")
    p.add_argument("--trust-current", action="store_true",
                   help="with --tight-only: take the current positions as open even if not restored")
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

    ui: Prompter = ScriptedPrompter() if args.scripted else Prompter()
    step_delay = 0.0 if args.scripted else 0.08
    if args.step_ticks is None and args.tight_only:
        cfg.defaults.calib_step_ticks = 20  # slower: we are looking for the limit, not taking up slack

    if args.tight_only or args.tight_auto:
        tight_only(args, cfg, raw, out, ui, step_delay)
        return

    if args.mock:
        from dexkit.hw.mock import MockFeetechSerial, mock_servos_for

        transport = MockFeetechSerial(mock_servos_for(cfg), baudrate=cfg.baud)
    else:
        transport = open_serial(cfg.port, cfg.baud, cfg.timeout_s)
    d = FeetechDriver(FeetechBus(transport, timeout_s=cfg.timeout_s), cfg.register_map())

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
    except EStopTripped as e:
        print(f"\nEMERGENCY STOP: {e}. Nothing written. Restart to continue.")
        sys.exit(3)
    except SafetyTrip as e:
        print(f"SAFETY: {e}")
        sys.exit(1)
    finally:
        try:
            d.set_torque_all(cfg.ids, False)
        finally:
            transport.close()

    raw["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Each captured servo was left at its slack with torque off: remember that absolute position.
    from dexkit.hw.feetech_hand import save_positions_state

    save_positions_state({sid: {"pos": by_id[sid]["slack"], "slack": by_id[sid]["slack"]}
                          for sid in captured if sid != cfg.roll.id},
                         roll_center=raw["roll"]["center"] if cfg.roll.id in captured else None)
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
