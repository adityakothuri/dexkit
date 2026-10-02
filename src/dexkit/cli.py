"""Shared CLI plumbing: common flags, hardware session, startup checklist."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dexkit.config import (
    GantryConfig,
    HandConfig,
    config_dir,
    load_gantry_config,
    load_hand_config,
)
from dexkit.hw.base import GantryInterface, HandInterface
from dexkit.hw.ports import PortNotFound
from dexkit.hw.safety import ESTOP, EStop, SafetyTrip, estop_flag_path, install_shutdown_handlers
from dexkit.util import setup_logging

log = logging.getLogger(__name__)


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mock", action="store_true", help="use simulated hardware")
    p.add_argument("--yes", "-y", action="store_true", help="skip the 'go' confirmation (scripted runs)")
    p.add_argument("--verbose", "-v", action="store_true", help="debug logging, including serial bytes")
    p.add_argument("--config-dir", default=None, help="directory holding hand.yaml / gantry.yaml / poses.yaml")
    p.add_argument("--mock-speed", type=float, default=1.0, help="mock gantry time multiplier")
    p.add_argument("--speed-scale", type=float, default=1.0, metavar="S",
                   help="scale hand speed, accel and per-tick step by S (0.1-1.0; e.g. 0.5 = half speed)")
    return p


def add_gantry_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-gantry", action="store_true", help="do not connect the gantry")
    p.add_argument("--restore-zero", action="store_true",
                   help="restore the last saved gantry zero without asking (gantry must not have moved)")


def init(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)
    if args.config_dir:
        import os

        os.environ["DEXKIT_CONFIG"] = str(Path(args.config_dir).resolve())


Prompt = Callable[[str], str]


@dataclass
class Session:
    hand: HandInterface | None = None
    gantry: GantryInterface | None = None
    hand_cfg: HandConfig | None = None
    gantry_cfg: GantryConfig | None = None
    estop: EStop = field(default_factory=lambda: ESTOP)
    mock: bool = False

    def close(self) -> None:
        for dev in (self.gantry, self.hand):
            if dev is not None:
                try:
                    dev.close()
                except Exception as e:  # noqa: BLE001
                    log.debug("close: %s", e)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_session(
    args: argparse.Namespace,
    need_hand: bool = True,
    need_gantry: bool = False,
    want_gantry: bool = False,
    prompt: Prompt = input,
    require_calibration: bool = True,
) -> Session:
    """Connect hardware (real or mock), register e-stop and signal handlers, run the checklist."""
    mock = getattr(args, "mock", False)
    s = Session(mock=mock)
    use_gantry = need_gantry or (want_gantry and not getattr(args, "no_gantry", False))
    lines: list[str] = []
    flag = estop_flag_path()
    if flag.exists():
        raise SafetyTrip(f"e-stop flag is set ({flag.read_text().strip()}); "
                         "check the hardware, then run: dexkit-estop --clear")
    try:
        if need_hand:
            s.hand_cfg = load_hand_config()
            apply_speed_scale(s.hand_cfg, getattr(args, "speed_scale", 1.0))
            if require_calibration and not mock and not s.hand_cfg.is_calibrated:
                raise SafetyTrip(f"{s.hand_cfg.source}: servos {s.hand_cfg.uncalibrated_ids} are not calibrated; "
                                 "run dexkit-calibrate-hand (it skips the ones already done)")
            if mock:
                from dexkit.hw.mock import MockHand

                s.hand = MockHand(s.hand_cfg)
            else:
                from dexkit.hw.feetech_hand import FeetechHand

                s.hand = FeetechHand(s.hand_cfg)
            s.hand.connect()
            volts = s.hand.read_voltages()
            port = "MOCK" if mock else getattr(s.hand, "port_name", s.hand_cfg.port)
            lines.append(f"[ok] hand on {port}: {len(volts)} servos answered")
            vs = ", ".join(f"{k}:{v:.1f}" for k, v in sorted(volts.items()))
            lo, hi = s.hand_cfg.voltage_window
            lines.append(f"[ok] servo voltages (window {lo}-{hi} V): {vs}")
            lines.append(f"[ok] {s.hand_cfg.side} hand, {len(s.hand_cfg.servos)} tendons + roll")
            cal = s.hand_cfg.calibrated_at or "NOT CALIBRATED (placeholder values)"
            lines.append(f"[{'ok' if s.hand_cfg.is_calibrated else '!!'}] calibration {s.hand_cfg.source}: {cal}")
            if not mock:
                lines.append("[!!] the fingers' CURRENT positions are now 'open' (servos lose their turn count at "
                             "power-off). If any finger is not relaxed/open, type 'home' in dexkit-pose")
        if use_gantry:
            s.gantry_cfg = load_gantry_config()
            if mock:
                from dexkit.hw.mock import MockGantry

                s.gantry = MockGantry(s.gantry_cfg, speed_factor=getattr(args, "mock_speed", 1.0))
            else:
                from dexkit.hw.grbl_gantry import GrblGantry

                s.gantry = GrblGantry(s.gantry_cfg)
            try:
                s.gantry.connect()
            except PortNotFound as e:
                if need_gantry:
                    raise
                # Optional gantry that isn't plugged in: carry on hand-only.
                s.gantry = None
                lines.append(f"[--] gantry not connected, continuing without it ({e})")
            if s.gantry is not None:
                port = "MOCK" if mock else getattr(s.gantry, "port_name", s.gantry_cfg.port)
                lines.append(f"[ok] gantry on {port}: GRBL {s.gantry.version}")
                establish_frame(s, args, prompt)
                fv = s.gantry.frame_valid
                lines.append(f"[{'ok' if fv else '!!'}] gantry frame "
                             f"{'valid' if fv else 'NOT zeroed: jog to your reference corner, then zero it'}")
        s.estop.register(hand=s.hand, gantry=s.gantry)
        install_shutdown_handlers(s.hand, s.gantry)
    except Exception:
        s.close()
        raise

    print("\nStartup checklist")
    for line in lines:
        print("  " + line)
    if not getattr(args, "yes", False):
        try:
            answer = prompt("Type 'go' to enable motion: ").strip().lower()
        except EOFError:
            s.close()
            print("\nno keyboard to confirm with (hand relaxed). Run this from a normal terminal window, "
                  "or add --yes to skip the prompt.")
            sys.exit(1)
        if answer != "go":
            s.close()
            print("aborted")
            sys.exit(1)
    return s


def apply_speed_scale(cfg: HandConfig, scale: float) -> None:
    """Slow the hand down uniformly: servo speed and accel registers plus the per-tick step limit."""
    if not 0.05 <= scale <= 1.0:
        raise ValueError("--speed-scale must be between 0.05 and 1.0")
    if scale == 1.0:
        return
    cfg.defaults.speed = max(20, int(cfg.defaults.speed * scale))
    cfg.defaults.accel = max(1, int(cfg.defaults.accel * scale))
    for servo in cfg.servos:
        servo.max_delta_ticks = max(2, int(servo.max_delta_ticks * scale))
    cfg.roll.max_delta_ticks = max(2, int(cfg.roll.max_delta_ticks * scale))
    log.info("speed scale %.2f: speed %d, accel %d, step %d ticks/tick", scale, cfg.defaults.speed,
             cfg.defaults.accel, cfg.servos[0].max_delta_ticks)


def establish_frame(s: Session, args: argparse.Namespace, prompt: Prompt) -> None:
    """Make the gantry frame valid if we safely can: home, restore saved zero, or (mock) zero here."""
    g, cfg = s.gantry, s.gantry_cfg
    assert g is not None and cfg is not None
    if cfg.zeroing == "home":
        try:
            g.home()
            return
        except Exception as e:  # noqa: BLE001
            log.error("homing failed: %s", e)
    if s.mock:
        log.info("mock gantry: zeroing at the current position")
        g.set_zero()
        return
    saved = g.load_saved_frame() if hasattr(g, "load_saved_frame") else None
    if saved and hasattr(g, "restore_frame"):
        xyz = saved["xyz"]
        if getattr(args, "restore_zero", False):
            g.restore_frame(tuple(xyz))
            return
        if not getattr(args, "yes", False):
            try:
                ans = prompt(f"Restore saved gantry zero (position {xyz}, assumes the gantry has not moved)? [y/N] ")
            except EOFError:
                ans = "n"
            if ans.strip().lower().startswith("y"):
                g.restore_frame(tuple(xyz))
                return


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def config_path(name: str) -> Path:
    return config_dir() / name
