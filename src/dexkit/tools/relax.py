"""Bring the hand back to normal.

    dexkit-relax              move every driven tendon to its known open position, wrist to
                              neutral, then torque off (SPACE = e-stop)
    dexkit-relax --now        torque off immediately, nothing moves
    dexkit-relax --unwind     REVERSE ASSIST: all driven tendons slowly unwind (release
                              direction) together while you watch. Press any key the moment
                              the hand looks open: it stops, records that as "open" for
                              every later command, and relaxes. SPACE = e-stop (nothing saved).
                              --only pinky_flex,ring_flex limits it to some tendons.

It never asks for 'go', so it also works from a shell without a keyboard (then --unwind
stops on its own after each tendon's safety cap).
"""

from __future__ import annotations

import os
import select
import sys
import time

import numpy as np

from dexkit.control.estop_key import SpaceWatch
from dexkit.control.poses import Pose, go_to_pose
from dexkit.hw.base import N_FINGERS
from dexkit.hw.feetech_hand import FeetechHand
from dexkit.hw.safety import EStopTripped

UNWIND_TICKS_PER_S = 350      # gentle: about one turn per 12 s
UNWIND_CAP_FRACTION = 1.25    # never unwind more than this times the tendon's calibrated span
UNWIND_TORQUE = 300


def _key_reader():
    """Returns a poll(timeout) -> bytes|None over the terminal in cbreak mode, and a restore()."""
    if not sys.stdin.isatty():
        return (lambda _t: None), (lambda: None)
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def poll(timeout: float) -> bytes | None:
        r, _, _ = select.select([fd], [], [], timeout)
        return os.read(fd, 1) if r else None

    def restore() -> None:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    return poll, restore


def unwind(hand: FeetechHand, only: list[str] | None, rate_hz: float = 20.0) -> None:
    """Reverse assist: step the chosen tendons in their release direction until a key is pressed."""
    assert hand.driver is not None
    cfg = hand.cfg
    chosen = [s for s in cfg.servos if s.enabled and (only is None or s.name in only)]
    if not chosen:
        raise SystemExit(f"no driven tendon matches {only}")
    present = hand.driver.read_positions([s.id for s in chosen])
    start = {s.id: present[s.id] for s in chosen}
    cap = {s.id: int(UNWIND_CAP_FRACTION * abs(s.span)) for s in chosen}
    sign = {s.id: -1 if s.span >= 0 else 1 for s in chosen}  # release = opposite to the pull direction
    for s in chosen:
        hand.driver.set_goal_torque(s.id, UNWIND_TORQUE)
        hand.driver.set_speed(s.id, UNWIND_TICKS_PER_S * 2)
    print("REVERSE ASSIST: unwinding", ", ".join(s.name for s in chosen))
    print("   press ANY key the moment the hand looks open   |   SPACE = e-stop (nothing saved)")
    poll, restore = _key_reader()
    step = UNWIND_TICKS_PER_S / rate_hz
    goal = dict(start)
    stopped_by_key = False
    try:
        t_next = time.monotonic()
        while True:
            key = poll(0.0)
            if key == b" ":
                hand.estop.trip("operator pressed SPACE") if hasattr(hand, "estop") else hand.relax()
                raise EStopTripped("operator pressed SPACE during reverse assist")
            if key:
                stopped_by_key = True
                break
            moving = False
            for s in chosen:
                if abs(goal[s.id] - start[s.id]) < cap[s.id]:
                    goal[s.id] += sign[s.id] * step
                    moving = True
            if not moving:
                print("   safety cap reached on every tendon; stopping")
                break
            hand.driver.sync_write_positions({sid: int(round(g)) for sid, g in goal.items()})
            t_next += 1.0 / rate_hz
            time.sleep(max(0.0, t_next - time.monotonic()))
    finally:
        restore()
    time.sleep(0.3)
    pos = hand.driver.read_positions(cfg.ids)
    moved = {s.name: abs(pos[s.id] - start[s.id]) for s in chosen}
    print("   unwound (ticks):", moved, "| stopped by", "key" if stopped_by_key else "cap")
    # Record this as "open" for the chosen tendons and keep everything else as it was.
    hand.rebase_partial(pos, [s.id for s in chosen])
    print("   saved as the open position for:", ", ".join(s.name for s in chosen))


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import common_parser, init, open_session

    p = common_parser(__doc__ or "")
    p.add_argument("--now", action="store_true", help="torque off immediately without moving first")
    p.add_argument("--unwind", action="store_true", help="reverse assist: unwind until you press a key")
    p.add_argument("--only", help="with --unwind: comma-separated tendon names")
    p.add_argument("--duration", type=float, default=2.0, help="seconds for the gentle release move")
    args = p.parse_args(argv)
    args.yes = True  # the "make it safe" command never waits for 'go'
    init(args)
    try:
        with open_session(args, need_hand=True) as s:
            assert s.hand is not None and s.hand_cfg is not None
            if args.unwind:
                assert isinstance(s.hand, FeetechHand)
                s.hand.estop = s.estop  # type: ignore[attr-defined]
                unwind(s.hand, args.only.split(",") if args.only else None, rate_hz=s.hand_cfg.rate_hz)
            elif not args.now:
                print("releasing every tendon and centering the wrist (SPACE = e-stop)")
                with SpaceWatch(s.estop):
                    go_to_pose(s.hand, Pose(np.zeros(N_FINGERS), 0.0), args.duration, s.hand_cfg.rate_hz,
                               estop=s.estop)
            s.hand.relax()
            print("hand relaxed: all torque off. It is safe to handle.")
    except EStopTripped as e:
        print(f"\nEMERGENCY STOP: {e}. Hand relaxed.")
        sys.exit(3)


if __name__ == "__main__":
    main()
