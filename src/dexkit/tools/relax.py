"""Bring the hand back to normal: every tendon released, wrist to neutral, then torque off.

    dexkit-relax            gently release everything (SPACE = e-stop), then relax
    dexkit-relax --now      skip the gentle move: torque off immediately, nothing moves

Use it after a sequence, a pose you don't like, or before putting the hand down.
It never asks for 'go', so it also works from a shell without a keyboard.
"""

from __future__ import annotations

import sys

import numpy as np

from dexkit.control.estop_key import SpaceWatch
from dexkit.control.poses import Pose, go_to_pose
from dexkit.hw.base import N_FINGERS
from dexkit.hw.safety import EStopTripped


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import common_parser, init, open_session

    p = common_parser(__doc__ or "")
    p.add_argument("--now", action="store_true", help="torque off immediately without moving first")
    p.add_argument("--duration", type=float, default=2.0, help="seconds for the gentle release move")
    args = p.parse_args(argv)
    args.yes = True  # the "make it safe" command never waits for 'go'
    init(args)
    try:
        with open_session(args, need_hand=True) as s:
            assert s.hand is not None and s.hand_cfg is not None
            if not args.now:
                print("releasing every tendon and centering the wrist (SPACE = e-stop)")
                with SpaceWatch(s.estop):
                    go_to_pose(s.hand, Pose(np.zeros(N_FINGERS), 0.0), args.duration, s.hand_cfg.rate_hz,
                               estop=s.estop)
            s.hand.relax()
            print("hand relaxed: all torque off, wrist at neutral. It is safe to handle.")
    except EStopTripped as e:
        print(f"\nEMERGENCY STOP: {e}. Hand relaxed.")
        sys.exit(3)


if __name__ == "__main__":
    main()
