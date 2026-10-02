"""Standalone emergency stop: open both ports and trip, from a second terminal.

Sends GRBL '!' (feed hold), 0x85 (jog cancel), 0x18 (soft reset), then torque
OFF to every servo (broadcast + per-ID sync write). Does not need the main
process to be responsive. Opening the GRBL port may itself reset the board,
which also stops motion.
"""

from __future__ import annotations

import argparse
import logging
import sys

from dexkit.config import load_gantry_config, load_hand_config
from dexkit.hw.feetech_protocol import BROADCAST_ID, FeetechBus
from dexkit.hw.grbl_protocol import RT_FEED_HOLD, RT_JOG_CANCEL, RT_SOFT_RESET
from dexkit.hw.safety import clear_estop_flag, set_estop_flag
from dexkit.util import setup_logging

log = logging.getLogger(__name__)


def stop_gantry(transport: object) -> None:
    for b in (RT_FEED_HOLD, RT_JOG_CANCEL, RT_SOFT_RESET):
        transport.write(b)  # type: ignore[attr-defined]


def stop_hand(transport: object, ids: list[int], torque_addr: int) -> None:
    bus = FeetechBus(transport)  # type: ignore[arg-type]
    bus.write(BROADCAST_ID, torque_addr, [0], expect_reply=False)
    bus.sync_write(torque_addr, 1, {sid: [0] for sid in ids})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--clear", action="store_true", help="clear the e-stop flag so motion can resume")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    if args.clear:
        print("e-stop flag cleared" if clear_estop_flag() else "no e-stop flag was set")
        return
    # First: the flag makes any running dexkit loop trip on its next tick (and stop re-commanding).
    flag = set_estop_flag("dexkit-estop")
    print(f"e-stop flag set: {flag}")
    hcfg = load_hand_config()
    gcfg = load_gantry_config()
    ok = True

    try:
        if args.mock:
            from dexkit.hw.mock import MockGrblSerial

            gt = MockGrblSerial()
        else:
            from dexkit.hw.grbl_gantry import open_grbl_serial

            gt = open_grbl_serial(gcfg.port, gcfg.baud)
        stop_gantry(gt)
        gt.close()
        print(f"gantry: feed hold, jog cancel, soft reset sent ({'MOCK' if args.mock else gcfg.port})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"gantry: FAILED ({e})")

    try:
        if args.mock:
            from dexkit.hw.mock import MockFeetechSerial, mock_servos_for

            ht = MockFeetechSerial(mock_servos_for(hcfg), baudrate=hcfg.baud)
            for servo in ht.servos.values():  # simulate a hand that is holding torque
                servo.mem[hcfg.register_map()["torque_enable"]] = 1
        else:
            from dexkit.hw.feetech_hand import open_serial

            ht = open_serial(hcfg.port, hcfg.baud, hcfg.timeout_s)
        stop_hand(ht, hcfg.ids, hcfg.register_map()["torque_enable"])
        if args.mock:
            on = [s.id for s in ht.servos.values() if s.torque_on]  # type: ignore[attr-defined]
            assert not on, f"torque still on: {on}"
        ht.close()
        print(f"hand: torque OFF sent to {len(hcfg.ids)} servos ({'MOCK' if args.mock else hcfg.port})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"hand: FAILED ({e})")

    print("motion stays blocked until: dexkit-estop --clear")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
