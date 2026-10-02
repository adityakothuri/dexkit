"""Find Feetech servos on the bus: IDs, model numbers, firmware, position, voltage.

    dexkit-scan                     scan IDs 0..253 at 1 Mbps, then 115200 if nothing answers
    dexkit-scan --assign N --i-know with ONE servo connected, change its ID to N
"""

from __future__ import annotations

import argparse
import logging
import sys

from dexkit.config import load_hand_config
from dexkit.hw.feetech_hand import FeetechDriver, open_serial
from dexkit.hw.feetech_protocol import FeetechBus, Transport
from dexkit.util import setup_logging

log = logging.getLogger(__name__)

EXPECTED_SERVOS = 13

REID_PROCEDURE = """
Every servo answered at ID 1 (factory default), so they collide on a shared bus.
Re-ID them one at a time:
  1. Power off. Connect ONE servo to the adapter.
  2. Power on and run:   dexkit-scan --assign N --i-know     (N = 1..13)
  3. Power-cycle that servo, label it with N, disconnect it.
  4. Repeat for the next servo. Use 13 for the HLS3640M roll servo.
Never re-ID with more than one servo connected.
"""


def scan(driver: FeetechDriver, ids: range) -> list[dict]:
    found = []
    for sid in ids:
        model = driver.ping(sid)
        if model is None:
            continue
        found.append({
            "id": sid,
            "model": model,
            "firmware": driver.read_firmware(sid),
            "position": driver.read_position(sid),
            "voltage": driver.read_voltage(sid),
        })
    return found


def open_bus(args: argparse.Namespace, baud: int) -> tuple[FeetechDriver, Transport]:
    cfg = load_hand_config()
    if args.mock:
        from dexkit.hw.mock import MockFeetechSerial, mock_servos_for

        if getattr(args, "_mock_bus", None) is None:
            servos = mock_servos_for(cfg)
            if args.mock_factory_ids:
                servos = servos[:1]
                servos[0].id = 1
            args._mock_bus = MockFeetechSerial(servos, baudrate=baud)
        t = args._mock_bus
        t.baudrate = baud
    else:
        t = open_serial(args.port or cfg.port, baud, cfg.timeout_s)
    timeout = 0.002 if args.mock else cfg.timeout_s  # the simulator answers instantly
    return FeetechDriver(FeetechBus(t, timeout_s=timeout), cfg.register_map()), t


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="serial port (default from hand.yaml)")
    p.add_argument("--baud", type=int, help="only try this baud rate")
    p.add_argument("--max-id", type=int, default=253)
    p.add_argument("--assign", type=int, metavar="N", help="change the single connected servo's ID to N")
    p.add_argument("--i-know", action="store_true", help="confirm the EEPROM write for --assign")
    p.add_argument("--expect", type=int, default=EXPECTED_SERVOS)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--mock-factory-ids", action="store_true", help="mock: one servo at factory ID 1")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_hand_config()

    bauds = [args.baud] if args.baud else [cfg.baud, cfg.fallback_baud]
    found: list[dict] = []
    driver = None
    used_baud = None
    echo = False
    for baud in bauds:
        driver, t = open_bus(args, baud)
        print(f"scanning IDs 0..{args.max_id} at {baud} bps ...")
        found = scan(driver, range(0, args.max_id + 1))
        echo = echo or driver.bus.echo_seen
        if found:
            used_baud = baud
            break
        t.close()
    if not found or driver is None:
        print("no servos answered at", ", ".join(map(str, bauds)), "bps")
        if echo:
            print("the adapter echoes what we send and no servo replied: on the Waveshare board this\n"
                  "means the jumper caps are lying sideways. Stand both caps vertically on B.")
        print("check: power supply ON and wired to the adapter's +/- terminal, servo cables seated,\n"
              "       jumper: BOTH caps vertical on the middle+bottom pins (B), USB cable")
        sys.exit(1)

    print(f"\nfound {len(found)} servo(s) at {used_baud} bps:")
    print(f"  {'ID':>3}  {'model':>6}  {'fw':>6}  {'pos':>5}  {'volts':>5}")
    for s in found:
        roll = "  <- roll (per hand.yaml)" if s["id"] == cfg.roll.id else ""
        v = f"{s['voltage']:.1f}" if s["voltage"] is not None else "?"
        print(f"  {s['id']:>3}  {s['model']:>6}  {s['firmware'] or '?':>6}  {s['position']!s:>5}  {v:>5}{roll}")
    if used_baud != cfg.baud:
        print(f"  NOTE: servos answered at {used_baud}, but hand.yaml says baud: {cfg.baud}. "
              f"Edit config/hand.yaml and set  baud: {used_baud}  or no other command will find them.")
    models = {s["model"] for s in found}
    if len(models) > 1:
        print("  distinct model numbers:", sorted(models), "(the odd one out is the HLS3640M roll servo)")

    if args.assign is not None:
        if len(found) != 1:
            print(f"refusing --assign: {len(found)} servos answered; connect exactly ONE servo")
            sys.exit(2)
        if not args.i_know:
            print("--assign writes EEPROM; re-run with --i-know to confirm")
            sys.exit(2)
        old = found[0]["id"]
        ok = driver.set_id(old, args.assign, i_know=True)
        print(f"servo {old} -> {args.assign}: {'OK' if ok else 'FAILED'}. Power-cycle the servo now.")
        sys.exit(0 if ok else 1)

    if len(found) == 1 and found[0]["id"] == 1 and args.expect > 1:
        print(REID_PROCEDURE)
    expected = set(cfg.ids)
    present = {s["id"] for s in found}
    if present != expected:
        print(f"hand.yaml expects IDs {sorted(expected)}; missing {sorted(expected - present)}, "
              f"unexpected {sorted(present - expected)}")
    if len(found) > max(args.expect, 1) * 2:
        print(f"FAIL: {len(found)} IDs answered; that is not a real bus. Re-run with -v and share the output.")
        sys.exit(1)
    if len(found) < args.expect:
        print(f"FAIL: {args.expect - len(found)} servo(s) missing (found {len(found)} of {args.expect})")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
