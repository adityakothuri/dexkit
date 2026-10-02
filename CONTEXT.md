# CONTEXT: where this project stands (2026-10-02)

Read this first. It records what has actually been verified on the bench, what is still
open, and the traps we already fell into. `HOW_TO_USE.md` is the step-by-step operator
guide; `README.md` is the technical reference.

## What this is

Python control stack for the CMU DexKit foam hand (12 finger servos + 1 forearm-roll
servo, Feetech HLS bus servos on a Waveshare Bus Servo Adapter (A)) and a 3018-style
GRBL gantry. Works natively on macOS and Linux; every command also runs on simulated
hardware with `--mock`.

```bash
cd dexkit && source .venv/bin/activate   # or: make setup  (fresh machine)
make test        # 112 tests, no hardware
make mock-all    # every CLI end-to-end on simulated hardware
make ports       # which USB port is the hand / the gantry
dexkit-scan      # list the servos on the bus
```

## Verified on the real hardware

| Item | Finding |
|---|---|
| Adapter | Waveshare Bus Servo Adapter (A), WCH CH343, shows up on macOS as `/dev/cu.usbmodem…` (auto-detected by USB ID 1a86:55d3) |
| Servo bus | **All 13 servos answer at 1,000,000 bps.** 12 finger servos report model 3082 (fw 3.43), the roll servo model 3594 (fw 3.41). |
| Servo IDs | **Fingers are 0–11, roll is 12** (not 1–13 as the original docs assumed). `config/hand.yaml` has been renumbered to match. |
| Supply | Drok PSU at 7.4 V. Servos read 7.1–7.3 V. Voltage gate (6.0–8.4 V) passes. |
| Jumper caps | **Both caps must stand vertically on the middle + bottom pins (the "B" = USB→SERVO rows).** Laid sideways (joining the two columns) they short the adapter's TX to RX: the PC sees a perfect echo of its own packets and no servo replies. This cost us an hour; the scan now detects the echo and says so. |
| Servo registers | Read from a live servo: angle limits min=max=0 (multi-turn mode), max torque (reg 16) 980, mode (reg 33) 0 on fingers, **mode 4 on the roll servo**, lock (55) = 1. A direct position write moves a finger servo correctly (200 ticks out and back). |
| Gantry | **Not yet connected or tested.** GRBL driver only exercised in mock. |

## Open problem: calibration doesn't visibly move the fingers

`dexkit-calibrate-hand` runs, but the operator reports the fingers do not move (earlier:
"barely move"). A direct register write *does* spin the thumb servo (ID 0). Leading
hypothesis, not yet confirmed: the tendon spools need more than half a turn to take up
slack, and calibration stops early because of two limits in `capture_servo()`:

- it aborts after 2,000 ticks (~176°) of travel "for safety";
- it refuses to step past position 0 or 4095 (ID 1 sits at 4082, ID 2 at 385).

The servos' EEPROM angle limits are 0/0, which on Feetech STS/HLS means multi-turn
mode, i.e. the spools are designed to rotate more than once. If that is confirmed
(turn a spool by hand with power off and see how far before the finger moves), the fix
is to let calibration travel multiple turns: raise/remove the 2,000-tick cap, and handle
positions beyond 0..4095 (the protocol encodes 15-bit magnitudes; `decode_position` /
`encode_position` in `hw/feetech_protocol.py` already support that, but
`config.py` validation and `TickClamp` assume 0..4095).

Also unverified: whether the roll servo's mode 4 is a problem (mode 0 = position; the
driver refuses to connect if mode != 0, so `dexkit-pose` will currently reject the hand
until this is understood or the mode is changed).

Diagnostic snippets that proved useful are in the git history of this file's author's
session; the simplest is:

```python
from dexkit.config import load_hand_config
from dexkit.hw.feetech_hand import FeetechDriver, open_serial
from dexkit.hw.feetech_protocol import FeetechBus
cfg = load_hand_config(); t = open_serial(cfg.port, cfg.baud, cfg.timeout_s)
d = FeetechDriver(FeetechBus(t, timeout_s=0.05), cfg.register_map())
print(d.read_position(0), d.read_load(0), d.read_mode(12))
```

## What changed in this session (all tested, 112 tests pass)

- `hw/ports.py`: USB auto-detection by vendor/product ID; `DEXKIT_HAND_PORT` /
  `DEXKIT_GANTRY_PORT` override; `make ports`.
- `hw/feetech_protocol.py`: skips adapter echoes; skips malformed packets instead of
  crashing (`ff ff 02 01 23` was seen on the bench).
- `tools/scan_bus.py`: reports echo-only buses, refuses impossible servo counts, tells
  you when the baud in `hand.yaml` needs changing.
- `control/poses.py`: `dexkit-pose --gantry` adds `where / jog / zero / goto`; `f 0 0.5`
  no longer silently moves finger 12.
- `cli.py`: teleop continues hand-only if the gantry is unplugged; one-line friendly
  errors (use `-v` for tracebacks).
- `tools/calibrate_hand.py`: default step 40 ticks (was 10); `f`/`s` keys change speed
  live; `--step-ticks`, `--torque` flags.
- `config/hand.yaml`: IDs 0–11 + 12. Tests pin their own IDs (`tests/conftest.py`).
- Docs: `HOW_TO_USE.md` (new), README updates, jumper diagram.

## Still placeholder / untested

- `config/hand.yaml` has **placeholder slack/tight values** (`calibrated_at: null`); real
  motion commands refuse to run until `dexkit-calibrate-hand` writes real ones.
- Finger names/groups in `hand.yaml` (thumb_flex, …) are guesses; rename after you learn
  which servo drives which tendon.
- `config/gantry.yaml` travel box is a stock-3018 default; gantry never connected.
- The policy layer (`src/dexkit/policy/`) is ported but unused.

## Hardware notes

- The hand can be calibrated and driven **off the gantry**; hand and gantry configs are
  independent. Clamp the arm down: the roll servo is strong.
- Both white servo cables go into the adapter's servo ports (they share one bus).
- Power the servos from the PSU terminal, never from USB.
- `dexkit-estop` from a second terminal kills everything; `dexkit-estop --clear` re-arms.
