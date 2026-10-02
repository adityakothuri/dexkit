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

## Resolved: why calibration didn't visibly move the fingers

The servos are in **multi-turn mode** (EEPROM angle limits 0/0). Verified on the thumb
servo (ID 0): a goal of 5200 drove the position register to 5175, i.e. it counts past
4095 instead of wrapping. The tendon spools therefore need more than one revolution, and
the old calibration gave up long before that: it refused to pass 0/4095 (the thumb sat
at 3604, so it quit after 12 steps) and capped travel at 2,000 ticks. The servos also
shipped with speed register 46 = 100, so goals were followed at a crawl.

Fixed (2026-10-02): config accepts 15-bit positions (`MAX_TICKS` = 32767),
`calib_max_travel_ticks` (default 3 turns) replaces the 2,000-tick cap, the 0..4095 stop
is gone, and calibration sets speed/accel from `defaults` before stepping.
**Power-cycle finding (verified):** the multi-turn count is lost when the PSU is switched off
(servo 0 read 22191, then 1711 = 22191 mod 4096 after a cycle). So absolute slack/tight
values in `hand.yaml` only matter as a *span*. `FeetechHand.connect()` now re-bases: each
finger's present position becomes open (0) and the roll center snaps to its nearest
equivalent turn. Operating rule: fingers relaxed/open before `go`; `home` at the pose
prompt re-bases later. `hand.yaml` carries per-servo `calibrated:` flags; servo 0 is the
**pinky** (observed), labelled `pinky_1`; the other names are still guesses.

**Not yet re-tested on the hand after the fix.** Next step is literally
`dexkit-calibrate-hand --servo 0` while watching the thumb: it should now keep winding
until the finger curls, and `r` reverses if the tendon tightens the other way.

Roll servo mode: an early dump showed register 33 = 4, but with the servo plugged in on
its own it reads 0 (position mode) consistently, and a +100-tick test turn at torque 300
moved it 441 -> 542 -> 442 with load 3. No change was made. The earlier 4 was most
likely a misread (the 70-byte block read fell back to 64 bytes that day).

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

## Tendon layout and wiring (from the lab's description, 2026-10-02)

**This is a RIGHT hand** (`side: right` in hand.yaml). Roll sign and the adduct directions
(`index_adduct` toward the thumb, `thumb_adduct` toward mid-palm) are described for a right hand.

12 tendons, one per servo, pull only. Palm side: 5 flexors + `thumb_adduct`. Back side:
5 extensors + `index_adduct`. Flexor/extensor of a finger are antagonists; the driver
scales a pair so it never sums past 1.0 (`AntagonistLimit`). Canonical names in
`config.TENDONS` set `finger`/`role`; `dexkit-calibrate-hand --wiring` prints the
channel table; `--label-only --name X` labels a calibrated channel without moving it.
**Wiring confirmed on the bench (2026-10-02), all 12 hand channels calibrated;** only the
wrist (ch 12) remains. Every tendon winds toward higher counts (`inverted: false`).

| ch | lab's name | code name | motion |
|---|---|---|---|
| 0 | pinky_curl | `pinky_flex` | pinky curls in |
| 1 | thumb_back | `thumb_extend` | thumb bends back |
| 2 | ring_curl | `ring_flex` | ring curls in |
| 3 | pointer_turn | `index_adduct` | index curls toward the thumb |
| 4 | point_curl | `index_flex` | index curls in |
| 5 | middle_back | `middle_extend` | middle bends back |
| 6 | middle_curl | `middle_flex` | middle curls in |
| 7 | pointer_back | `index_extend` | index bends back |
| 8 | thumb_curl | `thumb_flex` | thumb curls to the upper palm |
| 9 | ring_back | `ring_extend` | ring bends back |
| 10 | thumb_turn | `thumb_adduct` | thumb turns in toward mid-palm |
| 11 | pinky_back | `pinky_extend` | pinky bends back |
| 12 | | forearm roll | wrist twist |


## Poses, choreography, speed (2026-10-02)

- Pose library is `config/poses/*.yaml` (basic/signs/digits/grasps/custom), merged by
  `PoseLibrary.load`; duplicate names across files are an error; `save` writes custom.yaml.
- Sequences in `examples/`: show_off, count_to_five, rock_paper_scissors, wave_hello,
  finger_ripple, **team_7503** (7-5-0-3 with a fist between digits). `dexkit-run --loop N`.
- `go_to_pose` now waits for the servos to physically arrive (tolerance 0.05, 12 s timeout)
  before returning, so sequences hold their timing at any speed.
- Speed: `defaults.speed` 1500 -> 800, accel 50 -> 20, `max_delta_ticks` 120 -> 80 after the
  operator reported fast motion straining the hand; `--speed-scale S` scales all three per run.
  Note `pinky_flex` has a 2-turn span (8149 ticks): a full curl takes ~10 s at 800.

## Hardware notes

- The hand can be calibrated and driven **off the gantry**; hand and gantry configs are
  independent. Clamp the arm down: the roll servo is strong.
- Both white servo cables go into the adapter's servo ports (they share one bus).
- Power the servos from the PSU terminal, never from USB.
- `dexkit-estop` from a second terminal kills everything; `dexkit-estop --clear` re-arms.
