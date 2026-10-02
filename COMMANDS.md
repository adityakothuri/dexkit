# DexKit commands

Every session starts with:
```bash
cd ~/dexkit && source .venv/bin/activate
```
Run commands from a normal Terminal window (they ask you to type `go` before anything moves). The software remembers where "open" is between sessions; if the checklist says `open position unknown`, do `dexkit-pose` → `home` once.

## Emergency stop

| situation | do |
|---|---|
| anything is moving | **SPACE**. Stops within one control tick, hand goes limp, gantry holds, the command exits. Works in `dexkit-pose`, `dexkit-run`, `dexkit-replay`, `dexkit-teleop`, `dexkit-infer`, and both calibration tools. Nothing overrides it; restart the command to continue. |
| at a prompt, or SPACE didn't take | **Ctrl+C**. Same effect. |
| from a second window | `dexkit-estop` — stops everything and blocks every command until `dexkit-estop --clear` |
| last resort | PSU off |

## Every day

| command | what it does |
|---|---|
| `dexkit-pose` | Interactive prompt. Type a pose name (`fist`, `open`, `peace`, …), `f 3 0.5` (one tendon), `roll 30` (wrist, degrees), `normal` (release all, wrist 0), `state`, `save NAME`, `list`, `help`, `q`. |
| `dexkit-pose fist` | One-shot: go to the pose, hold 2 s (`--hold N`), relax, exit. |
| `dexkit-pose --list` | All pose names, grouped by file. |
| `dexkit-pose --gantry` | Same prompt plus `where`, `jog x 10`, `zero`, `goto X Y Z`. |
| `dexkit-relax` | **Back to normal:** release every tendon, wrist to neutral, torque off. `--now` = torque off without moving. |
| `dexkit-run FILE.yaml` | Run a routine. `--loop N` repeats (0 = until SPACE/Ctrl+C), `--dry-run` only checks it. |
| `dexkit-teleop` | Live keyboard control (keys printed on start). |

Options every command takes: `--speed-scale 0.5` (slower), `--mock` (simulate), `--yes` (skip `go`), `-v` (debug).

## Fixing a pose (in this order)

| step | command | what you do |
|---|---|---|
| 1. set open | `dexkit-relax --unwind` | motors unwind; press a key when every finger looks open (number keys stop single tendons) |
| 2. set curl limits | `dexkit-calibrate-hand --tight-only` | each driven tendon winds slowly from open; press `t` at a firm, unstrained curl (`x` skip, SPACE stop). Add `--only ring_flex` for one tendon |
| 3. review | `dexkit-pose --review` | every pose in turn, slowly, with its "should look like" line. Enter = next, `t` = tune it, `q` = quit |
| 4. tune | `dexkit-teleop --tune peace` | keys `1`–`7` pick a tendon, `[`/`]` nudge it, `0` zero, `o` release all, `,`/`.` wrist, **`v` saves** into the pose's file, `n`/`p` next/previous pose, Esc quit. Slow and low-torque |

## Routines (`examples/`)

| file | does |
|---|---|
| `team_7503.yaml` | 7 – 5 – 0 – 3 |
| `show_off.yaml` | tour of every pose |
| `count_to_five.yaml` | 1 … 5 |
| `rock_paper_scissors.yaml` | three wrist pumps, then scissors |
| `wave_hello.yaml` | open hand, wrist wave |
| `finger_ripple.yaml` | pinky → thumb → pinky curl ripple (try `--loop 3`) |
| `pick_and_show.yaml` | hand + gantry demo (needs the gantry) |

## Poses (`config/poses/`)

| file | poses |
|---|---|
| `basic.yaml` | relax, open, fist, point, thumbs_up, wave_a, wave_b |
| `signs.yaml` | peace, rock_on, shaka, spidey, finger_gun, ok |
| `digits.yaml` | zero, one, two, three, four, five, seven |
| `grasps.yaml` | pinch, claw, tripod, cross_thumb |
| `custom.yaml` | yours (`save NAME` at the prompt) |
| generated | `<tendon>_only` (e.g. `pinky_flex_only`), `finger_N_curl` |

## Setup and calibration (once)

| command | what it does |
|---|---|
| `make ports` | Which USB port is the hand, which is the gantry. |
| `dexkit-scan` | List the servos on the bus (expect 13). |
| `dexkit-calibrate-hand --wiring` | Table: every motor, its tendon name, calibrated or not. |
| `dexkit-calibrate-hand --servo N --name TENDON` | Calibrate one motor (see `CALIBRATION.md`). Keys while stepping: `t` save, `x` abort, `r` reverse, `u` back up, `f`/`s` speed, SPACE e-stop. |
| `dexkit-calibrate-hand --servo N --label-only --name TENDON` | Rename a motor without moving it. |
| `dexkit-calibrate-hand --servo 12` | Wrist: hold neutral, Enter. |
| `dexkit-calibrate-gantry` | Jog, set zero (`z`), mark corner (`m`), save (`x`). SPACE stops. |

## Config files

| file | holds |
|---|---|
| `config/hand.yaml` | servo IDs, tendon names, calibration, `speed`/`accel`, which servos are `enabled` |
| `config/poses/*.yaml` | the pose library |
| `config/gantry.yaml` | gantry port, travel box |
| `examples/*.yaml` | routines |

## Checks without hardware

```bash
make test        # unit tests
make mock-all    # every command end to end on simulated hardware
```
