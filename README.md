# DexKit foam hand

This repo drives the DexKit foam hand and its 3-axis GRBL gantry from plain Python:

- **Hand:** 12× Feetech HLS3620M finger servos and 1× HLS3640M forearm-roll servo, connected through a Waveshare Bus Servo Adapter (A).
- **Gantry:** a 3018-style frame on a "CNC Pro V5" GRBL board.
- **Diffusion policy:** CMU's `diff_foam` policy ships alongside as a dormant second layer, wired to the same environment interface.

Everything runs against simulated hardware (`--mock`), so the whole stack can be checked before any USB device is plugged in.

**New here? Read [HOW_TO_USE.md](HOW_TO_USE.md)**: a step-by-step guide from plugging in to running poses.

It runs natively on macOS as well as Linux. Ports are auto-detected by USB ID (`make ports` shows what was found), so the udev rule and the VM below are optional. To force a port, set `DEXKIT_HAND_PORT` or `DEXKIT_GANTRY_PORT`.

```
src/dexkit/
  hw/        base.py (interfaces, 16-dim action), feetech_protocol.py, feetech_hand.py,
             grbl_protocol.py, grbl_gantry.py, safety.py, mock.py, camera.py
  env/       dexkit_env.py   step(action) / reset() / observe()
  control/   poses.py, teleop.py, recorder.py, sequence.py
  tools/     scan_bus.py, calibrate_hand.py, calibrate_gantry.py, estop.py
  policy/    model/ (verbatim from diff_foam), dataset.py, train.py, infer.py, collect.py, smoke.py
  ros/       bridge_node.py  (optional, rclpy imported lazily)
config/      hand.yaml, gantry.yaml, poses.yaml, policy.yaml
scripts/     99-dexkit.rules
examples/    pick_and_show.yaml
```

## Conventions

The action vector has 16 entries, and the order is fixed in `hw/base.py`:

| Index | Meaning | Units |
|---|---|---|
| 0–11 | Finger servos | 0 = slack, 1 = tight |
| 12 | Forearm roll | degrees |
| 13–15 | Gantry X, Y, Z | mm in the zeroed frame |

Other conventions:

- Ticks and G-code exist only inside `hw/`, and nothing outside `hw/` imports `serial`.
- Every command goes through `hw/safety.py`: voltage gate, tick clamp and slew limit, load watch, temperature watch, travel box, and e-stop.
- Config is YAML, loaded once at startup and validated into dataclasses.

## Install

```bash
make setup         # venv + pip install -e .[policy,dev] (+ udev rule on Linux)
make setup-core    # lighter: hardware layer + tests, no torch
make test          # pytest, no hardware needed (policy/torch tests opt-in: DEXKIT_POLICY_TESTS=1)
make mock-all      # every CLI end to end on simulated hardware
make policy-smoke  # synthetic data -> train 2 epochs -> infer 20 steps (GPU: cuda/mps; refuses CPU)
```

## Commands

Every command accepts `--mock` (simulated hardware), `--yes` (skip the `go` prompt) and `-v` (debug logging, including serial bytes).

| Command | What it does |
|---|---|
| `dexkit-scan` | Find servos: ID, model, firmware, position, voltage. Tries 1 Mbps, then 115200. |
| `dexkit-scan --assign N --i-know` | Re-ID the single connected servo (EEPROM write). |
| `dexkit-calibrate-hand [--servo N] [--force]` | Capture slack/tight per finger and the roll center, then write `config/hand.yaml`. |
| `dexkit-calibrate-gantry` | Jog, zero and mark corners, then write the travel box to `config/gantry.yaml`. |
| `dexkit-pose` | Interactive pose prompt. Stays connected: type `fist`, `open`, `f 3 0.5`, `roll 20`, `save NAME`, `state`, `quit`. |
| `dexkit-pose --gantry` | Same prompt, plus gantry commands: `where`, `jog x 10`, `zero`, `goto X Y Z`. |
| `dexkit-pose <name>` / `--list` | One-shot: move, hold 2 s (`--hold`), relax. `--minjerk` optional. |
| `dexkit-teleop` | Keyboard teleop with a live status line (keys below). |
| `dexkit-replay <recording>` | Replay a `data/recordings/*.npz` recording, with a 1 s lead-in. |
| `dexkit-run <sequence.yaml> [--dry-run]` | Validate every step, then execute the sequence. |
| `dexkit-estop` / `dexkit-estop --clear` | Standalone kill from a second terminal. It sets a flag that stops any running dexkit loop and blocks new sessions until you run `--clear`. |
| `dexkit-collect`, `dexkit-train`, `dexkit-infer` | The policy layer (Phase 6, see below). |
| `dexkit-ros-bridge` | Optional ROS 2 bridge on CMU's topic names. |

**Teleop keys:**

| Key | Action |
|---|---|
| `1`–`9`, `0`, `-`, `=` | Select finger servo 1–12 |
| `[` / `]` | Selected finger −/+ 0.05 |
| `,` / `.` | Roll −/+ 5° |
| `w` `a` `s` `d` | Gantry Y+ / X− / Y− / X+, 1 mm steps (Shift = 10 mm) |
| `q` / `e` | Gantry Z up / down |
| `z` | Set gantry zero |
| `h` | Home the gantry (only when homing is enabled) |
| `o` / `f` / `p` | Open / fist / pinch |
| `r` | Start/stop recording |
| **Space** | **Emergency stop** |
| Esc | Relax and quit |

The default terminal key backend (blessed) reports key presses only. Jogs are therefore discrete steps that finish on their own. With `--keys pynput` (needs X11), releasing a key also sends GRBL jog-cancel.

## VM setup (Ubuntu 22.04 on the Mac)

1. Install Ubuntu 22.04 LTS. On Apple Silicon, use the **arm64** image in UTM or Parallels. On an Intel Mac, any VM or dual boot works.
2. Pass both USB serial devices through to the VM. Both are WCH chips with vendor ID `1a86`:
   - Waveshare adapter: CH343, `1a86:55d3`, appears as `/dev/ttyACM0`.
   - GRBL board: CH340, `1a86:7523`, appears as `/dev/ttyUSB0`.

   After passthrough, `ls /dev/ttyACM* /dev/ttyUSB*` must show both.
3. Add yourself to the `dialout` group with `sudo usermod -aG dialout $USER`, then log out and back in.
4. Install the udev rule. This gives you the stable names `/dev/dexkit_hand` and `/dev/dexkit_gantry`:
   ```bash
   sudo cp scripts/99-dexkit.rules /etc/udev/rules.d/
   sudo udevadm control --reload && sudo udevadm trigger
   ```
   `make setup` does this on Linux. If your board's product ID differs, edit the rule using `udevadm info -a -n /dev/ttyACM0`.
5. Install Python 3.10+ (Ubuntu 22.04 ships 3.10), then run `make setup`.
6. ROS 2 Humble is optional and only needed for `dexkit-ros-bridge`. Install it from the official apt repo, then:
   ```bash
   sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-cv-bridge
   ```

## Power: the Drok PSU procedure (do this first)

The servos are rated 6.0–7.4 V. The Waveshare adapter accepts 5–8.4 V on its DC jack or screw terminal. **USB does not power the servos.**

1. Disconnect the PSU from everything. Power it on, turn the trim pot until the display reads **7.4 V**, then power it off.
2. Wire PSU V+ to the adapter terminal **+** and PSU V− to **−**.
3. Set the adapter's two jumper caps to **B** (USB control): each cap vertical, on the middle and bottom pins of its column. Caps laid sideways loop TX back to RX, so the bus only echoes.

The software also enforces this. At connect it reads register 62 (present voltage) from every servo and **refuses to enable torque outside 6.0–8.4 V**. During operation it re-checks every 2 s and relaxes the hand if the voltage leaves that window.

## Hardware bring-up order

Do these in order. Every software-only check (`make test`, `make mock-all`) should pass before you start.

1. Set the PSU to 7.4 V as above, then power it off.
2. Wire the PSU to the adapter's screw terminal, set the jumper on B and connect USB. `ls /dev/ttyACM*` should show the adapter.
3. Run `dexkit-scan` and expect 13 servos. If every servo reports ID 1, follow the one-at-a-time re-ID procedure that the tool prints (`dexkit-scan --assign N --i-know`, one servo on the bus at a time).
4. Run `dexkit-calibrate-hand` and walk every finger, pressing `t` + Enter at the tight pose. The first run on the shipped placeholder file needs no flag. Use `--servo N` to redo one entry.
5. Run `dexkit-pose open`, then `dexkit-pose fist`. If any finger moves the wrong way, flip its `inverted` flag in `hand.yaml`.
6. Connect the GRBL board's USB and confirm `ls /dev/ttyUSB*` shows it. Run `dexkit-calibrate-gantry`, check the banner, jog each axis a few mm, set zero with `z`, mark the far corners with `m`, and write the box with `x`.
7. Run `dexkit-teleop` and drive the fingers, roll and gantry. **Press Space once on purpose** to test the e-stop.
8. Record a 10 s motion with `r`, then replay it with `dexkit-replay <name>`.
9. Write a sequence YAML (start from `examples/pick_and_show.yaml`) and run it with `dexkit-run`. This is the demo.

Hand loop timing (one 13-servo sync write plus one sync read) is logged. The target is under 50 ms at 1 Mbps.

### Gantry zeroing

Stock 3018 boards usually have no limit switches, so `zeroing: manual` is the default:

- Motion is refused until the operator jogs to the reference corner and presses `z` (G92 X0 Y0 Z0).
- Opening the GRBL port resets the board (via DTR), which loses the zero. When the frame is valid at exit, the driver saves the last position to `data/state/gantry_frame.json`. The next command then offers to restore it; `--restore-zero` restores it without asking. Only restore if nobody moved the gantry in between. Setting `$1=255` keeps the steppers energised so the gantry holds position.
- If switches are installed and `$22=1`, set `zeroing: home` and the driver uses `$H`.

## Safety summary

| Guard | Where | Behaviour |
|---|---|---|
| VoltageGate | connect and every 2 s | Refuses torque (or relaxes and trips) outside 6.0–8.4 V |
| TickClamp | every `set_targets` | Per-servo [slack, tight] limits and at most `max_delta_ticks` per tick |
| LoadWatch | every state read | A finger above `stall_load` for 0.5 s backs off 5% toward slack |
| TempWatch | every 2 s | Trips above 65 °C |
| TravelBox | `move_to`, jogs | Rejects targets outside the box; jogs are clamped to it |
| EStop | Space, `dexkit-estop`, any trip | GRBL `!`, then 0x85, then 0x18, then torque off; a global flag stops every loop |
| Signals | SIGINT, SIGTERM, atexit | Hand relax and gantry feed hold |

Before any motion, every CLI prints a startup checklist and waits for you to type `go`. The checklist covers ports, servo voltages, the calibration timestamp and whether the gantry frame is valid.

## Phase 6: the dormant diffusion policy

`policy/` is CMU `diff_foam` ported to this hand and to current libraries (verified with torch 2.14 and diffusers 0.40):

- `model/noise_pred_net.py` (ConditionalUnet1D) and `model/visual_encoder.py` (ResNet18, with BatchNorm swapped for GroupNorm) are copied **verbatim**.
- The action and low-dimensional observation are 16-dim, normalized to [−1, 1] from `hand.yaml` and `gantry.yaml` limits rather than hard-coded arrays.
- CMU's hyperparameters are kept: pred/obs/action horizons of 16/2/8, DDPM with 100 steps, the `squaredcos_cap_v2` schedule, clip_sample, epsilon prediction, EMA power 0.75, AdamW at 1e-4 (weight decay 1e-6), and a cosine schedule with 500 warmup steps.
- Images are stored at 240×320 and center-cropped to 216×288.
- Porting note: in current diffusers, `EMAModel.step()` takes parameters, so CMU's `ema.step(nets)` became `ema.step(nets.parameters())`.

**Collecting demos** (needs a webcam):

1. Mount the webcam so it sees the hand and the workspace. Never move it between demos and training.
2. Run `dexkit-collect --task <name>`. This is teleop with the camera on; each `r` toggle saves one episode to `data/demos/<name>/episode_XXXX.pkl`.
3. Collect 50–100 episodes of one task, with small variations in object position.
4. Train on a GPU (Colab or a cloud box) with `dexkit-train --data data/demos/<name> --task <name> --device cuda`, then copy `data/checkpoints/<name>.pt` back.
5. Run `dexkit-infer --checkpoint data/checkpoints/<name>.pt --scheduler ddim --num-inference-steps 16`. With DDIM at 16 steps this is light enough for a CPU at 10 Hz.

Later direction (not built): replace the from-scratch ResNet conditioning with a frozen VLA backbone (pi0.5 / OpenVLA-OFT) and keep this diffusion head, following the InDex paper (arXiv 2606.12109).

## Open questions for the bench

Each of these changes a config value, not the code. The code handles both answers.

- **Servo IDs:** unique 1–13, or all at factory ID 1? `dexkit-scan` tells you.
- **Servo baud:** 1 Mbps or 115200? The scan tries both.
- **HLS register map:** does it match STS at addresses 40–63? The SDK's `scservo_sdk/hls.py` agrees on 40, 41, 42, 46, 55, 56, 58, 60, 62 and 63, **but address 44 is Goal Torque on HLS (Goal Time on STS)**. The driver writes `goal_torque` at connect and never streams to 44. The torque limit at 48 is not listed in `hls.py`, so verify it against Feetech's "Magnetic Encoding Version" memory table. Any address can be overridden under `registers:` in `hand.yaml`.
- **Roll servo:** which ID is the HLS3640M? Its model number in the scan output differs from the others.
- **Finger mapping:** which servo drives which finger and joint? The names and groups in `hand.yaml` are provisional. Rename them after calibration, then cite the thesis's DexKit chapter once it has been read.
- **GRBL:** version (shown in the banner on connect), limit switches (`$21`/`$22` plus a physical look), and actual travel (it may not match a stock 3018).
- **The Mac** is Apple Silicon, so use the arm64 Ubuntu image. `--device mps` works for local training on macOS.

## Sources

- [CMU-Foam-Hands-Lab/diff_foam](https://github.com/CMU-Foam-Hands-Lab/diff_foam), [realsense_ros2_humble](https://github.com/CMU-Foam-Hands-Lab/realsense_ros2_humble), [dexkit-page](https://github.com/CMU-Foam-Hands-Lab/dexkit-page)
- J. P. King, *Accessible Dexterous Manipulation with Soft Hands…*, CMU-RI-TR-26-15 (2026)
- Chi et al., *Diffusion Policy*, RSS 2023
- Feetech SDK (`scservo_sdk`, including `hls.py`) via `vassar-feetech-servo-sdk`, the Waveshare Bus Servo Adapter (A) wiki, and the GRBL 1.1 interface docs
