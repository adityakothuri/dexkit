# How to use the DexKit hand + gantry

Plain steps, in order. Copy each grey box into the Terminal and press Enter.

You only do **Part A** once. After that, every session is just **Part B**.

---

## What's what

| Thing | What it is | Its cable |
|---|---|---|
| **Hand** | 12 finger motors + 1 wrist-twist motor | USB from the small **Waveshare servo board** |
| **Power supply (PSU)** | Powers the hand motors. USB does **not**. | Two wires to the Waveshare board's screw terminal |
| **Gantry** | The X/Y/Z frame that carries the hand | USB from the **gantry's control board** |

Both USB cables go into your Mac. A USB hub is fine.

---

## Doing the hand first, before it goes on the gantry

You can, and it's a good idea. Every hand step (A1–A6) works without the gantry. Leave the gantry's USB unplugged and skip A7 until later.

**The two white servo cables**

- One cable comes from the **12 chained finger motors**.
- The other comes from the **wrist-twist motor** (the big, strong one, ID 12).

Plug **both** into servo ports on the Waveshare board. All its servo ports share one connection, so the software sees all 13 motors either way. If you only have one free port, plug the finger chain into the spare connector on the wrist motor, then plug the wrist motor into the board.

**Before you power on**

- **Clamp or hold the arm down.** When the wrist motor turns, an unmounted arm can twist itself across the table.
- Keep fingers clear of anything they could snag on.

**Testing one cable at a time** (handy for finding a bad connection)

`dexkit-scan` lists exactly which motor numbers answer:

- Only the finger chain plugged in: you should see IDs 0–11 found and 12 listed as missing.
- Only the wrist plugged in: you should see 12 found.

Plug both in before you run `dexkit-calibrate-hand`.

**After you mount it on the gantry**

The finger calibration stays valid. If the wrist motor's horn or coupling was taken off or turned during mounting, redo only the wrist's center:

```bash
dexkit-calibrate-hand --servo 12
```

Then do A7 for the gantry.

---

## Part A: one-time setup

### A1. Set the power supply to 7.4 V (important: wrong voltage can kill the motors)

1. Unplug the PSU's output wires from everything.
2. Turn the PSU on and turn its little screw (trim pot) until the display reads **7.4 V**.
3. Turn the PSU off.
4. Wire PSU **+** to the Waveshare board's **+** terminal, and PSU **−** to **−**.
5. Set the **two yellow jumper caps** to **B** (B = the USB port controls the motors). The jumper block is 3 rows × 2 columns of pins. Each cap stands **up and down (vertically)**, one in each column, covering the **middle and bottom** pins. The top two pins stay bare:
   ```
           left  right
   top      o     o      <- bare
   middle  [o]   [o]
   bottom  [o]   [o]     <- each [ ] pair is one cap, standing vertically
   ```
   ⚠️ **Never lay the caps sideways** (joining left to right). That loops the computer's messages back on itself: the scan sees an "echo" and no motors.

If the voltage is ever outside 6.0–8.4 V, the software refuses to move. That's on purpose.

### A2. Open a Terminal in the project

```bash
cd ~/dexkit
source .venv/bin/activate
```

Everything is already installed. Do this at the start of every session (Part B repeats it).

> If you ever start fresh on a new computer, run `make setup` once first.

### A3. Check both devices are detected

Plug in **both** USB cables, turn the PSU on, then run:

```bash
make ports
```

You want to see:

```
hand   -> /dev/cu.usbmodem....
gantry -> /dev/cu.usbserial-....
```

If one says `NOT FOUND`, unplug and replug that USB cable, then run it again.

### A4. Check all 13 hand motors answer

Your hand's motors are numbered **0–11 for the fingers** and **12 for the wrist**. `config/hand.yaml` is already set up to match.

```bash
dexkit-scan
```

- **`found 13 servo(s)` and `OK`**: go to A5.
- **It says to edit `baud:`**: open `config/hand.yaml`, change the `baud:` line to the number it printed, save, and run `dexkit-scan` again.
- **Only 1 servo found, at ID 1**: the motors are all set to the factory ID, so they clash. Follow the steps it prints. You'll connect one motor at a time and give each a number with `dexkit-scan --assign N --i-know`: 0–11 for the fingers, 12 for the wrist. (Your current hand is already numbered this way.)

### A5. Calibration (one motor at a time)

There are 13 motors: 12 pull tendons in the hand (IDs 0–11) and 1 twists the wrist (ID 12). You calibrate each one on its own. The finished list is always one command away:

```bash
dexkit-calibrate-hand --wiring      # shows every motor: its name, and whether it's done
```

**For each hand motor (IDs 0–11), do these 4 steps:**

**Step 1. Find out what it moves.**
```bash
dexkit-calibrate-hand --servo 3
```
Press **Enter**, then **Enter** again. The motor starts winding. Watch the hand:
- a finger **curls in** toward the palm → that's a **flex** tendon
- a finger **bends back** → an **extend** tendon
- the thumb or index moves **sideways** → an **adduct** tendon

Now type **`x`** and press **Enter**. The motor stops and nothing is saved.

**Step 2. Calibrate it with its name.**
```bash
dexkit-calibrate-hand --servo 3 --name index_flex
```
Use the name that matches what you saw. The 12 allowed names:

| finger | curls in | bends back | sideways |
|---|---|---|---|
| thumb | `thumb_flex` | `thumb_extend` | `thumb_adduct` |
| index | `index_flex` | `index_extend` | `index_adduct` |
| middle | `middle_flex` | `middle_extend` | |
| ring | `ring_flex` | `ring_extend` | |
| pinky | `pinky_flex` | `pinky_extend` | |

**Step 3. First prompt: "RELAXED position".** Let the finger sit naturally, not pulled in or back. Press **Enter**.

**Step 4. Second prompt: "fully PULLED pose".** Press **Enter** and watch the motor wind the tendon. When the finger is as far as you want it to ever go (firmly curled / bent back / across, but not straining), type **`t`** and press **Enter**. Done: it saves and prints the updated table.

Helpful keys while it's winding (each followed by Enter): **`r`** reverse, **`u`** back up a bit, **`f`** faster, **`s`** slower, **`x`** stop without saving.

**Already calibrated a motor but didn't name it?** Just label it, no movement:
```bash
dexkit-calibrate-hand --servo 3 --label-only --name index_flex
```

**The wrist (ID 12):**
```bash
dexkit-calibrate-hand --servo 12
```
Twist the forearm to its neutral position by hand, press **Enter**. Done.

**When `--wiring` shows all 13 lines as `calibrated` with real names, you're finished.** To redo any one motor later, run its `--servo N --name …` command again.

### A6. Test it

```bash
dexkit-pose open
dexkit-pose fist
```

Each one shows a checklist. Type **`go`** and press Enter, and the hand moves, holds for 2 seconds, then goes limp.

If a finger moves **backwards** (opens when it should close), open `config/hand.yaml`, find that motor's line, change `inverted: false` to `inverted: true`, and save.

### A7. (Recommended) Measure the gantry's safe area

```bash
dexkit-calibrate-gantry
```

1. Use the keys to move. **`a`/`d`** = left/right (X), **`w`/`s`** = forward/back (Y), **`q`/`e`** = up/down (Z). Hold **Shift** for 10 mm steps instead of 1 mm.
2. Drive to your **home corner**: all the way to the low-X and low-Y end, with Z at the **top**. Press **`z`** to set zero there.
3. Drive to the **opposite corner**: high X, high Y, Z as low as is safe. Press **`m`** to mark it.
4. Press **`x`** to save.

After this, the gantry refuses to go outside that box. If you skip this step, a safe default box for a standard 3018 frame is used.

---

## Part B: every session

**Rule for every session: before you type `go`, every finger must be relaxed/open and the wrist in its neutral position.** The motors forget how many turns they've made whenever the power is off, so the software takes "wherever the fingers are right now" as *open* each time it connects. If you connect with a finger half-curled, that finger will never fully open and could over-tighten. Already connected and not sure? Type `home` at the `pose>` prompt: it switches the motors off, lets you pull the fingers open by hand, and re-bases when you press Enter.

```bash
cd ~/dexkit
source .venv/bin/activate
```

Turn the PSU on, plug in both USB cables, then pick one of the three ways below.

### Way 1 (easiest): type commands to the hand and gantry

```bash
dexkit-pose --gantry
```

Type **`go`** at the checklist. You now get a `pose>` prompt and can type any of these:

**Hand**

| Type this | What happens |
|---|---|
| `open` | open hand |
| `fist` | close hand |
| `pinch`, `point`, `thumbs_up`, `ok` | other poses |
| `list` | show every pose name |
| `f 3 0.5` | finger motor 3 to half-curled (0 = open, 1 = fully curled) |
| `roll 30` | twist the wrist to +30° (use `roll -30` for the other way) |
| `save myname` | save the hand's current shape as a new pose called `myname` |
| `state` | show where the fingers actually are, plus voltage |
| `home` | motors off → you pull every finger open → Enter: that becomes *open* again |
| `help` | show this list |
| `q` | relax the hand and quit |

**Gantry**

| Type this | What happens |
|---|---|
| `where` | show gantry position in mm |
| `jog x 10` | move X by +10 mm (`jog z -5` = down 5 mm) |
| `zero` | "this spot is 0,0,0" (do this at your home corner) |
| `goto 50 30 -10` | go to X=50, Y=30, Z=−10 mm |

**Each time you start, the gantry forgets its zero** (connecting resets its board). It will ask:

```
Restore saved gantry zero ...? [y/N]
```

- Answer **`y`** if nobody moved the gantry since last time.
- Otherwise answer **`n`**, `jog` to your home corner, and type `zero`.

Z goes **negative downwards**: `goto 50 30 0` is at the top and `goto 50 30 -20` is 20 mm lower.

Don't need the gantry? Just run `dexkit-pose` (without `--gantry`).

### Way 2: drive live with the keyboard

```bash
dexkit-teleop
```

| Key | Does |
|---|---|
| `1` … `9`, `0`, `-`, `=` | pick finger motor 1–12 |
| `]` / `[` | curl / uncurl the picked finger |
| `.` / `,` | twist wrist one way / the other |
| `w` `a` `s` `d` | gantry forward / left / back / right (Shift = bigger steps) |
| `q` / `e` | gantry up / down |
| `z` | set gantry zero here |
| `o` / `f` / `p` | open / fist / pinch |
| `r` | start/stop recording a motion (replay it with `dexkit-replay <name>`) |
| **SPACE** | **EMERGENCY STOP** |
| `Esc` | relax and quit |

If the gantry isn't plugged in, teleop just runs the hand.

### Way 3: run a saved routine

Write the steps in a file. Start by copying `examples/pick_and_show.yaml`:

```yaml
name: my_routine
steps:
  - {pose: open, duration: 1.0}
  - {gantry: {x: 50, y: 30, z: 0}, feed: 800}
  - {gantry: {z: -20}, feed: 300}
  - {pose: fist, duration: 1.5}
  - {gantry: {z: 0}, feed: 300}
  - {roll: 45, duration: 1.0}
  - {wait: 2.0}
  - {pose: open, duration: 1.0}
```

Check it without moving anything, then run it:

```bash
dexkit-run my_routine.yaml --dry-run
dexkit-run my_routine.yaml
```

The gantry must be zeroed first (Way 1 or Way 2), or answer `y` to "Restore saved gantry zero".

---

## Making your own poses

There are two ways:

- **Easiest:** in `dexkit-pose`, shape the hand with `f` and `roll` commands until it looks right, then type `save wave`. Now `wave` is a pose everywhere.
- **By editing a file:** open `config/poses.yaml` and add a line. Values go from 0 (open) to 1 (curled), and `default` covers every finger you don't name:
  ```yaml
  peace: {fingers: {default: 1.0, index: 0.0, middle: 0.0}, roll: 0}
  ```

---

## Emergency stop

| Situation | Do this |
|---|---|
| Inside `dexkit-teleop` | press **SPACE** |
| Anywhere else | press **Ctrl+C**. The hand goes limp and the gantry stops. |
| Need it from a second Terminal window | `dexkit-estop` |
| Absolute last resort | turn the PSU off and unplug the gantry's power |

After `dexkit-estop`, nothing will move again until you run `dexkit-estop --clear`. That's deliberate.

---

## When something goes wrong

| Message | Fix |
|---|---|
| `could not find the hand` / `could not find the gantry` | That USB cable isn't detected. Replug it, run `make ports`. |
| `no servos answered` + "the adapter is reachable … but no servo replied" | Jumper: **both caps vertical**, on the middle + bottom pins (see A1). If the log also says "adapter echoes", the caps are sideways. |
| `servo voltage outside 6.0-8.4 V` | PSU is off, unplugged, or set wrong. Re-do A1. |
| `servos not responding: [...]` | Those motor numbers aren't answering. Check the PSU is on, check the motor cables, and run `dexkit-scan`. |
| `placeholder calibration; run dexkit-calibrate-hand first` | Do step A5. |
| `gantry frame not valid` / `not zeroed` | Jog to your home corner and `zero` (or press `z` in teleop). |
| `outside travel box` | That position is outside the safe area. Pick a smaller number, or redo A7. |
| `e-stop flag is set` | Run `dexkit-estop --clear`. |
| A finger moves the wrong way | Flip `inverted:` for that motor in `config/hand.yaml`. |
| Finger gets "stuck" and backs off a bit | That's the overload guard. Its tendon is jammed or the target is too far. |
| Want more detail on any error | Add `-v` to the command, e.g. `dexkit-pose -v fist`. |

**Practice without hardware:** add `--mock` to any command, e.g. `dexkit-pose --mock --gantry`. Nothing real moves.

---

## Cheat sheet

```bash
cd ~/dexkit && source .venv/bin/activate   # every time
make ports                                  # are both USBs seen?
dexkit-scan                                 # are all 13 motors seen?
dexkit-calibrate-hand                       # once (finger ranges)
dexkit-calibrate-gantry                     # once (safe gantry area)
dexkit-pose --gantry                        # type poses + gantry moves
dexkit-teleop                               # keyboard live control
dexkit-run examples/pick_and_show.yaml      # run a routine
dexkit-estop                                # emergency stop (2nd window)
dexkit-estop --clear                        # allow motion again
```
