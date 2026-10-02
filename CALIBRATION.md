# Calibrating the hand

Do this once. It teaches the software, for each of the 13 motors, where "relaxed" is and how far it may pull.

## Before you start

- Power supply on (7.4 V), adapter USB plugged in, arm clamped down.
- In a terminal:
  ```bash
  cd ~/dexkit
  source .venv/bin/activate
  dexkit-calibrate-hand --wiring
  ```
  This prints the 13 motors and whether each is done. Run it any time to see where you are.

## The idea

There are 12 hand motors (IDs 0–11). Each winds one tendon. A tendon does one of three things:

| what you see when it winds | call it |
|---|---|
| a finger **curls in** toward the palm | `<finger>_flex` |
| a finger **bends back** | `<finger>_extend` |
| the thumb or index moves **sideways** | `thumb_adduct` or `index_adduct` |

Finger names: `thumb`, `index`, `middle`, `ring`, `pinky`. So the 12 names are:
`thumb_flex index_flex middle_flex ring_flex pinky_flex thumb_extend index_extend middle_extend ring_extend pinky_extend thumb_adduct index_adduct`

Motor 12 is the wrist.

This hand's wiring, as confirmed on the bench (the lab's own names on the left):

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

## For each hand motor (ID 0 to 11)

### Step 1: see what it moves
```bash
dexkit-calibrate-hand --servo 3
```
Press **Enter**. Press **Enter** again. The motor winds. Watch the hand and decide: which finger, and does it curl in, bend back, or go sideways?

Press **`x`**. The motor stops. Nothing is saved.

### Step 2: run it again with its name
```bash
dexkit-calibrate-hand --servo 3 --name index_flex
```
(Use whatever name matches what you saw.)

### Step 3: first prompt
It says *put this finger in its RELAXED position*. Leave the finger sitting naturally, not pulled in or back. Press **Enter**.

### Step 4: second prompt
It says *press Enter, then t at the fully PULLED pose*. Press **Enter**. The motor winds. When the finger is as far as you ever want it to go (firmly curled / bent back / across, but not straining), press **`t`**.

Saved. It prints the table. Next motor.

Keys while it winds (no Enter needed):

| key | does |
|---|---|
| `t` | save here |
| `x` | stop, save nothing |
| `r` | reverse direction |
| `u` | back up a bit |
| `f` / `s` | faster / slower |
| **SPACE** | emergency stop: all torque off, exits |

## The wrist (ID 12)
```bash
dexkit-calibrate-hand --servo 12
```
Turn the forearm to its neutral position by hand. Press **Enter**. Done.

## Already calibrated a motor but didn't give it a name?
Do Step 1 to see what it moves, then:
```bash
dexkit-calibrate-hand --servo 3 --label-only --name index_flex
```
No movement, just the name.

## Finished?
```bash
dexkit-calibrate-hand --wiring
```
All 13 lines say `calibrated`, and the 12 hand motors have real names (no `ch1`, `ch2`…). Then the hand is ready:
```bash
dexkit-pose
```
(let the fingers relax before you type `go`, then try `fist`, then `open`).

## If something goes wrong

| problem | do |
|---|---|
| the motor winds but nothing on the hand moves for a while | normal: it's taking up slack. It gives up on its own after 3 turns. |
| the finger gets looser instead of pulling | press `r` to reverse |
| it stopped by itself ("moved … ticks without 't'") | nothing saved; the tendon may be slack or off its spool. Check it, run the motor again |
| I pressed `t` in the wrong place | run the same `--servo N --name …` command again; it overwrites |
| I named it wrong | `--servo N --label-only --name <right name>` |
| "could not find the hand" | plug the adapter USB in |
| "no servos answered" | power supply off, or jumper caps wrong (see HOW_TO_USE.md A1) |
