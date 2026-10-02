"""Keyboard teleop for fingers, forearm roll and gantry.

Keys
  1-9 0 - =   select finger servo 1..12        [ / ]  selected finger -/+ 0.05
  , / .       roll -/+ 5 deg
  w a s d     gantry Y+ / X- / Y- / X+ jog     q / e  Z up / down
  (Shift = 10 mm steps; key release sends jog cancel where the key backend reports releases)
  z  set gantry zero      h  home (if enabled)
  o  open   f  fist   p  pinch
  r  start/stop recording
  Space  EMERGENCY STOP   Esc  relax and quit
"""

from __future__ import annotations

import logging
import queue
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from dexkit.config import GantryConfig, HandConfig
from dexkit.control.poses import Pose, PoseLibrary, pose_trajectory
from dexkit.control.recorder import Recorder
from dexkit.hw.base import N_FINGERS, GantryInterface, HandInterface, join_action
from dexkit.hw.safety import EStop, EStopTripped, SafetyTrip
from dexkit.util import Rate

log = logging.getLogger(__name__)

FINGER_KEYS = {k: i for i, k in enumerate("1234567890-=")}
JOG_KEYS = {"w": (0, 1, 0), "a": (-1, 0, 0), "s": (0, -1, 0), "d": (1, 0, 0), "q": (0, 0, 1), "e": (0, 0, -1)}
POSE_KEYS = {"o": "open", "f": "fist", "p": "pinch"}
FINGER_STEP = 0.05
ROLL_STEP = 5.0


@dataclass
class KeyEvent:
    key: str            # single character, "space", "esc"; uppercase letter = Shift held
    pressed: bool = True


# --------------------------------------------------------------------------- key sources


class KeySource:
    def poll(self) -> list[KeyEvent]:
        return []

    def close(self) -> None:
        pass


class ScriptedKeys(KeySource):
    """Replays (seconds_from_start, key) pairs; used by --mock runs and tests."""

    def __init__(self, events: Iterable[tuple[float, str]], clock: Callable[[], float] = time.monotonic) -> None:
        self.events = sorted(events)
        self.clock = clock
        self.t0 = clock()

    @classmethod
    def parse(cls, spec: str) -> ScriptedKeys:
        events = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            t, _, key = item.partition(":")
            events.append((float(t), key))
        return cls(events)

    def poll(self) -> list[KeyEvent]:
        now = self.clock() - self.t0
        out = []
        while self.events and self.events[0][0] <= now:
            _, k = self.events.pop(0)
            out.append(KeyEvent(k, True))
        return out


class TerminalKeys(KeySource):
    """blessed-based; works over SSH and in a VM terminal. Reports presses only."""

    def __init__(self) -> None:
        from blessed import Terminal

        self.term = Terminal()
        self._ctx = self.term.cbreak()
        self._ctx.__enter__()

    def poll(self) -> list[KeyEvent]:
        out = []
        while True:
            k = self.term.inkey(timeout=0)
            if not k:
                break
            if k.name == "KEY_ESCAPE":
                out.append(KeyEvent("esc"))
            elif str(k) == " ":
                out.append(KeyEvent("space"))
            elif not k.is_sequence:
                out.append(KeyEvent(str(k)))
        return out

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)


class PynputKeys(KeySource):
    """Global keyboard hook with press and release (needs X11 on Linux, Accessibility on macOS)."""

    def __init__(self) -> None:
        from pynput import keyboard

        self._q: queue.Queue[KeyEvent] = queue.Queue()
        self._kb = keyboard

        def name(k: object) -> str | None:
            if k == keyboard.Key.space:
                return "space"
            if k == keyboard.Key.esc:
                return "esc"
            ch = getattr(k, "char", None)
            return ch if ch else None

        def on_press(k: object) -> None:
            n = name(k)
            if n:
                self._q.put(KeyEvent(n, True))

        def on_release(k: object) -> None:
            n = name(k)
            if n:
                self._q.put(KeyEvent(n, False))

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

    def poll(self) -> list[KeyEvent]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self.listener.stop()


def make_keys(kind: str, script: str | None) -> KeySource:
    if script:
        return ScriptedKeys.parse(script)
    if kind == "none":
        return KeySource()
    if kind == "pynput":
        return PynputKeys()
    if kind in ("terminal", "auto"):
        if sys.stdin.isatty():
            return TerminalKeys()
        if kind == "terminal":
            raise RuntimeError("terminal key backend needs a TTY")
        log.warning("no TTY: keyboard input disabled")
        return KeySource()
    raise ValueError(f"unknown key backend {kind}")


# --------------------------------------------------------------------------- controller


class TeleopController:
    def __init__(
        self,
        hand: HandInterface,
        gantry: GantryInterface | None,
        hand_cfg: HandConfig,
        gantry_cfg: GantryConfig | None,
        poses: PoseLibrary,
        estop: EStop,
        recorder: Recorder | None = None,
        on_record_stop: Callable[[Recorder], None] | None = None,
    ) -> None:
        self.hand = hand
        self.gantry = gantry
        self.hand_cfg = hand_cfg
        self.gantry_cfg = gantry_cfg
        self.poses = poses
        self.estop = estop
        self.recorder = recorder or Recorder()
        self.on_record_stop = on_record_stop
        f, r = hand.last_command
        self.fingers = np.clip(f, 0, 1)
        self.roll = r
        self.selected = 0
        self.quit = False
        self.message = ""
        self._traj: list[Pose] = []
        self.gantry_xyz = np.zeros(3)
        self.gantry_status = "-"
        self.last_saved: str | None = None

    def handle(self, ev: KeyEvent) -> None:
        k = ev.key
        if not ev.pressed:
            if k.lower() in JOG_KEYS and self.gantry is not None:
                self.gantry.jog_cancel()
            return
        if k == "space":
            self.estop.trip("operator")
            self.message = "EMERGENCY STOP - press Esc to quit"
            return
        if k == "esc":
            self.quit = True
            return
        if self.estop.tripped:
            return
        try:
            self._handle_press(k)
        except SafetyTrip:
            raise
        except Exception as e:  # noqa: BLE001 - keep teleop alive on a rejected command
            self.message = f"{type(e).__name__}: {e}"
            log.warning(self.message)

    def _handle_press(self, k: str) -> None:
        if k in FINGER_KEYS:
            self.selected = FINGER_KEYS[k]
            self.message = f"selected {self.hand_cfg.servos[self.selected].name}"
        elif k in "[]":
            self._traj = []
            d = FINGER_STEP if k == "]" else -FINGER_STEP
            self.fingers[self.selected] = float(np.clip(self.fingers[self.selected] + d, 0, 1))
        elif k in ",.":
            self._traj = []
            d = ROLL_STEP if k == "." else -ROLL_STEP
            lim = self.hand_cfg.roll.range_deg
            self.roll = float(np.clip(self.roll + d, -lim, lim))
        elif k.lower() in JOG_KEYS:
            if self.gantry is None or self.gantry_cfg is None:
                self.message = "no gantry"
                return
            step = self.gantry_cfg.jog_step_big_mm if k.isupper() else self.gantry_cfg.jog_step_mm
            dx, dy, dz = (v * step for v in JOG_KEYS[k.lower()])
            if not self.gantry.jog(dx, dy, dz):
                self.message = "jog blocked by travel box"
        elif k == "z":
            if self.gantry is not None:
                self.gantry.set_zero()
                self.message = "gantry zero set"
        elif k == "h":
            if self.gantry is not None:
                self.gantry.home()
                self.message = "gantry homed"
        elif k in POSE_KEYS:
            target = self.poses[POSE_KEYS[k]]
            self._traj = pose_trajectory(Pose(self.fingers, self.roll), target, 1.0, self.hand_cfg.rate_hz)
            self.message = f"pose {POSE_KEYS[k]}"
        elif k == "r":
            if self.recorder.active:
                if self.on_record_stop:
                    self.on_record_stop(self.recorder)
                else:
                    path = self.recorder.stop_and_save()
                    self.last_saved = str(path) if path else None
                    self.message = f"saved {path}" if path else "empty recording discarded"
            else:
                self.recorder.start()
                self.message = "recording..."

    def tick(self, poll_gantry: bool) -> np.ndarray:
        """Advance pose animation, command the hand, return the 16-dim action sent."""
        self.estop.check()
        if self._traj:
            wp = self._traj.pop(0)
            self.fingers, self.roll = wp.fingers.copy(), wp.roll
        sent_f, sent_r = self.hand.set_targets(self.fingers, self.roll)
        if poll_gantry and self.gantry is not None:
            gs = self.gantry.get_state()
            self.gantry_xyz = gs.xyz
            self.gantry_status = gs.state + ("" if gs.frame_valid else " (not zeroed)")
        return join_action(sent_f, sent_r, self.gantry_xyz)


def status_text(c: TeleopController, hs: object, loop_ms: float) -> str:
    names = c.hand_cfg.names
    cells = []
    for i in range(N_FINGERS):
        mark = ">" if i == c.selected else " "
        cells.append(f"{mark}{i + 1:>2} {names[i][:11]:<11} {c.fingers[i]:.2f}")
    rows = ["   ".join(cells[j : j + 4]) for j in range(0, N_FINGERS, 4)]
    v = getattr(hs, "min_voltage", None)
    rec = "REC" if c.recorder.active else "   "
    lines = rows + [
        f"roll {c.roll:+6.1f} deg | gantry {np.round(c.gantry_xyz, 2).tolist()} [{c.gantry_status}] | "
        f"V {v if v is None else round(v, 1)} | loop {loop_ms:5.1f} ms | {rec}",
        "ESTOP TRIPPED" if c.estop.tripped else c.message,
    ]
    return "\n".join(lines)


def run_teleop(
    session: object,
    keys: KeySource,
    duration: float | None = None,
    per_tick: Callable[[np.ndarray, np.ndarray], None] | None = None,
    on_record_stop: Callable[[Recorder], None] | None = None,
    live: bool = True,
    recorder: Recorder | None = None,
) -> TeleopController:
    from dexkit.cli import Session

    assert isinstance(session, Session) and session.hand is not None and session.hand_cfg is not None
    s = session
    poses = PoseLibrary.load(s.hand_cfg)
    c = TeleopController(s.hand, s.gantry, s.hand_cfg, s.gantry_cfg, poses, s.estop,
                         recorder=recorder, on_record_stop=on_record_stop)
    rate_hz = s.hand_cfg.rate_hz
    rate = Rate(rate_hz)
    gantry_every = max(1, int(round(rate_hz / (s.gantry_cfg.status_hz if s.gantry_cfg else 10))))
    t_end = time.monotonic() + duration if duration else None
    tick = 0
    loop_ms = 0.0
    worst_ms = 0.0

    display = None
    if live and sys.stdout.isatty():
        from rich.live import Live

        display = Live("", refresh_per_second=10, transient=False)
        display.start()
    last_print = 0.0
    try:
        while not c.quit:
            if t_end and time.monotonic() >= t_end:
                break
            for ev in keys.poll():
                c.handle(ev)
            t0 = time.perf_counter()
            if s.estop.tripped:
                hs = None
            else:
                try:
                    action = c.tick(poll_gantry=(tick % gantry_every == 0))
                    hs = s.hand.get_state()
                    state = join_action(np.clip(hs.fingers, 0, 1), hs.roll_deg, c.gantry_xyz)
                    c.recorder.add(action, state)
                    if per_tick:
                        per_tick(action, state)
                except EStopTripped:
                    hs = None
            loop_ms = (time.perf_counter() - t0) * 1000
            worst_ms = max(worst_ms, loop_ms)
            text = status_text(c, hs, loop_ms)
            if display is not None:
                display.update(text)
            elif time.monotonic() - last_print > 1.0:
                last_print = time.monotonic()
                log.info("teleop | %s", text.splitlines()[-2])
            tick += 1
            rate.sleep()
    finally:
        if display is not None:
            display.stop()
        if c.recorder.active:
            if on_record_stop:
                on_record_stop(c.recorder)
            else:
                c.recorder.stop_and_save()
        keys.close()
    log.info("teleop finished: %d ticks, worst hand read+write %.1f ms", tick, worst_ms)
    c.worst_loop_ms = worst_ms  # type: ignore[attr-defined]
    return c


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import add_gantry_flags, common_parser, init, open_session

    p = common_parser("Keyboard teleop for the DexKit hand and gantry.\n" + (__doc__ or ""))
    add_gantry_flags(p)
    p.add_argument("--keys", choices=["auto", "terminal", "pynput", "none"], default="auto")
    p.add_argument("--script", help="scripted keys 't:key,...' e.g. '0.5:3,1:],2:w,3:f,9:esc'")
    p.add_argument("--duration", type=float, default=None, help="quit after N seconds")
    args = p.parse_args(argv)
    init(args)
    with open_session(args, need_hand=True, want_gantry=True) as s:
        print(__doc__)
        keys = make_keys(args.keys, args.script)  # after the 'go' prompt: cbreak mode eats line input
        c = run_teleop(s, keys, duration=args.duration)
        if s.estop.tripped:
            print(f"e-stop was tripped: {s.estop.reason}")
        if c.last_saved:
            print(f"last recording: {c.last_saved}")


if __name__ == "__main__":
    main()
