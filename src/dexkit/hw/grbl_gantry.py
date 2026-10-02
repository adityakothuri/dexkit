"""GRBL gantry driver (3018-style frame on a CNC Pro V5 board).

Millimetres in the zeroed frame are converted to G-code only here. Motion is
refused until the frame is valid (manual G92 zero, restored zero, or $H home).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from dexkit.config import GantryConfig, data_dir
from dexkit.hw.base import GantryInterface, GantryState
from dexkit.hw.grbl_protocol import (
    RT_FEED_HOLD,
    RT_JOG_CANCEL,
    RT_RESUME,
    RT_SOFT_RESET,
    RT_STATUS,
    STATE_WORDS,
    StatusReport,
    jog_command,
    machine_position,
    move_command,
    parse_line,
    work_position,
)
from dexkit.hw.safety import SafetyTrip, TravelBox

log = logging.getLogger(__name__)

LOGGED_SETTINGS = (10, 20, 21, 22, 110, 111, 112, 120, 121, 122, 130, 131, 132)


class GrblError(IOError):
    pass


class LineTransport(Protocol):
    dtr: bool

    def write(self, data: bytes) -> int | None: ...
    def readline(self) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


def open_grbl_serial(port: str, baud: int) -> LineTransport:
    import serial  # only hw/ imports pyserial

    from dexkit.hw.ports import resolve_port

    s = serial.Serial()
    s.port = resolve_port(port, "gantry")
    s.baudrate = baud
    s.timeout = 0.05
    s.write_timeout = 0.5
    s.open()
    return s


def frame_state_path() -> Path:
    return data_dir() / "state" / "gantry_frame.json"


class GrblGantry(GantryInterface):
    def __init__(self, cfg: GantryConfig, transport: LineTransport | None = None) -> None:
        self.cfg = cfg
        self._transport = transport
        self.port_name = cfg.port
        self.t: LineTransport | None = None
        self.box = TravelBox(cfg.travel_min, cfg.travel_max)
        self.version: str | None = None
        self.legacy = False
        self.settings: dict[int, float] = {}
        self.alarm: int | None = None
        self._frame_valid = False
        self._wco: tuple[float, float, float] | None = None
        self._last_status: StatusReport | None = None
        self._last_state: GantryState | None = None
        self._planned: np.ndarray | None = None  # end of queued jog motion, frame coords
        self.connected = False

    # ---------------------------------------------------------------- I/O

    def _write(self, data: bytes) -> None:
        if self.t is None:
            raise RuntimeError("gantry not connected")
        log.debug("GRBL TX %r", data)
        self.t.write(data)

    def _readline(self) -> str | None:
        assert self.t is not None
        raw = self.t.readline()
        if not raw:
            return None
        line = raw.decode("ascii", errors="replace").strip()
        if line:
            log.debug("GRBL RX %s", line)
        return line

    def _handle_async(self, line: str) -> None:
        r = parse_line(line)
        if r.kind == "status" and r.status is not None:
            self._last_status = r.status
            if r.status.wco is not None:
                self._wco = r.status.wco
        elif r.kind == "alarm":
            self.alarm = r.code
            log.error("GRBL %s", line)
        elif r.kind == "message":
            log.info("GRBL message: %s", r.text)
        elif r.kind == "banner":
            log.warning("GRBL reset detected (%s); frame invalidated", line)
            self._frame_valid = False

    def _command(self, line: str, timeout: float | None = None) -> list[str]:
        """Send one line, wait for ok/error. Returns the other lines received meanwhile."""
        self._write(line.encode("ascii") + b"\n")
        deadline = time.monotonic() + (timeout or self.cfg.command_timeout_s)
        lines: list[str] = []
        while time.monotonic() < deadline:
            got = self._readline()
            if got is None or got == "":
                continue
            r = parse_line(got)
            if r.kind == "ok":
                return lines
            if r.kind == "error":
                raise GrblError(f"'{line}' -> {got}")
            self._handle_async(got)
            if r.kind == "alarm":
                raise GrblError(f"'{line}' -> {got}")
            lines.append(got)
        raise TimeoutError(f"no 'ok' for '{line}' within timeout")

    def _wait_banner(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            got = self._readline()
            if not got:
                continue
            r = parse_line(got)
            if r.kind == "banner":
                return r.text
            self._handle_async(got)
        return None

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        self.t = self._transport or open_grbl_serial(self.cfg.port, self.cfg.baud)
        self.port_name = getattr(self.t, "port", self.cfg.port)
        if self.cfg.reset_on_connect:
            self.t.dtr = False
            time.sleep(0.1)
            self.t.dtr = True
        version = self._wait_banner(self.cfg.banner_timeout_s)
        if version is None:
            log.info("no banner after open; sending soft reset")
            self._write(RT_SOFT_RESET)
            version = self._wait_banner(self.cfg.banner_timeout_s)
        if version is None:
            raise GrblError("no GRBL banner (wrong port or baud?)")
        self._on_banner(version)
        self.connected = True

    def _on_banner(self, version: str) -> None:
        self.version = version
        self.legacy = version.startswith("0.")
        self._frame_valid = False
        log.info("GRBL %s connected", version)
        if self.legacy:
            log.warning("GRBL %s has no $J jogging; falling back to G91 G0 moves", version)
        time.sleep(0.05)
        self.t.reset_input_buffer()  # type: ignore[union-attr]
        self.read_settings()
        mask = int(self.cfg.status_report_mask)
        if int(self.settings.get(10, -1)) != mask:
            self._command(f"$10={mask}")
            self.settings[10] = float(mask)
            log.info("set $10=%d", mask)
        st = self.get_state()
        if st.state == "Alarm" or self.alarm is not None:
            self._command("$X")
            self.alarm = None
            log.warning("GRBL alarm cleared with $X (position not homed)")
        if self.cfg.zeroing == "home" and int(self.settings.get(22, 0)) != 1:
            log.error("zeroing: home requested but $22=%s (homing disabled); use manual zeroing",
                      self.settings.get(22))

    def read_settings(self) -> dict[int, float]:
        lines = self._command("$$")
        settings = {}
        for line in lines:
            r = parse_line(line)
            if r.kind == "setting" and r.setting:
                settings[r.setting[0]] = r.setting[1]
        self.settings = settings
        summary = ", ".join(f"${k}={settings[k]:g}" for k in LOGGED_SETTINGS if k in settings)
        log.info("GRBL settings: %s", summary)
        return settings

    def close(self) -> None:
        if self.t is None:
            return
        try:
            if self._frame_valid and self._last_state is not None:
                self.save_frame()
            self._write(RT_FEED_HOLD)
            self._write(RT_JOG_CANCEL)
        except Exception as e:  # noqa: BLE001
            log.debug("close: %s", e)
        try:
            self.t.close()
        finally:
            self.t = None
            self.connected = False

    # ---------------------------------------------------------------- frame

    @property
    def frame_valid(self) -> bool:
        return self._frame_valid

    def set_zero(self) -> None:
        st = self.wait_idle(timeout=10.0)
        self._command("G92 X0 Y0 Z0")
        self._wco = tuple(float(v) for v in st.mpos)  # type: ignore[assignment]
        self._planned = None
        self._frame_valid = True
        log.info("gantry zero set at MPos %s", np.round(st.mpos, 3).tolist())
        self.get_state()

    def restore_frame(self, xyz: tuple[float, float, float]) -> None:
        """Declare the current physical position to be `xyz` in the zeroed frame."""
        st = self.wait_idle(timeout=10.0)
        x, y, z = xyz
        self._command(f"G92 X{x:.3f} Y{y:.3f} Z{z:.3f}")
        self._wco = tuple(float(m - w) for m, w in zip(st.mpos, xyz, strict=True))  # type: ignore[assignment]
        self._planned = None
        self._frame_valid = True
        log.info("gantry frame restored: current position = %s", [round(v, 3) for v in xyz])

    def save_frame(self) -> None:
        st = self._last_state
        if st is None or not self._frame_valid:
            return
        p = frame_state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"xyz": [float(v) for v in st.xyz], "saved_at": time.time()}))

    @staticmethod
    def load_saved_frame() -> dict | None:
        p = frame_state_path()
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def home(self) -> None:
        if int(self.settings.get(22, 0)) != 1:
            raise GrblError("homing is disabled ($22=0); use manual zeroing")
        self._command("$H", timeout=120.0)
        st = self.wait_idle(timeout=10.0)
        self._wco = self._wco or (0.0, 0.0, 0.0)
        self._planned = None
        self._frame_valid = True
        log.info("gantry homed at MPos %s", st.mpos.tolist())

    # ---------------------------------------------------------------- motion

    def _clamp_feed(self, feed: float | None, default: float) -> float:
        f = default if feed is None else float(feed)
        return float(np.clip(f, 1.0, self.cfg.feed_max_mm_min))

    def jog(self, dx: float, dy: float, dz: float, feed: float | None = None) -> bool:
        """Relative jog. Clamped against where queued motion will END (not the stale current
        position), and skipped while more than jog_max_backlog_mm is still queued, so a held
        key with auto-repeat cannot pile up motion that keeps going after release."""
        delta = np.array([dx, dy, dz], dtype=float)
        st = self.get_state()
        moving = st.state in ("Jog", "Run")
        base = self._planned if (moving and self._planned is not None) else st.xyz.copy()
        if moving and np.linalg.norm(base - st.xyz) > self.cfg.jog_max_backlog_mm:
            return False
        if self._frame_valid:
            delta = self.box.clamp_jog(base, delta)
            if not np.any(delta):
                log.info("jog blocked by travel box")
                return False
        self._planned = base + delta
        f = self._clamp_feed(feed, self.cfg.jog_feed_mm_min)
        if self.legacy:
            parts = " ".join(f"{a}{d:.3f}" for a, d in zip("XYZ", delta, strict=True) if d)
            self._command(f"G91 G0 {parts}")
            self._command("G90")
        else:
            self._command(jog_command(*delta, f))
        return True

    def jog_cancel(self) -> None:
        self._write(RT_JOG_CANCEL)
        self._planned = None

    def move_to(self, x: float, y: float, z: float, feed: float | None = None, wait: bool = False) -> bool:
        if not self._frame_valid:
            raise SafetyTrip("gantry frame not valid: zero (manual) or home before moving")
        self.box.check(x, y, z)
        f = self._clamp_feed(feed, self.cfg.feed_default_mm_min)
        if not wait and self._last_status is not None and self._last_status.planner_free is not None:
            if self._last_status.planner_free < self.cfg.planner_min_free:
                log.debug("planner nearly full (%d free); skipping send", self._last_status.planner_free)
                return False
        self._command(move_command(x, y, z, f))
        if wait:
            self.wait_idle()
        return True

    def wait_idle(self, timeout: float = 30.0) -> GantryState:
        deadline = time.monotonic() + timeout
        period = 1.0 / self.cfg.status_hz
        while True:
            st = self.get_state()
            if st.state == "Idle":
                return st
            if st.state == "Alarm":
                raise GrblError(f"gantry in alarm (code {self.alarm})")
            if time.monotonic() > deadline:
                raise TimeoutError(f"gantry not idle after {timeout:.1f}s (state {st.state})")
            time.sleep(period)

    def get_state(self) -> GantryState:
        self._write(RT_STATUS)
        deadline = time.monotonic() + 0.5
        status: StatusReport | None = None
        while time.monotonic() < deadline:
            got = self._readline()
            if not got:
                continue
            r = parse_line(got)
            self._handle_async(got)
            if r.kind == "status":
                status = r.status
                break
        now = time.monotonic()
        if status is None:
            prev = self._last_state
            return GantryState(
                state="Unknown",
                xyz=prev.xyz if prev else np.zeros(3),
                mpos=prev.mpos if prev else np.zeros(3),
                frame_valid=self._frame_valid,
                timestamp=now,
            )
        if not status.known_state:
            log.warning("unknown GRBL state '%s'; treating as not idle", status.state)
        mpos = machine_position(status, self._wco)
        wpos = work_position(status, self._wco)
        mpos_a = np.array(mpos if mpos is not None else (wpos or (0.0, 0.0, 0.0)), dtype=float)
        wpos_a = np.array(wpos if wpos is not None else mpos_a, dtype=float)
        st = GantryState(
            state=status.state if status.state in STATE_WORDS else "Unknown",
            substate=status.substate,
            xyz=wpos_a,
            mpos=mpos_a,
            feed=status.feed or 0.0,
            planner_free=status.planner_free,
            frame_valid=self._frame_valid,
            timestamp=now,
        )
        self._last_state = st
        return st

    def feed_hold(self) -> None:
        self._write(RT_FEED_HOLD)

    def resume(self) -> None:
        self._write(RT_RESUME)

    def soft_reset(self, wait: bool = True) -> None:
        self._write(RT_SOFT_RESET)
        self._frame_valid = False
        self._planned = None
        if wait:
            version = self._wait_banner(self.cfg.banner_timeout_s)
            if version is None:
                raise GrblError("no banner after soft reset")
            self._on_banner(version)
