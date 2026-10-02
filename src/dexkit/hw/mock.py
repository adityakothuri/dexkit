"""Simulated hardware at the wire level.

MockFeetechSerial speaks the Feetech packet protocol and MockGrblSerial speaks
the GRBL line protocol, so MockHand / MockGantry are the *real* drivers running
on fake transports. Every byte written is recorded for tests.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from dexkit.config import BASE_REGISTERS, FAMILY_REGISTERS, GantryConfig, HandConfig
from dexkit.hw.feetech_hand import FeetechHand
from dexkit.hw.feetech_protocol import (
    BROADCAST_ID,
    INST_PING,
    INST_READ,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    INST_WRITE,
    checksum,
)
from dexkit.hw.grbl_gantry import GrblGantry

# Placeholder model numbers: the real HLS3620M / HLS3640M values are read on the bench.
MOCK_MODEL_FINGER = 3620
MOCK_MODEL_ROLL = 3640
HLS_TICKS_PER_S = 6000.0  # plan default: "1200 ticks in 0.2 s" (true 60 deg is 683 ticks)

REG = {**BASE_REGISTERS, **FAMILY_REGISTERS["hls"]}


# =========================================================================== Feetech


@dataclass
class MockServo:
    id: int
    model: int = MOCK_MODEL_FINGER
    position: float = 2048.0
    voltage: float = 7.4
    temperature: float = 32.0
    baud: int = 1_000_000
    speed_tps: float = HLS_TICKS_PER_S
    stall_tick: float | None = None   # load rises when pushed past this (toward stall_dir)
    stall_dir: int = 1
    load_override: int | None = None
    mem: bytearray = field(default_factory=lambda: bytearray(128))
    _t: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.mem[REG["model"]] = self.model & 0xFF
        self.mem[REG["model"] + 1] = (self.model >> 8) & 0xFF
        self.mem[REG["firmware_major"]] = 3
        self.mem[REG["firmware_minor"]] = 10
        self.mem[REG["id"]] = self.id
        self.mem[REG["lock"]] = 1
        self._set16(REG["goal_position"], int(self.position))

    def _get16(self, addr: int) -> int:
        return self.mem[addr] | (self.mem[addr + 1] << 8)

    def _set16(self, addr: int, v: int) -> None:
        self.mem[addr] = v & 0xFF
        self.mem[addr + 1] = (v >> 8) & 0xFF

    @property
    def torque_on(self) -> bool:
        return bool(self.mem[REG["torque_enable"]])

    @property
    def goal(self) -> int:
        raw = self._get16(REG["goal_position"])
        return -(raw & 0x7FFF) if raw & 0x8000 else raw

    def update(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        dt = max(0.0, now - self._t)
        self._t = now
        if self.torque_on:
            err = self.goal - self.position
            step = self.speed_tps * dt
            self.position += float(np.clip(err, -step, step))
        pos = int(round(self.position))
        self._set16(REG["present_position"], pos & 0x7FFF if pos >= 0 else ((-pos) | 0x8000))
        load = self.load_override
        if load is None:
            load = 15
            if self.torque_on and self.stall_tick is not None:
                past = (self.goal - self.stall_tick) * self.stall_dir
                if past > 0:
                    load = int(min(1000, 15 + 4 * past))
        self._set16(REG["present_load"], int(load) & 0x3FF)
        self.mem[REG["present_voltage"]] = int(round(self.voltage * 10)) & 0xFF
        self.mem[REG["present_temperature"]] = int(self.temperature) & 0xFF

    def write(self, addr: int, data: bytes) -> int | None:
        """Apply a write; returns a new ID if the ID register changed."""
        new_id = None
        for i, b in enumerate(data):
            a = addr + i
            if a == REG["id"]:
                if self.mem[REG["lock"]] == 0:
                    new_id = b
                continue
            self.mem[a] = b
        if new_id is not None:
            self.mem[REG["id"]] = new_id
        return new_id


class MockFeetechSerial:
    """A Feetech bus with simulated servos. Implements the Transport protocol."""

    def __init__(self, servos: list[MockServo], baudrate: int = 1_000_000, latency_s: float = 0.0,
                 echo: bool = False) -> None:
        self.servos: dict[int, MockServo] = {s.id: s for s in servos}
        self.echo = echo  # loop transmitted bytes back to RX, like some half-duplex adapters
        self.baudrate = baudrate
        self.latency_s = latency_s
        self._rx = bytearray()
        self.log: list[bytes] = []
        self.closed = False
        self._lock = threading.Lock()

    def _status(self, sid: int, params: bytes = b"", err: int = 0) -> bytes:
        body = [sid, len(params) + 2, err, *params]
        return b"\xff\xff" + bytes(body + [checksum(body)])

    def _responders(self, sid: int) -> list[MockServo]:
        return [s for s in self.servos.values() if s.id == sid and s.baud == self.baudrate]

    def write(self, data: bytes) -> int:
        with self._lock:
            self.log.append(bytes(data))
            if self.latency_s:
                time.sleep(self.latency_s)
            buf = bytes(data)
            if self.echo:
                self._rx += buf
            while len(buf) >= 6:
                if buf[:2] != b"\xff\xff":
                    buf = buf[1:]
                    continue
                length = buf[3]
                pkt, buf = buf[: length + 4], buf[length + 4 :]
                if len(pkt) < length + 4 or checksum(pkt[2:-1]) != pkt[-1]:
                    continue
                self._handle(pkt[2], pkt[4], pkt[5:-1])
            return len(data)

    def _handle(self, sid: int, inst: int, params: bytes) -> None:
        now = time.monotonic()
        for s in self.servos.values():
            s.update(now)
        if inst == INST_PING:
            for s in self._responders(sid):
                self._rx += self._status(s.id)
        elif inst == INST_READ:
            addr, n = params[0], params[1]
            for s in self._responders(sid):
                self._rx += self._status(s.id, bytes(s.mem[addr : addr + n]))
        elif inst == INST_WRITE:
            addr, data = params[0], params[1:]
            targets = [s for s in self.servos.values() if s.baud == self.baudrate
                       and (sid == BROADCAST_ID or s.id == sid)]
            for s in targets:
                old = s.id
                new_id = s.write(addr, data)
                if new_id is not None:
                    s.id = new_id
                    self.servos.pop(old, None)
                    self.servos[new_id] = s
                if sid != BROADCAST_ID:
                    self._rx += self._status(s.id)
        elif inst == INST_SYNC_WRITE:
            addr, n = params[0], params[1]
            rest = params[2:]
            for i in range(0, len(rest), n + 1):
                tid, data = rest[i], rest[i + 1 : i + 1 + n]
                for s in self._responders(tid):
                    s.write(addr, data)
        elif inst == INST_SYNC_READ:
            addr, n = params[0], params[1]
            for tid in params[2:]:
                for s in self._responders(tid):
                    self._rx += self._status(s.id, bytes(s.mem[addr : addr + n]))

    def read(self, size: int = 1) -> bytes:
        with self._lock:
            out = bytes(self._rx[:size])
            del self._rx[:size]
            return out

    def reset_input_buffer(self) -> None:
        with self._lock:
            self._rx.clear()

    def close(self) -> None:
        self.closed = True

    # test helpers
    def set_voltage(self, volts: float) -> None:
        for s in self.servos.values():
            s.voltage = volts

    def packets(self, instruction: int | None = None) -> list[bytes]:
        return [p for p in self.log if instruction is None or (len(p) > 4 and p[4] == instruction)]


def mock_servos_for(cfg: HandConfig, start_ticks: dict[int, int] | None = None) -> list[MockServo]:
    servos = []
    for s in cfg.servos:
        pos = (start_ticks or {}).get(s.id, s.slack)
        servos.append(MockServo(id=s.id, position=float(pos), stall_tick=float(s.effective_tight) + 40,
                                stall_dir=1 if s.span >= 0 else -1))
    servos.append(MockServo(id=cfg.roll.id, model=MOCK_MODEL_ROLL, position=float(cfg.roll.center)))
    return servos


class MockHand(FeetechHand):
    """The real FeetechHand driving simulated servos (2 ms bus latency by default)."""

    def __init__(self, cfg: HandConfig, latency_s: float = 0.002, bus: MockFeetechSerial | None = None) -> None:
        self.bus_sim = bus or MockFeetechSerial(mock_servos_for(cfg), baudrate=cfg.baud, latency_s=latency_s)
        super().__init__(cfg, transport=self.bus_sim)

    @property
    def calls(self) -> list[bytes]:
        return self.bus_sim.log


# =========================================================================== GRBL

DEFAULT_GRBL_SETTINGS: dict[int, float] = {
    0: 10, 1: 25, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 10: 1, 11: 0.010, 12: 0.002, 13: 0,
    20: 0, 21: 0, 22: 0, 23: 0, 24: 25, 25: 500, 26: 250, 27: 1, 30: 1000, 31: 0, 32: 0,
    100: 800, 101: 800, 102: 800, 110: 2000, 111: 2000, 112: 2000,
    120: 100, 121: 100, 122: 100, 130: 300, 131: 180, 132: 45,
}

AXIS_RE = re.compile(r"([XYZF])([-+]?\d*\.?\d+)")


class MockGrblSerial:
    """GRBL 1.1f simulator: banner, $$, ?, jogging, G90/G91/G92, hold/resume/reset."""

    def __init__(self, version: str = "1.1f", homing: bool = False, speed_factor: float = 1.0) -> None:
        self.version = version
        self.settings = dict(DEFAULT_GRBL_SETTINGS)
        if homing:
            self.settings[22] = 1
        self.speed_factor = speed_factor
        self.mpos = np.zeros(3)
        self.wco = np.zeros(3)
        self.queue: deque[tuple[np.ndarray, float, str]] = deque()
        self.hold = False
        self.alarm = homing  # GRBL boots into alarm when homing is enabled
        self._dtr = True
        self._line = bytearray()
        self._out: deque[bytes] = deque()
        self._t = time.monotonic()
        self._wco_dirty = True
        self._reports = 0
        self.log: list[bytes] = []
        self.lines: list[str] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ serial API

    @property
    def dtr(self) -> bool:
        return self._dtr

    @dtr.setter
    def dtr(self, value: bool) -> None:
        if value and not self._dtr:
            self._reset()
        self._dtr = value

    def write(self, data: bytes) -> int:
        with self._lock:
            self.log.append(bytes(data))
            for b in data:
                c = bytes([b])
                if c == b"?":
                    self._update()
                    self._out.append(self._status().encode() + b"\r\n")
                elif c == b"!":
                    self._update()
                    if self.queue:
                        self.hold = True
                elif c == b"~":
                    self._update()
                    self.hold = False
                elif c == b"\x18":
                    self._reset(during_motion=bool(self.queue))
                elif c == b"\x85":
                    self._update()
                    self.queue = deque(s for s in self.queue if s[2] != "jog")
                    if not self.queue:
                        self.hold = False
                elif c == b"\n":
                    line = self._line.decode(errors="replace").strip()
                    self._line.clear()
                    if line:
                        self.lines.append(line)
                        self._process(line)
                elif c != b"\r":
                    self._line += c
            return len(data)

    def readline(self) -> bytes:
        with self._lock:
            if self._out:
                return self._out.popleft()
        time.sleep(0.0005)
        return b""

    def reset_input_buffer(self) -> None:
        with self._lock:
            self._out.clear()

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ simulation

    def _reset(self, during_motion: bool = False) -> None:
        self._update()
        self.queue.clear()
        self.hold = False
        self.wco = np.zeros(3)  # matches the driver's assumption that zero is lost
        self._wco_dirty = True
        if during_motion or self.settings.get(22) == 1:
            self.alarm = True
        self._out.append(f"Grbl {self.version} ['$' for help]\r\n".encode())
        if self.alarm:
            self._out.append(b"[MSG:'$H'|'$X' to unlock]\r\n")

    def _state(self) -> str:
        if self.alarm:
            return "Alarm"
        if self.hold:
            return "Hold:0"
        if self.queue:
            return "Jog" if self.queue[0][2] == "jog" else "Run"
        return "Idle"

    def _update(self) -> None:
        now = time.monotonic()
        dt = (now - self._t) * self.speed_factor
        self._t = now
        while dt > 0 and self.queue and not self.hold:
            target, feed, _kind = self.queue[0]
            vec = target - self.mpos
            dist = float(np.linalg.norm(vec))
            step = feed / 60.0 * dt
            if dist <= step:
                self.mpos = target.copy()
                self.queue.popleft()
                dt -= dist / (feed / 60.0) if feed > 0 else dt
            else:
                self.mpos = self.mpos + vec / dist * step
                dt = 0

    def _status(self) -> str:
        mask = int(self.settings.get(10, 1))
        pos = self.mpos if mask & 1 else self.mpos - self.wco
        label = "MPos" if mask & 1 else "WPos"
        feed = self.queue[0][1] if self.queue and not self.hold else 0
        fields = [self._state(), f"{label}:{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}"]
        if mask & 2:
            fields.append(f"Bf:{max(0, 15 - len(self.queue))},128")
        fields.append(f"FS:{feed:.0f},0")
        self._reports += 1
        if self._wco_dirty or self._reports % 10 == 0:
            fields.append(f"WCO:{self.wco[0]:.3f},{self.wco[1]:.3f},{self.wco[2]:.3f}")
            self._wco_dirty = False
        return "<" + "|".join(fields) + ">"

    def _ok(self) -> None:
        self._out.append(b"ok\r\n")

    def _err(self, code: int) -> None:
        self._out.append(f"error:{code}\r\n".encode())

    def _planned_end(self) -> np.ndarray:
        return self.queue[-1][0].copy() if self.queue else self.mpos.copy()

    def _process(self, line: str) -> None:
        self._update()
        up = line.upper()
        if up == "$$":
            for k, v in sorted(self.settings.items()):
                val = int(v) if float(v).is_integer() else v
                self._out.append(f"${k}={val}\r\n".encode())
            return self._ok()
        if up == "$X":
            self.alarm = False
            self._out.append(b"[MSG:Caution: Unlocked]\r\n")
            return self._ok()
        if up == "$H":
            if self.settings.get(22) != 1:
                return self._err(5)
            self.mpos = np.zeros(3)
            self.alarm = False
            return self._ok()
        m = re.match(r"^\$(\d+)=([-+]?\d*\.?\d+)$", up)
        if m:
            self.settings[int(m[1])] = float(m[2])
            return self._ok()
        if self.alarm:
            return self._err(9)
        words = dict((a, float(v)) for a, v in AXIS_RE.findall(up))
        if up.startswith("$J="):
            if self.version.startswith("0."):
                return self._err(3)
            if "G91" not in up or "F" not in words:
                return self._err(3)
            target = self._planned_end()
            for i, a in enumerate("XYZ"):
                if a in words:
                    target[i] += words[a]
            self.queue.append((target, words["F"], "jog"))
            return self._ok()
        if "G92" in up:
            for i, a in enumerate("XYZ"):
                if a in words:
                    self.wco[i] = self.mpos[i] - words[a]
            self._wco_dirty = True
            return self._ok()
        if "G1" in up or "G0" in up:
            feed = words.get("F", 500.0 if "G1" in up else float(self.settings[110]))
            target = self._planned_end()
            for i, a in enumerate("XYZ"):
                if a in words:
                    target[i] = (target[i] + words[a]) if "G91" in up else (words[a] + self.wco[i])
            self.queue.append((target, feed, "move"))
            return self._ok()
        if re.fullmatch(r"(G9[01]|G2[01]|\s)+", up):
            return self._ok()
        return self._err(20)


class MockGantry(GrblGantry):
    """The real GrblGantry driving the GRBL simulator."""

    def __init__(self, cfg: GantryConfig, homing: bool = False, speed_factor: float = 1.0,
                 sim: MockGrblSerial | None = None) -> None:
        self.sim = sim or MockGrblSerial(homing=homing, speed_factor=speed_factor)
        super().__init__(cfg, transport=self.sim)

    @property
    def calls(self) -> list[bytes]:
        return self.sim.log

    def save_frame(self) -> None:
        """Never persist a simulated zero over the real gantry's saved frame."""


# =========================================================================== camera


class MockCamera:
    """Synthetic 240x320 RGB frames: a colored square whose position tracks `target`."""

    def __init__(self, height: int = 240, width: int = 320) -> None:
        self.h, self.w = height, width
        self.target = np.zeros(2)  # in [-1, 1]
        self._i = 0

    def set_target(self, xy: np.ndarray) -> None:
        self.target = np.clip(np.asarray(xy, dtype=float)[:2], -1, 1)

    def read(self) -> np.ndarray:
        self._i += 1
        img = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        cx = int((self.target[0] + 1) / 2 * (self.w - 40)) + 20
        cy = int((self.target[1] + 1) / 2 * (self.h - 40)) + 20
        color = np.array([220, 60 + (self._i * 7) % 120, 60], dtype=np.uint8)
        img[max(0, cy - 15) : cy + 15, max(0, cx - 15) : cx + 15] = color
        return img

    def close(self) -> None:
        pass
