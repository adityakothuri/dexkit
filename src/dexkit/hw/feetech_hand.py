"""Feetech HLS hand driver: register-level access plus the normalized HandInterface.

Normalized finger values ([0,1], 0 = slack) and roll degrees are converted to
ticks only here, using hand.yaml.

The servos run in multi-turn mode and LOSE the turn count at power-off (bench:
22191 read back as 1711 = 22191 mod 4096). So the absolute slack/tight numbers in
hand.yaml cannot be trusted across sessions; only their difference (the span) is.
On connect the hand is re-based: each finger's current position becomes "open" (0),
which is why the operator must leave the fingers relaxed/open before typing `go`.
The roll servo stays within one turn, so its center is snapped to the equivalent
angle nearest the current position.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np

from dexkit.config import TICKS_PER_REV, HandConfig, RollConfig, data_dir
from dexkit.hw.base import N_FINGERS, HandInterface, HandState
from dexkit.hw.feetech_protocol import (
    BROADCAST_ID,
    FeetechBus,
    Transport,
    decode_load,
    decode_position,
    encode_position,
    from_le16,
    le16,
)
from dexkit.hw.safety import AntagonistLimit, LoadWatch, SafetyTrip, TempWatch, TickClamp, VoltageGate

log = logging.getLogger(__name__)

# One block read covers present position(2) speed(2) load(2) voltage(1) temperature(1).
STATE_BLOCK_LEN = 8

# A finger may move this much (ticks) between sessions and still be recognised after a
# power cycle shifts the count by a multiple of 4096. Beyond it, we cannot tell turns apart.
RESTORE_DRIFT_TICKS = 1024


def positions_state_path() -> Path:
    return data_dir() / "state" / "hand_positions.json"


def load_positions_state() -> dict:
    p = positions_state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def save_positions_state(entries: dict[int, dict[str, int]], roll_center: int | None = None) -> None:
    """Merge {servo id: {"pos": last reading, "slack": absolute open position}} into the state file."""
    state = load_positions_state()
    servos = state.setdefault("servos", {})
    for sid, e in entries.items():
        servos[str(sid)] = {"pos": int(e["pos"]), "slack": int(e["slack"])}
    if roll_center is not None:
        state["roll_center"] = int(roll_center)
    state["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p = positions_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1))


def open_serial(port: str, baud: int, timeout_s: float) -> Transport:
    import serial  # only hw/ imports pyserial

    from dexkit.hw.ports import resolve_port

    return serial.Serial(resolve_port(port, "hand"), baudrate=baud, timeout=timeout_s, write_timeout=0.1)


class FeetechDriver:
    """Named-register access to Feetech servos. The rest of the code never sees addresses."""

    def __init__(self, bus: FeetechBus, registers: dict[str, int]) -> None:
        self.bus = bus
        self.reg = registers

    def ping(self, sid: int) -> int | None:
        """Model number if the servo answers, else None."""
        if not self.bus.ping(sid):
            return None
        data = self.bus.read(sid, self.reg["model"], 2)
        return from_le16(data[0], data[1]) if data else 0

    def read_firmware(self, sid: int) -> str | None:
        data = self.bus.read(sid, self.reg["firmware_major"], 2)
        return f"{data[0]}.{data[1]}" if data else None

    def _read_u8(self, sid: int, name: str) -> int | None:
        data = self.bus.read(sid, self.reg[name], 1)
        return data[0] if data else None

    def _read_u16(self, sid: int, name: str) -> int | None:
        data = self.bus.read(sid, self.reg[name], 2)
        return from_le16(data[0], data[1]) if data else None

    def read_position(self, sid: int) -> int | None:
        data = self.bus.read(sid, self.reg["present_position"], 2)
        return decode_position(data[0], data[1]) if data else None

    def read_positions(self, ids: list[int]) -> dict[int, int]:
        block = self.bus.read_block(self.reg["present_position"], 2, ids)
        return {sid: decode_position(d[0], d[1]) for sid, d in block.items()}

    def read_voltage(self, sid: int) -> float | None:
        v = self._read_u8(sid, "present_voltage")
        return v / 10.0 if v is not None else None

    def read_load(self, sid: int) -> int | None:
        data = self.bus.read(sid, self.reg["present_load"], 2)
        return decode_load(data[0], data[1]) if data else None

    def read_temperature(self, sid: int) -> int | None:
        return self._read_u8(sid, "present_temperature")

    def read_mode(self, sid: int) -> int | None:
        return self._read_u8(sid, "mode")

    def read_state_block(self, ids: list[int]) -> dict[int, dict[str, float]]:
        base = self.reg["present_position"]
        if (
            self.reg["present_speed"] != base + 2
            or self.reg["present_load"] != base + 4
            or self.reg["present_voltage"] != base + 6
            or self.reg["present_temperature"] != base + 7
        ):
            raise RuntimeError("register map is not contiguous at 56..63; block read needs updating")
        out = {}
        for sid, d in self.bus.read_block(base, STATE_BLOCK_LEN, ids).items():
            out[sid] = {
                "position": decode_position(d[0], d[1]),
                "load": decode_load(d[4], d[5]),
                "voltage": d[6] / 10.0,
                "temperature": float(d[7]),
            }
        return out

    def set_torque(self, sid: int, on: bool) -> bool:
        return self.bus.write(sid, self.reg["torque_enable"], [1 if on else 0])

    def set_torque_all(self, ids: list[int], on: bool) -> None:
        self.bus.sync_write(self.reg["torque_enable"], 1, {sid: [1 if on else 0] for sid in ids})

    def set_accel(self, sid: int, accel: int) -> bool:
        return self.bus.write(sid, self.reg["accel"], [int(accel) & 0xFF])

    def set_speed(self, sid: int, speed: int) -> bool:
        return self.bus.write(sid, self.reg["goal_speed"], le16(int(speed)))

    def set_torque_limit(self, sid: int, limit: int) -> bool:
        """Reg 48 on STS. Skipped on HLS unless configured (HLS uses goal torque, reg 44)."""
        if "torque_limit" not in self.reg:
            return True
        return self.bus.write(sid, self.reg["torque_limit"], le16(int(limit)))

    def set_goal_torque(self, sid: int, torque: int) -> bool:
        """HLS only: goal torque at address 44 caps force in position mode."""
        if "goal_torque" not in self.reg:
            return True
        return self.bus.write(sid, self.reg["goal_torque"], le16(int(torque)))

    def write_position(self, sid: int, ticks: int) -> bool:
        return self.bus.write(sid, self.reg["goal_position"], encode_position(int(ticks)))

    def sync_write_positions(self, targets: dict[int, int]) -> None:
        """The hot path: one 0x83 packet for all servos."""
        self.bus.sync_write(
            self.reg["goal_position"], 2, {sid: encode_position(int(t)) for sid, t in targets.items()}
        )

    def set_id(self, old: int, new: int, i_know: bool = False) -> bool:
        """Change a servo ID in EEPROM. Power-cycle afterwards. Only with one servo on the bus."""
        if not i_know:
            raise PermissionError("set_id writes EEPROM; pass i_know=True (CLI: --i-know)")
        if old == BROADCAST_ID or not 0 <= new <= 253:
            raise ValueError("invalid ID")
        lock = self.reg["lock"]
        if not self.bus.write(old, lock, [0]):
            return False
        # The reply to an ID write may come from either ID, so don't wait for it; verify by ping.
        self.bus.write(old, self.reg["id"], [new], expect_reply=False)
        time.sleep(0.05)
        if not self.bus.ping(new):
            return False
        return self.bus.write(new, lock, [1])


class FeetechHand(HandInterface):
    def __init__(self, cfg: HandConfig, transport: Transport | None = None) -> None:
        self.cfg = cfg
        self._transport = transport
        self.port_name = cfg.port
        self.driver: FeetechDriver | None = None
        self.ids = cfg.ids
        self.active_ids = cfg.active_ids  # commanded + torque on; the rest stay relaxed
        self._enabled = np.array([s.enabled for s in cfg.servos])
        self.voltage_gate = VoltageGate(cfg.voltage_window)
        self.temp_watch = TempWatch(cfg.max_temp_c)
        self.load_watch = LoadWatch([s.stall_load for s in cfg.servos], cfg.defaults.stall_time_s)
        self.antagonists = AntagonistLimit(cfg.antagonist_pairs(), cfg.defaults.antagonist_max_sum)
        self._pairs = cfg.antagonist_pairs()
        self._payout = max(0.0, cfg.defaults.antagonist_payout)
        self._payout_cap = max(0, cfg.defaults.antagonist_payout_max_ticks)
        self.roll = cfg.roll  # replaced by a re-based copy on connect
        self.restore_report: dict = {"restored": [], "assumed": [], "max_drift": 0}
        self._span = np.array([s.span for s in cfg.servos], dtype=float)
        self._set_slack(np.array([s.slack for s in cfg.servos], dtype=float), cfg.roll)
        self._last_ticks: np.ndarray | None = None
        self._last_cmd: tuple[np.ndarray, float] = (np.zeros(N_FINGERS), 0.0)
        self._load_cap = np.ones(N_FINGERS)
        self._last_voltage_check = 0.0
        self._last_state: HandState | None = None
        self.torque_on = False
        self.connected = False
        self.last_loop_ms = 0.0
        self.last_read_ms = 0.0

    # ---------------------------------------------------------------- re-basing

    def _payout_ticks(self, i: int, other: int) -> float:
        """How far servo i may unwind past slack while its antagonist `other` is fully pulled."""
        return min(self._payout * abs(self._span[other]), float(self._payout_cap))

    def _set_slack(self, slack: np.ndarray, roll: RollConfig) -> None:
        self._slack = np.asarray(slack, dtype=float)
        self.roll = roll
        tight = self._slack + self._span
        lo, hi = np.minimum(self._slack, tight), np.maximum(self._slack, tight)
        for a, b in self._pairs:  # room to pay out: beyond slack, opposite to the pull direction
            for i, other in ((a, b), (b, a)):
                room = self._payout_ticks(i, other)
                if self._span[i] >= 0:
                    lo[i] -= room
                else:
                    hi[i] += room
        self.clamp = TickClamp(
            lo=lo.tolist() + [roll.lo],
            hi=hi.tolist() + [roll.hi],
            max_delta=[s.max_delta_ticks for s in self.cfg.servos] + [roll.max_delta_ticks],
        )

    def _roll_center_for(self, present_roll: int) -> int:
        """The calibrated roll center on whichever turn is nearest the present reading."""
        p, c = present_roll, self.cfg.roll.center
        return int(p + ((c - p + TICKS_PER_REV // 2) % TICKS_PER_REV) - TICKS_PER_REV // 2)

    def rebase(self, present: dict[int, int]) -> None:
        """Current finger positions become 'open' (the operator has opened the hand by hand)."""
        slack = np.array([present[s.id] for s in self.cfg.servos], dtype=float)
        self._set_slack(slack, dataclasses.replace(self.cfg.roll, center=self._roll_center_for(present[self.cfg.roll.id])))
        self.restore_report = {"restored": [], "assumed": [s.name for s in self.cfg.servos], "max_drift": 0}
        log.info("hand re-based: open = %s, roll center = %d", slack.astype(int).tolist(), self.roll.center)
        self._save_positions(present)

    def restore_or_rebase(self, present: dict[int, int]) -> None:
        """Recover each finger's absolute open position from the last session.

        The multi-turn count survives between commands and is only lost at power-off, where it
        collapses to the same angle on turn 0, i.e. shifts by a multiple of 4096. From the last
        saved reading we undo that shift. A finger that was moved more than RESTORE_DRIFT_TICKS
        by hand while powered off cannot be told apart from another turn: it falls back to
        'current position = open' and is reported.
        """
        saved = load_positions_state().get("servos", {})
        slack = np.zeros(len(self.cfg.servos))
        restored, assumed, drifts = [], [], []
        for i, s in enumerate(self.cfg.servos):
            now = present[s.id]
            e = saved.get(str(s.id))
            if e is None:
                slack[i] = now
                assumed.append(s.name)
                continue
            k = round((e["pos"] - now) / TICKS_PER_REV)
            drift = abs(e["pos"] - k * TICKS_PER_REV - now)
            if drift <= RESTORE_DRIFT_TICKS:
                slack[i] = e["slack"] - k * TICKS_PER_REV
                restored.append(s.name)
                drifts.append(drift)
            else:
                slack[i] = now
                assumed.append(s.name)
        self._set_slack(slack, dataclasses.replace(self.cfg.roll, center=self._roll_center_for(present[self.cfg.roll.id])))
        self.restore_report = {"restored": restored, "assumed": assumed, "max_drift": int(max(drifts, default=0))}
        log.info("open positions: restored %d, assumed-current %s (max drift %d ticks)",
                 len(restored), assumed or "none", self.restore_report["max_drift"])
        self._save_positions(present)

    def _save_positions(self, present: dict[int, int]) -> None:
        try:
            save_positions_state({s.id: {"pos": present[s.id], "slack": int(self._slack[i])}
                                  for i, s in enumerate(self.cfg.servos) if s.id in present},
                                 roll_center=self.roll.center)
        except OSError as e:
            log.warning("could not save hand positions: %s", e)

    def rehome(self) -> None:
        """After the operator has pulled the fingers open with torque off: re-base and hold here."""
        if self.driver is None:
            raise RuntimeError("hand not connected")
        present = self.driver.read_positions(self.ids)
        if len(present) != len(self.ids):
            raise SafetyTrip(f"could not read positions from {sorted(set(self.ids) - set(present))}")
        self.rebase(present)
        ticks = np.array([present[sid] for sid in self.ids], dtype=np.int64)
        self.driver.sync_write_positions(self._active(ticks))
        self.driver.set_torque_all(self.active_ids, True)
        self.torque_on = True
        self._last_ticks = ticks
        self._load_cap = np.ones(N_FINGERS)
        self.load_watch.reset()
        self._last_cmd = (np.zeros(N_FINGERS), self.roll.ticks_to_deg(int(ticks[N_FINGERS])))

    # ---------------------------------------------------------------- conversions

    def fingers_to_ticks(self, fingers: np.ndarray) -> np.ndarray:
        """Tick targets for normalized pulls, with antagonist pay-out: while one side of a
        pair winds, the other unwinds the same tendon length (capped) so it does not brake."""
        u = np.clip(np.asarray(fingers, dtype=float), 0.0, 1.0)
        t = self._slack + u * self._span
        for a, b in self._pairs:
            for i, other in ((a, b), (b, a)):
                release = min(self._payout * u[other] * abs(self._span[other]), float(self._payout_cap))
                t[i] -= np.sign(self._span[i]) * release
        return t

    def _active(self, ticks: np.ndarray) -> dict[int, int]:
        """Goal positions for the commanded servos only (disabled ones are never written)."""
        return {sid: int(t) for sid, t in zip(self.ids, ticks.tolist(), strict=True) if sid in self.active_ids}

    def expected_fingers(self, fingers: np.ndarray) -> np.ndarray:
        """Normalized position each finger servo will report once it reaches `fingers`
        (negative where a servo has paid out past slack)."""
        return np.where(self._enabled, self.ticks_to_fingers(self.fingers_to_ticks(fingers)), 0.0)

    def ticks_to_fingers(self, ticks: np.ndarray) -> np.ndarray:
        span = np.where(self._span == 0, 1.0, self._span)
        return (np.asarray(ticks, dtype=float) - self._slack) / span

    # ---------------------------------------------------------------- lifecycle

    def _open(self) -> Transport:
        if self._transport is not None:
            return self._transport
        t = open_serial(self.cfg.port, self.cfg.baud, self.cfg.timeout_s)
        self.port_name = getattr(t, "port", self.cfg.port)
        return t

    def connect(self) -> None:
        bus = FeetechBus(self._open(), timeout_s=self.cfg.timeout_s)
        self.driver = FeetechDriver(bus, self.cfg.register_map())
        d = self.driver

        missing = [sid for sid in self.ids if d.ping(sid) is None]
        if missing:
            raise SafetyTrip(f"servos not responding: {missing} (run dexkit-scan)")

        voltages = {sid: v for sid in self.ids if (v := d.read_voltage(sid)) is not None}
        self.voltage_gate.check(voltages, expected_ids=self.ids)  # refuses torque enable
        temps = {sid: float(t) for sid in self.ids if (t := d.read_temperature(sid)) is not None}
        self.temp_watch.check(temps)
        log.info("servo voltages OK: %s", ", ".join(f"{k}:{v:.1f}V" for k, v in voltages.items()))

        wrong_mode = {sid: m for sid in self.ids if (m := d.read_mode(sid)) not in (None, 0)}
        if wrong_mode:
            # Mode 2 on HLS is torque mode: position goals are ignored and goal torque spins the servo.
            raise SafetyTrip(f"servos not in position mode (reg 33 must be 0): {wrong_mode}")
        for sid in self.ids:
            d.set_accel(sid, self.cfg.defaults.accel)
            d.set_speed(sid, self.cfg.defaults.speed)
            d.set_torque_limit(sid, self.cfg.defaults.torque_limit)
            d.set_goal_torque(sid, self.cfg.defaults.goal_torque)

        present = d.read_positions(self.ids)
        if len(present) != len(self.ids):
            raise SafetyTrip(f"could not read positions from {sorted(set(self.ids) - set(present))}")
        self.restore_or_rebase(present)
        ticks = np.array([present[sid] for sid in self.ids], dtype=np.int64)
        # Hold the current position so enabling torque never jumps to a stale goal.
        d.sync_write_positions(self._active(ticks))
        d.set_torque_all(self.active_ids, True)
        disabled = [sid for sid in self.ids if sid not in self.active_ids]
        if disabled:
            d.set_torque_all(disabled, False)
            log.info("servos %s are disabled in hand.yaml: torque off, never commanded", disabled)
        self.torque_on = True
        self._last_ticks = ticks
        self._last_cmd = (np.clip(self.ticks_to_fingers(ticks[:N_FINGERS]), 0, 1),
                          self.roll.ticks_to_deg(int(ticks[N_FINGERS])))
        self._last_voltage_check = time.monotonic()
        self.connected = True
        log.info("hand connected: %d servos, torque on", len(self.ids))

    # ---------------------------------------------------------------- commands

    @property
    def last_command(self) -> tuple[np.ndarray, float]:
        return self._last_cmd[0].copy(), self._last_cmd[1]

    def set_targets(self, fingers: np.ndarray, roll_deg: float) -> tuple[np.ndarray, float]:
        if self.driver is None or not self.connected:
            raise RuntimeError("hand not connected")
        if not self.torque_on:
            raise SafetyTrip("torque is off (relaxed); reconnect to resume")
        f = np.clip(np.asarray(fingers, dtype=float), 0.0, 1.0)
        if f.shape != (N_FINGERS,):
            raise ValueError(f"fingers must have shape ({N_FINGERS},)")
        f = np.where(self._enabled, f, 0.0)  # disabled servos: nothing is ever asked of them
        f = self.antagonists.apply(f)
        if self.antagonists.limited:
            log.debug("antagonist limit applied")
        f = np.minimum(f, self._load_cap)
        roll = float(np.clip(roll_deg, -self.roll.range_deg, self.roll.range_deg))
        target = np.append(self.fingers_to_ticks(f), self.roll.deg_to_ticks(roll))
        prev_ticks = self._last_ticks
        ticks = self.clamp.apply(target, prev_ticks)
        t0 = time.perf_counter()
        self.driver.sync_write_positions(self._active(ticks))
        self.last_loop_ms = (time.perf_counter() - t0) * 1000
        self._last_ticks = ticks
        # What was actually commanded, in pull units: the requested pull, advanced only as far
        # as the slew limit let each servo move this tick. (A servo's ticks also include pay-out
        # for its antagonist, so ticks cannot simply be converted back to a pull value.)
        prev_u = self._last_cmd[0]
        if prev_ticks is None:
            sent_f = f
        else:
            want = target[:N_FINGERS] - prev_ticks[:N_FINGERS]
            got = ticks[:N_FINGERS] - prev_ticks[:N_FINGERS]
            frac = np.where(np.abs(want) < 1e-9, 1.0, np.clip(got / np.where(np.abs(want) < 1e-9, 1.0, want), 0, 1))
            sent_f = prev_u + frac * (f - prev_u)
        sent = (np.clip(sent_f, 0, 1), self.roll.ticks_to_deg(int(ticks[N_FINGERS])))
        self._last_cmd = sent
        # Release a load cap once the operator commands below it.
        self._load_cap = np.where(np.asarray(fingers) < self._load_cap - 1e-6, 1.0, self._load_cap)
        return sent[0].copy(), sent[1]

    def get_state(self) -> HandState:
        if self.driver is None:
            raise RuntimeError("hand not connected")
        t0 = time.perf_counter()
        block = self.driver.read_state_block(self.ids)
        read_ms = (time.perf_counter() - t0) * 1000
        missing = [sid for sid in self.ids if sid not in block]
        if missing:
            log.warning("no state from servos %s", missing)
        now = time.monotonic()
        ticks = {sid: int(v["position"]) for sid, v in block.items()}
        load = {sid: int(v["load"]) for sid, v in block.items()}
        voltage = {sid: float(v["voltage"]) for sid, v in block.items()}
        temp = {sid: float(v["temperature"]) for sid, v in block.items()}

        if self.torque_on and now - self._last_voltage_check >= self.cfg.voltage_check_period_s:
            self._last_voltage_check = now
            try:
                self.voltage_gate.check(voltage)
                self.temp_watch.check(temp)
            except SafetyTrip:
                self.relax()
                raise

        prev = self._last_state
        finger_ticks = np.array(
            [ticks.get(sid, prev.ticks.get(sid, 0) if prev else 0) for sid in self.cfg.finger_ids],
            dtype=float,
        )
        roll_ticks = ticks.get(self.cfg.roll.id, self.roll.center)

        if self.torque_on:
            loads = np.array([load.get(sid, 0) for sid in self.cfg.finger_ids], dtype=float)
            loads[~self._enabled] = 0.0  # free-wheeling servos report drag, not a stall
            for i in self.load_watch.update(loads, now):
                present_u = float(np.clip(self.ticks_to_fingers(finger_ticks[i : i + 1])[0], 0, 1))
                self._load_cap[i] = max(0.0, min(self._load_cap[i], present_u) - self.cfg.defaults.stall_backoff)
                log.warning("servo %s stalled (load %.0f): backing off to %.2f",
                            self.cfg.servos[i].name, loads[i], self._load_cap[i])

        state = HandState(
            fingers=np.where(self._enabled, self.ticks_to_fingers(finger_ticks), 0.0),
            roll_deg=self.roll.ticks_to_deg(roll_ticks),
            ticks=ticks,
            load=load,
            voltage=voltage,
            temperature=temp,
            timestamp=now,
        )
        self._last_state = state
        self.last_read_ms = read_ms
        return state

    def read_voltages(self) -> dict[int, float]:
        if self.driver is None:
            return {}
        return {sid: v for sid in self.ids if (v := self.driver.read_voltage(sid)) is not None}

    def relax(self) -> None:
        self.torque_on = False
        if self.driver is None:
            return
        try:
            self.driver.set_torque_all(self.ids, False)
            self.driver.bus.write(BROADCAST_ID, self.cfg.register_map()["torque_enable"], [0], expect_reply=False)
        except Exception as e:  # noqa: BLE001 - relax must never raise
            log.error("relax failed: %s", e)

    def close(self) -> None:
        if self.driver is not None and self.connected:
            try:
                self._save_positions(self.driver.read_positions(self.ids))
            except Exception as e:  # noqa: BLE001 - closing must never fail
                log.debug("save positions on close: %s", e)
        self.relax()
        if self.driver is not None:
            try:
                self.driver.bus.close()
            except Exception as e:  # noqa: BLE001
                log.debug("close: %s", e)
        self.driver = None  # a later relax() (atexit, e-stop) is then a no-op instead of a port error
        self.connected = False
