"""Feetech STS/HLS serial bus protocol, implemented directly on pyserial.

Packet:  FF FF  ID  LEN  INSTR  P1..Pn  CHK
  LEN = n + 2,  CHK = ~(ID + LEN + INSTR + sum(P)) & 0xFF
Status:  FF FF  ID  LEN  ERR    P1..Pn  CHK
Multi-byte values are little-endian (STS/HLS; SCS is big-endian, not supported).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Protocol

log = logging.getLogger(__name__)

HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83


class FeetechError(IOError):
    pass


class Transport(Protocol):
    """The subset of serial.Serial the bus uses (lets mocks stand in)."""

    baudrate: int

    def write(self, data: bytes) -> int | None: ...
    def read(self, size: int = 1) -> bytes: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...


def checksum(body: Iterable[int]) -> int:
    """`body` is ID..last param (everything after the FF FF header)."""
    return (~sum(body)) & 0xFF


def encode_packet(servo_id: int, instruction: int, params: Iterable[int] = ()) -> bytes:
    params = list(params)
    if not 0 <= servo_id <= 0xFE:
        raise ValueError(f"servo id {servo_id} out of range")
    for p in params:
        if not 0 <= p <= 0xFF:
            raise ValueError(f"param byte {p} out of range")
    body = [servo_id, len(params) + 2, instruction, *params]
    return HEADER + bytes(body + [checksum(body)])


def le16(value: int) -> list[int]:
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"value {value} does not fit in 16 bits")
    return [value & 0xFF, (value >> 8) & 0xFF]


def from_le16(lo: int, hi: int) -> int:
    return lo | (hi << 8)


def encode_position(ticks: int) -> list[int]:
    """Goal position bytes. Bit 15 is the sign bit on STS/HLS; we only use 0..4095."""
    if ticks < 0:
        return le16((-ticks) | 0x8000)
    return le16(ticks)


def decode_position(lo: int, hi: int) -> int:
    raw = from_le16(lo, hi)
    return -(raw & 0x7FFF) if raw & 0x8000 else raw


def decode_load(lo: int, hi: int) -> int:
    """Present load: bits 0..9 magnitude (0..1000, 0.1 %), bit 10 direction. Returns magnitude."""
    return from_le16(lo, hi) & 0x3FF


def ping_packet(servo_id: int) -> bytes:
    return encode_packet(servo_id, INST_PING)


def read_packet(servo_id: int, address: int, length: int) -> bytes:
    return encode_packet(servo_id, INST_READ, [address, length])


def write_packet(servo_id: int, address: int, data: Iterable[int]) -> bytes:
    return encode_packet(servo_id, INST_WRITE, [address, *data])


def sync_write_packet(address: int, data_len: int, items: dict[int, list[int]]) -> bytes:
    params = [address, data_len]
    for sid, data in items.items():
        if len(data) != data_len:
            raise ValueError(f"servo {sid}: expected {data_len} bytes, got {len(data)}")
        params += [sid, *data]
    return encode_packet(BROADCAST_ID, INST_SYNC_WRITE, params)


def sync_read_packet(address: int, length: int, ids: Iterable[int]) -> bytes:
    return encode_packet(BROADCAST_ID, INST_SYNC_READ, [address, length, *ids])


def decode_status(packet: bytes) -> tuple[int, int, bytes]:
    """Validate a full status packet; return (id, error_byte, params)."""
    if len(packet) < 6 or packet[:2] != HEADER:
        raise FeetechError(f"malformed status packet {packet.hex(' ')}")
    sid, length, err = packet[2], packet[3], packet[4]
    if len(packet) != length + 4:
        raise FeetechError(f"length mismatch in {packet.hex(' ')}")
    params = packet[5:-1]
    if checksum(packet[2:-1]) != packet[-1]:
        raise FeetechError(f"bad checksum in {packet.hex(' ')}")
    return sid, err, bytes(params)


class FeetechBus:
    """Request/response access to a Feetech bus. No register names here, only addresses."""

    def __init__(self, transport: Transport, timeout_s: float = 0.02) -> None:
        self.t = transport
        self.timeout_s = timeout_s
        self.sync_read_supported: bool | None = None
        # Some half-duplex adapters loop every transmitted byte back to RX. Those echoes
        # decode as valid status packets (a ping echo is FF FF id 02 01 chk), so they are
        # recognised by being byte-identical to what was just sent, and skipped.
        self.echo_seen = False
        self._last_tx = b""
        self._last_rx = b""

    # ---------------------------------------------------------------- low level

    def _send(self, packet: bytes) -> None:
        self.t.reset_input_buffer()
        log.debug("TX %s", packet.hex(" "))
        self._last_tx = packet
        self.t.write(packet)

    def _read_exact(self, n: int, deadline: float) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.t.read(n - len(buf))
            if chunk:
                buf += chunk
            elif time.monotonic() > deadline:
                break
        return buf

    def _receive(self, deadline: float) -> tuple[int, int, bytes] | None:
        """Next valid status packet before `deadline`; garbled packets (line noise, two servos
        answering at once) are logged and skipped rather than raised."""
        while True:
            try:
                return self._receive_one(deadline)
            except FeetechError as e:
                log.debug("discarding %s", e)

    def _receive_one(self, deadline: float) -> tuple[int, int, bytes] | None:
        # Hunt for FF FF, tolerating stray bytes.
        prev = b""
        while True:
            b = self._read_exact(1, deadline)
            if not b:
                return None
            if prev == b"\xff" and b == b"\xff":
                break
            prev = b
        head = self._read_exact(2, deadline)
        if len(head) < 2:
            return None
        sid, length = head[0], head[1]
        if sid == 0xFF:  # three FFs in a row: shift and retry once
            head = head[1:] + self._read_exact(1, deadline)
            sid, length = head[0], head[1]
        rest = self._read_exact(length, deadline)
        if len(rest) < length:
            return None
        packet = HEADER + head + rest
        log.debug("RX %s", packet.hex(" "))
        self._last_rx = packet
        return decode_status(packet)

    def _is_echo(self, skipped: bool) -> bool:
        """The first packet identical to the request is an echo, not a reply."""
        if skipped or self._last_rx != self._last_tx:
            return False
        if not self.echo_seen:
            log.info("adapter echoes transmitted bytes; ignoring the echoes")
        self.echo_seen = True
        return True

    def _transact(self, packet: bytes, expect_id: int) -> tuple[int, bytes] | None:
        self._send(packet)
        deadline = time.monotonic() + self.timeout_s
        skipped = False
        while True:
            resp = self._receive(deadline)
            if resp is None:
                return None
            if self._is_echo(skipped):
                skipped = True
                continue
            sid, err, params = resp
            if sid == expect_id:
                if err:
                    log.debug("servo %d status error byte 0x%02x", sid, err)
                return err, params

    # ---------------------------------------------------------------- operations

    def ping(self, servo_id: int) -> bool:
        return self._transact(ping_packet(servo_id), servo_id) is not None

    def read(self, servo_id: int, address: int, length: int) -> bytes | None:
        resp = self._transact(read_packet(servo_id, address, length), servo_id)
        if resp is None or len(resp[1]) != length:
            return None
        return resp[1]

    def write(self, servo_id: int, address: int, data: list[int], expect_reply: bool = True) -> bool:
        packet = write_packet(servo_id, address, data)
        if not expect_reply or servo_id == BROADCAST_ID:
            self._send(packet)
            return True
        return self._transact(packet, servo_id) is not None

    def sync_write(self, address: int, data_len: int, items: dict[int, list[int]]) -> None:
        if items:
            self._send(sync_write_packet(address, data_len, items))

    def sync_read(self, address: int, length: int, ids: list[int]) -> dict[int, bytes]:
        """Returns whatever servos answered; caller falls back for the missing ones."""
        self._send(sync_read_packet(address, length, ids))
        out: dict[int, bytes] = {}
        deadline = time.monotonic() + self.timeout_s * max(1, len(ids) // 4 + 1)
        while len(out) < len(ids):
            resp = self._receive(deadline)
            if resp is None:
                break
            sid, _err, params = resp
            if sid in ids and len(params) == length:
                out[sid] = params
        return out

    def read_block(self, address: int, length: int, ids: list[int]) -> dict[int, bytes]:
        """Sync read if the servos support it, else sequential reads."""
        out: dict[int, bytes] = {}
        if self.sync_read_supported is not False:
            out = self.sync_read(address, length, ids)
            if self.sync_read_supported is None:
                self.sync_read_supported = len(out) > 0
                log.info("sync read %s", "supported" if self.sync_read_supported else "unsupported, using sequential reads")
        for sid in ids:
            if sid not in out:
                data = self.read(sid, address, length)
                if data is not None:
                    out[sid] = data
        return out

    def close(self) -> None:
        self.t.close()
