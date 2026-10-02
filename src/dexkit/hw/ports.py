"""Find the hand adapter and the GRBL board among the USB serial ports.

The configured port is used when it exists (e.g. /dev/dexkit_hand from the udev
rule on Linux). Otherwise the device is found by USB vendor/product ID, which is
what makes plug-and-play work on macOS (/dev/cu.usbmodem*, /dev/cu.usbserial-*)
and on Linux without the udev rule. $DEXKIT_HAND_PORT / $DEXKIT_GANTRY_PORT
override everything.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

WCH = 0x1A86
# Waveshare Bus Servo Adapter (A): WCH CH343 (55d3); CH9102/CH9101 variants also seen.
HAND_IDS = {(WCH, 0x55D3), (WCH, 0x55D4), (WCH, 0x55D2)}
# GRBL boards: WCH CH340/CH341 on 3018 / CNC Pro boards.
GANTRY_IDS = {(WCH, 0x7523), (WCH, 0x5523)}
# Other USB-serial chips a GRBL board might use (CP210x, FTDI, Arduino); fallback only.
GANTRY_FALLBACK_VIDS = {0x10C4, 0x0403, 0x2341, 0x2A03}

ENV = {"hand": "DEXKIT_HAND_PORT", "gantry": "DEXKIT_GANTRY_PORT"}


class PortNotFound(OSError):
    pass


@dataclass
class PortInfo:
    device: str
    vid: int | None
    pid: int | None
    description: str = ""


def list_usb_ports() -> list[PortInfo]:
    from serial.tools import list_ports  # only hw/ imports pyserial

    out = []
    for p in list_ports.comports():
        # macOS lists each device as /dev/cu.* (non-blocking open); never the /dev/tty.* twin.
        if p.vid is None:
            continue
        out.append(PortInfo(p.device, p.vid, p.pid, p.description or ""))
    return out


def find_port(kind: str, ports: list[PortInfo]) -> str | None:
    if kind == "hand":
        hits = [p for p in ports if (p.vid, p.pid) in HAND_IDS]
    elif kind == "gantry":
        hits = [p for p in ports if (p.vid, p.pid) in GANTRY_IDS]
        if not hits:
            hits = [p for p in ports if p.vid in GANTRY_FALLBACK_VIDS]
    else:
        raise ValueError(f"unknown device kind {kind}")
    if len(hits) > 1:
        log.warning("several %s candidates: %s; using %s", kind, [p.device for p in hits], hits[0].device)
    return hits[0].device if hits else None


def describe_ports(ports: list[PortInfo]) -> str:
    if not ports:
        return "no USB serial devices are visible"
    return "visible USB serial devices: " + "; ".join(
        f"{p.device} ({p.vid:04x}:{p.pid or 0:04x} {p.description})" for p in ports
    )


def resolve_port(configured: str | None, kind: str, ports: list[PortInfo] | None = None) -> str:
    """Configured path if it exists, else auto-detect by USB ID. Raises PortNotFound."""
    env = os.environ.get(ENV[kind])
    if env:
        return env
    if configured and configured != "auto" and Path(configured).exists():
        return configured
    ports = list_usb_ports() if ports is None else ports
    found = find_port(kind, ports)
    if found:
        if configured and configured != "auto":
            log.info("%s: %s not present; auto-detected %s", kind, configured, found)
        return found
    label = "hand (Waveshare servo adapter)" if kind == "hand" else "gantry (GRBL board)"
    raise PortNotFound(
        f"could not find the {label}. Is its USB cable plugged in? {describe_ports(ports)}. "
        f"To force a port: export {ENV[kind]}=/dev/cu.XXXX"
    )
