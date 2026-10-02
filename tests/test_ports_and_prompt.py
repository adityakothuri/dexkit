from __future__ import annotations

import numpy as np
import pytest

from dexkit.control.poses import PoseLibrary, interactive
from dexkit.hw.ports import PortInfo, PortNotFound, find_port, resolve_port

MAC_PORTS = [
    PortInfo("/dev/cu.usbmodem5A7A0123451", 0x1A86, 0x55D3, "USB Single Serial"),
    PortInfo("/dev/cu.usbserial-1410", 0x1A86, 0x7523, "USB Serial"),
]


def test_autodetects_hand_and_gantry_by_usb_id(monkeypatch):
    monkeypatch.delenv("DEXKIT_HAND_PORT", raising=False)
    monkeypatch.delenv("DEXKIT_GANTRY_PORT", raising=False)
    assert resolve_port("/dev/dexkit_hand", "hand", MAC_PORTS) == "/dev/cu.usbmodem5A7A0123451"
    assert resolve_port("/dev/dexkit_gantry", "gantry", MAC_PORTS) == "/dev/cu.usbserial-1410"


def test_existing_configured_port_wins(tmp_path):
    dev = tmp_path / "fake_tty"
    dev.touch()
    assert resolve_port(str(dev), "hand", MAC_PORTS) == str(dev)


def test_env_override(monkeypatch):
    monkeypatch.setenv("DEXKIT_GANTRY_PORT", "/dev/cu.custom")
    assert resolve_port("/dev/dexkit_gantry", "gantry", []) == "/dev/cu.custom"


def test_gantry_fallback_chip_and_missing_device(monkeypatch):
    monkeypatch.delenv("DEXKIT_HAND_PORT", raising=False)
    cp2102 = [PortInfo("/dev/cu.SLAB_USBtoUART", 0x10C4, 0xEA60)]
    assert find_port("gantry", cp2102) == "/dev/cu.SLAB_USBtoUART"
    assert find_port("hand", cp2102) is None
    with pytest.raises(PortNotFound, match="plugged in"):
        resolve_port("/dev/dexkit_hand", "hand", cp2102)


def test_prompt_drives_hand_and_gantry(mock_hand, mock_gantry, hand_cfg, estop):
    lib = PoseLibrary.load(hand_cfg)
    cmds = iter(["jog x 5", "zero", "goto 10 5 -2", "where", "f 0 0.5", "fist", "q"])
    interactive(mock_hand, lib, 50, 0.1, "linear", estop, read=lambda _: next(cmds), gantry=mock_gantry)
    assert mock_gantry.frame_valid
    np.testing.assert_allclose(mock_gantry.get_state().xyz, [10, 5, -2], atol=0.05)
    f, _ = mock_hand.last_command
    np.testing.assert_allclose(f, 1.0)  # "f 0" was rejected, fist applied


def test_echoing_adapter_is_not_mistaken_for_servos(hand_cfg):
    from dexkit.hw.feetech_hand import FeetechDriver
    from dexkit.hw.feetech_protocol import FeetechBus
    from dexkit.hw.mock import MockFeetechSerial, mock_servos_for
    from dexkit.tools.scan_bus import scan

    # Echo only, no servos powered: nothing must be "found".
    bus = FeetechBus(MockFeetechSerial([], echo=True), timeout_s=0.002)
    d = FeetechDriver(bus, hand_cfg.register_map())
    assert scan(d, range(0, 20)) == []
    assert bus.echo_seen

    # Echo plus real servos: the real replies come through.
    t = MockFeetechSerial(mock_servos_for(hand_cfg), echo=True)
    d = FeetechDriver(FeetechBus(t, timeout_s=0.002), hand_cfg.register_map())
    found = scan(d, range(0, 20))
    assert [s["id"] for s in found] == hand_cfg.ids
    assert all(s["voltage"] is not None and s["model"] != 515 for s in found)


def test_garbled_bytes_do_not_crash_the_bus(hand_cfg):
    from dexkit.hw.feetech_protocol import FeetechBus
    from dexkit.hw.mock import MockFeetechSerial, mock_servos_for

    t = MockFeetechSerial(mock_servos_for(hand_cfg))
    real_write = t.write

    def noisy_write(data: bytes) -> int:
        t._rx += bytes.fromhex("ff ff 02 01 23")  # the malformed packet seen on the bench
        return real_write(data)

    t.write = noisy_write
    bus = FeetechBus(t, timeout_s=0.002)
    assert bus.ping(1)


def test_multi_turn_ticks_are_valid_config(hand_cfg):
    from dexkit.config import ConfigError, hand_config_from_dict, load_yaml

    raw = load_yaml(hand_cfg.source)
    raw["servos"][0].update(slack=3604, tight=5175)  # past 4095, as measured on the bench
    raw["servos"][1].update(slack=200, tight=-900, inverted=True)
    cfg = hand_config_from_dict(raw)
    assert cfg.servos[0].hi == 5175 and cfg.servos[1].lo == -900
    raw["servos"][2].update(tight=40000)
    with pytest.raises(ConfigError, match="out of"):
        hand_config_from_dict(raw)


def test_calibration_walks_past_4095(hand_cfg):
    from dexkit.hw.feetech_hand import FeetechDriver
    from dexkit.hw.feetech_protocol import FeetechBus
    from dexkit.hw.mock import MockFeetechSerial, mock_servos_for
    from dexkit.tools.calibrate_hand import ScriptedPrompter, capture_servo

    servos = mock_servos_for(hand_cfg)
    servos[0].position = 3900.0
    t = MockFeetechSerial(servos)
    d = FeetechDriver(FeetechBus(t, timeout_s=0.002), hand_cfg.register_map())
    res = capture_servo(d, servos[0].id, "x", hand_cfg, ScriptedPrompter(tight_steps=20), step_delay=0.0)
    assert res is not None and res["tight"] > 4095
