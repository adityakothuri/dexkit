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


def test_partial_calibration_is_not_calibrated(hand_cfg):
    assert not hand_cfg.is_calibrated and len(hand_cfg.uncalibrated_ids) == 13
    hand_cfg.calibrated_at = "2026-10-02T00:00:00"
    hand_cfg.servos[0].calibrated = True
    assert not hand_cfg.is_calibrated and hand_cfg.uncalibrated_ids == hand_cfg.ids[1:]
    for s in hand_cfg.servos:
        s.calibrated = True
    hand_cfg.roll.calibrated = True
    assert hand_cfg.is_calibrated


def test_calibration_run_skips_done_servos_and_renames(tmp_path, monkeypatch):
    from dexkit.config import load_hand_config
    from dexkit.tools.calibrate_hand import main

    out = tmp_path / "hand.yaml"
    main(["--mock", "--scripted", "--servo", "3", "--name", "ring_x", "--finger", "ring", "--out", str(out)])
    cfg = load_hand_config(out)
    s3 = next(s for s in cfg.servos if s.id == 3)
    assert s3.calibrated and s3.name == "ring_x" and s3.finger == "ring"
    assert not cfg.is_calibrated and 3 not in cfg.uncalibrated_ids
    main(["--mock", "--scripted", "--out", str(out)])  # does the remaining 12, skips servo 3
    cfg = load_hand_config(out)
    assert cfg.is_calibrated and next(s for s in cfg.servos if s.id == 3).name == "ring_x"


def test_connect_rebases_to_present_position(hand_cfg):
    """Servos lose their turn count at power-off; 'open' must be wherever they are at connect."""
    from dexkit.hw.mock import MockFeetechSerial, MockHand, mock_servos_for

    s0 = hand_cfg.servos[0]
    start = {s0.id: 1711, hand_cfg.roll.id: hand_cfg.roll.center + 4096 - 100}  # roll one turn up, 100 ticks off
    h = MockHand(hand_cfg, latency_s=0, bus=MockFeetechSerial(mock_servos_for(hand_cfg, start_ticks=start)))
    h.connect()
    for _ in range(12):
        h.set_targets(np.ones(12), 0.0)
    assert h.bus_sim.servos[s0.id].goal == 1711 + s0.span          # full curl = present + span
    assert h.bus_sim.servos[hand_cfg.roll.id].goal == hand_cfg.roll.center + 4096  # roll 0 = nearest equivalent center
    f, r = h.last_command
    assert np.allclose(f, 1.0)
    h.close()


def test_rehome_makes_current_pose_open(mock_hand):
    for _ in range(12):
        mock_hand.set_targets(np.full(12, 0.5), 0.0)
    mock_hand.relax()
    mock_hand.rehome()
    f, _ = mock_hand.last_command
    assert np.allclose(f, 0.0) and mock_hand.torque_on
    sid = mock_hand.cfg.servos[0].id
    half = mock_hand.bus_sim.servos[sid].goal
    for _ in range(12):
        mock_hand.set_targets(np.ones(12), 0.0)
    assert mock_hand.bus_sim.servos[sid].goal == half + mock_hand.cfg.servos[0].span
