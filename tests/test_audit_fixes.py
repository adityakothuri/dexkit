"""Regression tests for the real-hardware audit fixes."""

import numpy as np
import pytest

from dexkit.hw.mock import MockFeetechSerial, MockGantry, MockHand, mock_servos_for
from dexkit.hw.safety import EStop, EStopTripped, SafetyTrip, clear_estop_flag, set_estop_flag


def test_refuses_servo_not_in_position_mode(hand_cfg):
    bus = MockFeetechSerial(mock_servos_for(hand_cfg))
    bus.servos[4].mem[33] = 2  # HLS torque mode
    h = MockHand(hand_cfg, latency_s=0, bus=bus)
    with pytest.raises(SafetyTrip, match="position mode"):
        h.connect()
    assert not any(s.torque_on for s in bus.servos.values())


def test_hls_never_writes_register_48(mock_hand):
    writes_48 = [p for p in mock_hand.calls if len(p) > 5 and p[4] in (0x03, 0x83) and p[5] == 48]
    assert writes_48 == []
    goal_torque = [p for p in mock_hand.calls if len(p) > 5 and p[4] == 0x03 and p[5] == 44]
    assert len(goal_torque) == 13


def test_external_estop_flag_trips_running_loop(mock_hand):
    e = EStop()
    e.register(hand=mock_hand)
    e.check()
    set_estop_flag("test")
    with pytest.raises(EStopTripped, match="external"):
        e.check()
    assert not any(s.torque_on for s in mock_hand.bus_sim.servos.values())
    assert clear_estop_flag()


def test_session_refuses_while_flag_set():
    import argparse

    from dexkit.cli import open_session

    set_estop_flag("test")
    args = argparse.Namespace(mock=True, yes=True, no_gantry=True)
    with pytest.raises(SafetyTrip, match="--clear"):
        open_session(args, need_hand=True)
    clear_estop_flag()


def test_held_jog_key_cannot_pile_up_motion(gantry_cfg):
    g = MockGantry(gantry_cfg, speed_factor=1.0)
    g.connect()
    sent = sum(g.jog(1, 0, 0, feed=600) for _ in range(30))  # 30 rapid auto-repeats
    assert sent <= int(gantry_cfg.jog_max_backlog_mm) + 2
    g.jog_cancel()
    assert g.get_state().state == "Idle"


def test_jog_clamp_uses_queued_end_not_stale_position(gantry_cfg):
    g = MockGantry(gantry_cfg, speed_factor=1.0)
    g.connect()
    g.set_zero()
    g.restore_frame((288.0, 10.0, -1.0))  # 2 mm from the X max of 290
    for _ in range(5):
        g.jog(1, 0, 0, feed=300)
    g.sim.speed_factor = 100
    st = g.wait_idle()
    assert st.xyz[0] <= gantry_cfg.travel_max[0] + 1e-6


def test_interactive_pose_prompt(mock_hand, hand_cfg):
    from dexkit.control.poses import PoseLibrary, interactive

    lib = PoseLibrary.load(hand_cfg)
    cmds = iter(["fist", "f 1 0.3", "roll 15", "bogus", "quit"])
    interactive(mock_hand, lib, 50, 0.1, "linear", EStop(), read=lambda _: next(cmds))
    f, r = mock_hand.last_command
    assert f[0] == pytest.approx(0.3, abs=0.01) and np.allclose(f[1:], 1.0) and r == pytest.approx(15, abs=0.1)
