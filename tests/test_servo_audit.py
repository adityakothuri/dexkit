"""Audit tests for the servo path: multi-turn re-basing, inverted spans, antagonists, settle."""
from __future__ import annotations

import numpy as np
import pytest

from dexkit.config import hand_config_from_dict, load_yaml
from dexkit.control.poses import Pose, go_to_pose
from dexkit.hw.mock import MockFeetechSerial, MockHand, mock_servos_for
from dexkit.util import Rate


def _hand(cfg, start=None):
    h = MockHand(cfg, latency_s=0, bus=MockFeetechSerial(mock_servos_for(cfg, start_ticks=start)))
    h.connect()
    return h


def _fist(cfg):
    return np.array([1.0 if s.role == "flex" else 0.0 for s in cfg.servos])


def test_inverted_negative_multiturn_servo_round_trips(hand_cfg):
    """A servo calibrated with slack above tight, crossing zero, after a power cycle."""
    raw = load_yaml(hand_cfg.source)
    raw["servos"][0].update(slack=284, tight=-464, inverted=True)  # keep the file's own name
    cfg = hand_config_from_dict(raw)
    s0 = cfg.servos[0]
    assert s0.span == -748 and s0.effective_tight == -464
    h = _hand(cfg, start={s0.id: 3900})  # power-cycled: count reset to within 0..4095
    only = np.zeros(12)
    only[0] = 1.0
    for _ in range(12):
        h.set_targets(only, 0.0)
    assert h.bus_sim.servos[s0.id].goal == 3900 - 748  # pulls toward lower counts from wherever it woke up
    import time

    time.sleep(0.25)  # let the simulated servo arrive
    st = h.get_state()
    assert st.fingers[0] == pytest.approx(1.0, abs=0.02)
    h.close()


def test_go_to_pose_settles_fast_when_antagonists_scale(mock_hand, hand_cfg, estop):
    names = hand_cfg.names
    f = np.zeros(12)
    f[names.index("index_flex")] = 1.0
    f[names.index("index_extend")] = 0.8  # sums to 1.8: the driver will scale it to 1.0
    fast = Rate(1000, sleep=lambda _s: None)
    before = len(mock_hand.calls)
    go_to_pose(mock_hand, Pose(f, 0.0), 0.2, 50, estop=estop, rate=fast)
    moves = len(mock_hand.calls) - before  # 10 trajectory ticks + a few slew-limited settle ticks
    assert moves < 40, f"settle loop sent {moves} packets: it ran to its cap instead of stopping once nothing changes"
    got, _ = mock_hand.last_command
    assert got[names.index("index_flex")] + got[names.index("index_extend")] == pytest.approx(1.0)


def test_calibration_stall_floor_is_above_normal_holding_load(hand_cfg):
    from dexkit.hw.feetech_hand import FeetechDriver
    from dexkit.hw.feetech_protocol import FeetechBus
    from dexkit.tools.calibrate_hand import ScriptedPrompter, capture_servo

    servos = mock_servos_for(hand_cfg)
    servos[0].load_override = 90  # a light tendon: load ~90 at tight, as seen on the bench
    t = MockFeetechSerial(servos)
    d = FeetechDriver(FeetechBus(t, timeout_s=0.002), hand_cfg.register_map())
    res = capture_servo(d, servos[0].id, "x", hand_cfg, ScriptedPrompter(tight_steps=10), step_delay=0.0)
    assert res["stall_load"] >= 400, "holding at goal_torque 600 must not look like a stall"
