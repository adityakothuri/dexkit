import numpy as np
import pytest

from dexkit.env.dexkit_env import DexKitEnv
from dexkit.hw.base import ACTION_DIM, join_action, split_action
from dexkit.hw.safety import EStopTripped, SafetyTrip


def test_action_layout_round_trip():
    a = np.arange(16, dtype=float)
    f, r, xyz = split_action(a)
    assert f.tolist() == list(range(12)) and r == 12 and xyz.tolist() == [13, 14, 15]
    assert np.array_equal(join_action(f, r, xyz), a)
    with pytest.raises(ValueError):
        split_action(np.zeros(15))


def test_reset_then_50_steps_converges(mock_hand, mock_gantry, estop):
    mock_gantry.set_zero()
    estop.register(hand=mock_hand, gantry=mock_gantry)
    env = DexKitEnv(mock_hand, mock_gantry, estop=estop)
    obs = env.reset(duration_s=0.1, rate_hz=50)
    assert obs["agent_pos"].shape == (ACTION_DIM,)
    target = join_action(np.full(12, 0.4), 20.0, np.array([10.0, 5.0, -2.0]))  # 0.4+0.4 stays under the antagonist limit
    errs = []
    import time

    for _ in range(50):
        obs = env.step(target)
        errs.append(np.abs(obs["agent_pos"][:12] - 0.4).max())
        time.sleep(0.01)
    assert errs[-1] < errs[0]
    assert errs[-1] < 0.05
    mock_gantry.wait_idle()
    assert np.allclose(env.observe()["agent_pos"][13:], [10, 5, -2], atol=0.1)
    assert obs["action_sent"].shape == (ACTION_DIM,)


def test_estop_mid_run_leaves_torque_off(mock_hand, mock_gantry, estop):
    mock_gantry.set_zero()
    estop.register(hand=mock_hand, gantry=mock_gantry)
    env = DexKitEnv(mock_hand, mock_gantry, estop=estop)
    target = join_action(np.full(12, 0.8), 0.0, np.array([5.0, 5.0, -1.0]))
    for i in range(20):
        if i == 10:
            estop.trip("test")
        if i >= 10:
            with pytest.raises(EStopTripped):
                env.step(target)
        else:
            env.step(target)
    assert not any(s.torque_on for s in mock_hand.bus_sim.servos.values())


def test_step_refuses_when_gantry_frame_invalid(mock_hand, mock_gantry, estop):
    env = DexKitEnv(mock_hand, mock_gantry, estop=estop)
    with pytest.raises(SafetyTrip, match="frame"):
        env.step(np.zeros(ACTION_DIM))


def test_step_rejects_non_finite(mock_hand, estop):
    env = DexKitEnv(mock_hand, None, estop=estop)
    a = np.zeros(ACTION_DIM)
    a[3] = np.nan
    with pytest.raises(SafetyTrip):
        env.step(a)


def test_hand_only_env(mock_hand, estop):
    env = DexKitEnv(mock_hand, None, estop=estop)
    obs = env.step(join_action(np.full(12, 0.2), 0, np.array([999.0, 0, 0])))
    assert obs["gantry_state"] is None and np.all(obs["agent_pos"][13:] == 0)
