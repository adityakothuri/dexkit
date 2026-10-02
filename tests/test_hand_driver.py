import numpy as np
import pytest

from dexkit.config import hand_config_from_dict, load_yaml
from dexkit.hw.mock import MockFeetechSerial, MockHand, mock_servos_for
from dexkit.hw.safety import SafetyTrip


def test_connect_refuses_torque_outside_voltage_window(hand_cfg):
    bus = MockFeetechSerial(mock_servos_for(hand_cfg))
    bus.set_voltage(12.0)
    h = MockHand(hand_cfg, latency_s=0, bus=bus)
    with pytest.raises(SafetyTrip, match="voltage"):
        h.connect()
    assert not any(s.torque_on for s in bus.servos.values())


def test_connect_reports_missing_servo(hand_cfg):
    servos = [s for s in mock_servos_for(hand_cfg) if s.id != 7]
    h = MockHand(hand_cfg, latency_s=0, bus=MockFeetechSerial(servos))
    with pytest.raises(SafetyTrip, match=r"\[7\]"):
        h.connect()


def test_connect_holds_position_then_enables_torque(mock_hand):
    servos = mock_hand.bus_sim.servos
    assert all(s.torque_on for s in servos.values())
    assert all(s.goal == round(s.position) for s in servos.values())


def test_set_targets_maps_clamps_and_rate_limits(mock_hand, hand_cfg):
    s0 = hand_cfg.servos[0]
    fist = np.array([1.0 if s.role == "flex" else 0.0 for s in hand_cfg.servos])  # flexors only: no antagonist scaling
    f, r = mock_hand.set_targets(fist, 200.0)
    # One call moves at most max_delta_ticks from the start position (slack).
    assert mock_hand.bus_sim.servos[s0.id].goal == s0.slack + s0.max_delta_ticks
    assert f[0] == pytest.approx(s0.max_delta_ticks / abs(s0.tight - s0.slack))
    for _ in range(20):
        f, r = mock_hand.set_targets(fist, 200.0)
    assert mock_hand.bus_sim.servos[s0.id].goal == s0.tight
    assert np.allclose(f, fist)
    assert r == pytest.approx(hand_cfg.roll.range_deg, abs=0.1)  # roll clamped to range
    f, _ = mock_hand.set_targets(np.full(12, 5.0), 0.0)  # out-of-range input is clipped
    assert np.all(f <= 1.0)


def test_inverted_servo_maps_toward_lower_ticks(hand_cfg):
    raw = load_yaml(hand_cfg.source)
    raw["servos"][0].update(slack=2500, tight=1900, inverted=True)
    cfg = hand_config_from_dict(raw)
    h = MockHand(cfg, latency_s=0)
    h.connect()
    only_first = np.zeros(12)
    only_first[0] = 1.0  # one tendon, so the antagonist limit does not scale it
    for _ in range(10):
        h.set_targets(only_first, 0)
    assert h.bus_sim.servos[cfg.servos[0].id].goal == 1900
    assert h.ticks_to_fingers(np.array([1900.0] + [2048.0] * 11))[0] == pytest.approx(1.0)
    h.close()


def test_get_state_normalizes_and_reports_voltage(mock_hand):
    pull = np.array([0.5 if s.role != "extend" else 0.0 for s in mock_hand.cfg.servos])  # one side of each pair
    for _ in range(15):
        mock_hand.set_targets(pull, 30.0)
    import time

    time.sleep(0.3)
    st = mock_hand.get_state()
    expect = mock_hand.expected_fingers(pull)  # extensors read negative: they paid out for the flexors
    assert np.allclose(st.fingers, expect, atol=0.02) and np.all(expect[pull == 0] < 0)
    assert st.roll_deg == pytest.approx(30.0, abs=0.5)
    assert st.min_voltage == pytest.approx(7.4)
    assert len(st.ticks) == 13


def test_voltage_drop_during_operation_relaxes_and_trips(mock_hand, hand_cfg):
    mock_hand.bus_sim.set_voltage(5.0)
    mock_hand._last_voltage_check = -1e9
    with pytest.raises(SafetyTrip):
        mock_hand.get_state()
    assert not any(s.torque_on for s in mock_hand.bus_sim.servos.values())
    with pytest.raises(SafetyTrip):
        mock_hand.set_targets(np.zeros(12), 0)


def test_stalled_finger_is_backed_off(mock_hand, hand_cfg):
    fist = np.array([1.0 if s.role == "flex" else 0.0 for s in hand_cfg.servos])  # flexors only: no antagonist scaling
    for _ in range(20):
        mock_hand.set_targets(fist, 0)
    mock_hand.bus_sim.servos[1].load_override = 1000
    mock_hand.get_state()
    mock_hand.load_watch._since[0] -= 1.0  # pretend the stall has lasted > 0.5 s
    mock_hand.get_state()
    assert mock_hand._load_cap[0] < 1.0
    f, _ = mock_hand.set_targets(fist, 0)
    assert f[0] < 1.0 and f[1] == pytest.approx(1.0)


def test_relax_turns_all_torque_off(mock_hand):
    mock_hand.relax()
    assert not any(s.torque_on for s in mock_hand.bus_sim.servos.values())


def test_set_id_requires_flag_and_single_servo(hand_cfg):
    from dexkit.hw.feetech_hand import FeetechDriver
    from dexkit.hw.feetech_protocol import FeetechBus

    bus = MockFeetechSerial(mock_servos_for(hand_cfg)[:1])
    d = FeetechDriver(FeetechBus(bus), hand_cfg.register_map())
    with pytest.raises(PermissionError):
        d.set_id(1, 5)
    assert d.set_id(1, 5, i_know=True)
    assert d.ping(5) is not None and d.ping(1) is None


def test_hand_loop_timing_under_50ms_on_mock(mock_hand):
    import time

    t0 = time.perf_counter()
    for _ in range(20):
        mock_hand.set_targets(np.full(12, 0.3), 0)
        mock_hand.get_state()
    assert (time.perf_counter() - t0) / 20 < 0.05
