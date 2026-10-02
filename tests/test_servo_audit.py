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
    go_to_pose(mock_hand, Pose(f, 0.0), 0.2, 50, estop=estop, rate=fast, arrive_timeout_s=0)  # settle loop only
    moves = len(mock_hand.calls) - before  # 10 trajectory ticks + a few slew-limited settle ticks
    assert moves < 40, f"settle loop sent {moves} packets: it ran to its cap instead of stopping once nothing changes"
    got, _ = mock_hand.last_command
    assert got[names.index("index_flex")] + got[names.index("index_extend")] == pytest.approx(1.0, abs=0.02)  # tick rounding


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


def test_go_to_pose_waits_for_servos_to_arrive(hand_cfg, estop):
    """Sequences must not start the next step before the servos have physically reached the pose."""
    from dexkit.hw.mock import MockFeetechSerial, MockHand, mock_servos_for

    servos = mock_servos_for(hand_cfg)
    for s in servos:
        s.speed_tps = 1500.0  # slow simulated servos
    h = MockHand(hand_cfg, latency_s=0, bus=MockFeetechSerial(servos))
    h.connect()
    go_to_pose(h, Pose(_fist(hand_cfg), 0.0), 0.1, 50, estop=estop, rate=Rate(1000, sleep=lambda _s: None))
    st = h.get_state()
    assert np.max(np.abs(np.clip(st.fingers, 0, 1) - _fist(hand_cfg))) < 0.06
    h.close()


def test_speed_scale_slows_everything(hand_cfg):
    from dexkit.cli import apply_speed_scale

    speed, accel = hand_cfg.defaults.speed, hand_cfg.defaults.accel
    steps = [s.max_delta_ticks for s in hand_cfg.servos] + [hand_cfg.roll.max_delta_ticks]
    apply_speed_scale(hand_cfg, 0.5)
    assert hand_cfg.defaults.speed == speed // 2 and hand_cfg.defaults.accel == accel // 2
    after = [s.max_delta_ticks for s in hand_cfg.servos] + [hand_cfg.roll.max_delta_ticks]
    assert after == [max(2, b // 2) for b in steps]
    with pytest.raises(ValueError):
        apply_speed_scale(hand_cfg, 3.0)


def test_antagonist_pays_out_when_the_other_side_pulls(hand_cfg):
    """fist must unwind the extensors by the flexors' travel (capped), or they brake the fingers."""
    cfg = hand_cfg
    cfg.defaults.antagonist_payout, cfg.defaults.antagonist_payout_max_ticks = 1.0, 1000
    h = _hand(cfg)
    names = cfg.names
    fi, ei = names.index("index_flex"), names.index("index_extend")
    for _ in range(20):
        h.set_targets(_fist(cfg), 0.0)
    flex, ext = cfg.servos[fi], cfg.servos[ei]
    assert h.bus_sim.servos[flex.id].goal == flex.tight
    assert h.bus_sim.servos[ext.id].goal == ext.slack - min(abs(flex.span), 1000)  # paid out past slack
    want = h.expected_fingers(_fist(cfg))
    assert want[fi] == pytest.approx(1.0) and want[ei] < 0
    # and back to relaxed: everything returns to slack
    for _ in range(20):
        h.set_targets(np.zeros(12), 0.0)
    assert h.bus_sim.servos[ext.id].goal == ext.slack and h.bus_sim.servos[flex.id].goal == flex.slack
    h.close()


def test_payout_is_capped_for_long_take_up_spans(hand_cfg):
    cfg = hand_cfg
    cfg.defaults.antagonist_payout_max_ticks = 300
    names = cfg.names
    fi, ei = names.index("pinky_flex"), names.index("pinky_extend")
    cfg.servos[fi].tight = cfg.servos[fi].slack + 8000  # a 2-turn flexor like the bench pinky
    h = _hand(cfg)
    only = np.zeros(12)
    only[fi] = 1.0
    for _ in range(120):
        h.set_targets(only, 0.0)
    assert h.bus_sim.servos[cfg.servos[ei].id].goal == cfg.servos[ei].slack - 300
    h.close()


def test_disabled_servos_are_never_commanded_or_energised(hand_cfg):
    """Extensors switched off in hand.yaml: torque stays off, no goal is ever written, state reads 0."""
    from dexkit.hw.feetech_protocol import INST_SYNC_WRITE

    cfg = hand_cfg
    off = [s for s in cfg.servos if s.role == "extend"]
    for s in off:
        s.enabled = False
    assert cfg.antagonist_pairs() == []
    h = _hand(cfg)
    assert all(not h.bus_sim.servos[s.id].torque_on for s in off)
    assert all(h.bus_sim.servos[s.id].torque_on for s in cfg.servos if s.enabled)
    for _ in range(15):
        f, _ = h.set_targets(np.ones(12), 0.0)  # ask for everything, including the disabled ones
    assert all(f[i] == 0.0 for i, s in enumerate(cfg.servos) if not s.enabled)
    goal_addr = cfg.register_map()["goal_position"]
    for pkt in h.calls:
        if len(pkt) > 6 and pkt[4] == INST_SYNC_WRITE and pkt[5] == goal_addr:
            written = {pkt[7 + 3 * k] for k in range((len(pkt) - 8) // 3)}
            assert not (written & {s.id for s in off}), "a disabled servo received a goal position"
    import time

    time.sleep(0.25)  # let the simulated servos arrive
    st = h.get_state()
    assert all(st.fingers[i] == 0.0 for i, s in enumerate(cfg.servos) if not s.enabled)
    assert all(st.fingers[i] > 0.9 for i, s in enumerate(cfg.servos) if s.enabled)
    h.close()


def _session(cfg, start):
    """One connect/close cycle on mock servos starting at `start`; returns the hand (closed)."""
    from dexkit.hw.mock import MockFeetechSerial, MockHand, mock_servos_for

    h = MockHand(cfg, latency_s=0, bus=MockFeetechSerial(mock_servos_for(cfg, start_ticks=start)))
    h.connect()
    return h


def test_open_position_survives_a_fist_and_a_power_cycle(hand_cfg):
    """Session 1 calibrates (slack known), session 2 ends with fingers curled, session 3 starts after a
    PSU cycle: 'open' must still be the real open, not wherever the curled fingers are."""
    import time

    from dexkit.hw.feetech_hand import save_positions_state

    cfg = hand_cfg
    s0 = cfg.servos[0]
    save_positions_state({s.id: {"pos": s.slack, "slack": s.slack} for s in cfg.servos}, roll_center=cfg.roll.center)
    # session 2: make a fist, leave it curled, close
    h = _session(cfg, None)
    only = np.zeros(12)
    only[0] = 1.0
    for _ in range(20):
        h.set_targets(only, 0.0)
    time.sleep(0.2)
    h.close()  # saves pos ~= tight (2700), slack 2048
    # power cycle: the count collapses to turn 0 -> same angle, but let's put it on another turn
    curled = s0.tight + 2 * 4096  # the servo "woke up" reading this
    h = _session(cfg, {s0.id: curled})
    assert s0.name in h.restore_report["restored"]
    assert h._slack[0] == s0.slack + 2 * 4096  # open recovered on the right turn
    for _ in range(20):
        h.set_targets(np.zeros(12), 0.0)
    assert h.bus_sim.servos[s0.id].goal == s0.slack + 2 * 4096  # 'relax' really goes back to open
    h.close()


def test_open_position_falls_back_when_moved_by_hand_too_far(hand_cfg):
    from dexkit.hw.feetech_hand import save_positions_state

    cfg = hand_cfg
    s0 = cfg.servos[0]
    save_positions_state({s.id: {"pos": s.slack, "slack": s.slack} for s in cfg.servos}, roll_center=cfg.roll.center)
    h = _session(cfg, {s0.id: s0.slack + 2000})  # moved by hand ~half a turn while off: ambiguous
    assert s0.name in h.restore_report["assumed"] and h._slack[0] == s0.slack + 2000
    assert all(n in h.restore_report["restored"] for n in cfg.names[1:])
    h.close()


def test_rehome_overrides_restored_open(hand_cfg):
    from dexkit.hw.feetech_hand import load_positions_state, save_positions_state

    cfg = hand_cfg
    save_positions_state({s.id: {"pos": s.slack, "slack": s.slack} for s in cfg.servos}, roll_center=cfg.roll.center)
    h = _session(cfg, {cfg.servos[0].id: cfg.servos[0].slack + 300})  # within drift: restored to slack
    assert h._slack[0] == cfg.servos[0].slack
    h.relax()
    h.rehome()  # operator says: where it is now IS open
    assert h._slack[0] == cfg.servos[0].slack + 300
    assert load_positions_state()["servos"][str(cfg.servos[0].id)]["slack"] == cfg.servos[0].slack + 300
    h.close()
