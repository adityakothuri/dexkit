import numpy as np
import pytest

from dexkit.hw.safety import (
    EStop,
    EStopTripped,
    LoadWatch,
    SafetyTrip,
    TempWatch,
    TickClamp,
    TravelBox,
    VoltageGate,
)


def test_tick_clamp_limits_and_delta():
    c = TickClamp(lo=[1000, 2000], hi=[2000, 3000], max_delta=[50, 100])
    assert c.clamp(np.array([500, 3500])).tolist() == [1000, 3000]
    out = c.apply(np.array([2000, 3000]), previous=np.array([1500, 2500]))
    assert out.tolist() == [1550, 2600]
    out = c.apply(np.array([1000, 2000]), previous=np.array([1500, 2500]))
    assert out.tolist() == [1450, 2400]


def test_tick_clamp_walks_back_into_range_without_jumping():
    c = TickClamp(lo=[1000], hi=[2000], max_delta=[100])
    # Previous command outside range (e.g. initial position): move toward range by max_delta only.
    assert c.apply(np.array([1500]), previous=np.array([2500])).tolist() == [2400]


def test_tick_clamp_rejects_bad_limits():
    with pytest.raises(ValueError):
        TickClamp(lo=[10], hi=[5], max_delta=[1])


def test_voltage_gate_both_directions():
    g = VoltageGate((6.0, 8.4))
    g.check({1: 7.4, 2: 6.0, 3: 8.4})
    with pytest.raises(SafetyTrip, match="12.0"):
        g.check({1: 7.4, 2: 12.0})  # Drok PSU left at 12 V
    with pytest.raises(SafetyTrip):
        g.check({1: 5.5})
    with pytest.raises(SafetyTrip, match="no voltage"):
        g.check({1: 7.4}, expected_ids=[1, 2])
    with pytest.raises(SafetyTrip):
        g.check({})


def test_temp_watch():
    TempWatch(65).check({1: 64.0})
    with pytest.raises(SafetyTrip):
        TempWatch(65).check({1: 70.0})


def test_load_watch_needs_sustained_stall():
    w = LoadWatch([500, 500], stall_time_s=0.5)
    assert w.update(np.array([600, 100]), now=0.0) == []
    assert w.update(np.array([600, 100]), now=0.3) == []
    assert w.update(np.array([600, 100]), now=0.6) == [0]
    assert w.update(np.array([100, 100]), now=0.7) == []
    assert w.update(np.array([600, 100]), now=0.8) == []  # timer restarted


def test_travel_box_edges():
    b = TravelBox([0, 0, -40], [290, 170, 0])
    b.check(0, 0, -40)
    b.check(290, 170, 0)
    for bad in [(-0.01, 0, 0), (0, 170.01, 0), (0, 0, 0.5), (0, 0, -40.1)]:
        with pytest.raises(SafetyTrip):
            b.check(*bad)
    assert b.clamp_jog([285, 0, 0], [10, 0, 0]).tolist() == [5, 0, 0]
    assert b.clamp_jog([290, 0, 0], [10, 0, 0]).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        TravelBox([0, 0, 0], [0, 1, 1])


class FakeGantry:
    def __init__(self, log):
        self.log = log

    def feed_hold(self):
        self.log.append("!")

    def jog_cancel(self):
        self.log.append("0x85")

    def soft_reset(self, wait=True):
        self.log.append("0x18")


class FakeHand:
    def __init__(self, log, fail=False):
        self.log, self.fail = log, fail

    def relax(self):
        self.log.append("torque_off")


def test_estop_ordering_and_idempotence():
    calls = []
    e = EStop()
    e.register(hand=FakeHand(calls), gantry=FakeGantry(calls))
    e.check()
    e.trip("operator")
    assert calls == ["!", "0x85", "0x18", "torque_off"]
    assert e.tripped and e.reason == "operator"
    e.trip("second")
    assert e.reason == "operator"
    with pytest.raises(EStopTripped):
        e.check()


def test_estop_continues_after_failing_step():
    calls = []

    class Broken(FakeGantry):
        def feed_hold(self):
            raise OSError("port gone")

    e = EStop()
    e.register(hand=FakeHand(calls), gantry=Broken(calls))
    e.trip("test")
    assert calls == ["0x85", "0x18", "torque_off"]
    assert "gantry.feed_hold:failed" in e.log


def test_estop_on_real_mock_drivers(mock_hand, mock_gantry):
    e = EStop()
    e.register(hand=mock_hand, gantry=mock_gantry)
    mock_gantry.jog(20, 0, 0)
    n = len(mock_gantry.calls)
    e.trip("test")
    rt = [c for c in mock_gantry.calls[n:] if c in (b"!", b"\x85", b"\x18")]
    assert rt == [b"!", b"\x85", b"\x18"]
    assert not any(s.torque_on for s in mock_hand.bus_sim.servos.values())
    assert not mock_gantry.frame_valid
