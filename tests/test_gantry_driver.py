import numpy as np
import pytest

from dexkit.hw.grbl_gantry import GrblError
from dexkit.hw.mock import MockGantry, MockGrblSerial
from dexkit.hw.safety import SafetyTrip


def test_connect_reads_banner_and_settings(mock_gantry):
    assert mock_gantry.version == "1.1f" and not mock_gantry.legacy
    assert mock_gantry.settings[130] == 300
    assert "$$" in mock_gantry.sim.lines


def test_status_mask_written_only_when_different(gantry_cfg):
    sim = MockGrblSerial(speed_factor=50)
    sim.settings[10] = 0
    g = MockGantry(gantry_cfg, sim=sim)
    g.connect()
    assert "$10=1" in sim.lines
    sim2 = MockGrblSerial(speed_factor=50)
    g2 = MockGantry(gantry_cfg, sim=sim2)
    g2.connect()
    assert "$10=1" not in sim2.lines


def test_motion_refused_until_zeroed(mock_gantry):
    assert not mock_gantry.frame_valid
    with pytest.raises(SafetyTrip, match="frame"):
        mock_gantry.move_to(10, 10, -5)
    mock_gantry.set_zero()
    assert mock_gantry.frame_valid
    assert "G92 X0 Y0 Z0" in mock_gantry.sim.lines


def test_move_square_returns_to_zero(mock_gantry):
    mock_gantry.jog(30, 40, -10)  # jogging allowed before zero
    mock_gantry.wait_idle()
    mock_gantry.set_zero()
    for x, y in [(20, 0), (20, 20), (0, 20), (0, 0)]:
        mock_gantry.move_to(x, y, 0, feed=500, wait=True)
    st = mock_gantry.get_state()
    assert np.allclose(st.xyz, 0, atol=0.1)
    assert np.allclose(st.mpos, [30, 40, -10], atol=0.1)


def test_travel_box_and_feed_cap(mock_gantry, gantry_cfg):
    mock_gantry.set_zero()
    with pytest.raises(SafetyTrip, match="outside"):
        mock_gantry.move_to(1000, 0, 0)
    mock_gantry.move_to(10, 10, -1, feed=99999, wait=True)
    assert any(f"F{gantry_cfg.feed_max_mm_min:.0f}" in line for line in mock_gantry.sim.lines)


def test_jog_clamped_to_box_after_zero(mock_gantry):
    mock_gantry.set_zero()
    assert mock_gantry.jog(0, 0, 5) is False  # z max is 0
    assert mock_gantry.jog(-5, 0, 0) is False
    assert mock_gantry.jog(5, 0, 0) is True
    st = mock_gantry.wait_idle()
    assert st.xyz[0] == pytest.approx(5.0, abs=0.01)


def test_jog_cancel_stops_motion(gantry_cfg):
    g = MockGantry(gantry_cfg, speed_factor=1.0)
    g.connect()
    g.jog(100, 0, 0, feed=600)
    assert g.get_state().state == "Jog"
    g.jog_cancel()
    assert g.get_state().state == "Idle"


def test_feed_hold_resume(gantry_cfg):
    g = MockGantry(gantry_cfg, speed_factor=1.0)
    g.connect()
    g.set_zero()
    g.move_to(100, 0, 0, feed=600)
    g.feed_hold()
    st = g.get_state()
    assert st.state == "Hold"
    x = st.xyz[0]
    import time

    time.sleep(0.1)
    assert g.get_state().xyz[0] == pytest.approx(x)
    g.resume()
    time.sleep(0.05)
    assert g.get_state().state == "Run"


def test_soft_reset_invalidates_frame_and_unlocks(mock_gantry):
    mock_gantry.set_zero()
    mock_gantry.move_to(50, 0, 0)
    mock_gantry.soft_reset()  # mid-motion reset -> GRBL alarm -> driver sends $X
    assert not mock_gantry.frame_valid
    assert "$X" in mock_gantry.sim.lines
    assert mock_gantry.get_state().state == "Idle"


def test_homing_mode(gantry_cfg):
    gantry_cfg.zeroing = "home"
    g = MockGantry(gantry_cfg, homing=True, speed_factor=50)
    g.connect()
    g.home()
    assert g.frame_valid and "$H" in g.sim.lines
    g2 = MockGantry(gantry_cfg, homing=False, speed_factor=50)
    g2.connect()
    with pytest.raises(GrblError, match="homing"):
        g2.home()


def test_grbl_09_falls_back_to_g91(gantry_cfg):
    sim = MockGrblSerial(version="0.9j", speed_factor=50)
    g = MockGantry(gantry_cfg, sim=sim)
    g.connect()
    assert g.legacy
    g.jog(2, 0, 0)
    assert any(line.startswith("G91 G0") for line in sim.lines)
    assert not any(line.startswith("$J=") for line in sim.lines)


def test_restore_frame(mock_gantry):
    mock_gantry.restore_frame((12.0, 3.0, -1.0))
    assert mock_gantry.frame_valid
    assert np.allclose(mock_gantry.get_state().xyz, [12, 3, -1], atol=1e-6)


def test_streaming_guard_skips_when_planner_full(mock_gantry, gantry_cfg):
    mock_gantry.set_zero()
    from dexkit.hw.grbl_protocol import StatusReport

    mock_gantry._last_status = StatusReport(state="Run", planner_free=1)
    assert mock_gantry.move_to(1, 1, -1) is False
    mock_gantry._last_status = StatusReport(state="Run", planner_free=10)
    assert mock_gantry.move_to(1, 1, -1) is True
