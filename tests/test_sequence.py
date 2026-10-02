import pytest

from dexkit.control.poses import PoseLibrary
from dexkit.control.sequence import SequenceError, execute, validate
from dexkit.hw.safety import TravelBox

BOX = TravelBox([0, 0, -40], [290, 170, 0])


def seq(*steps):
    return {"name": "t", "steps": list(steps)}


def test_valid_sequence_resolves_partial_targets(hand_cfg):
    lib = PoseLibrary.load(hand_cfg)
    s = validate(seq({"pose": "open"}, {"gantry": {"x": 50, "y": 30, "z": 0}, "feed": 800},
                     {"gantry": {"z": -20}, "feed": 300}, {"roll": 45}, {"wait": 0.5}), lib, BOX)
    assert [st.kind for st in s.steps] == ["pose", "gantry", "gantry", "roll", "wait"]
    assert s.steps[2].xyz == (50, 30, -20)


def test_rejects_unknown_pose(hand_cfg):
    with pytest.raises(SequenceError, match="unknown pose 'flex'"):
        validate(seq({"pose": "flex"}), PoseLibrary.load(hand_cfg), BOX)


def test_rejects_gantry_target_outside_box(hand_cfg):
    with pytest.raises(SequenceError, match="outside travel box"):
        validate(seq({"gantry": {"x": 500}}), PoseLibrary.load(hand_cfg), BOX)


def test_rejects_partial_target_that_leaves_box_from_start(hand_cfg):
    with pytest.raises(SequenceError, match="outside"):
        validate(seq({"gantry": {"y": 10}}), PoseLibrary.load(hand_cfg), BOX, start_xyz=(-5, 0, 0))


def test_reports_all_errors_and_bad_steps(hand_cfg):
    with pytest.raises(SequenceError) as e:
        validate(seq({"pose": "x"}, {"gantry": {"z": 10}}, {"jump": 1}, {"roll": 500},
                     {"gantry": {"x": 1}, "feed": 99999}), PoseLibrary.load(hand_cfg), BOX)
    msg = str(e.value)
    for frag in ["step 1", "step 2", "step 3", "step 4", "exceeds feed_max"]:
        assert frag in msg


def test_gantry_step_without_gantry(hand_cfg):
    with pytest.raises(SequenceError, match="no gantry"):
        validate(seq({"gantry": {"x": 1}}), PoseLibrary.load(hand_cfg), None)


def test_execute_on_mocks(hand_cfg, mock_hand, mock_gantry, estop):
    import numpy as np

    lib = PoseLibrary.load(hand_cfg)
    mock_gantry.set_zero()
    s = validate(seq({"pose": "fist", "duration": 0.2}, {"gantry": {"x": 10, "y": 5, "z": -2}, "feed": 1500},
                     {"roll": 20, "duration": 0.2}, {"wait": 0.1}), lib, BOX)
    execute(s, mock_hand, mock_gantry, lib, rate_hz=50, estop=estop)
    f, r = mock_hand.last_command
    assert np.allclose(f, 1.0) and r == pytest.approx(20, abs=0.1)
    assert np.allclose(mock_gantry.get_state().xyz, [10, 5, -2], atol=0.1)
