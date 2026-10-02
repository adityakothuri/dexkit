import time

import numpy as np
import pytest

from dexkit.control.poses import (
    Pose,
    PoseLibrary,
    go_to_pose,
    interpolate,
    min_jerk,
    pose_trajectory,
    resolve_fingers,
)


def test_defaults_load_with_generated_curls(hand_cfg):
    lib = PoseLibrary.load(hand_cfg)
    for name in ["open", "fist", "pinch", "point", "thumbs_up", "ok", "wave_a", "wave_b", "peace", "rock_on",
                 "shaka", "spidey", "finger_gun", "one", "two", "three", "four", "five", "claw", "tripod",
                 "cross_thumb"]:
        assert name in lib
    assert all(f"finger_{i}_curl" in lib for i in range(1, 13))
    assert all(f"{s.name}_only" in lib for s in hand_cfg.servos)
    only = lib["thumb_adduct_only"].fingers
    assert only[hand_cfg.names.index("thumb_adduct")] == 1.0 and only.sum() == 1.0
    # Every shipped pose must respect the flexor/extensor antagonist limit, or the driver
    # would silently scale it and the pose would not look as written.
    for name, pose in lib.poses.items():
        for a, b in hand_cfg.antagonist_pairs():
            assert pose.fingers[a] + pose.fingers[b] <= 1.0 + 1e-9, f"{name}: {hand_cfg.names[a]}+{hand_cfg.names[b]}"

    assert np.all(lib["relax"].fingers == 0)
    roles = [s.role for s in hand_cfg.servos]
    names = hand_cfg.names
    for i in range(12):
        assert lib["fist"].fingers[i] == (1.0 if roles[i] == "flex" else 0.0)
        assert lib["open"].fingers[i] == pytest.approx(0.3 if roles[i] == "extend" else 0.0)
        expect = {"thumb_flex": 0.8, "index_flex": 0.8, "thumb_adduct": 0.6, "index_adduct": 0.6}.get(names[i], 0.0)
        assert lib["pinch"].fingers[i] == pytest.approx(expect)
    point = lib["point"].fingers
    assert point[names.index("index_flex")] == 0 and point[names.index("index_extend")] == 0.5
    assert point[names.index("middle_flex")] == 1.0
    assert lib["finger_3_curl"].fingers.tolist() == [0, 0, 1] + [0] * 9
    with pytest.raises(KeyError):
        lib["nope"]


def test_resolve_fingers_precedence():
    names = [f"s{i}" for i in range(12)]
    groups = ["a"] * 6 + ["b"] * 6
    f = resolve_fingers({"default": 0.1, "b": 0.5, "s11": 0.9}, names, groups)
    assert f[0] == 0.1 and f[6] == 0.5 and f[11] == 0.9
    with pytest.raises(ValueError):
        resolve_fingers({"bogus": 1}, names, groups)


def test_yaml_round_trip(hand_cfg, tmp_path):
    import shutil

    d = tmp_path / "poses"
    shutil.copytree(hand_cfg.source.parent / "poses", d)  # the library is a folder of files
    lib = PoseLibrary.load(hand_cfg, d)
    pose = Pose(np.linspace(0, 1, 12), 12.5)
    lib.save_pose("my_pose", pose)  # goes to custom.yaml
    lib2 = PoseLibrary.load(hand_cfg, d)
    assert np.allclose(lib2["my_pose"].fingers, pose.fingers, atol=1e-4)
    assert lib2["my_pose"].roll == pytest.approx(12.5) and lib2.sources["my_pose"] == "custom"
    assert np.allclose(lib2["pinch"].fingers, lib["pinch"].fingers) and lib2.sources["pinch"] == "grasps"
    with pytest.raises(ValueError, match="defined in"):
        lib2.save_pose("pinch", pose)  # cannot shadow a pose from another file
    (d / "dup.yaml").write_text("poses:\n  fist: {fingers: {default: 0.0}}\n")
    with pytest.raises(ValueError, match="already defined"):
        PoseLibrary.load(hand_cfg, d)


def test_interpolation_endpoints_exact():
    a, b = Pose(np.zeros(12), -10), Pose(np.ones(12), 30)
    for mode in ("linear", "minjerk"):
        traj = pose_trajectory(a, b, 1.0, 20, mode)
        assert len(traj) == 20
        assert np.array_equal(traj[-1].fingers, b.fingers) and traj[-1].roll == 30
        assert np.array_equal(interpolate(a, b, 0.0, mode).fingers, a.fingers)
    assert interpolate(a, b, 0.5).roll == pytest.approx(10)
    assert min_jerk(0.0) == 0 and min_jerk(1.0) == 1 and min_jerk(0.5) == pytest.approx(0.5)


def test_go_to_pose_honors_duration(mock_hand, estop):
    target = Pose(np.full(12, 0.4), 10)
    t0 = time.monotonic()
    go_to_pose(mock_hand, target, duration_s=0.5, rate_hz=40, estop=estop)
    elapsed = time.monotonic() - t0
    assert 0.45 <= elapsed <= 0.8
    f, r = mock_hand.last_command
    assert np.allclose(f, 0.4, atol=0.01) and r == pytest.approx(10, abs=0.1)
