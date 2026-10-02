import numpy as np
import pytest

from dexkit.control.poses import PoseLibrary
from dexkit.control.recorder import Recorder, Recording, replay
from dexkit.control.teleop import KeyEvent, ScriptedKeys, TeleopController
from dexkit.env.dexkit_env import DexKitEnv


def test_recording_round_trip(isolated_data):
    r = Recorder()
    r.start()
    for i in range(5):
        r.add(np.full(16, i, dtype=float), np.full(16, -i, dtype=float), t=r._t0 + i * 0.05)
    path = r.stop_and_save("unit")
    rec = Recording.load("unit")
    assert path.exists() and len(rec) == 5
    assert rec.action[4, 0] == 4 and rec.state[4, 0] == -4
    assert rec.duration == pytest.approx(0.2)


def test_replay_streams_all_frames(mock_hand, mock_gantry, estop):
    mock_gantry.set_zero()
    env = DexKitEnv(mock_hand, mock_gantry, estop=estop)
    n = 10
    t = np.arange(n) * 0.02
    act = np.zeros((n, 16))
    act[:, :12] = np.linspace(0, 0.5, n)[:, None]
    act[:, 13] = np.linspace(0, 5, n)
    rec = Recording(t=t, action=act, state=act.copy())
    assert replay(env, rec, speed=2.0, lead_in_s=0.1, rate_hz=50) == n
    f, _ = mock_hand.last_command
    assert np.allclose(f, 0.5, atol=0.02)


def test_teleop_controller_keys(hand_cfg, gantry_cfg, mock_hand, mock_gantry, estop):
    lib = PoseLibrary.load(hand_cfg)
    c = TeleopController(mock_hand, mock_gantry, hand_cfg, gantry_cfg, lib, estop)
    c.handle(KeyEvent("3"))
    assert c.selected == 2
    c.handle(KeyEvent("]"))
    c.handle(KeyEvent("]"))
    assert c.fingers[2] == pytest.approx(0.1)
    c.handle(KeyEvent("="))
    assert c.selected == 11
    c.handle(KeyEvent("."))
    assert c.roll == pytest.approx(5.0)
    c.handle(KeyEvent("d"))
    c.handle(KeyEvent("D"))
    mock_gantry.wait_idle()
    assert mock_gantry.get_state().mpos[0] == pytest.approx(11.0, abs=0.01)
    c.handle(KeyEvent("d", pressed=False))
    assert mock_gantry.calls[-1] == b"\x85"
    c.handle(KeyEvent("z"))
    assert mock_gantry.frame_valid
    c.handle(KeyEvent("f"))
    for _ in range(25):
        c.tick(poll_gantry=False)
    assert np.allclose(c.fingers, 1.0)
    c.handle(KeyEvent("r"))
    assert c.recorder.active
    c.handle(KeyEvent("space"))
    assert estop.tripped
    c.handle(KeyEvent("]"))  # ignored after e-stop
    c.handle(KeyEvent("esc"))
    assert c.quit


def test_scripted_keys():
    clock = iter([0.0, 0.0, 0.5, 1.5]).__next__
    k = ScriptedKeys([(1.0, "a"), (0.2, "b")], clock=clock)
    assert [e.key for e in k.poll()] == []
    assert [e.key for e in k.poll()] == ["b"]
    assert [e.key for e in k.poll()] == ["a"]
