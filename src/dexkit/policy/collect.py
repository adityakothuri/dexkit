"""Collect demos: keyboard teleop with the camera on; one episode per 'r' toggle.

    dexkit-collect --task <name>

Writes data/demos/<task>/episode_XXXX.pkl with the CMU-style schema:
  images (N,240,320,3) uint8 RGB, agent_pos (N,16), action (N,16), episode_ends [N]
Frames are kept at policy rate_hz (subsampled from the teleop loop).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from dexkit.config import data_dir

log = logging.getLogger(__name__)


class EpisodeWriter:
    def __init__(self, out_dir: Path, camera, keep_every: int = 1) -> None:
        self.out_dir = out_dir
        self.camera = camera
        self.keep_every = max(1, keep_every)
        self.recording = False
        self._tick = 0
        self._imgs: list[np.ndarray] = []
        self._pos: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self.saved: list[Path] = []

    def per_tick(self, action: np.ndarray, state: np.ndarray) -> None:
        if not self.recording:
            return
        self._tick += 1
        if (self._tick - 1) % self.keep_every:
            return
        self._imgs.append(self.camera.read())
        self._pos.append(np.asarray(state, dtype=np.float64))
        self._act.append(np.asarray(action, dtype=np.float64))

    def start(self) -> None:
        self.recording = True
        self._tick = 0
        self._imgs, self._pos, self._act = [], [], []

    def stop(self) -> Path | None:
        self.recording = False
        n = len(self._pos)
        if n < 2:
            log.warning("episode too short (%d frames); discarded", n)
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(self.out_dir.glob("episode_*.pkl")))
        path = self.out_dir / f"episode_{idx:04d}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "images": np.stack(self._imgs).astype(np.uint8),
                "agent_pos": np.stack(self._pos),
                "action": np.stack(self._act),
                "episode_ends": np.array([n]),
            }, f)
        self.saved.append(path)
        log.info("saved %s (%d frames)", path, n)
        return path


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import add_gantry_flags, common_parser, init, open_session
    from dexkit.config import load_policy_config
    from dexkit.control.recorder import Recorder
    from dexkit.control.teleop import make_keys, run_teleop
    from dexkit.hw.camera import Camera
    from dexkit.hw.mock import MockCamera

    p = common_parser(__doc__ or "")
    add_gantry_flags(p)
    p.add_argument("--task", required=True)
    p.add_argument("--keys", choices=["auto", "terminal", "pynput", "none"], default="auto")
    p.add_argument("--script", help="scripted keys 't:key,...'")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--camera", default=None)
    args = p.parse_args(argv)
    init(args)
    pcfg = load_policy_config()
    out_dir = data_dir() / pcfg.paths.get("demos", "demos") / args.task
    camera = MockCamera() if args.mock else Camera(args.camera if args.camera is not None else pcfg.camera_index)
    with open_session(args, need_hand=True, want_gantry=True) as s:
        assert s.hand_cfg is not None
        writer = EpisodeWriter(out_dir, camera, keep_every=int(round(s.hand_cfg.rate_hz / pcfg.rate_hz)))

        recorder = Recorder()

        def per_tick(action: np.ndarray, state: np.ndarray) -> None:
            if recorder.active and not writer.recording:
                writer.start()
            writer.per_tick(action, state)

        def on_record_stop(rec: Recorder) -> None:
            rec.stop()
            writer.stop()

        keys = make_keys(args.keys, args.script)
        run_teleop(s, keys, duration=args.duration, per_tick=per_tick, on_record_stop=on_record_stop,
                   recorder=recorder)
        print(f"saved {len(writer.saved)} episode(s) to {out_dir}")
    camera.close()


if __name__ == "__main__":
    main()
