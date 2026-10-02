"""Synthetic end-to-end check of the policy pipeline (make policy-smoke).

10 fake episodes x 100 steps (smooth random walks in 16-dim, mock camera frames
whose square tracks the gantry XY), train 2 epochs on CPU, assert the loss
falls, then run `dexkit-infer --mock` for 20 steps.

Runs on a GPU (CUDA, or MPS on Apple Silicon). CPU is refused unless --allow-cpu:
ResNet18 training/inference on CPU is far too slow to be a useful check.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np

from dexkit.config import load_gantry_config, load_hand_config
from dexkit.hw.base import GANTRY_SLICE
from dexkit.hw.mock import MockCamera
from dexkit.policy.config import build_stats, load_policy_config
from dexkit.policy.normalize import normalize_data
from dexkit.util import setup_logging

log = logging.getLogger(__name__)


def make_synthetic_episodes(out: Path, stats: dict, n_episodes: int = 10, steps: int = 100, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    lo, hi = stats["min"], stats["max"]
    out.mkdir(parents=True, exist_ok=True)
    cam = MockCamera()
    for e in range(n_episodes):
        x = rng.uniform(-0.5, 0.5, size=16)
        v = np.zeros(16)
        states = []
        for _ in range(steps + 1):
            v = 0.9 * v + rng.normal(0, 0.02, size=16)
            x = np.clip(x + v, -1, 1)
            states.append(lo + (x + 1) / 2 * (hi - lo))
        states = np.array(states)
        imgs = []
        for s in states[:-1]:
            cam.set_target(normalize_data(s, stats)[GANTRY_SLICE][:2])
            imgs.append(cam.read())
        with open(out / f"episode_{e:04d}.pkl", "wb") as f:
            pickle.dump({
                "images": np.stack(imgs).astype(np.uint8),
                "agent_pos": states[:-1],
                "action": states[1:],  # next state as action, as CMU did
                "episode_ends": np.array([steps]),
            }, f)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--allow-cpu", action="store_true", help="permit running on CPU (slow)")
    p.add_argument("--infer-steps", type=int, default=20)
    p.add_argument("--workdir", default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)

    import torch

    from dexkit.policy.infer import main as infer_main
    from dexkit.policy.nets import pick_device
    from dexkit.policy.train import train

    device = pick_device(args.device)
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU (cuda/mps) found; refusing to train on CPU. Pass --allow-cpu to force.")
    log.info("device: %s", device)

    torch.manual_seed(0)
    pcfg = load_policy_config()
    stats = build_stats(load_hand_config(), load_gantry_config())
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="dexkit_smoke_"))
    demos = work / "demos"
    t0 = time.monotonic()
    make_synthetic_episodes(demos, stats, args.episodes, args.steps)
    log.info("synthetic demos in %s (%.1fs)", demos, time.monotonic() - t0)

    ckpt = work / "smoke.pt"
    history = train([demos], ckpt, pcfg, stats, epochs=args.epochs, device=device,
                    batch_size=args.batch_size, warmup_steps=10, checkpoint_every=args.epochs)
    first = float(np.mean(history[0][:3]))
    last = float(np.mean(history[-1][-3:]))
    print(f"loss: first batches {first:.4f} -> last batches {last:.4f}")
    if not last < first:
        raise SystemExit("FAIL: loss did not decrease")

    infer_main(["--mock", "--yes", "--checkpoint", str(ckpt), "--steps", str(args.infer_steps),
                "--device", device.type])
    print(f"policy smoke test PASSED in {time.monotonic() - t0:.0f}s (workdir {work})")


if __name__ == "__main__":
    main()
