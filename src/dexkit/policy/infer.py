"""Run a trained diffusion policy on the hand (ported from diff_foam inference.ipynb).

    dexkit-infer --checkpoint data/checkpoints/<task>.pt [--num-inference-steps 16 --scheduler ddim]

Loads EMA weights, keeps a 2-frame observation deque, denoises a 16-step action
chunk, executes 8 of them through DexKitEnv, repeats. FoamEnv/ROS/taichi removed.
"""

from __future__ import annotations

import collections
import logging
import time

import numpy as np
import torch

from dexkit.config import PolicyConfig
from dexkit.env.dexkit_env import DexKitEnv
from dexkit.hw.base import GANTRY_SLICE
from dexkit.hw.camera import preprocess_image
from dexkit.policy.nets import build_nets, make_scheduler, pick_device, predict_actions
from dexkit.policy.normalize import normalize_data, unnormalize_data
from dexkit.util import Rate

log = logging.getLogger(__name__)


def load_policy(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    pcfg = PolicyConfig(**ckpt["policy_config"])
    nets = build_nets(pcfg).to(device)
    nets.load_state_dict(ckpt.get("ema_state_dict") or ckpt["model_state_dict"])
    nets.eval()
    stats = {k: np.asarray(v, dtype=np.float64) for k, v in ckpt["stats"].items()}
    return nets, pcfg, stats


def run_policy(env: DexKitEnv, nets, pcfg: PolicyConfig, stats: dict[str, np.ndarray], device: torch.device,
               steps: int, num_inference_steps: int, scheduler_kind: str = "ddpm",
               box: tuple[np.ndarray, np.ndarray] | None = None, on_obs=None) -> int:
    scheduler = make_scheduler(pcfg, scheduler_kind)
    obs = env.observe()
    if obs["image"] is None:
        raise RuntimeError("policy needs a camera image")
    deque: collections.deque = collections.deque([obs] * pcfg.obs_horizon, maxlen=pcfg.obs_horizon)
    rate = Rate(pcfg.rate_hz)
    done = 0
    while done < steps:
        images = np.stack([preprocess_image(o["image"], tuple(pcfg.crop_size)) for o in deque])
        agent = np.stack([normalize_data(o["agent_pos"], stats) for o in deque])
        nimage = torch.from_numpy(images).unsqueeze(0).to(device)
        nagent = torch.from_numpy(agent).float().unsqueeze(0).to(device)
        t0 = time.perf_counter()
        naction = predict_actions(nets, scheduler, pcfg, nimage, nagent, num_inference_steps)
        log.debug("denoise %.0f ms", (time.perf_counter() - t0) * 1000)
        actions = unnormalize_data(naction, stats)
        start = pcfg.obs_horizon - 1
        for a in actions[start : start + pcfg.action_horizon]:
            a = a.copy()
            a[:12] = np.clip(a[:12], 0, 1)
            if box is not None:
                a[GANTRY_SLICE] = np.clip(a[GANTRY_SLICE], box[0], box[1])
            obs = env.step(a)
            if on_obs:
                on_obs(obs)
            deque.append(obs)
            done += 1
            rate.sleep()
            if done >= steps:
                break
    return done


def main(argv: list[str] | None = None) -> None:
    from dexkit.cli import add_gantry_flags, common_parser, init, open_session
    from dexkit.hw.camera import Camera
    from dexkit.hw.mock import MockCamera

    p = common_parser(__doc__ or "")
    add_gantry_flags(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=200, help="actions to execute before stopping")
    p.add_argument("--num-inference-steps", type=int, default=None, help="default: training diffusion steps")
    p.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--device", default="auto")
    p.add_argument("--camera", default=None, help="camera index or path (default from policy.yaml)")
    args = p.parse_args(argv)
    init(args)
    device = pick_device(args.device)
    nets, pcfg, stats = load_policy(args.checkpoint, device)
    k = args.num_inference_steps or pcfg.num_diffusion_iters
    if args.scheduler == "ddpm" and k != pcfg.num_diffusion_iters:
        log.warning("DDPM with %d < %d steps; prefer --scheduler ddim for fewer steps", k, pcfg.num_diffusion_iters)

    camera = MockCamera() if args.mock else Camera(args.camera if args.camera is not None else pcfg.camera_index)
    with open_session(args, need_hand=True, want_gantry=True) as s:
        assert s.hand is not None
        gantry = s.gantry if s.gantry is not None and s.gantry.frame_valid else None
        box = (np.asarray(s.gantry_cfg.travel_min), np.asarray(s.gantry_cfg.travel_max)) if gantry and s.gantry_cfg else None
        feed = s.gantry_cfg.feed_max_mm_min if s.gantry_cfg else None
        env = DexKitEnv(s.hand, gantry, camera=camera, estop=s.estop, gantry_feed=feed)
        n = run_policy(env, nets, pcfg, stats, device, args.steps, k, args.scheduler, box=box)
        print(f"policy executed {n} steps")
    camera.close()


if __name__ == "__main__":
    main()
