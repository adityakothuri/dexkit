"""Train the diffusion policy on collected demos (ported from diff_foam training_object.ipynb).

    dexkit-train --data data/demos/<task> --task <task> [--device cuda] [--epochs 100]

Keeps CMU's settings: pred 16 / obs 2 / action 8, DDPM 100 steps, squaredcos_cap_v2,
clip_sample, epsilon prediction, EMA power 0.75, AdamW 1e-4 / wd 1e-6, cosine LR
with 500 warmup steps, batch 64. Adds checkpoint-every-N, resume, device choice.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dexkit.config import data_dir, load_gantry_config, load_hand_config
from dexkit.policy.config import PolicyConfig, build_stats, load_policy_config
from dexkit.policy.dataset import DexKitDataset
from dexkit.policy.nets import build_nets, encode_obs, make_scheduler, pick_device
from dexkit.util import setup_logging

log = logging.getLogger(__name__)


def make_ema(nets: nn.Module, power: float):
    from diffusers.training_utils import EMAModel

    # diffusers >= 0.20 takes parameters=...; CMU's ema.step(nets) must become ema.step(nets.parameters()).
    return EMAModel(parameters=nets.parameters(), power=power)


def save_checkpoint(path: Path, nets, ema, optimizer, lr_scheduler, epoch: int, loss: float,
                    stats: dict, pcfg: PolicyConfig) -> None:
    ema_nets = copy.deepcopy(nets)
    ema.copy_to(ema_nets.parameters())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": nets.state_dict(),
        "ema_state_dict": ema_nets.state_dict(),
        "ema_internal": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "stats": {k: np.asarray(v).tolist() for k, v in stats.items()},
        "policy_config": pcfg.__dict__,
    }, path)


def train(
    data: list[str | Path],
    out: Path,
    pcfg: PolicyConfig,
    stats: dict[str, np.ndarray],
    epochs: int,
    device: torch.device,
    batch_size: int | None = None,
    warmup_steps: int | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
    num_workers: int = 0,
    max_batches_per_epoch: int | None = None,
) -> list[list[float]]:
    from diffusers.optimization import get_scheduler

    dataset = DexKitDataset(data, stats, pcfg.pred_horizon, pcfg.obs_horizon, pcfg.action_horizon,
                            crop_hw=tuple(pcfg.crop_size))
    log.info("dataset: %d episodes, %d samples", dataset.num_episodes, len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size or pcfg.batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    nets = build_nets(pcfg).to(device)
    noise_scheduler = make_scheduler(pcfg, "ddpm")
    ema = make_ema(nets, pcfg.ema_power)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=pcfg.lr, weight_decay=pcfg.weight_decay)
    steps_per_epoch = min(len(loader), max_batches_per_epoch or len(loader))
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=pcfg.warmup_steps if warmup_steps is None else warmup_steps,
        num_training_steps=steps_per_epoch * epochs,
    )
    start_epoch = 0
    if resume and out.exists():
        ckpt = torch.load(out, map_location=device, weights_only=False)
        nets.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        if "ema_internal" in ckpt:
            ema.load_state_dict(ckpt["ema_internal"])
        start_epoch = int(ckpt["epoch"]) + 1
        log.info("resumed from %s at epoch %d", out, start_epoch)

    every = checkpoint_every or pcfg.checkpoint_every
    history: list[list[float]] = []
    for epoch in range(start_epoch, epochs):
        t0 = time.monotonic()
        losses: list[float] = []
        nets.train()
        for b, batch in enumerate(loader):
            if max_batches_per_epoch and b >= max_batches_per_epoch:
                break
            nimage = batch["image"][:, : pcfg.obs_horizon].to(device, dtype=torch.float32)
            nagent = batch["agent_pos"][:, : pcfg.obs_horizon].to(device, dtype=torch.float32)
            naction = batch["action"].to(device, dtype=torch.float32)
            obs_cond = encode_obs(nets, nimage, nagent)
            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (naction.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(naction, noise, timesteps)
            noise_pred = nets["noise_pred_net"](noisy, timesteps, global_cond=obs_cond)
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(nets.parameters())
            losses.append(float(loss.item()))
        history.append(losses)
        mean = float(np.mean(losses)) if losses else float("nan")
        log.info("epoch %d: loss %.4f (%d batches, %.1fs)", epoch, mean, len(losses), time.monotonic() - t0)
        if (epoch + 1) % every == 0 or epoch == epochs - 1:
            save_checkpoint(out, nets, ema, optimizer, lr_scheduler, epoch, mean, stats, pcfg)
            log.info("checkpoint saved: %s", out)
    return history


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="folder(s) of episode .pkl files")
    p.add_argument("--task", default="task")
    p.add_argument("--out", help="checkpoint path (default data/checkpoints/<task>.pt)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None, help="LR warmup steps (default from policy.yaml)")
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)

    pcfg = load_policy_config()
    stats = build_stats(load_hand_config(), load_gantry_config())
    out = Path(args.out) if args.out else data_dir() / pcfg.paths.get("checkpoints", "checkpoints") / f"{args.task}.pt"
    device = pick_device(args.device)
    log.info("device: %s", device)
    train(args.data, out, pcfg, stats, args.epochs or pcfg.num_epochs, device, batch_size=args.batch_size,
          warmup_steps=args.warmup, checkpoint_every=args.checkpoint_every, resume=args.resume,
          num_workers=args.num_workers)


if __name__ == "__main__":
    main()
