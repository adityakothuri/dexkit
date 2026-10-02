"""Network construction, schedulers and the denoising loop shared by train and infer."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch
import torch.nn as nn

from dexkit.config import PolicyConfig
from dexkit.policy.model.noise_pred_net import ConditionalUnet1D
from dexkit.policy.model.visual_encoder import get_resnet, replace_bn_with_gn


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_nets(pcfg: PolicyConfig, quiet: bool = True) -> nn.ModuleDict:
    # IMPORTANT (from diff_foam): replace BatchNorm with GroupNorm to work with EMA.
    vision_encoder = replace_bn_with_gn(get_resnet("resnet18"))
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:  # ConditionalUnet1D prints its parameter count
        noise_pred_net = ConditionalUnet1D(
            input_dim=pcfg.action_dim, global_cond_dim=pcfg.obs_dim * pcfg.obs_horizon
        )
    return nn.ModuleDict({"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net})


def make_scheduler(pcfg: PolicyConfig, kind: str = "ddpm"):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    cls = DDIMScheduler if kind == "ddim" else DDPMScheduler
    return cls(
        num_train_timesteps=pcfg.num_diffusion_iters,
        beta_schedule=pcfg.beta_schedule,
        clip_sample=pcfg.clip_sample,
        prediction_type=pcfg.prediction_type,
    )


def encode_obs(nets: nn.ModuleDict, nimage: torch.Tensor, nagent_pos: torch.Tensor) -> torch.Tensor:
    """nimage (B, T, 3, H, W) float in [0,1]; nagent_pos (B, T, D) normalized -> (B, T*(512+D))."""
    feats = nets["vision_encoder"](nimage.flatten(end_dim=1))
    feats = feats.reshape(*nimage.shape[:2], -1)
    return torch.cat([feats, nagent_pos], dim=-1).flatten(start_dim=1)


@torch.no_grad()
def predict_actions(
    nets: nn.ModuleDict,
    scheduler,
    pcfg: PolicyConfig,
    nimage: torch.Tensor,
    nagent_pos: torch.Tensor,
    num_inference_steps: int,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Returns normalized actions (pred_horizon, action_dim)."""
    device = nimage.device
    obs_cond = encode_obs(nets, nimage, nagent_pos)
    naction = torch.randn((1, pcfg.pred_horizon, pcfg.action_dim), device=device, generator=generator)
    scheduler.set_timesteps(num_inference_steps)
    for k in scheduler.timesteps:
        noise_pred = nets["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
        naction = scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
    return naction[0].detach().cpu().numpy()
