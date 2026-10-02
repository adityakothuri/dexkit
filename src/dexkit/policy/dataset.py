"""Demo dataset (ported from diff_foam training_object.ipynb).

Each episode file is a pickle with the CMU-style schema:
  images (N,240,320,3) uint8 RGB, agent_pos (N,16), action (N,16), episode_ends [N]
Samples are pred_horizon windows, padded at episode boundaries by repetition.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms

from dexkit.hw.camera import CROP_HW, center_crop
from dexkit.policy.normalize import normalize_data


def create_sample_indices(
    episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0
) -> np.ndarray:
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    return np.array(indices)


def sample_sequence(
    train_data: dict[str, np.ndarray],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> dict[str, np.ndarray]:
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def add_noise(inputs: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(inputs) * 0.2 - 0.1
    return torch.clamp(inputs + noise, min=-1.0, max=1.0)


def load_episodes(folders: list[str | Path]) -> dict[str, np.ndarray]:
    images, states, actions, ends = [], [], [], []
    count = 0
    files: list[Path] = []
    for folder in folders:
        files += sorted(Path(folder).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no .pkl episodes in {[str(f) for f in folders]}")
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        n = len(d["agent_pos"])
        if not (len(d["images"]) == len(d["action"]) == n):
            raise ValueError(f"{f}: images/agent_pos/action lengths differ")
        images.append(np.asarray(d["images"], dtype=np.uint8))
        states.append(np.asarray(d["agent_pos"], dtype=np.float64))
        actions.append(np.asarray(d["action"], dtype=np.float64))
        count += n
        ends.append(count)
    return {
        "images": np.concatenate(images),
        "agent_pos": np.concatenate(states),
        "action": np.concatenate(actions),
        "episode_ends": np.asarray(ends),
    }


class DexKitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_folders: list[str | Path],
        stats: dict[str, np.ndarray],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        crop_hw: tuple[int, int] = CROP_HW,
        augment: bool = True,
    ) -> None:
        raw = load_episodes(data_folders)
        self.indices = create_sample_indices(
            episode_ends=raw["episode_ends"],
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )
        self.stats = stats
        self.normalized_train_data = {
            "agent_pos": normalize_data(raw["agent_pos"], stats).astype(np.float32),
            "action": normalize_data(raw["action"], stats).astype(np.float32),
            "image": np.moveaxis(raw["images"], -1, 1),  # (N, 3, 240, 320) uint8
        }
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.crop_hw = crop_hw
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5)
        self.num_episodes = len(raw["episode_ends"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        img = center_crop(nsample["image"][: self.obs_horizon], self.crop_hw).astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.ascontiguousarray(img))
        agent = torch.from_numpy(nsample["agent_pos"][: self.obs_horizon])
        if self.augment:
            img_t = self.color_jitter(img_t)
            agent = add_noise(agent)
        return {
            "image": img_t,
            "agent_pos": agent,
            "action": torch.from_numpy(nsample["action"][: self.pred_horizon]),
        }
