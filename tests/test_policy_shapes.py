import os

import numpy as np
import pytest

# Torch tests are opt-in: they run model forward passes and are slow on CPU.
pytestmark = pytest.mark.skipif(
    os.environ.get("DEXKIT_POLICY_TESTS") != "1", reason="set DEXKIT_POLICY_TESTS=1 to run policy tests"
)

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")


def test_network_forward_shapes():
    from dexkit.config import PolicyConfig
    from dexkit.policy.nets import build_nets, encode_obs

    pcfg = PolicyConfig()
    nets = build_nets(pcfg)
    img = torch.zeros((2, pcfg.obs_horizon, 3, *pcfg.crop_size))
    pos = torch.zeros((2, pcfg.obs_horizon, pcfg.lowdim_obs_dim))
    with torch.no_grad():
        cond = encode_obs(nets, img, pos)
        assert cond.shape == (2, pcfg.obs_horizon * (512 + 16))
        noise = nets["noise_pred_net"](torch.randn(2, pcfg.pred_horizon, pcfg.action_dim), torch.zeros(2), global_cond=cond)
    assert noise.shape == (2, pcfg.pred_horizon, pcfg.action_dim)


def test_predict_actions_ddim_shape():
    from dexkit.config import PolicyConfig
    from dexkit.policy.nets import build_nets, make_scheduler, predict_actions

    pcfg = PolicyConfig()
    nets = build_nets(pcfg).eval()
    out = predict_actions(nets, make_scheduler(pcfg, "ddim"), pcfg,
                          torch.zeros((1, 2, 3, *pcfg.crop_size)), torch.zeros((1, 2, 16)), num_inference_steps=4)
    assert out.shape == (16, 16) and np.all(np.abs(out) <= 1.0 + 1e-5)


def test_normalize_round_trip_from_configs(hand_cfg, gantry_cfg):
    from dexkit.policy.config import build_stats
    from dexkit.policy.normalize import normalize_data, unnormalize_data

    stats = build_stats(hand_cfg, gantry_cfg)
    assert stats["min"][12] == -hand_cfg.roll.range_deg
    assert list(stats["max"][13:]) == list(gantry_cfg.travel_max)
    x = np.random.default_rng(0).uniform(stats["min"], stats["max"], size=(5, 16))
    n = normalize_data(x, stats)
    assert n.min() >= -1 and n.max() <= 1
    assert np.allclose(unnormalize_data(n, stats), x)


def test_dataset_windows_and_padding(tmp_path, hand_cfg, gantry_cfg):
    from dexkit.policy.config import build_stats
    from dexkit.policy.dataset import DexKitDataset, create_sample_indices
    from dexkit.policy.smoke import make_synthetic_episodes

    idx = create_sample_indices(np.array([10, 25]), sequence_length=16, pad_before=1, pad_after=7)
    assert len(idx) == (10 - 16 + 7 + 1 + 1) + (15 - 16 + 7 + 1 + 1)
    stats = build_stats(hand_cfg, gantry_cfg)
    make_synthetic_episodes(tmp_path, stats, n_episodes=2, steps=20)
    ds = DexKitDataset([tmp_path], stats, 16, 2, 8)
    item = ds[0]
    assert item["image"].shape == (2, 3, 216, 288)
    assert item["agent_pos"].shape == (2, 16) and item["action"].shape == (16, 16)
    assert float(item["action"].abs().max()) <= 1.0 + 1e-6
