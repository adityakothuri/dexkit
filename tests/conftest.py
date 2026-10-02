from __future__ import annotations

import os
from pathlib import Path

import pytest

from dexkit.config import REPO_ROOT, load_gantry_config, load_hand_config
from dexkit.hw.safety import EStop


@pytest.fixture(autouse=True)
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never write into the real data/ or config/ directories from tests: every test gets its
    own copy of config/ (tools like --unwind and --label-only write hand.yaml in place)."""
    import shutil

    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    monkeypatch.setenv("DEXKIT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DEXKIT_CONFIG", str(tmp_path / "config"))
    return tmp_path / "data"


@pytest.fixture
def hand_cfg():
    """The shipped hand.yaml, renumbered to fixed IDs (fingers 1-12, roll 13) so tests don't
    depend on how the bench servos happen to be numbered."""
    from dexkit.config import TENDONS, config_dir

    cfg = load_hand_config(config_dir() / "hand.yaml")  # the per-test copy, never the real file

    for i, (s, name) in enumerate(zip(cfg.servos, TENDONS, strict=True)):
        s.id, s.slack, s.tight, s.inverted, s.stall_load = i + 1, 2048, 2700, False, 800
        s.name, (s.finger, s.role) = name, TENDONS[name]
        s.enabled = True  # tests exercise all 12 tendons regardless of what the bench file disables
    cfg.roll.id, cfg.roll.center, cfg.roll.inverted = 13, 2048, False
    cfg.calibrated_at = None
    for s in cfg.servos:
        s.calibrated = False
    cfg.roll.calibrated = False
    return cfg


@pytest.fixture
def gantry_cfg():
    from dexkit.config import config_dir

    cfg = load_gantry_config(config_dir() / "gantry.yaml")
    cfg.banner_timeout_s = 1.0
    return cfg


@pytest.fixture
def estop() -> EStop:
    return EStop()


@pytest.fixture
def mock_hand(hand_cfg):
    from dexkit.hw.mock import MockHand

    h = MockHand(hand_cfg, latency_s=0.0)
    h.connect()
    yield h
    h.close()


@pytest.fixture
def mock_gantry(gantry_cfg):
    from dexkit.hw.mock import MockGantry

    g = MockGantry(gantry_cfg, speed_factor=50.0)
    g.connect()
    yield g
    g.close()


os.environ.setdefault("PYTHONHASHSEED", "0")
