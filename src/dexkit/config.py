"""YAML configuration, loaded once at startup and validated into dataclasses."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


def config_dir(override: str | os.PathLike | None = None) -> Path:
    """Resolve the config directory: explicit arg > $DEXKIT_CONFIG > ./config > repo config/."""
    if override:
        return Path(override)
    env = os.environ.get("DEXKIT_CONFIG")
    if env:
        return Path(env)
    cwd = Path.cwd() / "config"
    if (cwd / "hand.yaml").exists():
        return cwd
    return REPO_ROOT / "config"


def data_dir() -> Path:
    env = os.environ.get("DEXKIT_DATA")
    return Path(env) if env else REPO_ROOT / "data"


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return d[key]


# --------------------------------------------------------------------------- hand

# DexKit tendon layout (CMU Foam Hands Lab): one servo winds one tendon, tendons only
# pull. Palm side: five flexors + thumb adduction. Back side: five extensors + index
# adduction. Flexor and extensor of the same finger are antagonists. Which servo
# channel drives which tendon is recorded in hand.yaml (`name`); a canonical name
# below fixes `finger` and `role` automatically.
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ROLES = ("flex", "extend", "adduct")
TENDONS: dict[str, tuple[str, str]] = {
    **{f"{f}_flex": (f, "flex") for f in FINGERS},
    **{f"{f}_extend": (f, "extend") for f in FINGERS},
    "thumb_adduct": ("thumb", "adduct"),
    "index_adduct": ("index", "adduct"),
}

TICKS_PER_REV = 4096
# The bench servos run in multi-turn mode (EEPROM angle limits 0/0): positions are
# 15-bit sign-magnitude and keep counting past 4095, so limits span several turns.
MAX_TICKS = 0x7FFF

# Feetech register map shared by STS and HLS. Address 44 differs: Goal Time on
# STS, Goal Torque on HLS (scservo_sdk/hls.py). Override per bench verification.
BASE_REGISTERS: dict[str, int] = {
    "model": 3,
    "firmware_major": 0,
    "firmware_minor": 1,
    "id": 5,
    "baud": 6,
    "mode": 33,
    "torque_enable": 40,
    "accel": 41,
    "goal_position": 42,
    "goal_speed": 46,
    "lock": 55,
    "present_position": 56,
    "present_speed": 58,
    "present_load": 60,
    "present_voltage": 62,
    "present_temperature": 63,
    "moving": 66,
}
# HLS: the vendor SDK (scservo_sdk/hls.py) limits force only via Goal Torque at 44
# and never writes 48, so 48 is not assumed on HLS; add `torque_limit: 48` under
# `registers:` in hand.yaml only after confirming it in the memory table.
FAMILY_REGISTERS: dict[str, dict[str, int]] = {
    "hls": {"goal_torque": 44},
    "sts": {"goal_time": 44, "torque_limit": 48},
}


@dataclass
class ServoConfig:
    id: int
    name: str
    slack: int
    tight: int
    finger: str = ""
    role: str = ""  # flex | extend | adduct (filled from a canonical name)
    inverted: bool = False
    max_delta_ticks: int = 120
    stall_load: int = 800
    calibrated: bool = False  # set by dexkit-calibrate-hand when slack/tight were measured

    def __post_init__(self) -> None:
        if self.name in TENDONS:
            self.finger, self.role = TENDONS[self.name]
        if self.role and self.role not in ROLES:
            raise ConfigError(f"hand.yaml: servo {self.id} role must be one of {ROLES}, got '{self.role}'")

    @property
    def span(self) -> int:
        """Signed tick distance from slack to tight, with direction from `inverted`."""
        mag = abs(self.tight - self.slack)
        return -mag if self.inverted else mag

    @property
    def effective_tight(self) -> int:
        return self.slack + self.span

    @property
    def lo(self) -> int:
        return min(self.slack, self.effective_tight)

    @property
    def hi(self) -> int:
        return max(self.slack, self.effective_tight)


@dataclass
class RollConfig:
    id: int
    center: int = 2048
    range_deg: float = 90.0
    inverted: bool = False
    model: str = "HLS3640M"
    max_delta_ticks: int = 120
    calibrated: bool = False

    def deg_to_ticks(self, deg: float) -> int:
        sign = -1.0 if self.inverted else 1.0
        return int(round(self.center + sign * deg * TICKS_PER_REV / 360.0))

    def ticks_to_deg(self, ticks: int) -> float:
        sign = -1.0 if self.inverted else 1.0
        return sign * (ticks - self.center) * 360.0 / TICKS_PER_REV

    @property
    def lo(self) -> int:
        return min(self.deg_to_ticks(-self.range_deg), self.deg_to_ticks(self.range_deg))

    @property
    def hi(self) -> int:
        return max(self.deg_to_ticks(-self.range_deg), self.deg_to_ticks(self.range_deg))


@dataclass
class HandDefaults:
    accel: int = 50
    speed: int = 1500
    torque_limit: int = 600
    goal_torque: int = 600
    calib_torque_limit: int = 300
    calib_step_ticks: int = 40
    calib_max_travel_ticks: int = 3 * TICKS_PER_REV  # give up finding 'tight' after this much travel
    calib_direction: int = 1  # +1: step toward higher counts, -1: lower; --reverse flips it
    calib_direction: int = 1  # +1: step toward higher counts, -1: lower (the bench hand winds on -1)
    stall_load_factor: float = 1.5
    stall_load_floor: int = 400  # never set a stall threshold below this (holding at goal_torque 600 is normal)
    stall_time_s: float = 0.5
    stall_backoff: float = 0.05
    antagonist_max_sum: float = 1.0  # flexor + extensor of one finger may never exceed this together


@dataclass
class HandConfig:
    servos: list[ServoConfig]
    roll: RollConfig
    port: str = "/dev/dexkit_hand"
    baud: int = 1_000_000
    fallback_baud: int = 115_200
    timeout_s: float = 0.02
    voltage_window: tuple[float, float] = (6.0, 8.4)
    voltage_check_period_s: float = 2.0
    max_temp_c: float = 65.0
    rate_hz: float = 20.0
    calibrated_at: str | None = None
    servo_family: str = "hls"
    registers: dict[str, int] = field(default_factory=dict)
    defaults: HandDefaults = field(default_factory=HandDefaults)
    source: Path | None = None

    def __post_init__(self) -> None:
        if len(self.servos) != 12:
            raise ConfigError(f"hand.yaml: expected 12 finger servos, got {len(self.servos)}")
        ids = [s.id for s in self.servos] + [self.roll.id]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"hand.yaml: servo IDs must be unique, got {ids}")
        names = [s.name for s in self.servos]
        if len(set(names)) != len(names):
            raise ConfigError(f"hand.yaml: servo names must be unique, got {names}")
        for i in ids:
            if not 0 <= i <= 253:
                raise ConfigError(f"hand.yaml: servo id {i} out of range 0..253")
        lo, hi = self.voltage_window
        if not lo < hi:
            raise ConfigError("hand.yaml: voltage_window must be [low, high]")
        if self.servo_family not in FAMILY_REGISTERS:
            raise ConfigError(f"hand.yaml: servo_family must be one of {list(FAMILY_REGISTERS)}")
        for s in self.servos:
            for v in (s.slack, s.tight):
                if not -MAX_TICKS <= v <= MAX_TICKS:
                    raise ConfigError(f"hand.yaml: servo {s.id} ticks {v} out of +/-{MAX_TICKS}")

    @property
    def ids(self) -> list[int]:
        return [s.id for s in self.servos] + [self.roll.id]

    @property
    def finger_ids(self) -> list[int]:
        return [s.id for s in self.servos]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.servos]

    def antagonist_pairs(self) -> list[tuple[int, int]]:
        """(flexor index, extensor index) for every finger that has both assigned."""
        by = {(s.finger, s.role): i for i, s in enumerate(self.servos) if s.finger and s.role}
        return [(by[(f, "flex")], by[(f, "extend")]) for f in FINGERS if (f, "flex") in by and (f, "extend") in by]

    @property
    def unassigned(self) -> list[int]:
        return [s.id for s in self.servos if s.name not in TENDONS]

    @property
    def uncalibrated_ids(self) -> list[int]:
        ids = [s.id for s in self.servos if not s.calibrated]
        if not self.roll.calibrated:
            ids.append(self.roll.id)
        return ids

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrated_at) and not self.uncalibrated_ids

    def register_map(self) -> dict[str, int]:
        regs = dict(BASE_REGISTERS)
        regs.update(FAMILY_REGISTERS[self.servo_family])
        regs.update(self.registers or {})
        return regs

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source", None)
        d["voltage_window"] = list(self.voltage_window)
        return d


def hand_config_from_dict(d: dict[str, Any], source: Path | None = None) -> HandConfig:
    where = str(source or "hand.yaml")
    servos = [ServoConfig(**s) for s in _require(d, "servos", where)]
    roll = RollConfig(**_require(d, "roll", where))
    defaults = HandDefaults(**(d.get("defaults") or {}))
    kwargs = {k: v for k, v in d.items() if k not in ("servos", "roll", "defaults")}
    if "voltage_window" in kwargs:
        kwargs["voltage_window"] = tuple(float(x) for x in kwargs["voltage_window"])
    try:
        return HandConfig(servos=servos, roll=roll, defaults=defaults, source=source, **kwargs)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from e


def load_hand_config(path: str | os.PathLike | None = None) -> HandConfig:
    p = Path(path) if path else config_dir() / "hand.yaml"
    return hand_config_from_dict(load_yaml(p), source=p)


# --------------------------------------------------------------------------- gantry


@dataclass
class GantryConfig:
    travel_min: tuple[float, float, float]
    travel_max: tuple[float, float, float]
    port: str = "/dev/dexkit_gantry"
    baud: int = 115_200
    reset_on_connect: bool = True
    banner_timeout_s: float = 3.0
    command_timeout_s: float = 5.0
    status_hz: float = 10.0
    status_report_mask: int = 1
    zeroing: str = "manual"
    calibrated_at: str | None = None
    margin_mm: float = 5.0
    feed_max_mm_min: float = 1500.0
    feed_default_mm_min: float = 800.0
    jog_feed_mm_min: float = 600.0
    jog_step_mm: float = 1.0
    jog_step_big_mm: float = 10.0
    jog_max_backlog_mm: float = 3.0
    planner_min_free: int = 2
    position_tolerance_mm: float = 0.1
    source: Path | None = None

    def __post_init__(self) -> None:
        if self.zeroing not in ("manual", "home"):
            raise ConfigError("gantry.yaml: zeroing must be 'manual' or 'home'")
        for a, b in zip(self.travel_min, self.travel_max, strict=True):
            if not a < b:
                raise ConfigError("gantry.yaml: travel_box min must be < max on every axis")
        if self.feed_max_mm_min <= 0:
            raise ConfigError("gantry.yaml: feed_max_mm_min must be positive")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source", None)
        d.pop("travel_min")
        d.pop("travel_max")
        d["travel_box"] = {"min": list(self.travel_min), "max": list(self.travel_max)}
        return d


def gantry_config_from_dict(d: dict[str, Any], source: Path | None = None) -> GantryConfig:
    where = str(source or "gantry.yaml")
    box = _require(d, "travel_box", where)
    kwargs = {k: v for k, v in d.items() if k != "travel_box"}
    try:
        return GantryConfig(
            travel_min=tuple(float(x) for x in box["min"]),
            travel_max=tuple(float(x) for x in box["max"]),
            source=source,
            **kwargs,
        )
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from e


def load_gantry_config(path: str | os.PathLike | None = None) -> GantryConfig:
    p = Path(path) if path else config_dir() / "gantry.yaml"
    return gantry_config_from_dict(load_yaml(p), source=p)


# --------------------------------------------------------------------------- policy


@dataclass
class PolicyConfig:
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    vision_feature_dim: int = 512
    action_dim: int = 16
    lowdim_obs_dim: int = 16
    image_size: tuple[int, int] = (240, 320)
    crop_size: tuple[int, int] = (216, 288)
    num_diffusion_iters: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"
    ema_power: float = 0.75
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    batch_size: int = 64
    num_epochs: int = 100
    checkpoint_every: int = 10
    rate_hz: float = 10.0
    camera_index: int = 0
    paths: dict[str, str] = field(default_factory=dict)

    @property
    def obs_dim(self) -> int:
        return self.vision_feature_dim + self.lowdim_obs_dim


def load_policy_config(path: str | os.PathLike | None = None) -> PolicyConfig:
    p = Path(path) if path else config_dir() / "policy.yaml"
    d = load_yaml(p)
    for k in ("image_size", "crop_size"):
        if k in d:
            d[k] = tuple(d[k])
    return PolicyConfig(**d)


def dump_yaml(data: dict[str, Any], path: str | os.PathLike, header: str = "") -> None:
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header + text)
