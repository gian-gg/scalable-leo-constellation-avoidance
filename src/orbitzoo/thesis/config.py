"""Versioned, reproducible experiment configuration for the thesis project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from orbitzoo.thesis.environments.rewards import REMOVED_REWARD_FIELDS, RewardConfig
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig
from orbitzoo.thesis.scenarios.config import ScenarioGeneratorConfig


def default_maneuver_config() -> ManeuverConfig:
    """Return the maneuver selected by the sizing study (docs/MANEUVER_SIZING_FINDINGS.md)."""
    return ManeuverConfig(
        commanded_delta_v_mps=0.5,
        maximum_thrust_newtons=7.0,
        specific_impulse_seconds=300.0,
        maximum_burn_duration_seconds=60.0,
    )


@dataclass(frozen=True)
class EnvironmentConfig:
    """Parameters that define one simulation environment."""

    num_agents: int = 16
    neighborhood_size: int = 1
    decision_interval_seconds: float = 120.0
    episode_horizon: int = 100
    scenario: str = "development"

    def validate(self) -> None:
        if not self.scenario:
            raise ValueError("scenario must be a non-empty name")
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        if self.neighborhood_size <= 0:
            raise ValueError("neighborhood_size must be positive")
        if self.decision_interval_seconds <= 0:
            raise ValueError("decision_interval_seconds must be positive")
        if self.episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")


@dataclass(frozen=True)
class PolicyConfig:
    """Architecture choices that must remain stable within a trained policy."""

    algorithm: str = "mappo"
    num_actions: int = 7
    local_observation_dim: int | None = None
    global_state_dim: int | None = None

    def validate(self) -> None:
        if self.algorithm.lower() != "mappo":
            raise ValueError("only the MAPPO policy configuration is currently supported")
        if self.num_actions != 7:
            raise ValueError("the thesis action contract currently defines exactly 7 actions")
        for name, value in (
            ("local_observation_dim", self.local_observation_dim),
            ("global_state_dim", self.global_state_dim),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")


TRAINING_DEVICES = ("cpu", "cuda", "mps", "auto")


@dataclass(frozen=True)
class TrainingConfig:
    """MAPPO hyperparameters and training-run schedule."""

    rollout_steps: int = 2048
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    total_updates: int = 500
    update_epochs: int = 4
    minibatch_size: int = 256
    actor_hidden_dims: tuple[int, ...] = (128, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    value_clip: float = 0.2
    max_gradient_norm: float = 0.5
    checkpoint_interval: int = 10
    device: str = "cpu"
    initial_actor_checkpoint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_hidden_dims", tuple(self.actor_hidden_dims))
        object.__setattr__(self, "critic_hidden_dims", tuple(self.critic_hidden_dims))

    def validate(self) -> None:
        for name in ("rollout_steps", "total_updates", "update_epochs", "minibatch_size", "checkpoint_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("actor_hidden_dims", "critic_hidden_dims"):
            dims = getattr(self, name)
            if not dims or any(dim <= 0 for dim in dims):
                raise ValueError(f"{name} must be a non-empty list of positive sizes")
        if self.value_clip <= 0 or self.max_gradient_norm <= 0:
            raise ValueError("value_clip and max_gradient_norm must be positive")
        if self.entropy_coefficient < 0 or self.value_coefficient < 0:
            raise ValueError("entropy_coefficient and value_coefficient cannot be negative")
        if self.device not in TRAINING_DEVICES:
            raise ValueError(f"device must be one of {TRAINING_DEVICES}")
        if self.actor_learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.ppo_clip <= 0:
            raise ValueError("ppo_clip must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """All values required to reproduce a single training or evaluation run."""

    seed: int = 42
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    maneuver: ManeuverConfig = field(default_factory=default_maneuver_config)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    scenario_generator: ScenarioGeneratorConfig = field(default_factory=ScenarioGeneratorConfig)
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported configuration schema version: {self.schema_version}")
        self.environment.validate()
        self.policy.validate()
        self.training.validate()
        self.maneuver.validate()
        self.safety.validate()
        self.rewards.validate()
        self.scenario_generator.validate()
        if self.rewards.shaping_discount != self.training.gamma:
            raise ValueError("rewards.shaping_discount must equal training.gamma for policy-invariant shaping")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the exact configuration used by a run as portable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load a configuration written by :meth:`save`."""
        raw = json.loads(Path(path).read_text())
        removed = sorted(set(raw.get("rewards", {})) & set(REMOVED_REWARD_FIELDS))
        if removed:
            raise ValueError(
                f"{path} uses removed reward fields {removed}; see docs/COLLISION_AVOIDANCE_ENVIRONMENT.md"
            )
        config = cls(
            seed=raw["seed"],
            schema_version=raw.get("schema_version", 1),
            environment=EnvironmentConfig(**raw["environment"]),
            policy=PolicyConfig(**raw["policy"]),
            training=TrainingConfig(**raw["training"]),
            maneuver=ManeuverConfig(**raw["maneuver"]) if "maneuver" in raw else default_maneuver_config(),
            safety=SafetyConfig(**raw["safety"]) if "safety" in raw else SafetyConfig(),
            rewards=RewardConfig(**raw["rewards"]) if "rewards" in raw else RewardConfig(),
            scenario_generator=ScenarioGeneratorConfig(**raw.get("scenario_generator", {})),
        )
        config.validate()
        return config
