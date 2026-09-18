"""Versioned, reproducible experiment configuration for the thesis project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from orbitzoo.thesis.environments.rewards import RewardConfig
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig


def default_maneuver_config() -> ManeuverConfig:
    """Return provisional maneuver defaults for development and test runs."""
    return ManeuverConfig(
        commanded_delta_v_mps=0.01,
        maximum_thrust_newtons=0.1,
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

    def validate(self) -> None:
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


@dataclass(frozen=True)
class TrainingConfig:
    """Initial MAPPO hyperparameters; later phases may extend this schema."""

    rollout_steps: int = 2048
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2

    def validate(self) -> None:
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
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
        config = cls(
            seed=raw["seed"],
            schema_version=raw.get("schema_version", 1),
            environment=EnvironmentConfig(**raw["environment"]),
            policy=PolicyConfig(**raw["policy"]),
            training=TrainingConfig(**raw["training"]),
            maneuver=ManeuverConfig(**raw["maneuver"]) if "maneuver" in raw else default_maneuver_config(),
            safety=SafetyConfig(**raw["safety"]) if "safety" in raw else SafetyConfig(),
            rewards=RewardConfig(**raw["rewards"]) if "rewards" in raw else RewardConfig(),
        )
        config.validate()
        return config
