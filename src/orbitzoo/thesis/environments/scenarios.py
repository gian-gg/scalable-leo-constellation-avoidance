"""Episode sources: the fixed development fixture or seeded generated scenarios."""

from __future__ import annotations

from typing import Any, Protocol

from orbitzoo.thesis.config import ExperimentConfig
from orbitzoo.thesis.environments.collision_avoidance import (
    CollisionAvoidanceEnv,
    development_environment_kwargs,
)

SCENARIOS = ("development", "generated")


class EpisodeSource(Protocol):
    """Provides the environment to play for a given episode seed."""

    def environment(self, seed: int) -> CollisionAvoidanceEnv: ...


def _make_environment(config: ExperimentConfig, orbitzoo_kwargs: dict[str, Any]) -> CollisionAvoidanceEnv:
    environment = config.environment
    env = CollisionAvoidanceEnv(
        maneuver_config=config.maneuver,
        safety_config=config.safety,
        reward_config=config.rewards,
        neighborhood_size=environment.neighborhood_size,
        decision_interval_seconds=environment.decision_interval_seconds,
        episode_horizon=environment.episode_horizon,
        **orbitzoo_kwargs,
    )
    agent_count = len(env.dynamics.spacecraft_names)
    if agent_count != environment.num_agents:
        raise ValueError(
            f"scenario {environment.scenario!r} has {agent_count} agents, "
            f"but the config expects {environment.num_agents}"
        )
    return env


class FixedEpisodeSource:
    """Replays the same environment every episode, reset with each seed."""

    def __init__(self, env: CollisionAvoidanceEnv) -> None:
        self.env = env

    def environment(self, seed: int) -> CollisionAvoidanceEnv:
        return self.env


class GeneratedEpisodeSource:
    """Builds a new environment per seed from real orbits and real close-call shapes."""

    def __init__(self, config: ExperimentConfig) -> None:
        from orbitzoo.thesis.scenarios.generator import ScenarioGenerator
        from orbitzoo.thesis.scenarios.pools import load_pools

        self.config = config
        self.generator = ScenarioGenerator(
            config.scenario_generator,
            load_pools(config.scenario_generator),
            config.environment.num_agents,
            config.maneuver,
            config.safety,
            config.environment.decision_interval_seconds,
        )

    def environment(self, seed: int) -> CollisionAvoidanceEnv:
        scenario = self.generator.generate(seed)
        env = _make_environment(self.config, scenario.orbitzoo_kwargs)
        env.scenario = scenario
        return env


def build_episode_source(config: ExperimentConfig) -> EpisodeSource:
    """Create the episode source named by ``config.environment.scenario``."""
    config.validate()
    scenario = config.environment.scenario
    if scenario == "development":
        return FixedEpisodeSource(_make_environment(config, development_environment_kwargs()))
    if scenario == "generated":
        return GeneratedEpisodeSource(config)
    raise ValueError(f"unknown scenario {scenario!r}; available: {list(SCENARIOS)}")


def build_environment(config: ExperimentConfig) -> CollisionAvoidanceEnv:
    """The environment for the config's first episode."""
    return build_episode_source(config).environment(config.seed)
