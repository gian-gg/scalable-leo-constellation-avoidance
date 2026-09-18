"""Named scenario sources that build collision-avoidance environments from a config."""

from __future__ import annotations

from typing import Any, Callable

from orbitzoo.thesis.config import ExperimentConfig
from orbitzoo.thesis.environments.collision_avoidance import (
    CollisionAvoidanceEnv,
    development_environment_kwargs,
)

SCENARIOS: dict[str, Callable[[], dict[str, Any]]] = {
    "development": development_environment_kwargs,
}


def build_environment(config: ExperimentConfig) -> CollisionAvoidanceEnv:
    """Create the environment named by ``config.environment.scenario``."""
    config.validate()
    environment = config.environment
    if environment.scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {environment.scenario!r}; available: {sorted(SCENARIOS)}"
        )
    env = CollisionAvoidanceEnv(
        maneuver_config=config.maneuver,
        safety_config=config.safety,
        reward_config=config.rewards,
        neighborhood_size=environment.neighborhood_size,
        decision_interval_seconds=environment.decision_interval_seconds,
        episode_horizon=environment.episode_horizon,
        **SCENARIOS[environment.scenario](),
    )
    agent_count = len(env.dynamics.spacecraft_names)
    if agent_count != environment.num_agents:
        raise ValueError(
            f"scenario {environment.scenario!r} has {agent_count} agents, "
            f"but the config expects {environment.num_agents}"
        )
    return env
