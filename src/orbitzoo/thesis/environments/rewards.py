"""Continuous, safety-first reward for collision avoidance.

See docs/COLLISION_AVOIDANCE_ENVIRONMENT.md#rewards-and-diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from orbitzoo.thesis.environments.observations import NEIGHBOR_FEATURE_DIM, OWN_FEATURE_DIM
from orbitzoo.thesis.maneuvers.contract import ManeuverResult

REMOVED_REWARD_FIELDS = ("unsafe_penalty", "resolution_reward", "unnecessary_maneuver_penalty")


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights; ``shaping_discount`` must equal the training discount."""

    collision_penalty: float = -100.0
    close_approach_penalty: float = -10.0
    shaping_weight: float = 10.0
    shaping_discount: float = 0.99
    delta_v_penalty_per_mps: float = 1.0
    infeasible_maneuver_penalty: float = -2.0

    def validate(self) -> None:
        if self.collision_penalty >= 0 or self.close_approach_penalty >= 0:
            raise ValueError("collision_penalty and close_approach_penalty must be negative")
        if self.shaping_weight < 0 or self.delta_v_penalty_per_mps < 0:
            raise ValueError("shaping_weight and delta_v_penalty_per_mps cannot be negative")
        if not 0 < self.shaping_discount <= 1:
            raise ValueError("shaping_discount must be in (0, 1]")
        if self.infeasible_maneuver_penalty > 0:
            raise ValueError("infeasible_maneuver_penalty cannot be positive")


def shortfall(miss_distance_meters: float, safe_separation_meters: float) -> float:
    """How far inside the safe separation a miss falls: 0 at or beyond it, 1 at zero miss."""
    return max(0.0, 1.0 - miss_distance_meters / safe_separation_meters)


def threat_potentials(local_observations: np.ndarray, safe_separation_meters: float) -> np.ndarray:
    """Minus the shortfall of each agent's worst still-approaching threat, read from its own observation."""
    blocks = local_observations[:, OWN_FEATURE_DIM:].reshape(len(local_observations), -1, NEIGHBOR_FEATURE_DIM)
    approaching = (blocks[:, :, 11] > 0.5) & (blocks[:, :, 6] > 0)
    misses = np.where(approaching, blocks[:, :, 7] * safe_separation_meters, np.inf).min(axis=1)
    return -np.clip(1.0 - misses / safe_separation_meters, 0.0, 1.0)


def calculate_rewards(
    agent_names: list[str],
    results: Mapping[str, ManeuverResult],
    potentials_before: np.ndarray,
    potentials_after: np.ndarray,
    collided_agents: set[str],
    close_approaches: Mapping[tuple[str, str], float],
    safe_separation_meters: float,
    config: RewardConfig,
    rejected_agents: set[str] | None = None,
) -> dict[str, float]:
    """Individual rewards from fuel, realized close approaches, collisions, and potential-based shaping.

    Potentials are in ``agent_names`` order; ``close_approaches`` maps each pair whose closest approach
    happened during the step to its miss distance.
    """
    config.validate()
    rejected_agents = rejected_agents or set()
    rewards: dict[str, float] = {}
    for index, agent in enumerate(agent_names):
        reward = -config.delta_v_penalty_per_mps * results[agent].actual_delta_v_mps
        if agent in rejected_agents:
            reward += config.infeasible_maneuver_penalty
        reward += config.close_approach_penalty * sum(
            shortfall(miss, safe_separation_meters) for pair, miss in close_approaches.items() if agent in pair
        )
        next_potential = 0.0
        if agent in collided_agents:
            reward += config.collision_penalty
        else:
            next_potential = float(potentials_after[index])
        reward += config.shaping_weight * (config.shaping_discount * next_potential - float(potentials_before[index]))
        rewards[agent] = reward
    return rewards
