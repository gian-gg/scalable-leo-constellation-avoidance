"""Decentralized policies compared during evaluation: no-op, rule-based, and trained MAPPO."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from orbitzoo.rl_algorithms.mappo import MAPPO
from orbitzoo.thesis.environments.observations import (
    OWN_FEATURE_DIM,
    POSITION_SCALE_METERS,
    VELOCITY_SCALE_MPS,
)
from orbitzoo.thesis.environments.prediction import predict_closest_approach
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.environments.vectorized_observations import rsw_bases
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig
from orbitzoo.thesis.maneuvers.sizing import BURN_ACTIONS, EncounterGeometry, miss_displacement_per_mps


class EvaluationPolicy(Protocol):
    """Chooses one action per agent from local observations only."""

    name: str

    def choose(self, local_observations: np.ndarray) -> np.ndarray: ...


class NoOpPolicy:
    """Never maneuvers."""

    name = "noop"

    def choose(self, local_observations: np.ndarray) -> np.ndarray:
        return np.zeros(local_observations.shape[0], dtype=np.int64)


class ClohessyWiltshireAvoidancePolicy:
    """Burns only when the top-ranked threat is predicted unsafe, choosing the burn that most widens its miss.

    The miss is predicted along curved J2 orbits reconstructed from the agent's own observation, and each
    burn's effect is the Clohessy–Wiltshire displacement across the relative velocity, as in the sizing study.
    """

    name = "rule"

    def __init__(self, maneuver_config: ManeuverConfig, safety_config: SafetyConfig) -> None:
        self.delta_v = maneuver_config.commanded_delta_v_mps
        self.safety_config = safety_config

    def choose(self, local_observations: np.ndarray) -> np.ndarray:
        actions = np.zeros(local_observations.shape[0], dtype=np.int64)
        blocks = local_observations[:, OWN_FEATURE_DIM:]
        threatened = np.flatnonzero((blocks[:, 11] > 0.5) & (blocks[:, 7] <= 1.0) & (blocks[:, 6] > 0))
        if threatened.size == 0:
            return actions
        rows = local_observations[threatened].astype(float)
        own_positions = rows[:, 0:3] * POSITION_SCALE_METERS
        own_velocities = rows[:, 3:6] * VELOCITY_SCALE_MPS
        bases = rsw_bases(own_positions, own_velocities)
        threat_positions = own_positions + np.einsum("aji,aj->ai", bases, rows[:, 7:10] * POSITION_SCALE_METERS)
        threat_velocities = own_velocities + np.einsum("aji,aj->ai", bases, rows[:, 10:13] * VELOCITY_SCALE_MPS)
        approach = predict_closest_approach(
            own_positions,
            own_velocities,
            threat_positions,
            threat_velocities,
            self.safety_config.screening_horizon_seconds,
            self.safety_config.prediction_step_seconds,
        )
        for index, row in enumerate(threatened):
            encounter = EncounterGeometry(
                event_id=0,
                maneuvering_norad_id=0,
                threat_norad_id=0,
                miss_vector_m=approach.miss_vectors_m[index],
                relative_velocity_mps=approach.relative_velocities_mps[index],
                agent_position_m=approach.first_positions_m[index],
                agent_velocity_mps=approach.first_velocities_mps[index],
            )
            best_action, best_miss = ManeuverAction.NO_OP, encounter.miss_distance_m
            for action in BURN_ACTIONS:
                gain = miss_displacement_per_mps(encounter, action, [float(approach.time_seconds[index])])
                miss = float(np.linalg.norm(encounter.miss_vector_m + self.delta_v * gain))
                if miss > best_miss:
                    best_action, best_miss = action, miss
            actions[row] = best_action
        return actions


class MAPPOActorPolicy:
    """Runs a trained shared actor deterministically, without the critic."""

    def __init__(self, policy: MAPPO, name: str = "mappo") -> None:
        self.policy = policy
        self.name = name

    def choose(self, local_observations: np.ndarray) -> np.ndarray:
        return self.policy.select_actions(local_observations, deterministic=True).numpy()
