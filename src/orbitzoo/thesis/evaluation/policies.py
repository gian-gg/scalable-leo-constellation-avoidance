"""Decentralized policies compared during evaluation: no-op, rule-based, and trained MAPPO."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from orbitzoo.rl_algorithms.mappo import MAPPO
from orbitzoo.thesis.environments.observations import (
    OWN_FEATURE_DIM,
    POSITION_SCALE_METERS,
    VELOCITY_SCALE_MPS,
)
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig

EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14
BURN_ACTIONS = tuple(action for action in ManeuverAction if action is not ManeuverAction.NO_OP)


class EvaluationPolicy(Protocol):
    """Chooses one action per agent from local observations only."""

    name: str

    def choose(self, local_observations: np.ndarray) -> np.ndarray: ...


class NoOpPolicy:
    """Never maneuvers."""

    name = "noop"

    def choose(self, local_observations: np.ndarray) -> np.ndarray:
        return np.zeros(local_observations.shape[0], dtype=np.int64)


def clohessy_wiltshire_displacement(
    delta_v_rsw: np.ndarray, mean_motion: float, elapsed_seconds: float
) -> np.ndarray:
    """RSW position change after an impulsive delta-v on a circular orbit."""
    radial, along_track, cross_track = delta_v_rsw
    angle = mean_motion * elapsed_seconds
    sine, cosine = math.sin(angle), math.cos(angle)
    return np.array(
        [
            radial / mean_motion * sine + 2 * along_track / mean_motion * (1 - cosine),
            2 * radial / mean_motion * (cosine - 1)
            + along_track / mean_motion * (4 * sine - 3 * angle),
            cross_track / mean_motion * sine,
        ]
    )


class ClohessyWiltshireAvoidancePolicy:
    """Burns only when the top-ranked threat is unsafe, choosing the burn that most increases its predicted miss distance."""

    name = "rule"

    def __init__(self, maneuver_config: ManeuverConfig, safety_config: SafetyConfig) -> None:
        self.delta_v = maneuver_config.commanded_delta_v_mps
        self.safety_config = safety_config

    def _action_for(self, observation: np.ndarray) -> ManeuverAction:
        block = observation[OWN_FEATURE_DIM:]
        is_valid, normalized_miss = block[11] > 0.5, block[7]
        if not is_valid or normalized_miss > 1.0:
            return ManeuverAction.NO_OP
        relative_position = block[0:3].astype(float) * POSITION_SCALE_METERS
        relative_velocity = block[3:6].astype(float) * VELOCITY_SCALE_MPS
        time_to_closest_approach = float(block[6]) * self.safety_config.screening_horizon_seconds
        if time_to_closest_approach <= 0:
            return ManeuverAction.NO_OP
        orbit_radius = np.linalg.norm(observation[0:3].astype(float) * POSITION_SCALE_METERS)
        mean_motion = math.sqrt(EARTH_GRAVITATIONAL_PARAMETER / orbit_radius**3)
        miss_vector = relative_position + relative_velocity * time_to_closest_approach
        best_action, best_miss = ManeuverAction.NO_OP, float(np.linalg.norm(miss_vector))
        for action in BURN_ACTIONS:
            displacement = clohessy_wiltshire_displacement(
                self.delta_v * np.asarray(action.rsw_unit_vector), mean_motion, time_to_closest_approach
            )
            miss = float(np.linalg.norm(miss_vector - displacement))
            if miss > best_miss:
                best_action, best_miss = action, miss
        return best_action

    def choose(self, local_observations: np.ndarray) -> np.ndarray:
        return np.array([self._action_for(row) for row in local_observations], dtype=np.int64)


class MAPPOActorPolicy:
    """Runs a trained shared actor deterministically, without the critic."""

    def __init__(self, policy: MAPPO, name: str = "mappo") -> None:
        self.policy = policy
        self.name = name

    def choose(self, local_observations: np.ndarray) -> np.ndarray:
        return self.policy.select_actions(local_observations, deterministic=True).numpy()
