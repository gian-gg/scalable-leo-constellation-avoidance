"""Fixed-size local observations and centralized training state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from orbitzoo.thesis.environments.safety import (
    PairSafetyAssessment,
    SafetyConfig,
    assess_all_pairs,
)


POSITION_SCALE_METERS = 10_000_000.0
VELOCITY_SCALE_MPS = 10_000.0
MAX_NORMALIZED_MISS_DISTANCE = 10.0

OWN_FEATURE_DIM = 7
NEIGHBOR_FEATURE_DIM = 12
GLOBAL_BODY_FEATURE_DIM = 9


@dataclass(frozen=True)
class ObservationState:
    """Actor observations and critic-only global state for one timestep."""

    local_observations: np.ndarray
    global_state: np.ndarray


def fuel_fraction(body: Any, maneuverable_names: set[str]) -> float:
    if body.name not in maneuverable_names:
        return 0.0
    initial_fuel = float(getattr(body, "initial_fuel_mass", 0.0))
    if initial_fuel <= 0.0:
        return 0.0
    return float(np.clip(body.get_fuel() / initial_fuel, 0.0, 1.0))


def _rsw_basis(body: Any) -> np.ndarray:
    """Return rows containing the body's radial, along-track, and cross-track axes."""
    position = np.asarray(body.position, dtype=float)
    velocity = np.asarray(body.velocity, dtype=float)
    radial_norm = float(np.linalg.norm(position))
    angular_momentum = np.cross(position, velocity)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if radial_norm <= 1e-12 or angular_momentum_norm <= 1e-12:
        raise ValueError(f"body {body.name!r} does not define a valid RSW frame")
    radial = position / radial_norm
    cross_track = angular_momentum / angular_momentum_norm
    along_track = np.cross(cross_track, radial)
    return np.stack((radial, along_track, cross_track))


class LocalObservationEncoder:
    """Encode a variable population into fixed-width shared-actor inputs."""

    def __init__(self, neighborhood_size: int, safety_config: SafetyConfig) -> None:
        if neighborhood_size <= 0:
            raise ValueError("neighborhood_size must be positive")
        safety_config.validate()
        self.neighborhood_size = neighborhood_size
        self.safety_config = safety_config

    @property
    def local_observation_dim(self) -> int:
        return OWN_FEATURE_DIM + self.neighborhood_size * NEIGHBOR_FEATURE_DIM

    @staticmethod
    def global_state_dim(num_bodies: int) -> int:
        if num_bodies <= 0:
            raise ValueError("num_bodies must be positive")
        return GLOBAL_BODY_FEATURE_DIM * num_bodies

    @staticmethod
    def _validate_bodies(bodies: Sequence[Any], agent_names: Sequence[str]) -> dict[str, Any]:
        names = [body.name for body in bodies]
        if any(name is None or name == "" for name in names):
            raise ValueError("every moving body must have a name for observation encoding")
        if len(names) != len(set(names)):
            raise ValueError("moving body names must be unique for observation encoding")
        body_by_name = dict(zip(names, bodies, strict=True))
        missing_agents = [name for name in agent_names if name not in body_by_name]
        if missing_agents:
            raise ValueError(
                f"agent bodies are missing from the moving-body list: {missing_agents}"
            )
        return body_by_name

    @staticmethod
    def _assessment_lookup(
        assessments: Iterable[PairSafetyAssessment],
    ) -> dict[frozenset[str], PairSafetyAssessment]:
        return {frozenset(assessment.pair): assessment for assessment in assessments}

    @staticmethod
    def _own_features(body: Any, maneuverable_names: set[str]) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(body.position, dtype=np.float32) / POSITION_SCALE_METERS,
                np.asarray(body.velocity, dtype=np.float32) / VELOCITY_SCALE_MPS,
                np.asarray([fuel_fraction(body, maneuverable_names)], dtype=np.float32),
            )
        )

    def _neighbor_features(
        self,
        observer: Any,
        neighbor: Any,
        assessment: PairSafetyAssessment,
        maneuverable_names: set[str],
    ) -> np.ndarray:
        rsw_basis = _rsw_basis(observer)
        relative_position = rsw_basis @ (
            np.asarray(neighbor.position, dtype=float) - np.asarray(observer.position, dtype=float)
        )
        relative_velocity = rsw_basis @ (
            np.asarray(neighbor.velocity, dtype=float) - np.asarray(observer.velocity, dtype=float)
        )
        normalized_miss_distance = min(
            assessment.predicted_miss_distance_meters
            / self.safety_config.safe_separation_meters,
            MAX_NORMALIZED_MISS_DISTANCE,
        )
        return np.concatenate(
            (
                relative_position.astype(np.float32) / POSITION_SCALE_METERS,
                relative_velocity.astype(np.float32) / VELOCITY_SCALE_MPS,
                np.asarray(
                    [
                        assessment.time_to_closest_approach_seconds
                        / self.safety_config.screening_horizon_seconds,
                        normalized_miss_distance,
                        assessment.combined_radius_meters
                        / self.safety_config.safe_separation_meters,
                        float(neighbor.name in maneuverable_names),
                        fuel_fraction(neighbor, maneuverable_names),
                        1.0,
                    ],
                    dtype=np.float32,
                ),
            )
        )

    @staticmethod
    def _relevance_key(
        assessment: PairSafetyAssessment, body_index: int
    ) -> tuple[bool, bool, float, float, int]:
        return (
            not assessment.is_collision,
            not assessment.is_unsafe,
            assessment.predicted_miss_distance_meters,
            assessment.time_to_closest_approach_seconds,
            body_index,
        )

    def encode(
        self,
        bodies: Sequence[Any],
        agent_names: Sequence[str],
        assessments: Sequence[PairSafetyAssessment] | None = None,
    ) -> ObservationState:
        """Build fixed-width local observations and the critic's full state."""
        bodies = list(bodies)
        agent_names = list(agent_names)
        if not agent_names:
            raise ValueError("at least one agent is required for observation encoding")
        body_by_name = self._validate_bodies(bodies, agent_names)
        maneuverable_names = set(agent_names)
        if assessments is None:
            assessments = assess_all_pairs(bodies, self.safety_config)
        assessment_by_pair = self._assessment_lookup(assessments)
        body_indices = {body.name: index for index, body in enumerate(bodies)}

        local_rows: list[np.ndarray] = []
        for agent_name in agent_names:
            observer = body_by_name[agent_name]
            ranked_neighbors: list[
                tuple[tuple[bool, bool, float, float, int], Any, PairSafetyAssessment]
            ] = []
            for neighbor in bodies:
                if neighbor.name == agent_name:
                    continue
                pair = frozenset((agent_name, neighbor.name))
                if pair not in assessment_by_pair:
                    raise ValueError(f"missing safety assessment for pair {tuple(sorted(pair))}")
                assessment = assessment_by_pair[pair]
                ranked_neighbors.append(
                    (
                        self._relevance_key(assessment, body_indices[neighbor.name]),
                        neighbor,
                        assessment,
                    )
                )
            ranked_neighbors.sort(key=lambda item: item[0])

            blocks = [
                self._neighbor_features(observer, neighbor, assessment, maneuverable_names)
                for _, neighbor, assessment in ranked_neighbors[: self.neighborhood_size]
            ]
            blocks.extend(
                np.zeros(NEIGHBOR_FEATURE_DIM, dtype=np.float32)
                for _ in range(self.neighborhood_size - len(blocks))
            )
            local_rows.append(
                np.concatenate((self._own_features(observer, maneuverable_names), *blocks))
            )

        return ObservationState(
            local_observations=np.stack(local_rows).astype(np.float32),
            global_state=self.global_state(bodies, agent_names),
        )

    def global_state(self, bodies: Sequence[Any], agent_names: Sequence[str]) -> np.ndarray:
        """Critic-only state: every moving body, agents first in ``agent_names`` order."""
        body_by_name = self._validate_bodies(bodies, agent_names)
        maneuverable_names = set(agent_names)
        ordered_bodies = [body_by_name[name] for name in agent_names]
        ordered_bodies.extend(body for body in bodies if body.name not in maneuverable_names)
        global_blocks = [
            np.concatenate(
                (
                    np.asarray(body.position, dtype=np.float32) / POSITION_SCALE_METERS,
                    np.asarray(body.velocity, dtype=np.float32) / VELOCITY_SCALE_MPS,
                    np.asarray(
                        [
                            float(body.radius) / self.safety_config.safe_separation_meters,
                            fuel_fraction(body, maneuverable_names),
                            float(body.name in maneuverable_names),
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            for body in ordered_bodies
        ]
        return np.concatenate(global_blocks).astype(np.float32)
