"""Fast, deterministic safety screening for collision-avoidance episodes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Protocol

import numpy as np


class KinematicBody(Protocol):
    name: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float


@dataclass(frozen=True)
class SafetyConfig:
    """Thresholds for fast conjunction screening, in SI units."""

    safe_separation_meters: float = 1_000.0
    screening_horizon_seconds: float = 1_800.0

    def validate(self) -> None:
        if self.safe_separation_meters <= 0 or self.screening_horizon_seconds <= 0:
            raise ValueError("safety thresholds must be positive")


@dataclass(frozen=True)
class PairSafetyAssessment:
    """Current and linearized future safety data for one body pair."""

    first_name: str
    second_name: str
    current_separation_meters: float
    combined_radius_meters: float
    time_to_closest_approach_seconds: float
    predicted_miss_distance_meters: float
    is_collision: bool
    is_unsafe: bool

    @property
    def pair(self) -> tuple[str, str]:
        return self.first_name, self.second_name


def assess_pair(first: KinematicBody, second: KinematicBody, config: SafetyConfig) -> PairSafetyAssessment:
    """Screen one pair using current range and bounded linear relative motion."""
    config.validate()
    relative_position = np.asarray(second.position, dtype=float) - np.asarray(first.position, dtype=float)
    relative_velocity = np.asarray(second.velocity, dtype=float) - np.asarray(first.velocity, dtype=float)
    current_separation = float(np.linalg.norm(relative_position))
    velocity_squared = float(np.dot(relative_velocity, relative_velocity))
    if velocity_squared <= 1e-12:
        time_to_closest_approach = 0.0
    else:
        unconstrained_tca = -float(np.dot(relative_position, relative_velocity)) / velocity_squared
        time_to_closest_approach = float(np.clip(unconstrained_tca, 0.0, config.screening_horizon_seconds))
    predicted_miss_distance = float(np.linalg.norm(relative_position + relative_velocity * time_to_closest_approach))
    combined_radius = float(first.radius + second.radius)
    return PairSafetyAssessment(
        first_name=first.name,
        second_name=second.name,
        current_separation_meters=current_separation,
        combined_radius_meters=combined_radius,
        time_to_closest_approach_seconds=time_to_closest_approach,
        predicted_miss_distance_meters=predicted_miss_distance,
        is_collision=current_separation <= combined_radius,
        is_unsafe=predicted_miss_distance <= config.safe_separation_meters,
    )


def assess_all_pairs(bodies: Iterable[KinematicBody], config: SafetyConfig) -> list[PairSafetyAssessment]:
    """Return safety assessments for every unique pair of moving bodies."""
    return [assess_pair(first, second, config) for first, second in combinations(bodies, 2)]


def recent_closest_approach(first: KinematicBody, second: KinematicBody, lookback_seconds: float) -> float | None:
    """Miss distance of a closest approach strictly within the last ``lookback_seconds``, from linear motion."""
    relative_position = np.asarray(second.position, dtype=float) - np.asarray(first.position, dtype=float)
    relative_velocity = np.asarray(second.velocity, dtype=float) - np.asarray(first.velocity, dtype=float)
    velocity_squared = float(np.dot(relative_velocity, relative_velocity))
    if velocity_squared <= 1e-12:
        return None
    offset = -float(np.dot(relative_position, relative_velocity)) / velocity_squared
    if not -lookback_seconds < offset < 0:
        return None
    return float(np.linalg.norm(relative_position + relative_velocity * offset))


def close_approaches_since(
    bodies: Iterable[KinematicBody], lookback_seconds: float, config: SafetyConfig
) -> dict[tuple[str, str], float]:
    """Pairs whose closest approach within the last ``lookback_seconds`` fell inside the safe separation."""
    approaches = {}
    for first, second in combinations(bodies, 2):
        miss = recent_closest_approach(first, second, lookback_seconds)
        if miss is not None and miss < config.safe_separation_meters:
            approaches[(first.name, second.name)] = miss
    return approaches


def involved_agents(assessments: Iterable[PairSafetyAssessment], predicate: str) -> set[str]:
    """Return body names involved in assessments whose boolean field is true."""
    return {
        name
        for assessment in assessments
        if getattr(assessment, predicate)
        for name in assessment.pair
    }
