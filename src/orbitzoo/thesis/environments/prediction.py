"""Closest approach of object pairs along curved J2 orbits, for the features of each agent's top threat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITATIONAL_PARAMETER = 3.986004418e14
EARTH_RADIUS_M = 6_378_137.0
J2 = 1.08262668e-3


@dataclass(frozen=True)
class ClosestApproach:
    """Predicted encounter per pair; ``miss_vectors`` point from the first object to the second."""

    time_seconds: np.ndarray
    miss_distance_m: np.ndarray
    miss_vectors_m: np.ndarray
    relative_velocities_mps: np.ndarray
    first_positions_m: np.ndarray
    first_velocities_mps: np.ndarray


def _acceleration(positions: np.ndarray) -> np.ndarray:
    """Point-mass gravity plus the J2 oblateness term, with the z axis as Earth's pole."""
    radius = np.linalg.norm(positions, axis=-1, keepdims=True)
    polar = (positions[..., 2:3] / radius) ** 2
    oblateness = -1.5 * J2 * GRAVITATIONAL_PARAMETER * EARTH_RADIUS_M**2 / radius**5
    factors = np.concatenate((1 - 5 * polar, 1 - 5 * polar, 3 - 5 * polar), axis=-1)
    return -GRAVITATIONAL_PARAMETER * positions / radius**3 + oblateness * positions * factors


def _rk4_step(positions: np.ndarray, velocities: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    k1v, k1p = _acceleration(positions), velocities
    k2v, k2p = _acceleration(positions + 0.5 * step * k1p), velocities + 0.5 * step * k1v
    k3v, k3p = _acceleration(positions + 0.5 * step * k2p), velocities + 0.5 * step * k2v
    k4v, k4p = _acceleration(positions + step * k3p), velocities + step * k3v
    return (
        positions + step / 6 * (k1p + 2 * k2p + 2 * k3p + k4p),
        velocities + step / 6 * (k1v + 2 * k2v + 2 * k3v + k4v),
    )


def predict_closest_approach(
    first_positions: np.ndarray,
    first_velocities: np.ndarray,
    second_positions: np.ndarray,
    second_velocities: np.ndarray,
    horizon_seconds: float,
    step_seconds: float = 10.0,
) -> ClosestApproach:
    """Fly both objects of each pair over the horizon and refine the closest approach between samples."""
    pair_count = len(first_positions)
    states = np.concatenate((first_positions, second_positions)).astype(float)
    rates = np.concatenate((first_velocities, second_velocities)).astype(float)
    best_distance = np.full(pair_count, np.inf)
    best_time = np.zeros(pair_count)
    best_miss = np.zeros((pair_count, 3))
    best_relative_velocity = np.zeros((pair_count, 3))
    best_first = np.zeros((pair_count, 3))
    best_first_velocity = np.zeros((pair_count, 3))
    steps = max(1, int(np.ceil(horizon_seconds / step_seconds)))
    step_seconds = horizon_seconds / steps
    half = step_seconds / 2
    for index in range(steps + 1):
        elapsed = index * step_seconds
        relative = states[pair_count:] - states[:pair_count]
        relative_velocity = rates[pair_count:] - rates[:pair_count]
        speed_squared = np.einsum("ij,ij->i", relative_velocity, relative_velocity)
        offset = np.zeros(pair_count)
        np.divide(-np.einsum("ij,ij->i", relative, relative_velocity), speed_squared, out=offset, where=speed_squared > 1e-12)
        offset = np.clip(offset, 0.0 if index == 0 else -half, 0.0 if index == steps else half)
        miss = relative + relative_velocity * offset[:, np.newaxis]
        distance = np.linalg.norm(miss, axis=1)
        closer = distance < best_distance
        best_distance = np.where(closer, distance, best_distance)
        best_time = np.where(closer, elapsed + offset, best_time)
        best_miss[closer] = miss[closer]
        best_relative_velocity[closer] = relative_velocity[closer]
        best_first[closer] = states[:pair_count][closer] + rates[:pair_count][closer] * offset[closer, np.newaxis]
        best_first_velocity[closer] = rates[:pair_count][closer]
        if index < steps:
            states, rates = _rk4_step(states, rates, step_seconds)
    return ClosestApproach(best_time, best_distance, best_miss, best_relative_velocity, best_first, best_first_velocity)
