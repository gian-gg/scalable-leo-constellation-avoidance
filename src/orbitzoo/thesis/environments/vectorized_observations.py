"""Vectorized actor observations for thousands of agents against a full catalog.

Produces the same rows as ``LocalObservationEncoder`` without building body objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from orbitzoo.thesis.environments.observations import (
    MAX_NORMALIZED_MISS_DISTANCE,
    NEIGHBOR_FEATURE_DIM,
    OWN_FEATURE_DIM,
    POSITION_SCALE_METERS,
    VELOCITY_SCALE_MPS,
)
from orbitzoo.thesis.environments.prediction import predict_closest_approach
from orbitzoo.thesis.environments.safety import SafetyConfig

DEFAULT_AGENT_BATCH_SIZE = 64


@dataclass(frozen=True)
class CatalogState:
    """Positions, velocities, and per-object attributes at one epoch (SI units)."""

    positions: np.ndarray
    velocities: np.ndarray
    radii: np.ndarray
    is_agent: np.ndarray
    fuel_fractions: np.ndarray


def rsw_bases(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Rows of radial, along-track, and cross-track unit vectors, shape [n, 3, 3]."""
    radial = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    cross_track = np.cross(positions, velocities)
    cross_track /= np.linalg.norm(cross_track, axis=1, keepdims=True)
    along_track = np.cross(cross_track, radial)
    return np.stack((radial, along_track, cross_track), axis=1)


@dataclass(frozen=True)
class RankedBatch:
    """Screening quantities for one batch of agents and their ranked neighbours."""

    start: int
    relative_positions: np.ndarray
    relative_velocities: np.ndarray
    tca: np.ndarray
    miss: np.ndarray
    combined_radii: np.ndarray
    neighbors: list[np.ndarray]


def ranked_batches(
    state: CatalogState,
    agent_indices: np.ndarray,
    retained: int,
    safety: SafetyConfig,
    agent_batch_size: int = DEFAULT_AGENT_BATCH_SIZE,
) -> Iterator[RankedBatch]:
    """Rank every object for each agent with the training encoder's threat order."""
    horizon = safety.screening_horizon_seconds
    object_order = np.arange(state.positions.shape[0])
    for start in range(0, agent_indices.size, agent_batch_size):
        batch = agent_indices[start : start + agent_batch_size]
        rows = np.arange(batch.size)
        relative_positions = state.positions[np.newaxis, :, :] - state.positions[batch, np.newaxis, :]
        relative_velocities = state.velocities[np.newaxis, :, :] - state.velocities[batch, np.newaxis, :]
        separations = np.linalg.norm(relative_positions, axis=2)
        speed_squared = np.einsum("bij,bij->bi", relative_velocities, relative_velocities)
        closing = np.einsum("bij,bij->bi", relative_positions, relative_velocities)
        tca = np.zeros_like(closing)
        np.divide(-closing, speed_squared, out=tca, where=speed_squared > 1e-12)
        tca = np.clip(tca, 0.0, horizon)
        miss = np.linalg.norm(relative_positions + relative_velocities * tca[:, :, np.newaxis], axis=2)
        combined_radii = state.radii[np.newaxis, :] + state.radii[batch, np.newaxis]
        not_collision = ~(separations <= combined_radii)
        not_unsafe = ~(miss <= safety.safe_separation_meters)
        not_collision[rows, batch] = True
        not_unsafe[rows, batch] = True
        miss[rows, batch] = np.inf
        tca[rows, batch] = np.inf

        # Monotone scalar key for (collision, unsafe, miss); exact ties are re-sorted below.
        priority = (not_collision.astype(np.float64) * 2 + not_unsafe) * 1e12 + miss
        priority[rows, batch] = np.inf
        cutoff = np.partition(priority, retained - 1, axis=1)[:, retained - 1]
        neighbors = []
        for row in rows:
            candidates = np.flatnonzero(priority[row] <= cutoff[row])
            order = np.lexsort(
                (
                    object_order[candidates],
                    tca[row, candidates],
                    miss[row, candidates],
                    not_unsafe[row, candidates],
                    not_collision[row, candidates],
                )
            )
            neighbors.append(candidates[order[:retained]])
        yield RankedBatch(start, relative_positions, relative_velocities, tca, miss, combined_radii, neighbors)


def top_neighbors(
    state: CatalogState, agent_indices: np.ndarray, neighborhood_size: int, safety: SafetyConfig
) -> np.ndarray:
    """Indices of each agent's top-``k`` ranked neighbours, padded with -1."""
    agent_indices = np.asarray(agent_indices, dtype=np.intp)
    result = np.full((agent_indices.size, neighborhood_size), -1, dtype=np.intp)
    retained = min(neighborhood_size, state.positions.shape[0] - 1)
    if retained <= 0:
        return result
    for ranked in ranked_batches(state, agent_indices, retained, safety):
        for row, neighbors in enumerate(ranked.neighbors):
            result[ranked.start + row, : neighbors.size] = neighbors
    return result


def encode_local_observations(
    state: CatalogState,
    agent_indices: np.ndarray,
    neighborhood_size: int,
    safety: SafetyConfig,
    *,
    agent_batch_size: int = DEFAULT_AGENT_BATCH_SIZE,
) -> np.ndarray:
    """Rank every catalog object for each agent and encode its top-``k`` threats."""
    agent_indices = np.asarray(agent_indices, dtype=np.intp)
    object_count = state.positions.shape[0]
    retained = min(neighborhood_size, object_count - 1)
    width = OWN_FEATURE_DIM + neighborhood_size * NEIGHBOR_FEATURE_DIM
    observations = np.zeros((agent_indices.size, width), dtype=np.float32)
    observations[:, 0:3] = state.positions[agent_indices].astype(np.float32) / POSITION_SCALE_METERS
    observations[:, 3:6] = state.velocities[agent_indices].astype(np.float32) / VELOCITY_SCALE_MPS
    observations[:, 6] = state.fuel_fractions[agent_indices]
    if retained <= 0:
        return observations

    horizon = safety.screening_horizon_seconds
    safe_separation = safety.safe_separation_meters
    bases = rsw_bases(state.positions[agent_indices], state.velocities[agent_indices])
    chosen: list[tuple[int, int, int]] = []
    for ranked in ranked_batches(state, agent_indices, retained, safety, agent_batch_size):
        for row, neighbors in enumerate(ranked.neighbors):
            basis = bases[ranked.start + row]
            for slot, neighbor in enumerate(neighbors):
                offset = OWN_FEATURE_DIM + slot * NEIGHBOR_FEATURE_DIM
                block = observations[ranked.start + row, offset : offset + NEIGHBOR_FEATURE_DIM]
                block[0:3] = (basis @ ranked.relative_positions[row, neighbor]).astype(np.float32) / POSITION_SCALE_METERS
                block[3:6] = (basis @ ranked.relative_velocities[row, neighbor]).astype(np.float32) / VELOCITY_SCALE_MPS
                block[6] = ranked.tca[row, neighbor] / horizon
                block[7] = min(ranked.miss[row, neighbor] / safe_separation, MAX_NORMALIZED_MISS_DISTANCE)
                block[8] = ranked.combined_radii[row, neighbor] / safe_separation
                block[9] = float(state.is_agent[neighbor])
                block[10] = state.fuel_fractions[neighbor] if state.is_agent[neighbor] else 0.0
                block[11] = 1.0
                chosen.append((ranked.start + row, slot, int(neighbor)))
    if safety.threat_prediction == "j2" and chosen:
        apply_curved_prediction(
            observations,
            chosen,
            state.positions[agent_indices],
            state.velocities[agent_indices],
            state.positions,
            state.velocities,
            safety,
        )
    return observations


def apply_curved_prediction(
    observations: np.ndarray,
    chosen: list[tuple[int, int, int]],
    agent_positions: np.ndarray,
    agent_velocities: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    safety: SafetyConfig,
) -> None:
    """Replace each chosen neighbour's time-to-closest-approach and miss features with the J2 prediction."""
    rows = np.array([item[0] for item in chosen])
    slots = np.array([item[1] for item in chosen])
    neighbors = np.array([item[2] for item in chosen])
    approach = predict_closest_approach(
        agent_positions[rows],
        agent_velocities[rows],
        positions[neighbors],
        velocities[neighbors],
        safety.screening_horizon_seconds,
        safety.prediction_step_seconds,
    )
    columns = OWN_FEATURE_DIM + slots * NEIGHBOR_FEATURE_DIM
    observations[rows, columns + 6] = approach.time_seconds / safety.screening_horizon_seconds
    observations[rows, columns + 7] = np.minimum(
        approach.miss_distance_m / safety.safe_separation_meters, MAX_NORMALIZED_MISS_DISTANCE
    )
