"""Fine-resolution conjunction detection between agents and the full catalog."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ConjunctionEvent:
    """One close approach below the safe separation; indices are catalog positions."""

    first_index: int
    second_index: int
    tca_seconds: float
    miss_distance_meters: float
    is_collision: bool


def close_approaches(
    positions: np.ndarray,
    velocities: np.ndarray,
    agent_indices: np.ndarray,
    search_radius_meters: float,
    half_window_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Agent-involved pairs, their linear minimum distance, and its time offset within ±half window."""
    agent_tree = cKDTree(positions[agent_indices])
    catalog_tree = cKDTree(positions)
    matches = agent_tree.sparse_distance_matrix(catalog_tree, search_radius_meters, output_type="ndarray")
    first = agent_indices[matches["i"]]
    second = matches["j"]
    keep = first != second
    first, second = first[keep], second[keep]
    low, high = np.minimum(first, second), np.maximum(first, second)
    pairs = np.unique(np.stack((low, high), axis=1), axis=0) if low.size else np.empty((0, 2), dtype=np.intp)
    low, high = pairs[:, 0], pairs[:, 1]
    relative_positions = positions[high] - positions[low]
    relative_velocities = velocities[high] - velocities[low]
    speed_squared = np.einsum("ij,ij->i", relative_velocities, relative_velocities)
    closing = np.einsum("ij,ij->i", relative_positions, relative_velocities)
    offsets = np.zeros_like(closing)
    np.divide(-closing, speed_squared, out=offsets, where=speed_squared > 1e-12)
    offsets = np.clip(offsets, -half_window_seconds, half_window_seconds)
    distances = np.linalg.norm(relative_positions + relative_velocities * offsets[:, np.newaxis], axis=1)
    return low, high, distances, offsets


class ConjunctionTracker:
    """Collects sub-threshold samples and merges them into one event per encounter."""

    def __init__(self, safe_separation_meters: float, radii: np.ndarray, merge_gap_seconds: float) -> None:
        self.safe_separation = safe_separation_meters
        self.radii = radii
        self.merge_gap = merge_gap_seconds
        self._samples: list[np.ndarray] = []

    def add(self, time_seconds: float, low: np.ndarray, high: np.ndarray, distances: np.ndarray, offsets: np.ndarray) -> None:
        close = distances <= self.safe_separation
        if close.any():
            self._samples.append(
                np.stack((low[close], high[close], time_seconds + offsets[close], distances[close]), axis=1)
            )

    def events(self) -> list[ConjunctionEvent]:
        if not self._samples:
            return []
        samples = np.concatenate(self._samples)
        samples = samples[np.lexsort((samples[:, 2], samples[:, 1], samples[:, 0]))]
        events: list[ConjunctionEvent] = []
        start = 0
        for index in range(1, len(samples) + 1):
            is_boundary = (
                index == len(samples)
                or samples[index, 0] != samples[start, 0]
                or samples[index, 1] != samples[start, 1]
                or samples[index, 2] - samples[index - 1, 2] > self.merge_gap
            )
            if not is_boundary:
                continue
            group = samples[start:index]
            closest = group[np.argmin(group[:, 3])]
            low, high = int(closest[0]), int(closest[1])
            events.append(
                ConjunctionEvent(
                    first_index=low,
                    second_index=high,
                    tca_seconds=float(closest[2]),
                    miss_distance_meters=float(closest[3]),
                    is_collision=bool(closest[3] <= self.radii[low] + self.radii[high]),
                )
            )
            start = index
        return sorted(events, key=lambda event: (event.tca_seconds, event.first_index, event.second_index))


def match_events(
    reference: list[ConjunctionEvent], candidate: list[ConjunctionEvent], tolerance_seconds: float
) -> tuple[list[bool], list[ConjunctionEvent]]:
    """Flag which reference events recur in ``candidate``, and return candidate events absent from ``reference``."""
    open_reference: dict[tuple[int, int], list[int]] = {}
    for index, event in enumerate(reference):
        open_reference.setdefault((event.first_index, event.second_index), []).append(index)
    recurs = [False] * len(reference)
    unmatched: list[ConjunctionEvent] = []
    for event in candidate:
        indices = open_reference.get((event.first_index, event.second_index), [])
        match = next(
            (index for index in indices if abs(reference[index].tca_seconds - event.tca_seconds) <= tolerance_seconds),
            None,
        )
        if match is None:
            unmatched.append(event)
        else:
            recurs[match] = True
            indices.remove(match)
    return recurs, unmatched
