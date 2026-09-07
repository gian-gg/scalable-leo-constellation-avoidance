"""Vectorized selected-agent neighbor ranking at shared decision epochs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence

import numpy as np

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import (
    CartesianStateFrame,
    CatalogObject,
    RankedNeighbor,
    RankedNeighborFrame,
)
from orbitzoo.thesis.calibration.propagation import (
    DEFAULT_BATCH_SIZE,
    SGP4Propagation,
    propagate_objects,
)


@dataclass(frozen=True)
class DecisionSchedule:
    """Union of all candidate decision-interval timelines."""

    start_epoch_utc: datetime
    end_epoch_utc: datetime
    epochs_by_interval: tuple[tuple[int, tuple[datetime, ...]], ...]
    shared_epochs: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if (
            self.start_epoch_utc.tzinfo is None
            or self.start_epoch_utc.utcoffset() is None
        ):
            raise ValueError("decision schedule start must be timezone-aware")
        if (
            self.end_epoch_utc.tzinfo is None
            or self.end_epoch_utc.utcoffset() is None
        ):
            raise ValueError("decision schedule end must be timezone-aware")
        start = self.start_epoch_utc.astimezone(timezone.utc)
        end = self.end_epoch_utc.astimezone(timezone.utc)
        if end <= start:
            raise ValueError("decision schedule end must follow its start")
        interval_values = tuple(interval for interval, _ in self.epochs_by_interval)
        if (
            not interval_values
            or any(
                isinstance(interval, bool)
                or not isinstance(interval, int)
                or interval <= 0
                for interval in interval_values
            )
            or interval_values != tuple(sorted(set(interval_values)))
        ):
            raise ValueError("decision intervals must be positive and sorted")
        normalized_groups: list[tuple[int, tuple[datetime, ...]]] = []
        for interval, epochs in self.epochs_by_interval:
            if any(
                epoch.tzinfo is None or epoch.utcoffset() is None
                for epoch in epochs
            ):
                raise ValueError("decision epochs must be timezone-aware")
            normalized_epochs = tuple(
                epoch.astimezone(timezone.utc) for epoch in epochs
            )
            if not normalized_epochs or normalized_epochs[0] != start:
                raise ValueError("every decision interval must begin at schedule start")
            if any(
                epoch < start or epoch >= end for epoch in normalized_epochs
            ) or any(
                first >= second
                for first, second in zip(normalized_epochs, normalized_epochs[1:])
            ):
                raise ValueError("decision epochs must be increasing within the schedule")
            normalized_groups.append((interval, normalized_epochs))
        expected_shared = tuple(
            sorted(
                {
                    epoch
                    for _, epochs in normalized_groups
                    for epoch in epochs
                }
            )
        )
        if any(
            epoch.tzinfo is None or epoch.utcoffset() is None
            for epoch in self.shared_epochs
        ):
            raise ValueError("shared decision epochs must be timezone-aware")
        normalized_shared = tuple(
            epoch.astimezone(timezone.utc) for epoch in self.shared_epochs
        )
        if normalized_shared != expected_shared:
            raise ValueError("shared epochs must equal the union of interval epochs")
        object.__setattr__(self, "start_epoch_utc", start)
        object.__setattr__(self, "end_epoch_utc", end)
        object.__setattr__(self, "epochs_by_interval", tuple(normalized_groups))
        object.__setattr__(self, "shared_epochs", normalized_shared)

    def epochs_for(self, decision_interval_seconds: int) -> tuple[datetime, ...]:
        for interval, epochs in self.epochs_by_interval:
            if interval == decision_interval_seconds:
                return epochs
        raise ValueError(
            f"unknown candidate decision interval {decision_interval_seconds}"
        )

    def includes(self, epoch_utc: datetime, decision_interval_seconds: int) -> bool:
        return epoch_utc in self.epochs_for(decision_interval_seconds)


def build_decision_schedule(
    start_epoch_utc: datetime,
    config: CalibrationConfig,
) -> DecisionSchedule:
    """Build each delta-t timeline and deduplicate their shared epochs."""
    config.validate()
    if start_epoch_utc.tzinfo is None or start_epoch_utc.utcoffset() is None:
        raise ValueError("decision schedule start must be timezone-aware")
    start = start_epoch_utc.astimezone(timezone.utc)
    duration_seconds = config.propagation.duration_seconds
    end = start + timedelta(seconds=duration_seconds)
    epochs_by_interval = tuple(
        (
            interval,
            tuple(
                start + timedelta(seconds=offset)
                for offset in range(0, duration_seconds, interval)
            ),
        )
        for interval in config.sweep.decision_intervals_seconds
    )
    shared_epochs = tuple(
        sorted(
            {
                epoch
                for _, interval_epochs in epochs_by_interval
                for epoch in interval_epochs
            }
        )
    )
    return DecisionSchedule(
        start_epoch_utc=start,
        end_epoch_utc=end,
        epochs_by_interval=epochs_by_interval,
        shared_epochs=shared_epochs,
    )


def _validated_agent_ids(
    agent_norad_ids: Iterable[int],
    object_by_id: dict[int, CatalogObject],
) -> tuple[int, ...]:
    identifiers = tuple(agent_norad_ids)
    if not identifiers:
        raise ValueError("at least one selected agent NORAD ID is required")
    if any(
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier <= 0
        for identifier in identifiers
    ):
        raise ValueError("selected agent NORAD IDs must be positive integers")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("selected agent NORAD IDs must be unique")
    absent_ids = set(identifiers).difference(object_by_id)
    if absent_ids:
        raise ValueError(
            f"selected agents are absent from the state frame: {sorted(absent_ids)}"
        )
    ineligible_ids = sorted(
        identifier
        for identifier in identifiers
        if not object_by_id[identifier].is_agent_candidate
    )
    if ineligible_ids:
        raise ValueError(
            "selected agents are not metadata-approved agent candidates: "
            f"{ineligible_ids}"
        )
    return tuple(sorted(identifiers))


def rank_neighbors_at_epoch(
    frame: CartesianStateFrame,
    objects: Sequence[CatalogObject],
    config: CalibrationConfig,
    *,
    agent_norad_ids: Iterable[int],
    maximum_neighbors: int | None = None,
    agent_batch_size: int = 16,
) -> RankedNeighborFrame:
    """Rank the full catalog for each selected agent using vectorized batches."""
    config.validate()
    if maximum_neighbors is None:
        maximum_neighbors = max(config.sweep.neighborhood_sizes)
    if (
        isinstance(maximum_neighbors, bool)
        or not isinstance(maximum_neighbors, int)
        or maximum_neighbors <= 0
    ):
        raise ValueError("maximum_neighbors must be a positive integer")
    if (
        isinstance(agent_batch_size, bool)
        or not isinstance(agent_batch_size, int)
        or agent_batch_size <= 0
    ):
        raise ValueError("agent_batch_size must be a positive integer")

    object_records = tuple(objects)
    object_by_id = {item.norad_id: item for item in object_records}
    if len(object_by_id) != len(object_records):
        raise ValueError("catalog object NORAD IDs must be unique")
    if set(frame.norad_ids) != set(object_by_id):
        raise ValueError("state frame and catalog objects must contain the same IDs")
    agent_ids = _validated_agent_ids(agent_norad_ids, object_by_id)
    index_by_id = {
        identifier: index for index, identifier in enumerate(frame.norad_ids)
    }
    agent_indices = np.asarray(
        [index_by_id[identifier] for identifier in agent_ids],
        dtype=np.intp,
    )
    catalog_ids = np.asarray(frame.norad_ids, dtype=np.int64)
    radii = np.asarray(
        [object_by_id[identifier].radius_meters for identifier in frame.norad_ids],
        dtype=np.float64,
    )
    retained_neighbor_count = min(maximum_neighbors, len(frame.norad_ids) - 1)
    horizon_seconds = config.safety.screening_horizon_seconds
    safe_separation_meters = config.safety.safe_separation_meters
    rankings: list[RankedNeighbor] = []

    for batch_start in range(0, len(agent_indices), agent_batch_size):
        batch_indices = agent_indices[batch_start : batch_start + agent_batch_size]
        relative_positions = (
            frame.positions_m[np.newaxis, :, :]
            - frame.positions_m[batch_indices, np.newaxis, :]
        )
        relative_velocities = (
            frame.velocities_mps[np.newaxis, :, :]
            - frame.velocities_mps[batch_indices, np.newaxis, :]
        )
        current_separations = np.linalg.norm(relative_positions, axis=2)
        relative_speed_squared = np.einsum(
            "bij,bij->bi",
            relative_velocities,
            relative_velocities,
        )
        relative_dot = np.einsum(
            "bij,bij->bi",
            relative_positions,
            relative_velocities,
        )
        unconstrained_tca = np.zeros_like(relative_dot)
        moving = relative_speed_squared > 1e-12
        np.divide(
            -relative_dot,
            relative_speed_squared,
            out=unconstrained_tca,
            where=moving,
        )
        tca_seconds = np.clip(unconstrained_tca, 0.0, horizon_seconds)
        predicted_positions = (
            relative_positions + relative_velocities * tca_seconds[:, :, np.newaxis]
        )
        predicted_miss_distances = np.linalg.norm(predicted_positions, axis=2)
        combined_radii = radii[np.newaxis, :] + radii[batch_indices, np.newaxis]
        collisions = current_separations <= combined_radii
        unsafe = predicted_miss_distances <= safe_separation_meters

        for row, agent_index in enumerate(batch_indices):
            self_index = int(agent_index)
            current_separations[row, self_index] = np.inf
            predicted_miss_distances[row, self_index] = np.inf
            tca_seconds[row, self_index] = np.inf
            collisions[row, self_index] = False
            unsafe[row, self_index] = False
            order = np.lexsort(
                (
                    catalog_ids,
                    tca_seconds[row],
                    predicted_miss_distances[row],
                    np.logical_not(unsafe[row]),
                    np.logical_not(collisions[row]),
                )
            )[:retained_neighbor_count]
            agent_id = int(catalog_ids[self_index])
            for rank, neighbor_index in enumerate(order, start=1):
                neighbor = int(neighbor_index)
                rankings.append(
                    RankedNeighbor(
                        agent_norad_id=agent_id,
                        neighbor_norad_id=int(catalog_ids[neighbor]),
                        rank=rank,
                        decision_epoch_utc=frame.epoch_utc,
                        current_separation_meters=float(
                            current_separations[row, neighbor]
                        ),
                        time_to_closest_approach_seconds=float(
                            tca_seconds[row, neighbor]
                        ),
                        predicted_miss_distance_meters=float(
                            predicted_miss_distances[row, neighbor]
                        ),
                        combined_radius_meters=float(combined_radii[row, neighbor]),
                        is_collision=bool(collisions[row, neighbor]),
                        is_unsafe=bool(unsafe[row, neighbor]),
                    )
                )

    rankings.sort(key=lambda item: (item.agent_norad_id, item.rank))
    return RankedNeighborFrame(
        decision_epoch_utc=frame.epoch_utc,
        agent_norad_ids=agent_ids,
        rankings=tuple(rankings),
    )


def iter_ranked_neighbor_frames(
    propagation: SGP4Propagation,
    config: CalibrationConfig,
    *,
    agent_norad_ids: Iterable[int],
    maximum_neighbors: int | None = None,
    agent_batch_size: int = 16,
    propagation_batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[RankedNeighborFrame]:
    """Stream rankings once at the union of all candidate delta-t epochs."""
    schedule = build_decision_schedule(propagation.start_epoch_utc, config)
    expected_end = propagation.start_epoch_utc + timedelta(
        seconds=config.propagation.duration_seconds
    )
    if propagation.end_epoch_utc != expected_end:
        raise ValueError("propagation timeline does not match calibration duration")
    identifiers = tuple(agent_norad_ids)
    frames = propagate_objects(
        propagation.objects,
        schedule.shared_epochs,
        batch_size=propagation_batch_size,
    )
    for frame in frames:
        yield rank_neighbors_at_epoch(
            frame,
            propagation.objects,
            config,
            agent_norad_ids=identifiers,
            maximum_neighbors=maximum_neighbors,
            agent_batch_size=agent_batch_size,
        )
