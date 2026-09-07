"""Validated records exchanged by the offline calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math

import numpy as np
from numpy.typing import NDArray


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _finite_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _utc(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware UTC datetime")
    offset = value.utcoffset()
    if offset is None:
        raise ValueError(f"{name} must be a timezone-aware UTC datetime")
    return value.astimezone(timezone.utc)


def _norad_ids(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(values)
    for value in normalized:
        _positive_integer(name, value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique identifiers")
    return normalized


class ObjectType(str, Enum):
    """Supported catalog object classifications."""

    PAYLOAD = "payload"
    ROCKET_BODY = "rocket_body"
    DEBRIS = "debris"
    UNKNOWN = "unknown"


class EvaluationSplit(str, Enum):
    """Independent seed group used to produce combination metrics."""

    CALIBRATION = "calibration"
    VALIDATION = "validation"


@dataclass(frozen=True)
class CatalogObject:
    """One validated TLE joined with optional project metadata."""

    norad_id: int
    name: str
    tle_name: str | None
    line1: str
    line2: str
    tle_epoch_utc: datetime
    object_type: ObjectType
    is_agent_candidate: bool
    radius_meters: float
    constellation: str | None
    has_metadata: bool

    def __post_init__(self) -> None:
        _positive_integer("norad_id", self.norad_id)
        if not self.name:
            raise ValueError("name cannot be empty")
        if not isinstance(self.object_type, ObjectType):
            raise ValueError("object_type must be an ObjectType")
        if not isinstance(self.is_agent_candidate, bool):
            raise ValueError("is_agent_candidate must be boolean")
        if self.is_agent_candidate and self.object_type is not ObjectType.PAYLOAD:
            raise ValueError("only payloads may be agent candidates")
        if not math.isfinite(self.radius_meters) or self.radius_meters <= 0.0:
            raise ValueError("radius_meters must be finite and positive")
        if not isinstance(self.has_metadata, bool):
            raise ValueError("has_metadata must be boolean")
        object.__setattr__(
            self,
            "tle_epoch_utc",
            _utc("tle_epoch_utc", self.tle_epoch_utc),
        )


@dataclass(frozen=True)
class LoadedCatalog:
    """Retained objects and freshness-filter audit information."""

    objects: tuple[CatalogObject, ...]
    latest_epoch_utc: datetime
    source_record_count: int
    stale_filtered_norad_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        if not objects:
            raise ValueError("a loaded catalog cannot be empty")
        retained_ids = tuple(item.norad_id for item in objects)
        _norad_ids("catalog object NORAD IDs", retained_ids)
        stale_ids = _norad_ids(
            "stale_filtered_norad_ids",
            tuple(self.stale_filtered_norad_ids),
        )
        if set(retained_ids).intersection(stale_ids):
            raise ValueError("retained and stale catalog identifiers must be disjoint")
        _nonnegative_integer("source_record_count", self.source_record_count)
        if self.source_record_count != len(objects) + len(stale_ids):
            raise ValueError("source_record_count must equal retained plus stale records")
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "stale_filtered_norad_ids", stale_ids)
        object.__setattr__(
            self,
            "latest_epoch_utc",
            _utc("latest_epoch_utc", self.latest_epoch_utc),
        )


@dataclass(frozen=True)
class CartesianStateFrame:
    """TEME Cartesian states for every retained object at one UTC epoch."""

    epoch_utc: datetime
    norad_ids: tuple[int, ...]
    positions_m: NDArray[np.float64]
    velocities_mps: NDArray[np.float64]
    reference_frame: str = "TEME"

    def __post_init__(self) -> None:
        norad_ids = _norad_ids("norad_ids", tuple(self.norad_ids))
        if not norad_ids:
            raise ValueError("a Cartesian state frame cannot be empty")
        if self.reference_frame != "TEME":
            raise ValueError("calibration Cartesian states must use the TEME frame")
        positions = np.asarray(self.positions_m, dtype=np.float64).copy()
        velocities = np.asarray(self.velocities_mps, dtype=np.float64).copy()
        expected_shape = (len(norad_ids), 3)
        if positions.shape != expected_shape:
            raise ValueError(f"positions_m must have shape {expected_shape}")
        if velocities.shape != expected_shape:
            raise ValueError(f"velocities_mps must have shape {expected_shape}")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must contain only finite values")
        if not np.all(np.isfinite(velocities)):
            raise ValueError("velocities_mps must contain only finite values")
        positions.flags.writeable = False
        velocities.flags.writeable = False
        object.__setattr__(self, "epoch_utc", _utc("epoch_utc", self.epoch_utc))
        object.__setattr__(self, "norad_ids", norad_ids)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)


@dataclass(frozen=True)
class AgentSelection:
    """Deterministic set of maneuverable payloads selected for one seed."""

    seed: int
    requested_agent_count: int
    agent_norad_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _nonnegative_integer("seed", self.seed)
        _positive_integer("requested_agent_count", self.requested_agent_count)
        identifiers = _norad_ids("agent_norad_ids", tuple(self.agent_norad_ids))
        if len(identifiers) != self.requested_agent_count:
            raise ValueError(
                "agent_norad_ids length must equal requested_agent_count"
            )
        object.__setattr__(self, "agent_norad_ids", identifiers)


@dataclass(frozen=True)
class ReferenceConjunction:
    """One propagated reference conjunction between a canonical object pair."""

    first_norad_id: int
    second_norad_id: int
    tca_epoch_utc: datetime
    miss_distance_meters: float
    relative_speed_mps: float
    combined_radius_meters: float

    def __post_init__(self) -> None:
        _positive_integer("first_norad_id", self.first_norad_id)
        _positive_integer("second_norad_id", self.second_norad_id)
        if self.first_norad_id >= self.second_norad_id:
            raise ValueError("conjunction NORAD IDs must be in ascending order")
        for name in (
            "miss_distance_meters",
            "relative_speed_mps",
        ):
            _finite_nonnegative(name, getattr(self, name))
        _finite_positive("combined_radius_meters", self.combined_radius_meters)
        object.__setattr__(
            self,
            "tca_epoch_utc",
            _utc("tca_epoch_utc", self.tca_epoch_utc),
        )


@dataclass(frozen=True)
class RankedNeighbor:
    """One neighbor's threat rank for an agent at a decision epoch."""

    agent_norad_id: int
    neighbor_norad_id: int
    rank: int
    decision_epoch_utc: datetime
    current_separation_meters: float
    time_to_closest_approach_seconds: float
    predicted_miss_distance_meters: float
    combined_radius_meters: float
    is_collision: bool
    is_unsafe: bool

    def __post_init__(self) -> None:
        _positive_integer("agent_norad_id", self.agent_norad_id)
        _positive_integer("neighbor_norad_id", self.neighbor_norad_id)
        if self.agent_norad_id == self.neighbor_norad_id:
            raise ValueError("an agent cannot be its own ranked neighbor")
        _positive_integer("rank", self.rank)
        for name in (
            "current_separation_meters",
            "time_to_closest_approach_seconds",
            "predicted_miss_distance_meters",
        ):
            _finite_nonnegative(name, getattr(self, name))
        _finite_positive("combined_radius_meters", self.combined_radius_meters)
        if not isinstance(self.is_collision, bool) or not isinstance(
            self.is_unsafe, bool
        ):
            raise ValueError("ranked-neighbor flags must be boolean")
        object.__setattr__(
            self,
            "decision_epoch_utc",
            _utc("decision_epoch_utc", self.decision_epoch_utc),
        )


@dataclass(frozen=True)
class CombinationMetrics:
    """Detection metrics for one k/delta-t combination and sampled population."""

    evaluation_split: EvaluationSplit
    selection_seed: int
    agent_count: int
    neighborhood_size: int
    decision_interval_seconds: int
    reference_conjunction_count: int
    detected_conjunction_count: int
    timely_detected_conjunction_count: int
    runtime_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_split, EvaluationSplit):
            raise ValueError("evaluation_split must be an EvaluationSplit")
        _nonnegative_integer("selection_seed", self.selection_seed)
        for name in (
            "agent_count",
            "neighborhood_size",
            "decision_interval_seconds",
        ):
            _positive_integer(name, getattr(self, name))
        for name in (
            "reference_conjunction_count",
            "detected_conjunction_count",
            "timely_detected_conjunction_count",
        ):
            _nonnegative_integer(name, getattr(self, name))
        if self.detected_conjunction_count > self.reference_conjunction_count:
            raise ValueError("detected conjunctions cannot exceed reference conjunctions")
        if self.timely_detected_conjunction_count > self.detected_conjunction_count:
            raise ValueError("timely detections cannot exceed detected conjunctions")
        _finite_nonnegative("runtime_seconds", self.runtime_seconds)

    @property
    def missed_conjunction_count(self) -> int:
        return self.reference_conjunction_count - self.detected_conjunction_count

    @property
    def threat_recall(self) -> float:
        if self.reference_conjunction_count == 0:
            return 0.0
        return self.detected_conjunction_count / self.reference_conjunction_count

    @property
    def timely_detection_fraction(self) -> float:
        if self.reference_conjunction_count == 0:
            return 0.0
        return self.timely_detected_conjunction_count / self.reference_conjunction_count


@dataclass(frozen=True)
class CalibrationRecommendation:
    """Final selected parameters with calibration and validation evidence."""

    neighborhood_size: int
    decision_interval_seconds: int
    minimum_threat_recall: float
    minimum_timely_detection_fraction: float
    minimum_decisions_before_tca: int
    calibration_metrics: tuple[CombinationMetrics, ...]
    validation_metrics: tuple[CombinationMetrics, ...]

    def __post_init__(self) -> None:
        _positive_integer("neighborhood_size", self.neighborhood_size)
        _positive_integer("decision_interval_seconds", self.decision_interval_seconds)
        _positive_integer("minimum_decisions_before_tca", self.minimum_decisions_before_tca)
        for name in (
            "minimum_threat_recall",
            "minimum_timely_detection_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        calibration_metrics = tuple(self.calibration_metrics)
        validation_metrics = tuple(self.validation_metrics)
        if not calibration_metrics or not validation_metrics:
            raise ValueError("final recommendations require both metric groups")
        self._validate_metrics(calibration_metrics, EvaluationSplit.CALIBRATION)
        self._validate_metrics(validation_metrics, EvaluationSplit.VALIDATION)
        object.__setattr__(self, "calibration_metrics", calibration_metrics)
        object.__setattr__(self, "validation_metrics", validation_metrics)

    def _validate_metrics(
        self,
        metrics: tuple[CombinationMetrics, ...],
        expected_split: EvaluationSplit,
    ) -> None:
        seen_samples: set[tuple[int, int]] = set()
        for metric in metrics:
            if metric.evaluation_split is not expected_split:
                raise ValueError(f"metrics must belong to the {expected_split.value} split")
            if (
                metric.neighborhood_size != self.neighborhood_size
                or metric.decision_interval_seconds != self.decision_interval_seconds
            ):
                raise ValueError("recommendation metrics must match selected k and delta-t")
            sample = metric.selection_seed, metric.agent_count
            if sample in seen_samples:
                raise ValueError("recommendation metrics cannot duplicate a sampled population")
            seen_samples.add(sample)

    @staticmethod
    def _aggregate_rates(
        metrics: tuple[CombinationMetrics, ...],
    ) -> tuple[float, float]:
        reference_count = sum(item.reference_conjunction_count for item in metrics)
        if reference_count == 0:
            return 0.0, 0.0
        detected_count = sum(item.detected_conjunction_count for item in metrics)
        timely_count = sum(item.timely_detected_conjunction_count for item in metrics)
        return detected_count / reference_count, timely_count / reference_count

    def _passes(self, metrics: tuple[CombinationMetrics, ...]) -> bool:
        recall, timely_fraction = self._aggregate_rates(metrics)
        return (
            recall >= self.minimum_threat_recall
            and timely_fraction >= self.minimum_timely_detection_fraction
        )

    @property
    def calibration_passed(self) -> bool:
        return self._passes(self.calibration_metrics)

    @property
    def validation_passed(self) -> bool:
        return self._passes(self.validation_metrics)

    @property
    def is_accepted(self) -> bool:
        return self.calibration_passed and self.validation_passed
