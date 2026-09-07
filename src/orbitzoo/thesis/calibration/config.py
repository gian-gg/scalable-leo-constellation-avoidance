"""Versioned configuration for offline TLE calibration runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

from orbitzoo.thesis.environments.safety import SafetyConfig


LATEST_TLE_EPOCH = "latest_tle_epoch"


def _validate_strictly_increasing(
    name: str,
    values: Sequence[int],
    *,
    minimum: int,
) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{name} must contain integers")
    if any(value < minimum for value in values):
        raise ValueError(f"{name} values must be at least {minimum}")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be unique and strictly increasing")


def _validate_seeds(name: str, seeds: Sequence[int]) -> None:
    if not seeds:
        raise ValueError(f"{name} cannot be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError(f"{name} must contain integers")
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"{name} cannot contain negative values")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{name} cannot contain duplicates")


@dataclass(frozen=True)
class CatalogConfig:
    """Paths and altitude bounds for one frozen TLE catalog."""

    tle_path: str = "../data/tle/catalog.tle"
    metadata_path: str = "../data/tle/objects.csv"
    minimum_altitude_meters: float = 200_000.0
    maximum_altitude_meters: float = 2_000_000.0
    maximum_tle_age_days: float = 14.0
    default_radius_meters: float = 1.0

    def validate(self) -> None:
        if not isinstance(self.tle_path, str) or not isinstance(self.metadata_path, str):
            raise ValueError("TLE and metadata paths must be strings")
        if not self.tle_path.strip() or not self.metadata_path.strip():
            raise ValueError("TLE and metadata paths cannot be empty")
        if not math.isfinite(self.minimum_altitude_meters) or not math.isfinite(
            self.maximum_altitude_meters
        ):
            raise ValueError("altitude bounds must be finite")
        if self.minimum_altitude_meters < 0:
            raise ValueError("minimum_altitude_meters cannot be negative")
        if self.maximum_altitude_meters <= self.minimum_altitude_meters:
            raise ValueError("maximum_altitude_meters must exceed minimum_altitude_meters")
        if not math.isfinite(self.maximum_tle_age_days) or self.maximum_tle_age_days <= 0:
            raise ValueError("maximum_tle_age_days must be finite and positive")
        if not math.isfinite(self.default_radius_meters) or self.default_radius_meters <= 0:
            raise ValueError("default_radius_meters must be finite and positive")


@dataclass(frozen=True)
class PropagationConfig:
    """Time model for generating reusable calibration trajectories."""

    start_epoch_mode: str = LATEST_TLE_EPOCH
    duration_seconds: int = 86_400
    reference_step_seconds: int = 10
    coarse_step_seconds: int = 60
    fine_window_padding_seconds: int = 60
    maximum_relative_speed_mps: float = 20_000.0

    def validate(self) -> None:
        if self.start_epoch_mode != LATEST_TLE_EPOCH:
            raise ValueError(
                f"unsupported propagation start_epoch_mode: {self.start_epoch_mode!r}"
            )
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, int
        ):
            raise ValueError("duration_seconds must be an integer")
        if isinstance(self.reference_step_seconds, bool) or not isinstance(
            self.reference_step_seconds, int
        ):
            raise ValueError("reference_step_seconds must be an integer")
        if isinstance(self.coarse_step_seconds, bool) or not isinstance(
            self.coarse_step_seconds, int
        ):
            raise ValueError("coarse_step_seconds must be an integer")
        if isinstance(self.fine_window_padding_seconds, bool) or not isinstance(
            self.fine_window_padding_seconds, int
        ):
            raise ValueError("fine_window_padding_seconds must be an integer")
        if (
            self.duration_seconds <= 0
            or self.reference_step_seconds <= 0
            or self.coarse_step_seconds <= 0
        ):
            raise ValueError("propagation duration and timesteps must be positive")
        if self.fine_window_padding_seconds < 0:
            raise ValueError("fine_window_padding_seconds cannot be negative")
        if self.duration_seconds % self.reference_step_seconds != 0:
            raise ValueError("duration_seconds must be divisible by reference_step_seconds")
        if self.duration_seconds % self.coarse_step_seconds != 0:
            raise ValueError("duration_seconds must be divisible by coarse_step_seconds")
        if self.coarse_step_seconds % self.reference_step_seconds != 0:
            raise ValueError("coarse_step_seconds must be divisible by reference_step_seconds")
        if self.fine_window_padding_seconds % self.reference_step_seconds != 0:
            raise ValueError(
                "fine_window_padding_seconds must be divisible by reference_step_seconds"
            )
        if (
            not math.isfinite(self.maximum_relative_speed_mps)
            or self.maximum_relative_speed_mps <= 0.0
        ):
            raise ValueError("maximum_relative_speed_mps must be finite and positive")


@dataclass(frozen=True)
class SweepConfig:
    """Candidate values and deterministic samples evaluated by calibration."""

    agent_counts: tuple[int, ...] = (16, 64, 256)
    neighborhood_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    decision_intervals_seconds: tuple[int, ...] = (60, 120, 300, 600)
    calibration_seeds: tuple[int, ...] = tuple(range(10))
    validation_seeds: tuple[int, ...] = tuple(range(100, 105))

    def validate(self, reference_step_seconds: int, duration_seconds: int) -> None:
        _validate_strictly_increasing("agent_counts", self.agent_counts, minimum=2)
        _validate_strictly_increasing(
            "neighborhood_sizes", self.neighborhood_sizes, minimum=1
        )
        _validate_strictly_increasing(
            "decision_intervals_seconds",
            self.decision_intervals_seconds,
            minimum=1,
        )
        if any(
            interval % reference_step_seconds != 0
            for interval in self.decision_intervals_seconds
        ):
            raise ValueError(
                "decision intervals must be divisible by reference_step_seconds"
            )
        if any(interval > duration_seconds for interval in self.decision_intervals_seconds):
            raise ValueError("decision intervals cannot exceed propagation duration")
        _validate_seeds("calibration_seeds", self.calibration_seeds)
        _validate_seeds("validation_seeds", self.validation_seeds)
        overlap = set(self.calibration_seeds).intersection(self.validation_seeds)
        if overlap:
            raise ValueError(
                f"calibration and validation seeds must be disjoint: {sorted(overlap)}"
            )


@dataclass(frozen=True)
class PassingThresholds:
    """Predeclared criteria for accepting a candidate parameter pair."""

    minimum_threat_recall: float = 0.999
    minimum_timely_detection_fraction: float = 0.99
    minimum_decisions_before_tca: int = 3

    def validate(self) -> None:
        for name, value in (
            ("minimum_threat_recall", self.minimum_threat_recall),
            (
                "minimum_timely_detection_fraction",
                self.minimum_timely_detection_fraction,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            isinstance(self.minimum_decisions_before_tca, bool)
            or not isinstance(self.minimum_decisions_before_tca, int)
            or self.minimum_decisions_before_tca <= 0
        ):
            raise ValueError("minimum_decisions_before_tca must be a positive integer")


@dataclass(frozen=True)
class CalibrationConfig:
    """All versioned inputs needed to define a k/delta-t calibration."""

    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    passing_thresholds: PassingThresholds = field(default_factory=PassingThresholds)
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported calibration configuration schema version: {self.schema_version}"
            )
        self.catalog.validate()
        self.propagation.validate()
        self.safety.validate()
        if not math.isfinite(self.safety.safe_separation_meters) or not math.isfinite(
            self.safety.screening_horizon_seconds
        ):
            raise ValueError("safety thresholds must be finite")
        self.sweep.validate(
            self.propagation.reference_step_seconds,
            self.propagation.duration_seconds,
        )
        self.passing_thresholds.validate()
        if self.safety.screening_horizon_seconds > self.propagation.duration_seconds:
            raise ValueError("screening horizon cannot exceed propagation duration")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the exact calibration configuration as portable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def resolve_catalog_paths(self, config_path: str | Path) -> tuple[Path, Path]:
        """Resolve catalog inputs relative to the containing JSON configuration."""
        base_directory = Path(config_path).resolve().parent

        def resolve(path: str) -> Path:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = base_directory / candidate
            return candidate.resolve()

        return resolve(self.catalog.tle_path), resolve(self.catalog.metadata_path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationConfig":
        """Load and validate a versioned calibration configuration."""
        raw = json.loads(Path(path).read_text())
        config = cls(
            catalog=CatalogConfig(**raw["catalog"]),
            propagation=PropagationConfig(**raw["propagation"]),
            safety=SafetyConfig(**raw["safety"]),
            sweep=SweepConfig(
                agent_counts=tuple(raw["sweep"]["agent_counts"]),
                neighborhood_sizes=tuple(raw["sweep"]["neighborhood_sizes"]),
                decision_intervals_seconds=tuple(
                    raw["sweep"]["decision_intervals_seconds"]
                ),
                calibration_seeds=tuple(raw["sweep"]["calibration_seeds"]),
                validation_seeds=tuple(raw["sweep"]["validation_seeds"]),
            ),
            passing_thresholds=PassingThresholds(**raw["passing_thresholds"]),
            schema_version=raw["schema_version"],
        )
        config.validate()
        return config
