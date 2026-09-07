from dataclasses import replace
import math
from pathlib import Path

import pytest

from orbitzoo.thesis.calibration.config import (
    CalibrationConfig,
    CatalogConfig,
    PassingThresholds,
    PropagationConfig,
    SweepConfig,
)
from orbitzoo.thesis.environments.safety import SafetyConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_configuration_matches_versioned_defaults() -> None:
    config = CalibrationConfig.load(PROJECT_ROOT / "configs" / "k_dt_calibration.json")

    assert config == CalibrationConfig()


def test_configuration_round_trip(tmp_path: Path) -> None:
    config = CalibrationConfig(
        catalog=CatalogConfig(tle_path="inputs/catalog.tle", metadata_path="inputs/objects.csv")
    )
    destination = tmp_path / "nested" / "calibration.json"

    config.save(destination)

    assert CalibrationConfig.load(destination) == config
    first_serialization = destination.read_text()
    config.save(destination)
    assert destination.read_text() == first_serialization


def test_catalog_paths_are_resolved_from_configuration_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "calibration.json"
    config = CalibrationConfig(
        catalog=CatalogConfig(
            tle_path="../data/catalog.tle",
            metadata_path="../data/objects.csv",
        )
    )

    tle_path, metadata_path = config.resolve_catalog_paths(config_path)

    assert tle_path == tmp_path / "data" / "catalog.tle"
    assert metadata_path == tmp_path / "data" / "objects.csv"


@pytest.mark.parametrize(
    "catalog",
    [
        CatalogConfig(tle_path=""),
        CatalogConfig(minimum_altitude_meters=-1),
        CatalogConfig(minimum_altitude_meters=500_000, maximum_altitude_meters=500_000),
        CatalogConfig(maximum_tle_age_days=0),
        CatalogConfig(maximum_tle_age_days=math.nan),
        CatalogConfig(default_radius_meters=0),
    ],
)
def test_invalid_catalog_configuration_is_rejected(catalog: CatalogConfig) -> None:
    with pytest.raises(ValueError):
        replace(CalibrationConfig(), catalog=catalog).validate()


@pytest.mark.parametrize(
    "propagation",
    [
        PropagationConfig(start_epoch_mode="explicit_utc"),
        PropagationConfig(duration_seconds=0),
        PropagationConfig(duration_seconds=101, reference_step_seconds=10),
        PropagationConfig(coarse_step_seconds=55),
        PropagationConfig(fine_window_padding_seconds=55),
        PropagationConfig(maximum_relative_speed_mps=math.nan),
    ],
)
def test_invalid_propagation_configuration_is_rejected(
    propagation: PropagationConfig,
) -> None:
    with pytest.raises(ValueError):
        replace(CalibrationConfig(), propagation=propagation).validate()


@pytest.mark.parametrize(
    "sweep",
    [
        SweepConfig(agent_counts=()),
        SweepConfig(neighborhood_sizes=(1, 4, 4)),
        SweepConfig(decision_intervals_seconds=(65, 120)),
        SweepConfig(calibration_seeds=(0, 0)),
        SweepConfig(calibration_seeds=(0, 100)),
    ],
)
def test_invalid_sweep_configuration_is_rejected(sweep: SweepConfig) -> None:
    with pytest.raises(ValueError):
        replace(CalibrationConfig(), sweep=sweep).validate()


@pytest.mark.parametrize(
    "thresholds",
    [
        PassingThresholds(minimum_threat_recall=1.01),
        PassingThresholds(minimum_timely_detection_fraction=-0.01),
        PassingThresholds(minimum_decisions_before_tca=0),
    ],
)
def test_invalid_passing_thresholds_are_rejected(
    thresholds: PassingThresholds,
) -> None:
    with pytest.raises(ValueError):
        replace(CalibrationConfig(), passing_thresholds=thresholds).validate()


def test_screening_horizon_cannot_exceed_propagation_duration() -> None:
    config = CalibrationConfig(
        propagation=PropagationConfig(duration_seconds=600, reference_step_seconds=10),
        safety=SafetyConfig(screening_horizon_seconds=601),
    )

    with pytest.raises(ValueError, match="screening horizon"):
        config.validate()


def test_decision_interval_cannot_exceed_propagation_duration() -> None:
    config = CalibrationConfig(
        propagation=PropagationConfig(duration_seconds=60, reference_step_seconds=10),
        sweep=SweepConfig(decision_intervals_seconds=(60, 120)),
    )

    with pytest.raises(ValueError, match="decision intervals"):
        config.validate()


def test_non_finite_safety_threshold_is_rejected() -> None:
    config = replace(
        CalibrationConfig(),
        safety=SafetyConfig(safe_separation_meters=math.nan),
    )

    with pytest.raises(ValueError, match="finite"):
        config.validate()


def test_unsupported_schema_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema version"):
        replace(CalibrationConfig(), schema_version=2).validate()
