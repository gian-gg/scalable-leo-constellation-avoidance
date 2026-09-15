from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitzoo.thesis.calibration import runner
from orbitzoo.thesis.calibration.config import (
    CalibrationConfig,
    CatalogConfig,
    PassingThresholds,
    PropagationConfig,
    SweepConfig,
)
from orbitzoo.thesis.calibration.metrics import pool_combination_metrics
from orbitzoo.thesis.calibration.models import (
    CandidateScreeningResult,
    CatalogObject,
    CombinationMetrics,
    EvaluationSplit,
    LoadedCatalog,
    ObjectType,
)
from orbitzoo.thesis.environments.safety import SafetyConfig


EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _config() -> CalibrationConfig:
    return CalibrationConfig(
        catalog=CatalogConfig(tle_path="catalog.tle", metadata_path="objects.csv"),
        propagation=PropagationConfig(
            duration_seconds=60,
            reference_step_seconds=10,
            coarse_step_seconds=60,
            fine_window_padding_seconds=60,
        ),
        safety=SafetyConfig(
            safe_separation_meters=1_000.0,
            screening_horizon_seconds=60.0,
        ),
        sweep=SweepConfig(
            agent_counts=(2,),
            neighborhood_sizes=(1,),
            decision_intervals_seconds=(60,),
            calibration_seeds=(0,),
            validation_seeds=(100,),
        ),
        passing_thresholds=PassingThresholds(
            minimum_threat_recall=1.0,
            minimum_timely_detection_fraction=1.0,
            minimum_decisions_before_tca=1,
        ),
    )


def _catalog() -> LoadedCatalog:
    objects = tuple(
        CatalogObject(
            norad_id=norad_id,
            name=f"SAT-{norad_id}",
            tle_name=f"SAT-{norad_id}",
            line1="line 1",
            line2="line 2",
            tle_epoch_utc=EPOCH,
            object_type=ObjectType.PAYLOAD,
            is_agent_candidate=True,
            radius_meters=1.0,
            constellation="test",
            has_metadata=True,
        )
        for norad_id in (1, 2)
    )
    return LoadedCatalog(
        objects=objects,
        latest_epoch_utc=EPOCH,
        source_record_count=2,
        stale_filtered_norad_ids=(),
    )


class _FakePropagation:
    def __init__(self, catalog: LoadedCatalog) -> None:
        self.coarse_propagation = SimpleNamespace(
            objects=catalog.objects,
            start_epoch_utc=EPOCH,
            end_epoch_utc=EPOCH + timedelta(seconds=60),
        )

    def screen_agent_selections(self, selections):
        return CandidateScreeningResult(
            screened_agent_norad_ids=selections.eligible_norad_ids,
            encounter_windows=(),
        )

    def iter_fine_trajectories(self, windows):
        assert tuple(windows) == ()
        return iter(())


def _metric(split: EvaluationSplit, detected: int) -> CombinationMetrics:
    return CombinationMetrics(
        evaluation_split=split,
        selection_seed=0 if split is EvaluationSplit.CALIBRATION else 100,
        agent_count=2,
        neighborhood_size=1,
        decision_interval_seconds=60,
        reference_conjunction_count=1,
        detected_conjunction_count=detected,
        timely_detected_conjunction_count=detected,
        runtime_seconds=0.0,
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    config: CalibrationConfig,
    metrics: tuple[CombinationMetrics, ...],
) -> None:
    catalog = _catalog()
    pooled = pool_combination_metrics(metrics, config)
    monkeypatch.setattr(runner, "load_catalog", lambda *_args: catalog)
    monkeypatch.setattr(
        runner,
        "build_two_resolution_propagation",
        lambda *_args: _FakePropagation(catalog),
    )
    monkeypatch.setattr(runner, "iter_ranked_neighbor_frames", lambda *_a, **_k: ())
    monkeypatch.setattr(runner, "evaluate_joint_combinations", lambda *_a, **_k: ())
    monkeypatch.setattr(
        runner,
        "aggregate_and_pool_detections",
        lambda *_args: (metrics, pooled),
    )


@pytest.mark.parametrize(
    ("calibration_detected", "validation_detected", "expected_status"),
    [
        (1, 1, runner.CalibrationRunStatus.ACCEPTED),
        (1, 0, runner.CalibrationRunStatus.REJECTED_VALIDATION),
        (0, 1, runner.CalibrationRunStatus.NO_PASSING_CALIBRATION),
    ],
)
def test_run_calibration_publishes_complete_artifacts_for_scientific_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_detected: int,
    validation_detected: int,
    expected_status: runner.CalibrationRunStatus,
) -> None:
    config = _config()
    config_path = tmp_path / "config.json"
    config.save(config_path)
    metrics = (
        _metric(EvaluationSplit.CALIBRATION, calibration_detected),
        _metric(EvaluationSplit.VALIDATION, validation_detected),
    )
    _patch_pipeline(monkeypatch, config, metrics)
    output = tmp_path / "run"

    result = runner.run_calibration(config_path, output)

    assert result.status is expected_status
    assert sorted(path.name for path in output.iterdir()) == [
        "agent_selections.json",
        "catalog_summary.json",
        "combination_metrics.json",
        "pooled_metrics.json",
        "recommendation.json",
        "reference_conjunctions.json",
        "resolved_config.json",
        "summary.txt",
    ]
    recommendation = json.loads((output / "recommendation.json").read_text())
    assert recommendation["status"] == expected_status.value
    assert recommendation["catalog_epoch_utc"] == EPOCH.isoformat()
    assert (output / "summary.txt").read_text().endswith("\n")


def test_run_calibration_refuses_existing_output_before_loading_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_calibration(tmp_path / "missing.json", output)


def test_run_calibration_removes_partial_directory_on_operational_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    _config().save(config_path)
    monkeypatch.setattr(
        runner,
        "load_catalog",
        lambda *_args: (_ for _ in ()).throw(ValueError("bad catalog")),
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="bad catalog"):
        runner.run_calibration(config_path, output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".run.tmp-*"))
