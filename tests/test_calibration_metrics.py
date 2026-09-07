from datetime import datetime, timedelta, timezone

import pytest

from orbitzoo.thesis.calibration import (
    CalibrationConfig,
    CatalogConfig,
    EvaluationSplit,
    PassingThresholds,
    PropagationConfig,
    SweepConfig,
    ThreatDetection,
    aggregate_and_pool_detections,
    aggregate_detection_metrics,
    pool_combination_metrics,
)
from orbitzoo.thesis.environments.safety import SafetyConfig


UTC_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _config() -> CalibrationConfig:
    return CalibrationConfig(
        catalog=CatalogConfig(),
        propagation=PropagationConfig(
            duration_seconds=60,
            reference_step_seconds=10,
            coarse_step_seconds=20,
            fine_window_padding_seconds=20,
        ),
        safety=SafetyConfig(
            safe_separation_meters=10.0,
            screening_horizon_seconds=20.0,
        ),
        sweep=SweepConfig(
            agent_counts=(2,),
            neighborhood_sizes=(1, 2),
            decision_intervals_seconds=(10,),
            calibration_seeds=(0, 1),
            validation_seeds=(100,),
        ),
        passing_thresholds=PassingThresholds(
            minimum_threat_recall=0.6,
            minimum_timely_detection_fraction=0.5,
            minimum_decisions_before_tca=2,
        ),
    )


def _detection(
    *,
    seed: int,
    neighborhood_size: int,
    event_index: int,
    first_visible_seconds: int | None,
) -> ThreatDetection:
    tca_seconds = 30 + event_index * 10
    return ThreatDetection(
        evaluation_split=EvaluationSplit.CALIBRATION,
        selection_seed=seed,
        agent_count=2,
        neighborhood_size=neighborhood_size,
        decision_interval_seconds=10,
        first_norad_id=10,
        second_norad_id=20 + event_index,
        reference_tca_epoch_utc=UTC_EPOCH + timedelta(seconds=tca_seconds),
        first_visible_epoch_utc=(
            None
            if first_visible_seconds is None
            else UTC_EPOCH + timedelta(seconds=first_visible_seconds)
        ),
        decisions_remaining=(
            0
            if first_visible_seconds is None
            else (tca_seconds - first_visible_seconds + 9) // 10
        ),
        minimum_decisions_before_tca=2,
    )


def _detections() -> tuple[ThreatDetection, ...]:
    detections = []
    for neighborhood_size in (1, 2):
        for event_index in range(3):
            if neighborhood_size == 1:
                first_visible = 0 if event_index == 0 else None
            else:
                first_visible = (0, 20, 40)[event_index]
            detections.append(
                _detection(
                    seed=0,
                    neighborhood_size=neighborhood_size,
                    event_index=event_index,
                    first_visible_seconds=first_visible,
                )
            )
        detections.append(
            _detection(
                seed=1,
                neighborhood_size=neighborhood_size,
                event_index=0,
                first_visible_seconds=10,
            )
        )
    return tuple(detections)


def test_aggregation_emits_all_cases_including_empty_samples() -> None:
    runtime_key = (EvaluationSplit.CALIBRATION, 0, 2, 1, 10)

    metrics = aggregate_detection_metrics(
        _detections(),
        _config(),
        runtime_seconds_by_case={runtime_key: 1.25},
    )

    assert len(metrics) == 6
    seed_zero_k1 = next(
        metric
        for metric in metrics
        if metric.evaluation_split is EvaluationSplit.CALIBRATION
        and metric.selection_seed == 0
        and metric.neighborhood_size == 1
    )
    assert seed_zero_k1.reference_conjunction_count == 3
    assert seed_zero_k1.detected_conjunction_count == 1
    assert seed_zero_k1.timely_detected_conjunction_count == 1
    assert seed_zero_k1.missed_conjunction_count == 2
    assert seed_zero_k1.runtime_seconds == pytest.approx(1.25)

    validation_rows = tuple(
        metric
        for metric in metrics
        if metric.evaluation_split is EvaluationSplit.VALIDATION
    )
    assert len(validation_rows) == 2
    assert all(metric.reference_conjunction_count == 0 for metric in validation_rows)
    assert all(metric.threat_recall == 0.0 for metric in validation_rows)


def test_pooled_rates_use_integer_counts_instead_of_averaging_samples() -> None:
    metrics, pooled = aggregate_and_pool_detections(_detections(), _config())

    assert len(metrics) == 6
    calibration_k1 = next(
        result
        for result in pooled
        if result.evaluation_split is EvaluationSplit.CALIBRATION
        and result.neighborhood_size == 1
    )
    assert calibration_k1.sample_count == 2
    assert calibration_k1.reference_conjunction_count == 4
    assert calibration_k1.detected_conjunction_count == 2
    assert calibration_k1.timely_detected_conjunction_count == 2
    assert calibration_k1.threat_recall == pytest.approx(0.5)
    assert calibration_k1.timely_detection_fraction == pytest.approx(0.5)
    assert calibration_k1.passed is False

    calibration_k2 = next(
        result
        for result in pooled
        if result.evaluation_split is EvaluationSplit.CALIBRATION
        and result.neighborhood_size == 2
    )
    assert calibration_k2.threat_recall == pytest.approx(1.0)
    assert calibration_k2.timely_detection_fraction == pytest.approx(0.75)
    assert calibration_k2.passed is True

    validation_results = tuple(
        result
        for result in pooled
        if result.evaluation_split is EvaluationSplit.VALIDATION
    )
    assert all(result.reference_conjunction_count == 0 for result in validation_results)
    assert all(result.passed is False for result in validation_results)


def test_aggregation_rejects_missing_or_duplicate_reference_evidence() -> None:
    detections = _detections()
    with pytest.raises(ValueError, match="same reference events"):
        aggregate_detection_metrics(detections[:-1], _config())

    with pytest.raises(ValueError, match="duplicate reference event"):
        aggregate_detection_metrics((*detections, detections[0]), _config())


def test_pooling_requires_every_configured_metric_row() -> None:
    metrics = aggregate_detection_metrics(_detections(), _config())

    with pytest.raises(ValueError, match="every configured case"):
        pool_combination_metrics(metrics[:-1], _config())
