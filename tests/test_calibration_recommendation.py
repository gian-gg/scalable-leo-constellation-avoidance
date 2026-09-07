from dataclasses import replace

import pytest

from orbitzoo.thesis.calibration import (
    CalibrationConfig,
    CatalogConfig,
    CombinationMetrics,
    EvaluationSplit,
    NoPassingCombinationError,
    PassingThresholds,
    PropagationConfig,
    SweepConfig,
    pool_combination_metrics,
    select_calibration_recommendation,
)
from orbitzoo.thesis.environments.safety import SafetyConfig


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
            decision_intervals_seconds=(10, 20),
            calibration_seeds=(0,),
            validation_seeds=(100,),
        ),
        passing_thresholds=PassingThresholds(
            minimum_threat_recall=0.9,
            minimum_timely_detection_fraction=0.8,
            minimum_decisions_before_tca=2,
        ),
    )


def _metrics(
    *,
    calibration_counts: dict[tuple[int, int], tuple[int, int]] | None = None,
    validation_counts: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[CombinationMetrics, ...]:
    calibration_counts = calibration_counts or {
        (1, 10): (10, 9),
        (1, 20): (8, 8),
        (2, 10): (10, 9),
        (2, 20): (10, 9),
    }
    validation_counts = validation_counts or {
        pair: (10, 9) for pair in calibration_counts
    }
    rows = []
    for evaluation_split, seed, counts in (
        (EvaluationSplit.CALIBRATION, 0, calibration_counts),
        (EvaluationSplit.VALIDATION, 100, validation_counts),
    ):
        for neighborhood_size in (1, 2):
            for interval in (10, 20):
                detected_count, timely_count = counts[
                    (neighborhood_size, interval)
                ]
                rows.append(
                    CombinationMetrics(
                        evaluation_split=evaluation_split,
                        selection_seed=seed,
                        agent_count=2,
                        neighborhood_size=neighborhood_size,
                        decision_interval_seconds=interval,
                        reference_conjunction_count=10,
                        detected_conjunction_count=detected_count,
                        timely_detected_conjunction_count=timely_count,
                        runtime_seconds=1.0,
                    )
                )
    return tuple(rows)


def test_selects_smallest_passing_k_before_largest_delta_t() -> None:
    config = _config()
    metrics = _metrics()
    pooled = pool_combination_metrics(metrics, config)

    recommendation = select_calibration_recommendation(metrics, pooled, config)

    assert recommendation.neighborhood_size == 1
    assert recommendation.decision_interval_seconds == 10
    assert recommendation.calibration_threat_recall == pytest.approx(1.0)
    assert recommendation.calibration_timely_detection_fraction == pytest.approx(0.9)
    assert recommendation.validation_passed is True
    assert recommendation.is_accepted is True


def test_validation_rejects_selected_pair_without_triggering_fallback() -> None:
    config = _config()
    calibration_counts = {
        (1, 10): (10, 9),
        (1, 20): (10, 9),
        (2, 10): (10, 9),
        (2, 20): (10, 9),
    }
    validation_counts = {
        (1, 10): (10, 9),
        (1, 20): (8, 8),
        (2, 10): (10, 9),
        (2, 20): (10, 9),
    }
    metrics = _metrics(
        calibration_counts=calibration_counts,
        validation_counts=validation_counts,
    )

    recommendation = select_calibration_recommendation(
        metrics,
        pool_combination_metrics(metrics, config),
        config,
    )

    assert recommendation.neighborhood_size == 1
    assert recommendation.decision_interval_seconds == 20
    assert recommendation.calibration_passed is True
    assert recommendation.validation_threat_recall == pytest.approx(0.8)
    assert recommendation.validation_passed is False
    assert recommendation.is_accepted is False


def test_no_calibration_pass_fails_without_inventing_a_recommendation() -> None:
    config = _config()
    failing = {pair: (8, 7) for pair in ((1, 10), (1, 20), (2, 10), (2, 20))}
    metrics = _metrics(calibration_counts=failing)

    with pytest.raises(NoPassingCombinationError, match="no k/delta-t"):
        select_calibration_recommendation(
            metrics,
            pool_combination_metrics(metrics, config),
            config,
        )


def test_rejects_pooled_results_that_contradict_sample_evidence() -> None:
    config = _config()
    metrics = _metrics()
    pooled = list(pool_combination_metrics(metrics, config))
    pooled[0] = replace(
        pooled[0],
        reference_conjunction_count=11,
    )

    with pytest.raises(ValueError, match="do not match"):
        select_calibration_recommendation(metrics, pooled, config)
