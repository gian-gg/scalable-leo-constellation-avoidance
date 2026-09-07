"""Leakage-free calibration selection and held-out validation."""

from __future__ import annotations

from typing import Iterable

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.metrics import pool_combination_metrics
from orbitzoo.thesis.calibration.models import (
    CalibrationRecommendation,
    CombinationMetrics,
    EvaluationSplit,
    PooledCombinationMetrics,
)


class NoPassingCombinationError(RuntimeError):
    """Raised when no candidate pair satisfies the calibration thresholds."""


def select_calibration_recommendation(
    metrics: Iterable[CombinationMetrics],
    pooled_metrics: Iterable[PooledCombinationMetrics],
    config: CalibrationConfig,
) -> CalibrationRecommendation:
    """Select on calibration only, then audit that exact pair on validation."""
    config.validate()
    metric_rows = tuple(metrics)
    supplied_pooled = tuple(pooled_metrics)
    expected_pooled = pool_combination_metrics(metric_rows, config)
    supplied_by_case = {
        (
            result.evaluation_split,
            result.neighborhood_size,
            result.decision_interval_seconds,
        ): result
        for result in supplied_pooled
    }
    expected_by_case = {
        (
            result.evaluation_split,
            result.neighborhood_size,
            result.decision_interval_seconds,
        ): result
        for result in expected_pooled
    }
    if (
        len(supplied_by_case) != len(supplied_pooled)
        or supplied_by_case != expected_by_case
    ):
        raise ValueError("pooled metrics do not match the per-sample evidence")

    passing_calibration = tuple(
        result
        for result in expected_pooled
        if result.evaluation_split is EvaluationSplit.CALIBRATION
        and result.passed
    )
    if not passing_calibration:
        raise NoPassingCombinationError(
            "no k/delta-t combination satisfies the calibration thresholds"
        )
    selected = min(
        passing_calibration,
        key=lambda result: (
            result.neighborhood_size,
            -result.decision_interval_seconds,
        ),
    )
    selected_metrics = tuple(
        metric
        for metric in metric_rows
        if metric.neighborhood_size == selected.neighborhood_size
        and metric.decision_interval_seconds
        == selected.decision_interval_seconds
    )
    calibration_metrics = tuple(
        metric
        for metric in selected_metrics
        if metric.evaluation_split is EvaluationSplit.CALIBRATION
    )
    validation_metrics = tuple(
        metric
        for metric in selected_metrics
        if metric.evaluation_split is EvaluationSplit.VALIDATION
    )
    thresholds = config.passing_thresholds
    recommendation = CalibrationRecommendation(
        neighborhood_size=selected.neighborhood_size,
        decision_interval_seconds=selected.decision_interval_seconds,
        minimum_threat_recall=thresholds.minimum_threat_recall,
        minimum_timely_detection_fraction=(
            thresholds.minimum_timely_detection_fraction
        ),
        minimum_decisions_before_tca=thresholds.minimum_decisions_before_tca,
        calibration_metrics=calibration_metrics,
        validation_metrics=validation_metrics,
    )
    if not recommendation.calibration_passed:
        raise ValueError("selected pooled result contradicts calibration evidence")
    return recommendation
