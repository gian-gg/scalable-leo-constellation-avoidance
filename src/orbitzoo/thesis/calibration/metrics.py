"""Detection aggregation and count-weighted calibration thresholds."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
from typing import Iterable, Mapping, TypeAlias

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import (
    CombinationMetrics,
    EvaluationSplit,
    PooledCombinationMetrics,
    ThreatDetection,
)


MetricCaseKey: TypeAlias = tuple[EvaluationSplit, int, int, int, int]
_EventKey: TypeAlias = tuple[int, int, datetime]


def _configured_samples(
    config: CalibrationConfig,
) -> tuple[tuple[EvaluationSplit, int, int], ...]:
    return tuple(
        (evaluation_split, seed, agent_count)
        for evaluation_split, seeds in (
            (EvaluationSplit.CALIBRATION, config.sweep.calibration_seeds),
            (EvaluationSplit.VALIDATION, config.sweep.validation_seeds),
        )
        for seed in seeds
        for agent_count in config.sweep.agent_counts
    )


def _configured_case_keys(config: CalibrationConfig) -> tuple[MetricCaseKey, ...]:
    return tuple(
        (evaluation_split, seed, agent_count, neighborhood_size, interval)
        for evaluation_split, seed, agent_count in _configured_samples(config)
        for neighborhood_size in config.sweep.neighborhood_sizes
        for interval in config.sweep.decision_intervals_seconds
    )


def aggregate_detection_metrics(
    detections: Iterable[ThreatDetection],
    config: CalibrationConfig,
    *,
    runtime_seconds_by_case: Mapping[MetricCaseKey, float] | None = None,
) -> tuple[CombinationMetrics, ...]:
    """Create every configured per-sample metric row, including empty cases."""
    config.validate()
    expected_case_keys = _configured_case_keys(config)
    expected_case_set = set(expected_case_keys)
    runtime_by_case = dict(runtime_seconds_by_case or {})
    unknown_runtime_keys = set(runtime_by_case).difference(expected_case_set)
    if unknown_runtime_keys:
        raise ValueError("runtime mapping contains an unconfigured metric case")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in runtime_by_case.values()
    ):
        raise ValueError("metric runtimes must be finite and nonnegative")

    detections_by_case: dict[MetricCaseKey, list[ThreatDetection]] = defaultdict(list)
    event_keys_by_case: dict[MetricCaseKey, set[_EventKey]] = defaultdict(set)
    for detection in detections:
        case_key = (
            detection.evaluation_split,
            detection.selection_seed,
            detection.agent_count,
            detection.neighborhood_size,
            detection.decision_interval_seconds,
        )
        if case_key not in expected_case_set:
            raise ValueError("detection references an unconfigured metric case")
        if (
            detection.minimum_decisions_before_tca
            != config.passing_thresholds.minimum_decisions_before_tca
        ):
            raise ValueError("detection uses a different timely-decision threshold")
        event_key = (
            detection.first_norad_id,
            detection.second_norad_id,
            detection.reference_tca_epoch_utc,
        )
        if event_key in event_keys_by_case[case_key]:
            raise ValueError("metric case contains a duplicate reference event")
        event_keys_by_case[case_key].add(event_key)
        detections_by_case[case_key].append(detection)

    for evaluation_split, seed, agent_count in _configured_samples(config):
        sample_event_sets = tuple(
            event_keys_by_case[
                (
                    evaluation_split,
                    seed,
                    agent_count,
                    neighborhood_size,
                    interval,
                )
            ]
            for neighborhood_size in config.sweep.neighborhood_sizes
            for interval in config.sweep.decision_intervals_seconds
        )
        reference_events = set().union(*sample_event_sets)
        if any(events != reference_events for events in sample_event_sets):
            raise ValueError(
                "every combination for a sample must contain the same reference events"
            )

    metrics: list[CombinationMetrics] = []
    for case_key in expected_case_keys:
        evaluation_split, seed, agent_count, neighborhood_size, interval = case_key
        case_detections = detections_by_case[case_key]
        metrics.append(
            CombinationMetrics(
                evaluation_split=evaluation_split,
                selection_seed=seed,
                agent_count=agent_count,
                neighborhood_size=neighborhood_size,
                decision_interval_seconds=interval,
                reference_conjunction_count=len(case_detections),
                detected_conjunction_count=sum(
                    detection.detected for detection in case_detections
                ),
                timely_detected_conjunction_count=sum(
                    detection.timely_detected for detection in case_detections
                ),
                runtime_seconds=runtime_by_case.get(case_key, 0.0),
            )
        )
    return tuple(metrics)


def pool_combination_metrics(
    metrics: Iterable[CombinationMetrics],
    config: CalibrationConfig,
) -> tuple[PooledCombinationMetrics, ...]:
    """Pool integer counts across seeds and agent counts before computing rates."""
    config.validate()
    metric_rows = tuple(metrics)
    expected_case_keys = _configured_case_keys(config)
    metric_by_case: dict[MetricCaseKey, CombinationMetrics] = {}
    for metric in metric_rows:
        case_key = (
            metric.evaluation_split,
            metric.selection_seed,
            metric.agent_count,
            metric.neighborhood_size,
            metric.decision_interval_seconds,
        )
        if case_key in metric_by_case:
            raise ValueError("combination metrics contain a duplicate case")
        metric_by_case[case_key] = metric
    if set(metric_by_case) != set(expected_case_keys):
        raise ValueError("combination metrics do not cover every configured case")

    thresholds = config.passing_thresholds
    pooled: list[PooledCombinationMetrics] = []
    for evaluation_split, seeds in (
        (EvaluationSplit.CALIBRATION, config.sweep.calibration_seeds),
        (EvaluationSplit.VALIDATION, config.sweep.validation_seeds),
    ):
        for neighborhood_size in config.sweep.neighborhood_sizes:
            for interval in config.sweep.decision_intervals_seconds:
                rows = tuple(
                    metric_by_case[
                        (
                            evaluation_split,
                            seed,
                            agent_count,
                            neighborhood_size,
                            interval,
                        )
                    ]
                    for seed in seeds
                    for agent_count in config.sweep.agent_counts
                )
                pooled.append(
                    PooledCombinationMetrics(
                        evaluation_split=evaluation_split,
                        neighborhood_size=neighborhood_size,
                        decision_interval_seconds=interval,
                        sample_count=len(rows),
                        reference_conjunction_count=sum(
                            row.reference_conjunction_count for row in rows
                        ),
                        detected_conjunction_count=sum(
                            row.detected_conjunction_count for row in rows
                        ),
                        timely_detected_conjunction_count=sum(
                            row.timely_detected_conjunction_count for row in rows
                        ),
                        runtime_seconds=sum(row.runtime_seconds for row in rows),
                        minimum_threat_recall=thresholds.minimum_threat_recall,
                        minimum_timely_detection_fraction=(
                            thresholds.minimum_timely_detection_fraction
                        ),
                    )
                )
    return tuple(pooled)


def aggregate_and_pool_detections(
    detections: Iterable[ThreatDetection],
    config: CalibrationConfig,
    *,
    runtime_seconds_by_case: Mapping[MetricCaseKey, float] | None = None,
) -> tuple[tuple[CombinationMetrics, ...], tuple[PooledCombinationMetrics, ...]]:
    """Build per-sample metrics and their split-specific pooled pass table."""
    metrics = aggregate_detection_metrics(
        detections,
        config,
        runtime_seconds_by_case=runtime_seconds_by_case,
    )
    return metrics, pool_combination_metrics(metrics, config)
