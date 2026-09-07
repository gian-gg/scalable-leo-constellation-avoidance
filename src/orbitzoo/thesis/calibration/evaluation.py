"""Single-pass joint evaluation of neighborhood sizes and decision intervals."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import (
    AgentSelection,
    AgentSelectionManifest,
    EvaluationSplit,
    RankedNeighborFrame,
    ReferenceConjunction,
    ReferenceConjunctionManifest,
    ThreatDetection,
)
from orbitzoo.thesis.calibration.ranking import (
    DecisionSchedule,
    build_decision_schedule,
)


@dataclass(frozen=True)
class _SelectionContext:
    evaluation_split: EvaluationSplit
    selection: AgentSelection
    references_by_pair: dict[tuple[int, int], tuple[ReferenceConjunction, ...]]


def _validate_selection_cases(
    manifest: AgentSelectionManifest,
    config: CalibrationConfig,
) -> None:
    expected_counts = set(config.sweep.agent_counts)
    expected_by_split = (
        (
            "calibration",
            manifest.calibration_selections,
            set(config.sweep.calibration_seeds),
        ),
        (
            "validation",
            manifest.validation_selections,
            set(config.sweep.validation_seeds),
        ),
    )
    for name, selections, expected_seeds in expected_by_split:
        actual_cases = {
            (selection.seed, selection.requested_agent_count)
            for selection in selections
        }
        expected_cases = {
            (seed, count)
            for seed in expected_seeds
            for count in expected_counts
        }
        if actual_cases != expected_cases:
            raise ValueError(
                f"{name} selections do not match the configured seeds and counts"
            )


def _build_contexts(
    selections: AgentSelectionManifest,
    references: ReferenceConjunctionManifest,
) -> tuple[_SelectionContext, ...]:
    contexts: list[_SelectionContext] = []
    for evaluation_split, group in (
        (EvaluationSplit.CALIBRATION, selections.calibration_selections),
        (EvaluationSplit.VALIDATION, selections.validation_selections),
    ):
        for selection in group:
            references_by_pair: dict[
                tuple[int, int], list[ReferenceConjunction]
            ] = defaultdict(list)
            for reference in references.conjunctions_for(selection):
                pair = reference.first_norad_id, reference.second_norad_id
                references_by_pair[pair].append(reference)
            contexts.append(
                _SelectionContext(
                    evaluation_split=evaluation_split,
                    selection=selection,
                    references_by_pair={
                        pair: tuple(sorted(events, key=lambda item: item.tca_epoch_utc))
                        for pair, events in references_by_pair.items()
                    },
                )
            )
    return tuple(contexts)


def evaluate_joint_combinations(
    ranking_frames: Iterable[RankedNeighborFrame],
    references: ReferenceConjunctionManifest,
    selections: AgentSelectionManifest,
    config: CalibrationConfig,
    *,
    schedule: DecisionSchedule | None = None,
) -> tuple[ThreatDetection, ...]:
    """Evaluate every configured k/delta-t case from one ranking-frame stream."""
    config.validate()
    _validate_selection_cases(selections, config)
    if selections.catalog_epoch_utc != references.catalog_epoch_utc:
        raise ValueError("selection and reference catalog epochs do not match")
    if references.safe_separation_meters != config.safety.safe_separation_meters:
        raise ValueError("reference and configured safe separations do not match")
    if references.reference_step_seconds != config.propagation.reference_step_seconds:
        raise ValueError("reference and configured timesteps do not match")
    selected_ids = {
        identifier
        for selection in (
            *selections.calibration_selections,
            *selections.validation_selections,
        )
        for identifier in selection.agent_norad_ids
    }
    if not selected_ids.issubset(references.screened_agent_norad_ids):
        raise ValueError("reference pass does not cover every selected agent")

    if schedule is None:
        schedule = build_decision_schedule(references.catalog_epoch_utc, config)
    if schedule.start_epoch_utc != references.catalog_epoch_utc:
        raise ValueError("decision schedule and reference catalog epochs do not match")
    configured_intervals = config.sweep.decision_intervals_seconds
    scheduled_intervals = tuple(
        interval for interval, _ in schedule.epochs_by_interval
    )
    if scheduled_intervals != configured_intervals:
        raise ValueError("decision schedule does not match configured intervals")

    contexts = _build_contexts(selections, references)
    context_indices_by_agent: dict[int, list[int]] = defaultdict(list)
    for context_index, context in enumerate(contexts):
        for agent_id in context.selection.agent_norad_ids:
            context_indices_by_agent[agent_id].append(context_index)
    mutable_intervals: dict[datetime, list[int]] = defaultdict(list)
    for interval, epochs in schedule.epochs_by_interval:
        for epoch in epochs:
            mutable_intervals[epoch].append(interval)
    intervals_by_epoch: dict[datetime, tuple[int, ...]] = {
        epoch: tuple(intervals)
        for epoch, intervals in mutable_intervals.items()
    }

    neighborhood_sizes = config.sweep.neighborhood_sizes
    maximum_neighborhood_size = max(neighborhood_sizes)
    horizon = timedelta(seconds=config.safety.screening_horizon_seconds)
    first_visibility: dict[
        tuple[int, ReferenceConjunction, int, int], datetime
    ] = {}
    processed_epoch_count = 0
    expected_agent_ids = tuple(sorted(selected_ids))

    for frame in ranking_frames:
        if processed_epoch_count >= len(schedule.shared_epochs):
            raise ValueError("ranking stream contains unexpected extra epochs")
        expected_epoch = schedule.shared_epochs[processed_epoch_count]
        if frame.decision_epoch_utc != expected_epoch:
            raise ValueError(
                "ranking stream must contain every shared decision epoch in order"
            )
        if not set(expected_agent_ids).issubset(frame.agent_norad_ids):
            raise ValueError("ranking frame does not cover every selected agent")
        if frame.maximum_neighbors < maximum_neighborhood_size:
            raise ValueError("ranking frame does not cover the largest configured k")
        active_intervals = intervals_by_epoch[frame.decision_epoch_utc]

        for ranking in frame.rankings:
            if ranking.rank > maximum_neighborhood_size:
                continue
            pair = tuple(
                sorted((ranking.agent_norad_id, ranking.neighbor_norad_id))
            )
            predicted_tca = frame.decision_epoch_utc + timedelta(
                seconds=ranking.time_to_closest_approach_seconds
            )
            for context_index in context_indices_by_agent.get(
                ranking.agent_norad_id,
                (),
            ):
                context = contexts[context_index]
                candidate_events = tuple(
                    reference
                    for reference in context.references_by_pair.get(pair, ())
                    if frame.decision_epoch_utc < reference.tca_epoch_utc
                    <= frame.decision_epoch_utc + horizon
                )
                if not candidate_events:
                    continue
                reference = min(
                    candidate_events,
                    key=lambda item: abs(
                        (item.tca_epoch_utc - predicted_tca).total_seconds()
                    ),
                )
                for interval in active_intervals:
                    for neighborhood_size in neighborhood_sizes:
                        if ranking.rank > neighborhood_size:
                            continue
                        key = (
                            context_index,
                            reference,
                            neighborhood_size,
                            interval,
                        )
                        first_visibility.setdefault(key, frame.decision_epoch_utc)
        processed_epoch_count += 1

    if processed_epoch_count != len(schedule.shared_epochs):
        raise ValueError("ranking stream ended before all shared epochs were evaluated")

    detections: list[ThreatDetection] = []
    minimum_decisions = config.passing_thresholds.minimum_decisions_before_tca
    for context_index, context in enumerate(contexts):
        selection = context.selection
        reference_events = tuple(
            reference
            for events in context.references_by_pair.values()
            for reference in events
        )
        reference_events = tuple(
            sorted(
                reference_events,
                key=lambda item: (
                    item.tca_epoch_utc,
                    item.first_norad_id,
                    item.second_norad_id,
                ),
            )
        )
        for neighborhood_size in neighborhood_sizes:
            for interval in configured_intervals:
                for reference in reference_events:
                    first_visible = first_visibility.get(
                        (context_index, reference, neighborhood_size, interval)
                    )
                    decisions_remaining = 0
                    if first_visible is not None:
                        decisions_remaining = math.ceil(
                            (reference.tca_epoch_utc - first_visible).total_seconds()
                            / interval
                        )
                    detections.append(
                        ThreatDetection(
                            evaluation_split=context.evaluation_split,
                            selection_seed=selection.seed,
                            agent_count=selection.requested_agent_count,
                            neighborhood_size=neighborhood_size,
                            decision_interval_seconds=interval,
                            first_norad_id=reference.first_norad_id,
                            second_norad_id=reference.second_norad_id,
                            reference_tca_epoch_utc=reference.tca_epoch_utc,
                            first_visible_epoch_utc=first_visible,
                            decisions_remaining=decisions_remaining,
                            minimum_decisions_before_tca=minimum_decisions,
                        )
                    )
    detections.sort(
        key=lambda item: (
            item.evaluation_split.value,
            item.selection_seed,
            item.agent_count,
            item.neighborhood_size,
            item.decision_interval_seconds,
            item.reference_tca_epoch_utc,
            item.first_norad_id,
            item.second_norad_id,
        )
    )
    return tuple(detections)
