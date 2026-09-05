from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from orbitzoo.thesis.calibration import (
    AgentSelection,
    CalibrationRecommendation,
    CartesianStateFrame,
    CombinationMetrics,
    EvaluationSplit,
    RankedNeighbor,
    ReferenceConjunction,
)


UTC_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _metrics(
    split: EvaluationSplit = EvaluationSplit.CALIBRATION,
    neighborhood_size: int = 4,
    decision_interval_seconds: int = 120,
) -> CombinationMetrics:
    return CombinationMetrics(
        evaluation_split=split,
        selection_seed=3,
        agent_count=16,
        neighborhood_size=neighborhood_size,
        decision_interval_seconds=decision_interval_seconds,
        reference_conjunction_count=10,
        detected_conjunction_count=9,
        timely_detected_conjunction_count=8,
        runtime_seconds=1.25,
    )


def test_cartesian_state_frame_owns_validated_float64_si_arrays() -> None:
    source_positions = np.array([[7_000_000, 0, 0], [0, 7_100_000, 0]])
    source_velocities = np.array([[0, 7_500, 0], [-7_400, 0, 0]])
    local_epoch = datetime(2026, 1, 1, 8, tzinfo=timezone(timedelta(hours=8)))

    frame = CartesianStateFrame(
        epoch_utc=local_epoch,
        norad_ids=(10, 20),
        positions_m=source_positions,
        velocities_mps=source_velocities,
    )
    source_positions[0, 0] = 1

    assert frame.epoch_utc == UTC_EPOCH
    assert frame.positions_m.dtype == np.float64
    assert frame.velocities_mps.dtype == np.float64
    assert frame.positions_m[0, 0] == 7_000_000
    assert frame.positions_m.flags.writeable is False
    assert frame.velocities_mps.flags.writeable is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"norad_ids": (10, 10)},
        {"positions_m": np.zeros((2, 2))},
        {"velocities_mps": np.array([[0, 0, 0], [0, np.nan, 0]])},
        {"epoch_utc": datetime(2026, 1, 1)},
    ],
)
def test_cartesian_state_frame_rejects_ambiguous_data(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "epoch_utc": UTC_EPOCH,
        "norad_ids": (10, 20),
        "positions_m": np.zeros((2, 3)),
        "velocities_mps": np.zeros((2, 3)),
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        CartesianStateFrame(**values)


def test_agent_selection_requires_the_requested_unique_population() -> None:
    selection = AgentSelection(
        seed=7,
        requested_agent_count=2,
        agent_norad_ids=(10, 20),
    )

    assert selection.agent_norad_ids == (10, 20)

    with pytest.raises(ValueError, match="requested_agent_count"):
        AgentSelection(seed=7, requested_agent_count=3, agent_norad_ids=(10, 20))
    with pytest.raises(ValueError, match="unique"):
        AgentSelection(seed=7, requested_agent_count=2, agent_norad_ids=(10, 10))


def test_reference_conjunction_uses_canonical_norad_pair_and_si_values() -> None:
    conjunction = ReferenceConjunction(
        first_norad_id=10,
        second_norad_id=20,
        tca_epoch_utc=UTC_EPOCH,
        miss_distance_meters=750.0,
        relative_speed_mps=12_000.0,
        combined_radius_meters=3.0,
    )

    assert conjunction.miss_distance_meters == 750.0
    assert conjunction.relative_speed_mps == 12_000.0

    with pytest.raises(ValueError, match="ascending"):
        ReferenceConjunction(
            first_norad_id=20,
            second_norad_id=10,
            tca_epoch_utc=UTC_EPOCH,
            miss_distance_meters=750.0,
            relative_speed_mps=12_000.0,
            combined_radius_meters=3.0,
        )


def test_ranked_neighbor_rejects_self_neighbors_and_invalid_ranks() -> None:
    neighbor = RankedNeighbor(
        agent_norad_id=10,
        neighbor_norad_id=20,
        rank=1,
        decision_epoch_utc=UTC_EPOCH,
        current_separation_meters=20_000.0,
        time_to_closest_approach_seconds=300.0,
        predicted_miss_distance_meters=500.0,
        combined_radius_meters=2.0,
        is_collision=False,
        is_unsafe=True,
    )

    assert neighbor.rank == 1

    with pytest.raises(ValueError, match="own ranked neighbor"):
        RankedNeighbor(**{**neighbor.__dict__, "neighbor_norad_id": 10})
    with pytest.raises(ValueError, match="rank"):
        RankedNeighbor(**{**neighbor.__dict__, "rank": 0})


def test_combination_metrics_derive_rates_from_consistent_counts() -> None:
    metrics = _metrics()

    assert metrics.missed_conjunction_count == 1
    assert metrics.threat_recall == 0.9
    assert metrics.timely_detection_fraction == 0.8

    with pytest.raises(ValueError, match="cannot exceed reference"):
        CombinationMetrics(
            **{
                **metrics.__dict__,
                "detected_conjunction_count": 11,
            }
        )


def test_zero_reference_events_do_not_produce_passing_rates() -> None:
    metrics = CombinationMetrics(
        evaluation_split=EvaluationSplit.CALIBRATION,
        selection_seed=0,
        agent_count=16,
        neighborhood_size=1,
        decision_interval_seconds=60,
        reference_conjunction_count=0,
        detected_conjunction_count=0,
        timely_detected_conjunction_count=0,
        runtime_seconds=0.1,
    )

    assert metrics.threat_recall == 0.0
    assert metrics.timely_detection_fraction == 0.0


def test_final_recommendation_requires_matching_independent_metric_groups() -> None:
    recommendation = CalibrationRecommendation(
        neighborhood_size=4,
        decision_interval_seconds=120,
        minimum_threat_recall=0.9,
        minimum_timely_detection_fraction=0.8,
        minimum_decisions_before_tca=3,
        calibration_metrics=(_metrics(),),
        validation_metrics=(_metrics(EvaluationSplit.VALIDATION),),
    )

    assert recommendation.calibration_passed is True
    assert recommendation.validation_passed is True
    assert recommendation.is_accepted is True

    with pytest.raises(ValueError, match="match selected"):
        CalibrationRecommendation(
            **{
                **recommendation.__dict__,
                "validation_metrics": (
                    _metrics(EvaluationSplit.VALIDATION, neighborhood_size=8),
                ),
            }
        )

    with pytest.raises(ValueError, match="duplicate a sampled population"):
        CalibrationRecommendation(
            **{
                **recommendation.__dict__,
                "calibration_metrics": (_metrics(), _metrics()),
            }
        )
