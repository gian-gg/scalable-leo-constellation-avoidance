from datetime import datetime, timedelta, timezone

import pytest

from orbitzoo.thesis.calibration import (
    AgentSelection,
    AgentSelectionManifest,
    CalibrationConfig,
    CatalogConfig,
    EvaluationSplit,
    PassingThresholds,
    PropagationConfig,
    RankedNeighbor,
    RankedNeighborFrame,
    ReferenceConjunction,
    ReferenceConjunctionManifest,
    SweepConfig,
    build_decision_schedule,
    evaluate_joint_combinations,
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


def _selections() -> AgentSelectionManifest:
    return AgentSelectionManifest(
        catalog_epoch_utc=UTC_EPOCH,
        eligible_norad_ids=(10, 11),
        calibration_selections=(
            AgentSelection(
                seed=0,
                requested_agent_count=2,
                agent_norad_ids=(10, 11),
            ),
        ),
        validation_selections=(
            AgentSelection(
                seed=100,
                requested_agent_count=2,
                agent_norad_ids=(10, 11),
            ),
        ),
    )


def _references(pair: tuple[int, int] = (10, 20)) -> ReferenceConjunctionManifest:
    return ReferenceConjunctionManifest(
        catalog_epoch_utc=UTC_EPOCH,
        safe_separation_meters=10.0,
        reference_step_seconds=10,
        screened_agent_norad_ids=(10, 11),
        conjunctions=(
            ReferenceConjunction(
                first_norad_id=pair[0],
                second_norad_id=pair[1],
                tca_epoch_utc=UTC_EPOCH + timedelta(seconds=35),
                miss_distance_meters=5.0,
                relative_speed_mps=100.0,
                combined_radius_meters=2.0,
            ),
        ),
    )


def _ranking(
    epoch_seconds: int,
    agent_id: int,
    neighbor_id: int,
    rank: int,
) -> RankedNeighbor:
    return RankedNeighbor(
        agent_norad_id=agent_id,
        neighbor_norad_id=neighbor_id,
        rank=rank,
        decision_epoch_utc=UTC_EPOCH + timedelta(seconds=epoch_seconds),
        current_separation_meters=1_000.0,
        time_to_closest_approach_seconds=max(0.0, 35.0 - epoch_seconds),
        predicted_miss_distance_meters=5.0,
        combined_radius_meters=2.0,
        is_collision=False,
        is_unsafe=True,
    )


def _frames(*, maximum_neighbors: int = 2) -> tuple[RankedNeighborFrame, ...]:
    frames = []
    for epoch_seconds in range(0, 60, 10):
        if epoch_seconds == 20:
            rankings = (
                _ranking(epoch_seconds, 10, 30, 1),
                _ranking(epoch_seconds, 10, 20, 2),
            )
        else:
            rankings = (
                _ranking(epoch_seconds, 10, 20, 1),
                _ranking(epoch_seconds, 10, 30, 2),
            )
        frames.append(
            RankedNeighborFrame(
                decision_epoch_utc=UTC_EPOCH + timedelta(seconds=epoch_seconds),
                agent_norad_ids=(10, 11),
                maximum_neighbors=maximum_neighbors,
                rankings=rankings[:maximum_neighbors],
            )
        )
    return tuple(frames)


def _calibration_detection(detections, k: int, delta_t: int):
    return next(
        item
        for item in detections
        if item.evaluation_split is EvaluationSplit.CALIBRATION
        and item.neighborhood_size == k
        and item.decision_interval_seconds == delta_t
    )


def test_joint_evaluator_updates_all_prefixes_and_schedules_in_one_pass() -> None:
    consumed_frames = 0

    def ranking_stream():
        nonlocal consumed_frames
        for frame in _frames():
            consumed_frames += 1
            yield frame

    detections = evaluate_joint_combinations(
        ranking_stream(),
        _references(),
        _selections(),
        _config(),
    )

    assert consumed_frames == 6
    assert len(detections) == 8

    k2_dt10 = _calibration_detection(detections, 2, 10)
    assert k2_dt10.first_visible_epoch_utc == UTC_EPOCH + timedelta(seconds=20)
    assert k2_dt10.decisions_remaining == 2
    assert k2_dt10.detected is True
    assert k2_dt10.timely_detected is True

    k1_dt10 = _calibration_detection(detections, 1, 10)
    assert k1_dt10.first_visible_epoch_utc == UTC_EPOCH + timedelta(seconds=30)
    assert k1_dt10.decisions_remaining == 1
    assert k1_dt10.timely_detected is False

    k2_dt20 = _calibration_detection(detections, 2, 20)
    assert k2_dt20.first_visible_epoch_utc == UTC_EPOCH + timedelta(seconds=20)
    assert k2_dt20.decisions_remaining == 1

    k1_dt20 = _calibration_detection(detections, 1, 20)
    assert k1_dt20.first_visible_epoch_utc is None
    assert k1_dt20.decisions_remaining == 0
    assert k1_dt20.detected is False


def test_agent_agent_visibility_is_not_counted_as_two_reference_events() -> None:
    frames = []
    for frame in _frames():
        if frame.decision_epoch_utc == UTC_EPOCH + timedelta(seconds=20):
            rankings = (
                _ranking(20, 10, 11, 1),
                _ranking(20, 11, 10, 1),
            )
        else:
            rankings = ()
        frames.append(
            RankedNeighborFrame(
                decision_epoch_utc=frame.decision_epoch_utc,
                agent_norad_ids=(10, 11),
                maximum_neighbors=2,
                rankings=rankings,
            )
        )

    detections = evaluate_joint_combinations(
        frames,
        _references((10, 11)),
        _selections(),
        _config(),
    )

    assert len(detections) == 8
    assert _calibration_detection(detections, 1, 10).first_visible_epoch_utc == (
        UTC_EPOCH + timedelta(seconds=20)
    )


def test_evaluator_rejects_incomplete_or_insufficient_ranking_streams() -> None:
    with pytest.raises(ValueError, match="ended before"):
        evaluate_joint_combinations(
            _frames()[:-1],
            _references(),
            _selections(),
            _config(),
        )

    with pytest.raises(ValueError, match="largest configured k"):
        evaluate_joint_combinations(
            _frames(maximum_neighbors=1),
            _references(),
            _selections(),
            _config(),
        )


def test_explicit_schedule_must_match_the_configuration() -> None:
    schedule = build_decision_schedule(UTC_EPOCH, _config())
    mismatched_schedule = type(schedule)(
        start_epoch_utc=schedule.start_epoch_utc,
        end_epoch_utc=schedule.end_epoch_utc,
        epochs_by_interval=(schedule.epochs_by_interval[0],),
        shared_epochs=schedule.epochs_by_interval[0][1],
    )

    with pytest.raises(ValueError, match="configured intervals"):
        evaluate_joint_combinations(
            _frames(),
            _references(),
            _selections(),
            _config(),
            schedule=mismatched_schedule,
        )
