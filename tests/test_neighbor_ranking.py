from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

import orbitzoo.thesis.calibration.ranking as ranking_module

from orbitzoo.thesis.calibration import (
    AgentSelection,
    CalibrationConfig,
    CartesianStateFrame,
    CatalogConfig,
    CatalogObject,
    ObjectType,
    PropagationConfig,
    SweepConfig,
    build_decision_schedule,
    iter_ranked_neighbor_frames,
    rank_neighbors_at_epoch,
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
            neighborhood_sizes=(1, 2, 3),
            decision_intervals_seconds=(10, 20, 30),
            calibration_seeds=(0,),
            validation_seeds=(100,),
        ),
    )


def _object(
    norad_id: int,
    *,
    candidate: bool = False,
    object_type: ObjectType = ObjectType.DEBRIS,
    radius_meters: float = 1.0,
) -> CatalogObject:
    return CatalogObject(
        norad_id=norad_id,
        name=f"OBJECT {norad_id}",
        tle_name=None,
        line1="test line 1",
        line2="test line 2",
        tle_epoch_utc=UTC_EPOCH,
        object_type=ObjectType.PAYLOAD if candidate else object_type,
        is_agent_candidate=candidate,
        radius_meters=radius_meters,
        constellation="test" if candidate else None,
        has_metadata=True,
    )


def _frame() -> CartesianStateFrame:
    return CartesianStateFrame(
        epoch_utc=UTC_EPOCH,
        norad_ids=(40, 10, 30, 20),
        positions_m=np.asarray(
            [
                [1_000.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        velocities_mps=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def _objects() -> tuple[CatalogObject, ...]:
    return (
        _object(20, candidate=True, radius_meters=2.0),
        _object(30),
        _object(10, candidate=True, radius_meters=2.0),
        _object(40, object_type=ObjectType.ROCKET_BODY),
    )


def test_decision_schedule_computes_shared_epochs_once() -> None:
    schedule = build_decision_schedule(UTC_EPOCH, _config())

    assert len(schedule.epochs_for(10)) == 6
    assert len(schedule.epochs_for(20)) == 3
    assert len(schedule.epochs_for(30)) == 2
    assert len(schedule.shared_epochs) == 6
    assert schedule.shared_epochs[-1] < schedule.end_epoch_utc
    assert schedule.includes(schedule.shared_epochs[2], 20) is True
    assert schedule.includes(schedule.shared_epochs[1], 20) is False


def test_ranking_prioritizes_collision_then_unsafe_threats() -> None:
    ranked = rank_neighbors_at_epoch(
        _frame(),
        _objects(),
        _config(),
        agent_norad_ids=(10,),
        maximum_neighbors=3,
        agent_batch_size=1,
    )

    assert [item.neighbor_norad_id for item in ranked.rankings] == [20, 30, 40]
    collision, unsafe, safe = ranked.rankings
    assert collision.is_collision is True
    assert collision.is_unsafe is True
    assert unsafe.is_collision is False
    assert unsafe.is_unsafe is True
    assert unsafe.time_to_closest_approach_seconds == pytest.approx(10.0)
    assert unsafe.predicted_miss_distance_meters == pytest.approx(0.0)
    assert safe.is_unsafe is False
    assert all(item.neighbor_norad_id != 10 for item in ranked.rankings)


def test_same_threat_scores_use_norad_id_as_final_tie_breaker() -> None:
    frame = _frame()
    positions = frame.positions_m.copy()
    velocities = frame.velocities_mps.copy()
    positions[3] = positions[2]
    velocities[3] = velocities[2]
    tied_frame = CartesianStateFrame(
        epoch_utc=frame.epoch_utc,
        norad_ids=frame.norad_ids,
        positions_m=positions,
        velocities_mps=velocities,
    )

    ranked = rank_neighbors_at_epoch(
        tied_frame,
        _objects(),
        _config(),
        agent_norad_ids=(10,),
        maximum_neighbors=2,
    )

    assert [item.neighbor_norad_id for item in ranked.rankings] == [20, 30]


def test_largest_ranking_frame_is_reused_for_nested_selection_and_k() -> None:
    ranked = rank_neighbors_at_epoch(
        _frame(),
        _objects(),
        _config(),
        agent_norad_ids=(20, 10),
        maximum_neighbors=3,
    )
    selection = AgentSelection(
        seed=0,
        requested_agent_count=1,
        agent_norad_ids=(10,),
    )

    filtered = ranked.rankings_for(selection, neighborhood_size=2)

    assert len(filtered) == 2
    assert all(item.agent_norad_id == 10 for item in filtered)
    assert [item.rank for item in filtered] == [1, 2]


def test_ranking_rejects_non_agent_observers() -> None:
    with pytest.raises(ValueError, match="not metadata-approved"):
        rank_neighbors_at_epoch(
            _frame(),
            _objects(),
            _config(),
            agent_norad_ids=(30,),
        )


def test_streaming_ranker_propagates_only_shared_decision_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    source_frame = _frame()
    captured_epochs: list[datetime] = []

    def fake_propagate_objects(objects, epochs, *, batch_size):
        del objects, batch_size
        captured_epochs.extend(epochs)
        return iter(
            CartesianStateFrame(
                epoch_utc=epoch,
                norad_ids=source_frame.norad_ids,
                positions_m=source_frame.positions_m,
                velocities_mps=source_frame.velocities_mps,
            )
            for epoch in captured_epochs
        )

    monkeypatch.setattr(ranking_module, "propagate_objects", fake_propagate_objects)
    propagation = SimpleNamespace(
        start_epoch_utc=UTC_EPOCH,
        end_epoch_utc=UTC_EPOCH + timedelta(seconds=60),
        objects=_objects(),
    )

    frames = tuple(
        iter_ranked_neighbor_frames(
            propagation,
            config,
            agent_norad_ids=(10,),
            maximum_neighbors=1,
        )
    )

    schedule = build_decision_schedule(UTC_EPOCH, config)
    assert captured_epochs == list(schedule.shared_epochs)
    assert [frame.decision_epoch_utc for frame in frames] == list(
        schedule.shared_epochs
    )
