from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orbitzoo.thesis.calibration import (
    AgentSelection,
    CalibrationConfig,
    CartesianStateFrame,
    CatalogObject,
    EncounterWindow,
    FineEncounterTrajectory,
    LoadedCatalog,
    ObjectType,
    generate_reference_conjunctions,
    load_reference_conjunctions,
    save_reference_conjunctions,
)


UTC_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _object(
    norad_id: int,
    *,
    candidate: bool = False,
    radius_meters: float = 1.0,
) -> CatalogObject:
    return CatalogObject(
        norad_id=norad_id,
        name=f"OBJECT {norad_id}",
        tle_name=None,
        line1="test line 1",
        line2="test line 2",
        tle_epoch_utc=UTC_EPOCH,
        object_type=ObjectType.PAYLOAD if candidate else ObjectType.DEBRIS,
        is_agent_candidate=candidate,
        radius_meters=radius_meters,
        constellation="test" if candidate else None,
        has_metadata=True,
    )


def _catalog(*objects: CatalogObject) -> LoadedCatalog:
    return LoadedCatalog(
        objects=objects,
        latest_epoch_utc=UTC_EPOCH,
        source_record_count=len(objects),
        stale_filtered_norad_ids=(),
    )


def _frame(
    seconds: int,
    pair: tuple[int, int],
    separation_meters: float,
    relative_speed_mps: float,
) -> CartesianStateFrame:
    return CartesianStateFrame(
        epoch_utc=UTC_EPOCH + timedelta(seconds=seconds),
        norad_ids=pair,
        positions_m=np.asarray(
            [[0.0, 0.0, 0.0], [separation_meters, 0.0, 0.0]]
        ),
        velocities_mps=np.asarray(
            [[0.0, 0.0, 0.0], [relative_speed_mps, 0.0, 0.0]]
        ),
    )


def _trajectory(
    pair: tuple[int, int] = (10, 20),
    *,
    start_seconds: int = 0,
    end_seconds: int = 10,
    start_separation_meters: float = -1_000.0,
    relative_speed_mps: float = 200.0,
) -> FineEncounterTrajectory:
    duration_seconds = end_seconds - start_seconds
    end_separation_meters = (
        start_separation_meters + relative_speed_mps * duration_seconds
    )
    window = EncounterWindow(
        first_norad_id=pair[0],
        second_norad_id=pair[1],
        start_epoch_utc=UTC_EPOCH + timedelta(seconds=start_seconds),
        end_epoch_utc=UTC_EPOCH + timedelta(seconds=end_seconds),
        minimum_coarse_miss_distance_meters=0.0,
        coarse_detection_count=1,
    )
    return FineEncounterTrajectory(
        window=window,
        frames=(
            _frame(
                start_seconds,
                pair,
                start_separation_meters,
                relative_speed_mps,
            ),
            _frame(
                end_seconds,
                pair,
                end_separation_meters,
                relative_speed_mps,
            ),
        ),
    )


def test_reference_tca_is_refined_between_fine_samples() -> None:
    catalog = _catalog(
        _object(10, candidate=True, radius_meters=2.0),
        _object(20, radius_meters=3.0),
    )

    result = generate_reference_conjunctions(
        (_trajectory(),),
        catalog,
        CalibrationConfig(),
        screened_agent_norad_ids=(10,),
    )

    assert len(result.conjunctions) == 1
    conjunction = result.conjunctions[0]
    assert conjunction.tca_epoch_utc == UTC_EPOCH + timedelta(seconds=5)
    assert conjunction.miss_distance_meters == pytest.approx(0.0)
    assert conjunction.relative_speed_mps == pytest.approx(200.0)
    assert conjunction.combined_radius_meters == pytest.approx(5.0)
    assert conjunction.is_collision is True
    assert result.collision_count == 1


def test_fine_pass_removes_coarse_false_positives() -> None:
    catalog = _catalog(_object(10, candidate=True), _object(20))
    trajectory = _trajectory(
        start_separation_meters=2_000.0,
        relative_speed_mps=0.0,
    )

    result = generate_reference_conjunctions(
        (trajectory,),
        catalog,
        CalibrationConfig(),
        screened_agent_norad_ids=(10,),
    )

    assert result.conjunctions == ()


def test_overlapping_windows_are_deduplicated_but_later_events_remain() -> None:
    catalog = _catalog(_object(10, candidate=True), _object(20))
    overlapping = _trajectory(
        start_seconds=5,
        end_seconds=15,
        start_separation_meters=-500.0,
        relative_speed_mps=200.0,
    )
    later = _trajectory(
        start_seconds=20,
        end_seconds=30,
        start_separation_meters=-1_000.0,
        relative_speed_mps=200.0,
    )

    result = generate_reference_conjunctions(
        (_trajectory(), overlapping, later),
        catalog,
        CalibrationConfig(),
        screened_agent_norad_ids=(10,),
    )

    assert tuple(item.tca_epoch_utc for item in result.conjunctions) == (
        UTC_EPOCH + timedelta(seconds=5),
        UTC_EPOCH + timedelta(seconds=25),
    )


def test_reference_manifest_round_trips_and_filters_by_selection(
    tmp_path: Path,
) -> None:
    catalog = _catalog(
        _object(10, candidate=True),
        _object(20, candidate=True),
        _object(30),
    )
    result = generate_reference_conjunctions(
        (
            _trajectory((10, 30)),
            _trajectory((20, 30), start_seconds=20, end_seconds=30),
        ),
        catalog,
        CalibrationConfig(),
        screened_agent_norad_ids=(20, 10),
    )
    destination = tmp_path / "reference_conjunctions.json"

    save_reference_conjunctions(result, destination)
    first_serialization = destination.read_text(encoding="utf-8")
    loaded = load_reference_conjunctions(destination)
    save_reference_conjunctions(loaded, destination)

    assert loaded == result
    assert destination.read_text(encoding="utf-8") == first_serialization
    assert '"is_collision": true' in first_serialization
    assert loaded.conjunctions_for(
        AgentSelection(seed=0, requested_agent_count=1, agent_norad_ids=(10,))
    ) == (result.conjunctions[0],)


def test_reference_generation_rejects_non_agent_screening_ids() -> None:
    catalog = _catalog(_object(10, candidate=True), _object(20))

    with pytest.raises(ValueError, match="not metadata-approved"):
        generate_reference_conjunctions(
            (),
            catalog,
            CalibrationConfig(),
            screened_agent_norad_ids=(20,),
        )
