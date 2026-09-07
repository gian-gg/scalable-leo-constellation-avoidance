from datetime import timezone

import pytest
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from sgp4.io import fix_checksum

from orbitzoo.thesis.calibration import (
    AgentSelection,
    AgentSelectionManifest,
    CalibrationConfig,
    CatalogConfig,
    CatalogObject,
    LoadedCatalog,
    ObjectType,
    PropagationConfig,
    PropagationError,
    SweepConfig,
    build_two_resolution_propagation,
)
from orbitzoo.thesis.environments.safety import SafetyConfig


BASE_LINE_1 = (
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
)
BASE_LINE_2 = (
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
)


def _object(
    norad_id: int,
    *,
    mean_anomaly: str = " 85.6398",
    is_agent_candidate: bool = False,
) -> CatalogObject:
    catalog_field = f"{norad_id:05d}"
    line1 = fix_checksum(
        (BASE_LINE_1[:2] + catalog_field + BASE_LINE_1[7:])[:68]
    )
    line2 = BASE_LINE_2[:2] + catalog_field + BASE_LINE_2[7:43]
    line2 += mean_anomaly + BASE_LINE_2[51:]
    line2 = fix_checksum(line2[:68])
    epoch = sat_epoch_datetime(Satrec.twoline2rv(line1, line2)).astimezone(timezone.utc)
    return CatalogObject(
        norad_id=norad_id,
        name=f"OBJECT {norad_id}",
        tle_name=None,
        line1=line1,
        line2=line2,
        tle_epoch_utc=epoch,
        object_type=(
            ObjectType.PAYLOAD if is_agent_candidate else ObjectType.DEBRIS
        ),
        is_agent_candidate=is_agent_candidate,
        radius_meters=1.0,
        constellation=None,
        has_metadata=True,
    )


def _catalog(*objects: CatalogObject) -> LoadedCatalog:
    return LoadedCatalog(
        objects=objects,
        latest_epoch_utc=max(item.tle_epoch_utc for item in objects),
        source_record_count=len(objects),
        stale_filtered_norad_ids=(),
    )


def _config(maximum_relative_speed_mps: float = 20_000.0) -> CalibrationConfig:
    return CalibrationConfig(
        catalog=CatalogConfig(
            minimum_altitude_meters=200_000.0,
            maximum_altitude_meters=1_000_000.0,
        ),
        propagation=PropagationConfig(
            duration_seconds=120,
            reference_step_seconds=10,
            coarse_step_seconds=60,
            fine_window_padding_seconds=60,
            maximum_relative_speed_mps=maximum_relative_speed_mps,
        ),
        safety=SafetyConfig(
            safe_separation_meters=1_000.0,
            screening_horizon_seconds=120,
        ),
        sweep=SweepConfig(
            agent_counts=(2,),
            neighborhood_sizes=(1,),
            decision_intervals_seconds=(10, 20, 60, 120),
            calibration_seeds=(0,),
            validation_seeds=(100,),
        ),
    )


def test_coarse_candidates_are_merged_then_propagated_as_fine_pairs() -> None:
    agent = _object(25544, is_agent_candidate=True)
    colocated_debris = _object(40909)
    parallel_debris = _object(43013, mean_anomaly=" 89.6398")
    propagation = build_two_resolution_propagation(
        _catalog(agent, colocated_debris, parallel_debris),
        _config(),
        batch_size=2,
    )

    windows = propagation.discover_encounter_windows((25544,))

    assert propagation.coarse_propagation.step_seconds == 60
    assert propagation.coarse_propagation.frame_count == 3
    assert propagation.coarse_search_radius_meters == 1_201_000.0
    assert propagation.coarse_candidate_threshold_meters < 50_000.0
    assert len(windows) == 1
    window = windows[0]
    assert (window.first_norad_id, window.second_norad_id) == (25544, 40909)
    assert window.start_epoch_utc == propagation.coarse_propagation.start_epoch_utc
    assert window.end_epoch_utc == propagation.coarse_propagation.end_epoch_utc
    assert window.coarse_detection_count == 2

    trajectories = tuple(propagation.iter_fine_trajectories(windows))

    assert len(trajectories) == 1
    assert len(trajectories[0].frames) == 13
    assert all(
        frame.norad_ids == (25544, 40909)
        for frame in trajectories[0].frames
    )
    assert all(
        43013 not in frame.norad_ids
        for frame in trajectories[0].frames
    )


def test_run_is_deterministic_and_combines_both_passes() -> None:
    propagation = build_two_resolution_propagation(
        _catalog(_object(25544, is_agent_candidate=True), _object(40909)),
        _config(),
    )

    first = tuple(propagation.run((25544,)))
    second = tuple(propagation.run((25544,)))

    assert [item.window for item in first] == [item.window for item in second]
    assert [frame.epoch_utc for frame in first[0].frames] == [
        frame.epoch_utc for frame in second[0].frames
    ]


def test_catalog_only_pairs_are_never_screened() -> None:
    propagation = build_two_resolution_propagation(
        _catalog(
            _object(
                25544,
                mean_anomaly="265.6398",
                is_agent_candidate=True,
            ),
            _object(40909),
            _object(43013),
        ),
        _config(),
    )

    assert propagation.discover_encounter_windows((25544,)) == ()
    assert tuple(propagation.run((25544,))) == ()


def test_selected_agent_pairs_are_canonical_and_deduplicated() -> None:
    propagation = build_two_resolution_propagation(
        _catalog(
            _object(25544, is_agent_candidate=True),
            _object(40909, is_agent_candidate=True),
        ),
        _config(),
    )

    windows = propagation.discover_encounter_windows((40909, 25544))

    assert len(windows) == 1
    assert (windows[0].first_norad_id, windows[0].second_norad_id) == (
        25544,
        40909,
    )
    assert windows[0].coarse_detection_count == 2


def test_nested_populations_reuse_one_selection_screening_pass() -> None:
    agents = tuple(
        _object(norad_id, is_agent_candidate=True)
        for norad_id in (25544, 40909, 43013, 44001)
    )
    propagation = build_two_resolution_propagation(_catalog(*agents), _config())
    small = AgentSelection(
        seed=0,
        requested_agent_count=2,
        agent_norad_ids=(25544, 40909),
    )
    large = AgentSelection(
        seed=0,
        requested_agent_count=4,
        agent_norad_ids=(25544, 40909, 43013, 44001),
    )
    validation_small = AgentSelection(
        seed=100,
        requested_agent_count=2,
        agent_norad_ids=(25544, 40909),
    )
    validation_large = AgentSelection(
        seed=100,
        requested_agent_count=4,
        agent_norad_ids=(25544, 40909, 43013, 44001),
    )
    manifest = AgentSelectionManifest(
        catalog_epoch_utc=propagation.coarse_propagation.start_epoch_utc,
        eligible_norad_ids=(25544, 40909, 43013, 44001),
        calibration_selections=(small, large),
        validation_selections=(validation_small, validation_large),
    )

    screening = propagation.screen_agent_selections(manifest)

    assert screening.screened_agent_norad_ids == (25544, 40909, 43013, 44001)
    assert len(screening.encounter_windows) == 6
    assert len(screening.windows_for(small)) == 5
    assert all(
        25544 in (window.first_norad_id, window.second_norad_id)
        or 40909 in (window.first_norad_id, window.second_norad_id)
        for window in screening.windows_for(small)
    )


def test_screening_rejects_absent_or_ineligible_selected_agents() -> None:
    propagation = build_two_resolution_propagation(
        _catalog(_object(25544, is_agent_candidate=True), _object(40909)),
        _config(),
    )

    with pytest.raises(ValueError, match="absent"):
        propagation.discover_encounter_windows((99999,))
    with pytest.raises(ValueError, match="not metadata-approved"):
        propagation.discover_encounter_windows((40909,))


def test_rejects_nonconservative_relative_speed_bound() -> None:
    propagation = build_two_resolution_propagation(
        _catalog(_object(25544, is_agent_candidate=True), _object(40909)),
        _config(maximum_relative_speed_mps=1_000.0),
    )

    with pytest.raises(PropagationError, match="not conservative"):
        propagation.discover_encounter_windows((25544,))
