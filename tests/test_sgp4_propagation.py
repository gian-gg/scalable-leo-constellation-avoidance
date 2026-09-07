from datetime import timedelta, timezone

import numpy as np
import pytest
from sgp4.api import Satrec, jday
from sgp4.conveniences import sat_epoch_datetime
from sgp4.io import fix_checksum

from orbitzoo.thesis.calibration import (
    CalibrationConfig,
    CatalogConfig,
    CatalogObject,
    LoadedCatalog,
    ObjectType,
    PropagationConfig,
    PropagationError,
    SweepConfig,
    propagate_catalog,
)
from orbitzoo.thesis.environments.safety import SafetyConfig


BASE_LINE_1 = (
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
)
BASE_LINE_2 = (
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
)


def _catalog_object(norad_id: int, mean_motion: str = "15.50103472") -> CatalogObject:
    catalog_field = f"{norad_id:05d}"
    line1 = fix_checksum(
        (BASE_LINE_1[:2] + catalog_field + BASE_LINE_1[7:])[:68]
    )
    line2 = BASE_LINE_2[:2] + catalog_field + BASE_LINE_2[7:52]
    line2 += mean_motion + BASE_LINE_2[63:]
    line2 = fix_checksum(line2[:68])
    epoch = sat_epoch_datetime(Satrec.twoline2rv(line1, line2)).astimezone(timezone.utc)
    return CatalogObject(
        norad_id=norad_id,
        name=f"OBJECT {norad_id}",
        tle_name=None,
        line1=line1,
        line2=line2,
        tle_epoch_utc=epoch,
        object_type=ObjectType.PAYLOAD,
        is_agent_candidate=True,
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


def _config(
    minimum_altitude_meters: float = 200_000.0,
    maximum_altitude_meters: float = 1_000_000.0,
) -> CalibrationConfig:
    return CalibrationConfig(
        catalog=CatalogConfig(
            minimum_altitude_meters=minimum_altitude_meters,
            maximum_altitude_meters=maximum_altitude_meters,
        ),
        propagation=PropagationConfig(
            duration_seconds=20,
            reference_step_seconds=10,
        ),
        safety=SafetyConfig(screening_horizon_seconds=20),
        sweep=SweepConfig(
            agent_counts=(2,),
            neighborhood_sizes=(1,),
            decision_intervals_seconds=(10, 20),
            calibration_seeds=(0,),
            validation_seeds=(100,),
        ),
    )


def test_propagates_inclusive_timeline_to_teme_si_frames() -> None:
    low_earth_object = _catalog_object(25544)
    medium_earth_object = _catalog_object(40909, mean_motion=" 2.00000000")
    propagation = propagate_catalog(
        _catalog(low_earth_object, medium_earth_object),
        _config(),
        batch_size=2,
    )

    frames = list(propagation)

    assert propagation.norad_ids == (25544,)
    assert propagation.altitude_filtered_norad_ids == (40909,)
    assert propagation.objects == (low_earth_object,)
    assert propagation.frame_count == 3
    assert propagation.end_epoch_utc == propagation.start_epoch_utc + timedelta(seconds=20)
    assert [frame.epoch_utc for frame in frames] == [
        propagation.start_epoch_utc + timedelta(seconds=offset)
        for offset in (0, 10, 20)
    ]
    assert all(frame.norad_ids == (25544,) for frame in frames)
    assert all(frame.reference_frame == "TEME" for frame in frames)
    assert frames[0].positions_m.shape == (1, 3)
    assert frames[0].velocities_mps.shape == (1, 3)
    assert np.linalg.norm(frames[0].positions_m[0]) > 6_000_000
    assert np.linalg.norm(frames[0].velocities_mps[0]) > 1_000


def test_sgp4_kilometer_output_is_converted_to_meters() -> None:
    object_record = _catalog_object(25544)
    propagation = propagate_catalog(_catalog(object_record), _config())

    frame = next(iter(propagation))
    epoch = propagation.start_epoch_utc
    jd, fraction = jday(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second + epoch.microsecond / 1_000_000.0,
    )
    error, position_km, velocity_kmps = Satrec.twoline2rv(
        object_record.line1,
        object_record.line2,
    ).sgp4(jd, fraction)

    assert error == 0
    assert np.allclose(frame.positions_m[0], np.asarray(position_km) * 1_000)
    assert np.allclose(frame.velocities_mps[0], np.asarray(velocity_kmps) * 1_000)


def test_propagation_is_repeatable_and_batch_size_independent() -> None:
    catalog = _catalog(_catalog_object(25544), _catalog_object(40909))

    single_frame_batches = list(propagate_catalog(catalog, _config(), batch_size=1))
    larger_batches = list(propagate_catalog(catalog, _config(), batch_size=64))

    assert len(single_frame_batches) == len(larger_batches)
    for first, second in zip(single_frame_batches, larger_batches, strict=True):
        assert first.epoch_utc == second.epoch_utc
        assert first.norad_ids == second.norad_ids
        assert np.array_equal(first.positions_m, second.positions_m)
        assert np.array_equal(first.velocities_mps, second.velocities_mps)


def test_rejects_catalog_when_altitude_filter_removes_every_object() -> None:
    catalog = _catalog(_catalog_object(25544))

    with pytest.raises(PropagationError, match="no catalog objects remain"):
        propagate_catalog(
            catalog,
            _config(
                minimum_altitude_meters=200_000.0,
                maximum_altitude_meters=300_000.0,
            ),
        )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_rejects_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        propagate_catalog(
            _catalog(_catalog_object(25544)),
            _config(),
            batch_size=batch_size,
        )
