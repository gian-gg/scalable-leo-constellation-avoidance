"""Streaming SGP4 propagation for frozen calibration catalogs."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec, SatrecArray, jday
from sgp4.conveniences import check_satrec
from sgp4.earth_gravity import wgs72

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import (
    CartesianStateFrame,
    CatalogObject,
    LoadedCatalog,
)


METERS_PER_KILOMETER = 1_000.0
DEFAULT_BATCH_SIZE = 64


class PropagationError(RuntimeError):
    """Raised when SGP4 cannot produce a complete calibration trajectory."""


def _julian_date(epoch: datetime) -> tuple[float, float]:
    seconds = epoch.second + epoch.microsecond / 1_000_000.0
    return jday(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        seconds,
    )


def _satellite(object_record: CatalogObject) -> Satrec:
    try:
        satellite = Satrec.twoline2rv(object_record.line1, object_record.line2)
        check_satrec(satellite)
        return satellite
    except (TypeError, ValueError) as error:
        raise PropagationError(
            f"cannot initialize SGP4 for NORAD {object_record.norad_id}: {error}"
        ) from error


def _raise_first_error(
    errors: np.ndarray,
    norad_ids: tuple[int, ...],
    epochs: tuple[datetime, ...],
) -> None:
    failures = np.argwhere(errors != 0)
    if failures.size == 0:
        return
    satellite_index, epoch_index = (int(value) for value in failures[0])
    error_code = int(errors[satellite_index, epoch_index])
    reason = SGP4_ERRORS.get(error_code, f"unknown SGP4 error {error_code}")
    raise PropagationError(
        f"SGP4 failed for NORAD {norad_ids[satellite_index]} at "
        f"{epochs[epoch_index].isoformat()}: {reason}"
    )


class SGP4Propagation:
    """Reusable, streaming propagation of one validated catalog."""

    def __init__(
        self,
        catalog: LoadedCatalog,
        config: CalibrationConfig,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        config.validate()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self._config = config
        self._batch_size = batch_size
        self._start_epoch_utc = catalog.latest_epoch_utc
        frame_count = (
            config.propagation.duration_seconds
            // config.propagation.reference_step_seconds
            + 1
        )
        self._epochs = tuple(
            self._start_epoch_utc
            + timedelta(seconds=index * config.propagation.reference_step_seconds)
            for index in range(frame_count)
        )

        source_objects = tuple(catalog.objects)
        source_ids = tuple(item.norad_id for item in source_objects)
        source_satellites = tuple(_satellite(item) for item in source_objects)
        start_jd, start_fraction = _julian_date(self._start_epoch_utc)
        errors, positions_km, _ = SatrecArray(source_satellites).sgp4(
            np.asarray([start_jd]),
            np.asarray([start_fraction]),
        )
        _raise_first_error(errors, source_ids, (self._start_epoch_utc,))

        earth_radius_meters = wgs72.radiusearthkm * METERS_PER_KILOMETER
        altitudes_meters = (
            np.linalg.norm(positions_km[:, 0, :], axis=1) * METERS_PER_KILOMETER
            - earth_radius_meters
        )
        minimum_altitude = config.catalog.minimum_altitude_meters
        maximum_altitude = config.catalog.maximum_altitude_meters
        retained_indices = tuple(
            index
            for index, altitude in enumerate(altitudes_meters)
            if minimum_altitude <= altitude <= maximum_altitude
        )
        if not retained_indices:
            raise PropagationError(
                "no catalog objects remain inside the configured altitude limits"
            )

        retained_index_set = set(retained_indices)
        self._objects = tuple(source_objects[index] for index in retained_indices)
        self._satellites = tuple(source_satellites[index] for index in retained_indices)
        self._norad_ids = tuple(item.norad_id for item in self._objects)
        self._altitude_filtered_norad_ids = tuple(
            item.norad_id
            for index, item in enumerate(source_objects)
            if index not in retained_index_set
        )

    @property
    def objects(self) -> tuple[CatalogObject, ...]:
        return self._objects

    @property
    def norad_ids(self) -> tuple[int, ...]:
        return self._norad_ids

    @property
    def altitude_filtered_norad_ids(self) -> tuple[int, ...]:
        return self._altitude_filtered_norad_ids

    @property
    def start_epoch_utc(self) -> datetime:
        return self._start_epoch_utc

    @property
    def end_epoch_utc(self) -> datetime:
        return self._epochs[-1]

    @property
    def frame_count(self) -> int:
        return len(self._epochs)

    def __iter__(self) -> Iterator[CartesianStateFrame]:
        satellite_array = SatrecArray(self._satellites)
        for batch_start in range(0, self.frame_count, self._batch_size):
            epochs = self._epochs[batch_start : batch_start + self._batch_size]
            julian_dates_and_fractions = tuple(_julian_date(epoch) for epoch in epochs)
            julian_dates = np.asarray(
                [value[0] for value in julian_dates_and_fractions],
                dtype=np.float64,
            )
            fractions = np.asarray(
                [value[1] for value in julian_dates_and_fractions],
                dtype=np.float64,
            )
            errors, positions_km, velocities_kmps = satellite_array.sgp4(
                julian_dates,
                fractions,
            )
            _raise_first_error(errors, self._norad_ids, epochs)
            for index, epoch in enumerate(epochs):
                yield CartesianStateFrame(
                    epoch_utc=epoch,
                    norad_ids=self._norad_ids,
                    positions_m=positions_km[:, index, :] * METERS_PER_KILOMETER,
                    velocities_mps=(
                        velocities_kmps[:, index, :] * METERS_PER_KILOMETER
                    ),
                )


def propagate_catalog(
    catalog: LoadedCatalog,
    config: CalibrationConfig,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SGP4Propagation:
    """Prepare a deterministic iterable of SI-unit Cartesian state frames."""
    return SGP4Propagation(catalog, config, batch_size=batch_size)
