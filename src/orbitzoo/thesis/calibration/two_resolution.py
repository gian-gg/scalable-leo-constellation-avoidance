"""Conservative coarse screening with fine pair-only SGP4 propagation."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Iterator

import numpy as np
from scipy.spatial import cKDTree
from sgp4.earth_gravity import wgs72

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import (
    AgentSelection,
    AgentSelectionManifest,
    CandidateScreeningResult,
    EncounterWindow,
    FineEncounterTrajectory,
    LoadedCatalog,
)
from orbitzoo.thesis.calibration.propagation import (
    DEFAULT_BATCH_SIZE,
    METERS_PER_KILOMETER,
    PropagationError,
    SGP4Propagation,
    propagate_catalog,
    propagate_objects,
)


class TwoResolutionPropagation:
    """Find candidate encounters coarsely, then propagate only their pairs finely."""

    def __init__(
        self,
        catalog: LoadedCatalog,
        config: CalibrationConfig,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        config.validate()
        self._config = config
        self._batch_size = batch_size
        self._coarse_propagation = propagate_catalog(
            catalog,
            config,
            batch_size=batch_size,
            step_seconds=config.propagation.coarse_step_seconds,
        )
        self._object_by_id = {
            item.norad_id: item for item in self._coarse_propagation.objects
        }

    @property
    def coarse_propagation(self) -> SGP4Propagation:
        return self._coarse_propagation

    @property
    def coarse_search_radius_meters(self) -> float:
        propagation = self._config.propagation
        return (
            self._config.safety.safe_separation_meters
            + propagation.maximum_relative_speed_mps
            * propagation.coarse_step_seconds
        )

    @property
    def coarse_curvature_margin_meters(self) -> float:
        """Bound two-body curvature omitted by a linear 60-second screen."""
        propagation = self._config.propagation
        earth_radius_meters = wgs72.radiusearthkm * METERS_PER_KILOMETER
        minimum_radius_meters = (
            earth_radius_meters + self._config.catalog.minimum_altitude_meters
        )
        gravitational_parameter_m3_s2 = wgs72.mu * METERS_PER_KILOMETER**3
        maximum_acceleration_mps2 = (
            gravitational_parameter_m3_s2 / minimum_radius_meters**2
        )
        return maximum_acceleration_mps2 * propagation.coarse_step_seconds**2

    @property
    def coarse_candidate_threshold_meters(self) -> float:
        return (
            self._config.safety.safe_separation_meters
            + self.coarse_curvature_margin_meters
        )

    def _validated_agent_ids(
        self,
        agent_norad_ids: Iterable[int],
    ) -> tuple[int, ...]:
        identifiers = tuple(agent_norad_ids)
        if not identifiers:
            raise ValueError("at least one selected agent NORAD ID is required")
        if any(
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier <= 0
            for identifier in identifiers
        ):
            raise ValueError("selected agent NORAD IDs must be positive integers")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("selected agent NORAD IDs must be unique")
        absent_ids = set(identifiers).difference(self._object_by_id)
        if absent_ids:
            raise ValueError(
                "selected agents are absent from the altitude-filtered catalog: "
                f"{sorted(absent_ids)}"
            )
        ineligible_ids = sorted(
            identifier
            for identifier in identifiers
            if not self._object_by_id[identifier].is_agent_candidate
        )
        if ineligible_ids:
            raise ValueError(
                "selected agents are not metadata-approved agent candidates: "
                f"{ineligible_ids}"
            )
        return tuple(sorted(identifiers))

    def discover_encounter_windows(
        self,
        agent_norad_ids: Iterable[int],
    ) -> tuple[EncounterWindow, ...]:
        """Screen selected agents against the spatially indexed full catalog."""
        selected_agent_ids = self._validated_agent_ids(agent_norad_ids)
        propagation = self._config.propagation
        global_start = self._coarse_propagation.start_epoch_utc
        global_end = self._coarse_propagation.end_epoch_utc
        padding = timedelta(seconds=propagation.fine_window_padding_seconds)
        coarse_interval = timedelta(seconds=propagation.coarse_step_seconds)
        windows_by_pair: dict[tuple[int, int], list[EncounterWindow]] = {}
        index_by_norad_id = {
            norad_id: index
            for index, norad_id in enumerate(self._coarse_propagation.norad_ids)
        }
        agent_indices = np.asarray(
            [index_by_norad_id[norad_id] for norad_id in selected_agent_ids],
            dtype=np.intp,
        )

        for frame in self._coarse_propagation:
            if frame.epoch_utc >= global_end:
                break
            maximum_object_speed = float(
                np.max(np.linalg.norm(frame.velocities_mps, axis=1))
            )
            if 2.0 * maximum_object_speed > propagation.maximum_relative_speed_mps:
                raise PropagationError(
                    "maximum_relative_speed_mps is not conservative for the "
                    f"coarse frame at {frame.epoch_utc.isoformat()}"
                )

            tree = cKDTree(frame.positions_m)
            neighbor_indices = tree.query_ball_point(
                frame.positions_m[agent_indices],
                self.coarse_search_radius_meters,
            )
            candidate_indices: set[tuple[int, int]] = set()
            for agent_index, neighbors in zip(agent_indices, neighbor_indices):
                for neighbor_index in neighbors:
                    if neighbor_index == agent_index:
                        continue
                    first_index, second_index = int(agent_index), int(neighbor_index)
                    if frame.norad_ids[first_index] > frame.norad_ids[second_index]:
                        first_index, second_index = second_index, first_index
                    candidate_indices.add((first_index, second_index))

            for first_index, second_index in sorted(
                candidate_indices,
                key=lambda pair: (
                    frame.norad_ids[pair[0]],
                    frame.norad_ids[pair[1]],
                ),
            ):
                first_id = frame.norad_ids[first_index]
                second_id = frame.norad_ids[second_index]
                pair = first_id, second_id
                relative_position = (
                    frame.positions_m[second_index]
                    - frame.positions_m[first_index]
                )
                relative_velocity = (
                    frame.velocities_mps[second_index]
                    - frame.velocities_mps[first_index]
                )
                relative_speed_squared = float(
                    np.dot(relative_velocity, relative_velocity)
                )
                if relative_speed_squared <= 1e-12:
                    time_to_closest_approach = 0.0
                else:
                    time_to_closest_approach = float(
                        np.clip(
                            -np.dot(relative_position, relative_velocity)
                            / relative_speed_squared,
                            0.0,
                            propagation.coarse_step_seconds,
                        )
                    )
                predicted_miss_distance = float(
                    np.linalg.norm(
                        relative_position
                        + relative_velocity * time_to_closest_approach
                    )
                )
                if predicted_miss_distance > self.coarse_candidate_threshold_meters:
                    continue
                window = EncounterWindow(
                    first_norad_id=pair[0],
                    second_norad_id=pair[1],
                    start_epoch_utc=max(global_start, frame.epoch_utc - padding),
                    end_epoch_utc=min(
                        global_end,
                        frame.epoch_utc + coarse_interval + padding,
                    ),
                    minimum_coarse_miss_distance_meters=predicted_miss_distance,
                    coarse_detection_count=1,
                )
                pair_windows = windows_by_pair.setdefault(pair, [])
                if pair_windows and window.start_epoch_utc <= pair_windows[-1].end_epoch_utc:
                    previous = pair_windows[-1]
                    pair_windows[-1] = EncounterWindow(
                        first_norad_id=pair[0],
                        second_norad_id=pair[1],
                        start_epoch_utc=previous.start_epoch_utc,
                        end_epoch_utc=max(
                            previous.end_epoch_utc,
                            window.end_epoch_utc,
                        ),
                        minimum_coarse_miss_distance_meters=min(
                            previous.minimum_coarse_miss_distance_meters,
                            predicted_miss_distance,
                        ),
                        coarse_detection_count=(
                            previous.coarse_detection_count + 1
                        ),
                    )
                else:
                    pair_windows.append(window)

        windows = [
            window
            for pair_windows in windows_by_pair.values()
            for window in pair_windows
        ]
        windows.sort(
            key=lambda item: (
                item.start_epoch_utc,
                item.first_norad_id,
                item.second_norad_id,
            )
        )
        return tuple(windows)

    def screen_agent_selections(
        self,
        manifest: AgentSelectionManifest,
    ) -> CandidateScreeningResult:
        """Screen the largest nested population for every configured seed once."""
        if manifest.catalog_epoch_utc != self._coarse_propagation.start_epoch_utc:
            raise ValueError(
                "agent-selection catalog epoch does not match propagation start"
            )
        largest_by_seed: list[AgentSelection] = []
        for selections in (
            manifest.calibration_selections,
            manifest.validation_selections,
        ):
            by_seed: dict[int, list] = {}
            for selection in selections:
                by_seed.setdefault(selection.seed, []).append(selection)
            largest_by_seed.extend(
                max(seed_selections, key=lambda item: item.requested_agent_count)
                for seed_selections in by_seed.values()
            )
        screened_agent_ids = tuple(
            sorted(
                {
                    norad_id
                    for selection in largest_by_seed
                    for norad_id in selection.agent_norad_ids
                }
            )
        )
        return CandidateScreeningResult(
            screened_agent_norad_ids=screened_agent_ids,
            encounter_windows=self.discover_encounter_windows(screened_agent_ids),
        )

    def iter_fine_trajectories(
        self,
        windows: Iterable[EncounterWindow],
    ) -> Iterator[FineEncounterTrajectory]:
        """Propagate only each candidate pair inside its merged fine window."""
        fine_step_seconds = self._config.propagation.reference_step_seconds
        global_start = self._coarse_propagation.start_epoch_utc
        global_end = self._coarse_propagation.end_epoch_utc
        for window in windows:
            if window.start_epoch_utc < global_start or window.end_epoch_utc > global_end:
                raise ValueError("encounter window lies outside the propagation timeline")
            try:
                objects = (
                    self._object_by_id[window.first_norad_id],
                    self._object_by_id[window.second_norad_id],
                )
            except KeyError as error:
                raise ValueError(
                    f"encounter window references absent NORAD ID {error.args[0]}"
                ) from error
            duration = (window.end_epoch_utc - window.start_epoch_utc).total_seconds()
            if not duration.is_integer():
                raise ValueError("encounter window must use whole-second boundaries")
            duration_seconds = int(duration)
            if duration_seconds % fine_step_seconds != 0:
                raise ValueError("encounter window is not aligned to the fine timestep")
            epochs = tuple(
                window.start_epoch_utc + timedelta(seconds=offset)
                for offset in range(
                    0,
                    duration_seconds + fine_step_seconds,
                    fine_step_seconds,
                )
            )
            frames = tuple(
                propagate_objects(
                    objects,
                    epochs,
                    batch_size=self._batch_size,
                )
            )
            yield FineEncounterTrajectory(window=window, frames=frames)

    def run(
        self,
        agent_norad_ids: Iterable[int],
    ) -> Iterator[FineEncounterTrajectory]:
        """Discover coarse windows and stream their fine pair trajectories."""
        return self.iter_fine_trajectories(
            self.discover_encounter_windows(agent_norad_ids)
        )


def build_two_resolution_propagation(
    catalog: LoadedCatalog,
    config: CalibrationConfig,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TwoResolutionPropagation:
    """Prepare conservative coarse-to-fine calibration propagation."""
    return TwoResolutionPropagation(catalog, config, batch_size=batch_size)
