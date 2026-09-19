"""Run a frozen decentralized policy for many agents against a full catalog.

See docs/SCALABILITY.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Protocol, Sequence

import numpy as np
from sgp4.api import SatrecArray

from orbitzoo.thesis.calibration.models import CatalogObject
from orbitzoo.thesis.calibration.propagation import METERS_PER_KILOMETER, _julian_date, _raise_first_error, _satellite
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.evaluation.policies import EvaluationPolicy
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import STANDARD_GRAVITY_MPS2, ManeuverConfig
from orbitzoo.thesis.scalability.dynamics import HillOffsets, inertial_states, mean_motions
from orbitzoo.thesis.environments.vectorized_observations import CatalogState, encode_local_observations
from orbitzoo.thesis.scalability.screening import ConjunctionEvent, ConjunctionTracker, close_approaches

ACTION_DIRECTIONS = np.array([action.rsw_unit_vector for action in ManeuverAction], dtype=np.float64)
TIMING_STAGES = ("propagation", "observation", "policy", "maneuver", "screening")


class ReferenceTrajectory(Protocol):
    """Unmaneuvered states of every object at times measured from the start epoch."""

    def states(self, times_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class SGP4Trajectory:
    """SGP4 TEME states for catalog objects, in metres and metres per second."""

    def __init__(self, objects: Sequence[CatalogObject], start_epoch_utc: datetime) -> None:
        self._norad_ids = tuple(item.norad_id for item in objects)
        self._satellites = SatrecArray([_satellite(item) for item in objects])
        self._start_jd, self._start_fraction = _julian_date(start_epoch_utc)
        self._start_epoch = start_epoch_utc

    def states(self, times_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        days = np.asarray(times_seconds, dtype=np.float64) / 86_400.0
        julian_dates = np.full(days.shape, self._start_jd)
        errors, positions_km, velocities_kmps = self._satellites.sgp4(julian_dates, self._start_fraction + days)
        _raise_first_error(errors, self._norad_ids, (self._start_epoch,) * len(days))
        return positions_km * METERS_PER_KILOMETER, velocities_kmps * METERS_PER_KILOMETER


@dataclass(frozen=True)
class SimulationSettings:
    """Physical and timing parameters shared by every scalability scenario."""

    neighborhood_size: int
    decision_interval_seconds: int
    duration_seconds: int
    fine_step_seconds: int
    safety: SafetyConfig
    maneuver: ManeuverConfig
    dry_mass_kg: float
    initial_fuel_mass_kg: float
    maximum_relative_speed_mps: float
    merge_gap_seconds: float = 600.0

    def validate(self) -> None:
        if self.decision_interval_seconds <= 0 or self.fine_step_seconds <= 0:
            raise ValueError("decision and fine step intervals must be positive")
        if self.duration_seconds % self.decision_interval_seconds:
            raise ValueError("duration_seconds must be a multiple of decision_interval_seconds")
        if self.decision_interval_seconds % self.fine_step_seconds:
            raise ValueError("decision_interval_seconds must be a multiple of fine_step_seconds")
        if self.dry_mass_kg <= 0 or self.initial_fuel_mass_kg < 0:
            raise ValueError("dry mass must be positive and fuel cannot be negative")


@dataclass
class SimulationResult:
    """Conjunctions, maneuver accounting, and stage timings for one policy run."""

    events: list[ConjunctionEvent]
    delta_v_mps: np.ndarray
    maneuvers: int
    rejected_actions: int
    decisions: int
    wall_seconds: float
    stage_seconds: dict[str, float] = field(default_factory=dict)
    burn_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    burn_times_seconds: np.ndarray = field(default_factory=lambda: np.empty(0))


def _execute_maneuvers(
    actions: np.ndarray, masses: np.ndarray, fuels: np.ndarray, maneuver: ManeuverConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return executed actions and propellant used, rejecting burns the contract would reject."""
    exhaust_velocity = maneuver.specific_impulse_seconds * STANDARD_GRAVITY_MPS2
    propellant = masses * (1 - np.exp(-maneuver.commanded_delta_v_mps / exhaust_velocity))
    burn_seconds = propellant / (maneuver.maximum_thrust_newtons / exhaust_velocity)
    requested = actions != ManeuverAction.NO_OP
    feasible = requested & (propellant <= fuels) & (burn_seconds <= maneuver.maximum_burn_duration_seconds)
    executed = np.where(feasible, actions, ManeuverAction.NO_OP)
    return executed, np.where(feasible, propellant, 0.0)


def simulate(
    trajectory: ReferenceTrajectory,
    radii: np.ndarray,
    agent_indices: np.ndarray,
    policy: EvaluationPolicy,
    settings: SimulationSettings,
) -> SimulationResult:
    """Step every agent at each decision epoch and screen agent-involved conjunctions."""
    settings.validate()
    agent_indices = np.asarray(agent_indices, dtype=np.intp)
    agent_count = agent_indices.size
    is_agent = np.zeros(radii.size, dtype=bool)
    is_agent[agent_indices] = True
    offsets = HillOffsets(agent_count)
    fuels = np.full(agent_count, settings.initial_fuel_mass_kg)
    masses = settings.dry_mass_kg + fuels
    delta_v = np.zeros(agent_count)
    maneuvers = rejected = 0
    burn_indices: list[np.ndarray] = []
    burn_times: list[np.ndarray] = []
    tracker = ConjunctionTracker(settings.safety.safe_separation_meters, radii, settings.merge_gap_seconds)
    fine = settings.fine_step_seconds
    search_radius = settings.safety.safe_separation_meters + settings.maximum_relative_speed_mps * fine / 2
    substeps = settings.decision_interval_seconds // fine
    decisions = settings.duration_seconds // settings.decision_interval_seconds
    stage = dict.fromkeys(TIMING_STAGES, 0.0)
    started = time.perf_counter()

    def agent_states(positions, velocities, substep_offsets, substep_rates, motion):
        full_positions, full_velocities = positions.copy(), velocities.copy()
        full_positions[agent_indices], full_velocities[agent_indices] = inertial_states(
            positions[agent_indices], velocities[agent_indices], substep_offsets, substep_rates, motion
        )
        return full_positions, full_velocities

    def screen(time_seconds, positions, velocities):
        clock = time.perf_counter()
        tracker.add(time_seconds, *close_approaches(positions, velocities, agent_indices, search_radius, fine / 2))
        stage["screening"] += time.perf_counter() - clock

    for decision in range(decisions):
        clock = time.perf_counter()
        start_time = decision * settings.decision_interval_seconds
        reference_positions, reference_velocities = trajectory.states(start_time + fine * np.arange(substeps + 1))
        motion = mean_motions(reference_positions[agent_indices, 0])
        stage["propagation"] += time.perf_counter() - clock

        clock = time.perf_counter()
        positions, velocities = agent_states(
            reference_positions[:, 0], reference_velocities[:, 0], offsets.offsets, offsets.rates, motion
        )
        fuel_fractions = np.zeros(radii.size)
        if settings.initial_fuel_mass_kg > 0:
            fuel_fractions[agent_indices] = np.clip(fuels / settings.initial_fuel_mass_kg, 0.0, 1.0)
        observations = encode_local_observations(
            CatalogState(positions, velocities, radii, is_agent, fuel_fractions),
            agent_indices,
            settings.neighborhood_size,
            settings.safety,
        )
        stage["observation"] += time.perf_counter() - clock

        clock = time.perf_counter()
        actions = np.asarray(policy.choose(observations), dtype=np.int64)
        stage["policy"] += time.perf_counter() - clock

        clock = time.perf_counter()
        executed, propellant = _execute_maneuvers(actions, masses, fuels, settings.maneuver)
        burned = executed != ManeuverAction.NO_OP
        maneuvers += int(burned.sum())
        burn_indices.append(agent_indices[burned])
        burn_times.append(np.full(int(burned.sum()), float(start_time)))
        rejected += int(((actions != ManeuverAction.NO_OP) & ~burned).sum())
        exhaust_velocity = settings.maneuver.specific_impulse_seconds * STANDARD_GRAVITY_MPS2
        delta_v[burned] += exhaust_velocity * np.log(masses[burned] / (masses[burned] - propellant[burned]))
        fuels -= propellant
        masses -= propellant
        offsets.apply_impulse(settings.maneuver.commanded_delta_v_mps * ACTION_DIRECTIONS[executed])
        stage["maneuver"] += time.perf_counter() - clock

        for substep in range(substeps):
            elapsed = substep * fine
            substep_offsets, substep_rates = offsets.advanced(motion, elapsed)
            screen(
                start_time + elapsed,
                *agent_states(
                    reference_positions[:, substep],
                    reference_velocities[:, substep],
                    substep_offsets,
                    substep_rates,
                    motion,
                ),
            )
        offsets.advance(motion, settings.decision_interval_seconds)

    if decisions:
        screen(
            settings.duration_seconds,
            *agent_states(
                reference_positions[:, substeps], reference_velocities[:, substeps], offsets.offsets, offsets.rates, motion
            ),
        )
    return SimulationResult(
        events=tracker.events(),
        delta_v_mps=delta_v,
        maneuvers=maneuvers,
        rejected_actions=rejected,
        decisions=decisions,
        wall_seconds=time.perf_counter() - started,
        stage_seconds=stage,
        burn_indices=np.concatenate(burn_indices) if burn_indices else np.empty(0, dtype=np.intp),
        burn_times_seconds=np.concatenate(burn_times) if burn_times else np.empty(0),
    )
