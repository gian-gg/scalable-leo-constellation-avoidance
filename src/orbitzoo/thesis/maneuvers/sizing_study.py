"""Maneuver-sizing study over the calibration's reference conjunctions, with an Orekit spot-check.

See docs/MANEUVER_SIZING.md.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sgp4.earth_gravity import wgs72

from orbitzoo.thesis.calibration.catalog import load_catalog
from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.reference import load_reference_conjunctions
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.sizing import (
    EncounterGeometry,
    Requirement,
    encounter_from_states,
    requirement_for,
    resolved_fraction,
    smallest_passing_delta_v,
)
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.environments.vectorized_observations import CatalogState, top_neighbors
from orbitzoo.thesis.scalability.simulator import SGP4Trajectory


@dataclass(frozen=True)
class SizingConfig:
    """Inputs, candidate values, and the declared selection rule for maneuver sizing."""

    calibration_config: str = "k_dt_calibration_72h.json"
    reference_conjunctions: str = "../runs/k_dt_calibration_20k_72h/reference_conjunctions.json"
    decision_interval_seconds: float = 120.0
    safe_separation_meters: float = 1_000.0
    lead_times_seconds: tuple[float, ...] = (360.0, 600.0, 1_200.0, 1_800.0)
    delta_v_candidates_mps: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    target_fraction: float = 0.95
    selection_lead_seconds: float = 360.0
    selection_burns: int = 3
    satellite_masses_kg: tuple[float, ...] = (300.0, 800.0)
    maximum_burn_duration_seconds: float = 60.0
    spot_check_events: int = 12
    spot_check_thrust_newtons: float = 20.0
    spot_check_tolerance_fraction: float = 0.05
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("lead_times_seconds", "delta_v_candidates_mps", "satellite_masses_kg"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported sizing schema version: {self.schema_version}")
        if self.decision_interval_seconds <= 0 or self.safe_separation_meters <= 0:
            raise ValueError("decision_interval_seconds and safe_separation_meters must be positive")
        if not self.lead_times_seconds or min(self.lead_times_seconds) < self.decision_interval_seconds:
            raise ValueError("lead times must be at least one decision interval")
        if not self.delta_v_candidates_mps or min(self.delta_v_candidates_mps) <= 0:
            raise ValueError("delta-v candidates must be positive")
        if not 0 < self.target_fraction <= 1:
            raise ValueError("target_fraction must be in (0, 1]")
        if self.selection_burns < 1 or self.selection_burns > self.max_burns(self.selection_lead_seconds):
            raise ValueError("selection_burns must fit within selection_lead_seconds")
        if min(self.satellite_masses_kg, default=0) <= 0 or self.maximum_burn_duration_seconds <= 0:
            raise ValueError("satellite masses and maximum burn duration must be positive")
        if self.spot_check_events < 0 or self.spot_check_thrust_newtons <= 0:
            raise ValueError("spot-check settings must be non-negative with positive thrust")

    def max_burns(self, lead_seconds: float) -> int:
        """Decisions available from ``lead_seconds`` before closest approach until it."""
        return int(lead_seconds // self.decision_interval_seconds)

    def resolve(self, config_path: Path, relative: str) -> Path:
        path = Path(relative).expanduser()
        return path if path.is_absolute() else (config_path.resolve().parent / path).resolve()

    @classmethod
    def load(cls, path: str | Path) -> "SizingConfig":
        config = cls(**json.loads(Path(path).read_text()))
        config.validate()
        return config


@dataclass(frozen=True)
class SpotCheck:
    """Orekit two-body verification of one computed requirement."""

    event_id: int
    lead_seconds: float
    burns: int
    action: str
    delta_v_mps: float
    coasting_miss_m: float
    maneuvered_miss_m: float
    passed: bool


@dataclass(frozen=True)
class DetectionLead:
    """When the threat first became the agent's top-ranked neighbour before closest approach."""

    event_id: int
    lead_seconds: float | None
    available_burns: int


@dataclass
class SizingResult:
    encounters: list[EncounterGeometry]
    requirements: list[Requirement]
    selected_delta_v_mps: float | None
    detected_delta_v_mps: float | None = None
    detection_leads: list[DetectionLead] = field(default_factory=list)
    spot_checks: list[SpotCheck] = field(default_factory=list)


def _closest_approach(
    agent: tuple[np.ndarray, np.ndarray],
    threat: tuple[np.ndarray, np.ndarray],
    max_offset_seconds: float = math.inf,
):
    """Shift both states to the linear closest approach, at most ``max_offset_seconds`` away."""
    miss = threat[0] - agent[0]
    relative_velocity = threat[1] - agent[1]
    offset = -float(miss @ relative_velocity) / max(float(relative_velocity @ relative_velocity), 1e-12)
    offset = float(np.clip(offset, -max_offset_seconds, max_offset_seconds))
    return (agent[0] + agent[1] * offset, agent[1]), (threat[0] + threat[1] * offset, threat[1])


def load_encounters(
    config: SizingConfig, config_path: Path
) -> tuple[list[EncounterGeometry], dict[int, object], list, object, CalibrationConfig]:
    """Geometry of every reference conjunction, with the screened agent as the maneuvering satellite."""
    calibration_path = config.resolve(config_path, config.calibration_config)
    calibration = CalibrationConfig.load(calibration_path)
    catalog = load_catalog(calibration, calibration_path)
    objects = {item.norad_id: item for item in catalog.objects}
    manifest = load_reference_conjunctions(config.resolve(config_path, config.reference_conjunctions))
    agents = set(manifest.screened_agent_norad_ids)
    encounters = []
    for event_id, event in enumerate(manifest.conjunctions):
        agent_id, threat_id = (
            (event.first_norad_id, event.second_norad_id)
            if event.first_norad_id in agents
            else (event.second_norad_id, event.first_norad_id)
        )
        positions, velocities = SGP4Trajectory([objects[agent_id], objects[threat_id]], event.tca_epoch_utc).states(
            np.zeros(1)
        )
        agent, threat = _closest_approach((positions[0, 0], velocities[0, 0]), (positions[1, 0], velocities[1, 0]))
        encounters.append(encounter_from_states(event_id, agent_id, threat_id, agent, threat))
    return encounters, objects, list(manifest.conjunctions), catalog, calibration


def measure_detection_leads(
    config: SizingConfig,
    encounters: Sequence[EncounterGeometry],
    references: Sequence,
    catalog,
    calibration: CalibrationConfig,
    progress: Callable[[str], None] | None = None,
) -> list[DetectionLead]:
    """Replay each agent's k=1 view on the calibration decision grid and find its first sighting of the threat."""
    band = calibration.catalog
    start = catalog.latest_epoch_utc
    objects = sorted(catalog.objects, key=lambda item: item.norad_id)
    trajectory = SGP4Trajectory(objects, start)
    positions, _ = trajectory.states(np.zeros(1))
    altitudes = np.linalg.norm(positions[:, 0], axis=1) - wgs72.radiusearthkm * 1_000.0
    keep = (altitudes >= band.minimum_altitude_meters) & (altitudes <= band.maximum_altitude_meters)
    objects = [item for item, kept in zip(objects, keep) if kept]
    trajectory = SGP4Trajectory(objects, start)
    index = {item.norad_id: position for position, item in enumerate(objects)}
    radii = np.asarray([item.radius_meters for item in objects])
    safety = SafetyConfig(
        safe_separation_meters=calibration.safety.safe_separation_meters,
        screening_horizon_seconds=calibration.safety.screening_horizon_seconds,
    )
    interval, horizon = config.decision_interval_seconds, safety.screening_horizon_seconds

    windows: dict[int, range] = {}
    requests: dict[int, list[int]] = {}
    for encounter in encounters:
        tca = (references[encounter.event_id].tca_epoch_utc - start).total_seconds()
        first = math.ceil(max(0.0, tca - horizon) / interval)
        last = math.ceil(tca / interval) - 1
        windows[encounter.event_id] = range(first, last + 1)
        for decision in windows[encounter.event_id]:
            requests.setdefault(decision, []).append(encounter.event_id)

    detected: dict[int, int] = {}
    by_event = {encounter.event_id: encounter for encounter in encounters}
    for count, decision in enumerate(sorted(requests)):
        pending = [event for event in requests[decision] if event not in detected]
        if not pending:
            continue
        positions, velocities = trajectory.states(np.array([decision * interval]))
        state = CatalogState(positions[:, 0], velocities[:, 0], radii, np.zeros(radii.size, bool), np.zeros(radii.size))
        agents = np.array([index[by_event[event].maneuvering_norad_id] for event in pending])
        top = top_neighbors(state, agents, 1, safety)[:, 0]
        for event, neighbor in zip(pending, top):
            if neighbor == index[by_event[event].threat_norad_id]:
                detected[event] = decision
        if progress and count % 250 == 0:
            progress(f"detection replay: decision {count + 1}/{len(requests)}")

    leads = []
    for encounter in encounters:
        window = windows[encounter.event_id]
        decision = detected.get(encounter.event_id)
        if decision is None:
            leads.append(DetectionLead(encounter.event_id, None, 0))
            continue
        tca = (references[encounter.event_id].tca_epoch_utc - start).total_seconds()
        leads.append(DetectionLead(encounter.event_id, tca - decision * interval, window.stop - decision))
    return leads


def detected_requirements(
    config: SizingConfig, encounters: Sequence[EncounterGeometry], leads: Sequence[DetectionLead], burns_mode: str
) -> list[Requirement]:
    """Requirements when each agent burns from its own first detection: once, or at every remaining decision."""
    requirements = []
    for encounter, lead in zip(encounters, leads):
        if lead.lead_seconds is None:
            requirements.append(Requirement(encounter.event_id, 0.0, 0, math.inf, ManeuverAction.NO_OP))
            continue
        burns = lead.available_burns if burns_mode == "all" else 1
        requirements.append(
            requirement_for(encounter, lead.lead_seconds, burns, config.decision_interval_seconds, config.safe_separation_meters)
        )
    return requirements


def compute_requirements(config: SizingConfig, encounters: Sequence[EncounterGeometry]) -> list[Requirement]:
    """Requirements for one burn and for every available burn, at each lead time."""
    requirements = []
    for lead in config.lead_times_seconds:
        for burns in sorted({1, config.max_burns(lead)}):
            requirements += [
                requirement_for(encounter, lead, burns, config.decision_interval_seconds, config.safe_separation_meters)
                for encounter in encounters
            ]
    if config.selection_lead_seconds not in config.lead_times_seconds or config.selection_burns not in {
        1,
        config.max_burns(config.selection_lead_seconds),
    }:
        requirements += [
            requirement_for(
                encounter,
                config.selection_lead_seconds,
                config.selection_burns,
                config.decision_interval_seconds,
                config.safe_separation_meters,
            )
            for encounter in encounters
        ]
    return requirements


def _select(requirements: Sequence[Requirement], lead: float, burns: int) -> list[Requirement]:
    return [item for item in requirements if item.lead_seconds == lead and item.burns == burns]


def sizing_table(config: SizingConfig, requirements: Sequence[Requirement]) -> list[dict[str, object]]:
    rows = []
    for lead, burns in sorted({(item.lead_seconds, item.burns) for item in requirements}):
        subset = _select(requirements, lead, burns)
        for candidate in config.delta_v_candidates_mps:
            rows.append(
                {
                    "lead_seconds": lead,
                    "burns": burns,
                    "delta_v_per_burn_mps": candidate,
                    "resolved_fraction": resolved_fraction(subset, candidate),
                    "encounters": len(subset),
                }
            )
    return rows


def _epoch_dict(epoch: datetime) -> dict[str, float]:
    return {
        "year": epoch.year,
        "month": epoch.month,
        "day": epoch.day,
        "hour": epoch.hour,
        "minute": epoch.minute,
        "second": epoch.second + epoch.microsecond / 1e6,
    }


def _fly(
    agent_state: np.ndarray,
    threat_state: np.ndarray,
    start: datetime,
    lead_seconds: float,
    config: SizingConfig,
    burn: tuple[ManeuverAction, float, int] | None,
    mass_kg: float,
) -> tuple[float, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Two-body Orekit flight through closest approach; returns the miss and both states at it."""
    from orbitzoo.env import OrbitZoo

    body = {"dry_mass": mass_kg, "initial_fuel_mass": 0.25 * mass_kg, "isp": 300.0, "radius": 1.0, "forces": ["gravity_newton"]}
    env = OrbitZoo(
        dynamics_library="orekit",
        step_size=1.0,
        initial_epoch=_epoch_dict(start),
        spacecrafts=[
            {**body, "name": "agent", "initial_state": agent_state.tolist()},
            {**body, "name": "threat", "initial_state": threat_state.tolist()},
        ],
    )
    env.reset(0)
    agent, threat = env.dynamics.spacecrafts
    interval = config.decision_interval_seconds
    fine_start, end = lead_seconds - 60.0, lead_seconds + 60.0
    burns: dict[float, float] = {}
    if burn:
        action, delta_v, count = burn
        duration = agent.get_mass() * delta_v / config.spot_check_thrust_newtons
        burns = {index * interval: duration for index in range(count)}
    timeline = sorted(
        {*np.arange(0.0, fine_start, interval), *np.arange(fine_start, end + 0.5, 1.0), *burns,
         *(time + duration for time, duration in burns.items())}
    )
    best: tuple[float, tuple, tuple] = (math.inf, (), ())
    for begin, finish in zip(timeline, timeline[1:]):
        actions = durations = None
        if begin in burns:
            actions = {"agent": list(config.spot_check_thrust_newtons * np.asarray(burn[0].rsw_unit_vector))}
            durations = {"agent": finish - begin}
        env.step(step_size=finish - begin, actions=actions, maneuver_durations=durations)
        if finish >= fine_start:
            shifted_agent, shifted_threat = _closest_approach(
                (agent.position.copy(), agent.velocity.copy()),
                (threat.position.copy(), threat.velocity.copy()),
                max_offset_seconds=0.5,
            )
            miss = float(np.linalg.norm(shifted_threat[0] - shifted_agent[0]))
            if miss < best[0]:
                best = (miss, shifted_agent, shifted_threat)
    return best


def spot_check(
    config: SizingConfig,
    encounters: Sequence[EncounterGeometry],
    references: Sequence,
    objects: dict[int, object],
    requirements: Sequence[Requirement],
) -> list[SpotCheck]:
    """Fly a sample of requirements in Orekit and check each ends at the safe separation."""
    by_event = {item.event_id: item for item in _select(requirements, config.selection_lead_seconds, config.selection_burns)}
    candidates = [encounter for encounter in encounters if math.isfinite(by_event[encounter.event_id].delta_v_mps)]
    stride = max(len(candidates) // max(config.spot_check_events, 1), 1)
    mass = min(config.satellite_masses_kg)
    checks = []
    for encounter in candidates[::stride][: config.spot_check_events]:
        reference = references[encounter.event_id]
        lead = config.selection_lead_seconds
        start = reference.tca_epoch_utc - timedelta(seconds=lead)
        positions, velocities = SGP4Trajectory(
            [objects[encounter.maneuvering_norad_id], objects[encounter.threat_norad_id]], start
        ).states(np.zeros(1))
        agent_state = np.concatenate((positions[0, 0], velocities[0, 0]))
        threat_state = np.concatenate((positions[1, 0], velocities[1, 0]))
        coasting_miss, agent_at_tca, threat_at_tca = _fly(agent_state, threat_state, start, lead, config, None, mass)
        agent_at_tca, threat_at_tca = _closest_approach(agent_at_tca, threat_at_tca)
        geometry = encounter_from_states(encounter.event_id, 0, 0, agent_at_tca, threat_at_tca)
        requirement = requirement_for(
            geometry, lead, config.selection_burns, config.decision_interval_seconds, config.safe_separation_meters
        )
        if not math.isfinite(requirement.delta_v_mps) or requirement.delta_v_mps == 0.0:
            continue
        maneuvered_miss, _, _ = _fly(
            agent_state, threat_state, start, lead, config,
            (requirement.action, requirement.delta_v_mps, config.selection_burns), mass,
        )
        checks.append(
            SpotCheck(
                event_id=encounter.event_id,
                lead_seconds=lead,
                burns=config.selection_burns,
                action=requirement.action.display_name,
                delta_v_mps=requirement.delta_v_mps,
                coasting_miss_m=coasting_miss,
                maneuvered_miss_m=maneuvered_miss,
                passed=abs(maneuvered_miss - config.safe_separation_meters)
                <= config.spot_check_tolerance_fraction * config.safe_separation_meters,
            )
        )
    return checks


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_sizing(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> SizingResult:
    """Compute requirements, the sizing table, the selection, thrust needs, and the Orekit spot-check."""
    config_path = Path(config_path)
    config = SizingConfig.load(config_path)
    encounters, objects, references, catalog, calibration = load_encounters(config, config_path)
    if progress:
        progress(f"loaded {len(encounters)} reference conjunctions")
    requirements = compute_requirements(config, encounters)
    selection = _select(requirements, config.selection_lead_seconds, config.selection_burns)
    selected = smallest_passing_delta_v(selection, config.delta_v_candidates_mps, config.target_fraction)
    leads = measure_detection_leads(config, encounters, references, catalog, calibration, progress)
    detected_all = detected_requirements(config, encounters, leads, "all")
    detected_single = detected_requirements(config, encounters, leads, "single")
    detected = smallest_passing_delta_v(detected_all, config.delta_v_candidates_mps, config.target_fraction)
    checks = spot_check(config, encounters, references, objects, requirements) if config.spot_check_events else []
    if progress:
        progress(f"spot-checked {len(checks)} encounters in Orekit")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "sizing_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    table = sizing_table(config, requirements)
    for label, subset in (("all", detected_all), ("single", detected_single)):
        table += [
            {
                "lead_seconds": "detected",
                "burns": label,
                "delta_v_per_burn_mps": candidate,
                "resolved_fraction": resolved_fraction(subset, candidate),
                "encounters": len(subset),
            }
            for candidate in config.delta_v_candidates_mps
        ]
    _write_csv(output_directory / "sizing_table.csv", table)
    _write_csv(
        output_directory / "detection_leads.csv",
        [
            {
                **asdict(lead),
                "required_delta_v_all_burns_mps": all_burns.delta_v_mps,
                "required_delta_v_single_burn_mps": single.delta_v_mps,
            }
            for lead, all_burns, single in zip(leads, detected_all, detected_single)
        ],
    )
    _write_csv(
        output_directory / "requirements.csv",
        [
            {
                "event_id": item.event_id,
                "maneuvering_norad_id": encounters[item.event_id].maneuvering_norad_id,
                "threat_norad_id": encounters[item.event_id].threat_norad_id,
                "miss_distance_m": encounters[item.event_id].miss_distance_m,
                "relative_speed_mps": float(np.linalg.norm(encounters[item.event_id].relative_velocity_mps)),
                "lead_seconds": item.lead_seconds,
                "burns": item.burns,
                "required_delta_v_per_burn_mps": item.delta_v_mps,
                "best_action": item.action.display_name,
            }
            for item in requirements
        ],
    )
    _write_csv(output_directory / "spot_checks.csv", [asdict(check) for check in checks])
    def thrust(delta_v: float | None) -> dict[str, float | None]:
        return {
            f"{mass:g}": (mass * delta_v / config.maximum_burn_duration_seconds if delta_v else None)
            for mass in config.satellite_masses_kg
        }

    detected_leads = sorted(lead.lead_seconds for lead in leads if lead.lead_seconds is not None)
    (output_directory / "recommendation.json").write_text(
        json.dumps(
            {
                "rule": (
                    f"smallest candidate resolving at least {config.target_fraction:.1%} of conjunctions with "
                    f"{config.selection_burns} burn(s) starting {config.selection_lead_seconds:g} s before closest approach"
                ),
                "selected_delta_v_per_burn_mps": selected,
                "resolved_fraction": resolved_fraction(selection, selected) if selected else None,
                "minimum_thrust_newtons_by_mass_kg": thrust(selected),
                "detected_warning": {
                    "rule": (
                        f"smallest candidate resolving at least {config.target_fraction:.1%} of conjunctions with a "
                        "burn at every decision from each agent's first k=1 detection until closest approach"
                    ),
                    "selected_delta_v_per_burn_mps": detected,
                    "resolved_fraction": resolved_fraction(detected_all, detected) if detected else None,
                    "minimum_thrust_newtons_by_mass_kg": thrust(detected),
                    "undetected_conjunctions": sum(lead.lead_seconds is None for lead in leads),
                    "median_lead_seconds": float(np.median(detected_leads)) if detected_leads else None,
                    "fifth_percentile_lead_seconds": float(np.percentile(detected_leads, 5)) if detected_leads else None,
                },
                "spot_checks_passed": sum(check.passed for check in checks),
                "spot_checks": len(checks),
            },
            indent=2,
        )
        + "\n"
    )
    return SizingResult(encounters, requirements, selected, detected, leads, checks)
