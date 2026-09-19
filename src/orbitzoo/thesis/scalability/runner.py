"""Catalog-size and agent-count sweeps for the frozen decentralized actor.

See docs/SCALABILITY.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import sys
from typing import Callable, Sequence

import numpy as np
from sgp4.earth_gravity import wgs72

from orbitzoo.thesis.calibration.catalog import load_catalog
from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import CatalogObject, ObjectType
from orbitzoo.thesis.config import ExperimentConfig
from orbitzoo.thesis.environments.observations import NEIGHBOR_FEATURE_DIM, OWN_FEATURE_DIM
from orbitzoo.thesis.evaluation.coordination import COORDINATION_COLUMNS, CoordinationCounts
from orbitzoo.thesis.evaluation.evaluator import build_policy
from orbitzoo.thesis.evaluation.policies import EvaluationPolicy, NoOpPolicy
from orbitzoo.thesis.runtime import environment_info
from orbitzoo.thesis.scalability.config import ScalabilityConfig
from orbitzoo.thesis.scalability.screening import match_events
from orbitzoo.thesis.scalability.simulator import (
    TIMING_STAGES,
    SGP4Trajectory,
    SimulationResult,
    SimulationSettings,
    simulate,
)

CATALOG_SWEEP = "catalog"
AGENT_SWEEP = "agents"
SWEEPS = (CATALOG_SWEEP, AGENT_SWEEP)
RESULT_FIELDS = (
    "sweep",
    "objects",
    "agents",
    "policy",
    "decisions",
    "conjunctions",
    "collisions",
    "resolved_reference_conjunctions",
    "secondary_conjunctions",
    "minimum_miss_distance_meters",
    "maneuvers",
    "rejected_actions",
    "total_delta_v_mps",
    "mean_delta_v_per_agent_mps",
    "wall_seconds",
    *(f"{stage}_seconds_per_decision" for stage in TIMING_STAGES),
    "observation_microseconds_per_agent_decision",
    "policy_microseconds_per_agent_decision",
    "process_peak_rss_mb",
    *COORDINATION_COLUMNS,
)
EVENT_FIELDS = (
    "sweep",
    "objects",
    "agents",
    "policy",
    "first_norad_id",
    "second_norad_id",
    "tca_seconds",
    "miss_distance_meters",
    "is_collision",
    "is_secondary",
)


@dataclass(frozen=True)
class Scenario:
    """One population: the objects simulated and which of them run the policy."""

    sweep: str
    objects: tuple[CatalogObject, ...]
    agent_indices: np.ndarray


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1_048_576 if sys.platform == "darwin" else peak / 1_024


def leo_objects(config: ScalabilityConfig, config_path: Path) -> tuple[tuple[CatalogObject, ...], object]:
    """Load the catalog and keep objects inside the altitude band at the latest TLE epoch."""
    catalog = load_catalog(CalibrationConfig(catalog=config.catalog), config_path)
    objects = tuple(sorted(catalog.objects, key=lambda item: item.norad_id))
    positions, _ = SGP4Trajectory(objects, catalog.latest_epoch_utc).states(np.zeros(1))
    altitudes = np.linalg.norm(positions[:, 0], axis=1) - wgs72.radiusearthkm * 1_000.0
    band = config.catalog
    retained = tuple(
        item
        for item, altitude in zip(objects, altitudes)
        if band.minimum_altitude_meters <= altitude <= band.maximum_altitude_meters
    )
    return retained, catalog.latest_epoch_utc


def build_scenarios(config: ScalabilityConfig, objects: Sequence[CatalogObject], sweeps: Sequence[str]) -> list[Scenario]:
    """Nested, seeded populations for each requested sweep."""
    scenarios: list[Scenario] = []
    if CATALOG_SWEEP in sweeps:
        candidates = [item for item in objects if item.is_agent_candidate]
        others = [item for item in objects if not item.is_agent_candidate]
        order = np.random.default_rng([config.seed, 0]).permutation(len(others))
        for size in config.catalog_sizes:
            if not len(candidates) <= size <= len(objects):
                raise ValueError(
                    f"catalog size {size} must lie between the {len(candidates)} agent candidates "
                    f"and the {len(objects)} retained objects"
                )
            chosen = sorted(candidates + [others[index] for index in order[: size - len(candidates)]], key=lambda item: item.norad_id)
            agents = np.flatnonzero([item.is_agent_candidate for item in chosen])
            scenarios.append(Scenario(CATALOG_SWEEP, tuple(chosen), agents))
    if AGENT_SWEEP in sweeps:
        payloads = np.flatnonzero([item.object_type is ObjectType.PAYLOAD for item in objects])
        order = np.random.default_rng([config.seed, 1]).permutation(payloads)
        for count in config.hypothetical_agent_counts:
            if count > payloads.size:
                raise ValueError(f"agent count {count} exceeds the {payloads.size} retained payloads")
            scenarios.append(Scenario(AGENT_SWEEP, tuple(objects), np.sort(order[:count])))
    return scenarios


def pair_coordination(
    reference: SimulationResult,
    result: SimulationResult,
    recurs: list[bool],
    is_agent: np.ndarray,
    window_seconds: float,
) -> CoordinationCounts:
    """Classify each reference agent-agent conjunction by how many members burned in the window before it."""
    counts = CoordinationCounts()
    for event, recurred in zip(reference.events, recurs):
        if not (is_agent[event.first_index] and is_agent[event.second_index]):
            continue
        in_window = (result.burn_times_seconds >= event.tca_seconds - window_seconds) & (
            result.burn_times_seconds <= event.tca_seconds
        )
        burned = set(result.burn_indices[in_window].tolist())
        counts = counts.add(len(burned & {event.first_index, event.second_index}), not recurred)
    return counts


def _result_row(
    scenario: Scenario,
    policy_name: str,
    result: SimulationResult,
    resolved: int | str,
    secondary: int | str,
    coordination: CoordinationCounts | None,
) -> dict[str, object]:
    agent_decisions = max(result.decisions * scenario.agent_indices.size, 1)
    misses = [event.miss_distance_meters for event in result.events]
    return {
        "sweep": scenario.sweep,
        "objects": len(scenario.objects),
        "agents": scenario.agent_indices.size,
        "policy": policy_name,
        "decisions": result.decisions,
        "conjunctions": len(result.events),
        "collisions": sum(event.is_collision for event in result.events),
        "resolved_reference_conjunctions": resolved,
        "secondary_conjunctions": secondary,
        "minimum_miss_distance_meters": min(misses) if misses else "",
        "maneuvers": result.maneuvers,
        "rejected_actions": result.rejected_actions,
        "total_delta_v_mps": float(result.delta_v_mps.sum()),
        "mean_delta_v_per_agent_mps": float(result.delta_v_mps.mean()) if result.delta_v_mps.size else 0.0,
        "wall_seconds": result.wall_seconds,
        **{
            f"{stage}_seconds_per_decision": result.stage_seconds[stage] / max(result.decisions, 1)
            for stage in TIMING_STAGES
        },
        "observation_microseconds_per_agent_decision": 1e6 * result.stage_seconds["observation"] / agent_decisions,
        "policy_microseconds_per_agent_decision": 1e6 * result.stage_seconds["policy"] / agent_decisions,
        "process_peak_rss_mb": _peak_rss_mb(),
        **(coordination.as_columns() if coordination else dict.fromkeys(COORDINATION_COLUMNS, "")),
    }


def _event_rows(scenario: Scenario, policy_name: str, result: SimulationResult, secondary: Sequence) -> list[dict]:
    secondary_ids = {id(event) for event in secondary}
    return [
        {
            "sweep": scenario.sweep,
            "objects": len(scenario.objects),
            "agents": scenario.agent_indices.size,
            "policy": policy_name,
            "first_norad_id": scenario.objects[event.first_index].norad_id,
            "second_norad_id": scenario.objects[event.second_index].norad_id,
            "tca_seconds": event.tca_seconds,
            "miss_distance_meters": event.miss_distance_meters,
            "is_collision": event.is_collision,
            "is_secondary": id(event) in secondary_ids,
        }
        for event in result.events
    ]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_scalability(
    config_path: str | Path,
    policy_specs: Sequence[str],
    output_directory: str | Path,
    *,
    sweeps: Sequence[str] = SWEEPS,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Run no-op plus each policy on every scenario and write results and events."""
    config_path = Path(config_path).resolve()
    config = ScalabilityConfig.load(config_path)
    experiment_path = config.resolve(config_path, config.experiment_config)
    experiment = ExperimentConfig.load(experiment_path)
    unknown = set(sweeps) - set(SWEEPS)
    if unknown:
        raise ValueError(f"unknown sweeps {sorted(unknown)}; choose from {SWEEPS}")
    width = OWN_FEATURE_DIM + experiment.environment.neighborhood_size * NEIGHBOR_FEATURE_DIM
    policies: list[EvaluationPolicy] = [NoOpPolicy()]
    policies += [build_policy(spec, experiment, width) for spec in policy_specs if spec != "noop"]
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"policy names must be unique: {names}")
    decision_interval = experiment.environment.decision_interval_seconds
    if not float(decision_interval).is_integer():
        raise ValueError("decision_interval_seconds must be a whole number of seconds")
    settings = SimulationSettings(
        neighborhood_size=experiment.environment.neighborhood_size,
        decision_interval_seconds=int(decision_interval),
        duration_seconds=config.duration_seconds,
        fine_step_seconds=config.fine_step_seconds,
        safety=experiment.safety,
        maneuver=experiment.maneuver,
        dry_mass_kg=config.dry_mass_kg,
        initial_fuel_mass_kg=config.initial_fuel_mass_kg,
        maximum_relative_speed_mps=config.maximum_relative_speed_mps,
    )
    settings.validate()

    objects, start_epoch = leo_objects(config, config_path)
    scenarios = build_scenarios(config, objects, sweeps)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    config.save(output_directory / "scalability_config.json")
    experiment.save(output_directory / "experiment_config.json")
    (output_directory / "scalability_info.json").write_text(
        json.dumps(
            {
                "policies": names,
                "sweeps": list(sweeps),
                "start_epoch_utc": start_epoch.isoformat(),
                "retained_objects": len(objects),
                "agent_candidates": sum(item.is_agent_candidate for item in objects),
                **environment_info(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    result_rows: list[dict[str, object]] = []
    event_rows: list[dict] = []
    for scenario in scenarios:
        trajectory = SGP4Trajectory(scenario.objects, start_epoch)
        radii = np.asarray([item.radius_meters for item in scenario.objects])
        is_agent = np.zeros(len(scenario.objects), dtype=bool)
        is_agent[scenario.agent_indices] = True
        reference: SimulationResult | None = None
        for policy in policies:
            result = simulate(trajectory, radii, scenario.agent_indices, policy, settings)
            if reference is None:
                reference, resolved, secondary, coordination = result, "", [], None
                secondary_count: int | str = ""
            else:
                recurs, secondary = match_events(reference.events, result.events, config.match_tolerance_seconds)
                resolved, secondary_count = recurs.count(False), len(secondary)
                coordination = pair_coordination(
                    reference, result, recurs, is_agent, experiment.safety.screening_horizon_seconds
                )
            result_rows.append(_result_row(scenario, policy.name, result, resolved, secondary_count, coordination))
            event_rows.extend(_event_rows(scenario, policy.name, result, secondary))
            _write_csv(output_directory / "results.csv", RESULT_FIELDS, result_rows)
            _write_csv(output_directory / "events.csv", EVENT_FIELDS, event_rows)
            if progress:
                progress(
                    f"{scenario.sweep}: {len(scenario.objects)} objects, {scenario.agent_indices.size} agents, "
                    f"{policy.name}: {len(result.events)} conjunctions, {result.wall_seconds:.1f} s"
                )
    return result_rows
