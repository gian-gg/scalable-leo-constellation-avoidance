"""Build one seeded episode: real agent orbits plus threats placed backwards from real close-call shapes.

See docs/TRAINING_SCENARIOS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from orbitzoo.thesis.environments.safety import SafetyConfig, safety_snapshot
from orbitzoo.thesis.environments.vectorized_observations import rsw_bases
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig
from orbitzoo.thesis.scalability.simulator import SGP4Trajectory
from orbitzoo.thesis.scenarios.config import SITUATIONS, ScenarioGeneratorConfig
from orbitzoo.thesis.scenarios.pools import CloseCallShape, ScenarioPools
from orbitzoo.thesis.scenarios.propagation import propagate, teme_to_inertial

EARTH_RADIUS_M = 6_378_137.0
GRAVITATIONAL_PARAMETER = 3.986004418e14
FORCES = ["gravity_hf"]
# (agents, extra objects) each situation occupies.
SITUATION_SLOTS = {
    "debris": (1, 1),
    "satellite_pair": (2, 0),
    "double_threat": (1, 2),
    "harmless": (1, 1),
    "quiet": (1, 0),
}


class ScenarioGenerationError(RuntimeError):
    """Raised when no valid scenario is found within the configured attempts."""


@dataclass(frozen=True)
class PlannedEncounter:
    """One intended close call: who meets whom, when, and at what miss distance."""

    agent: str
    other: str
    meeting_time_seconds: float
    planned_miss_meters: float
    shape_event_id: int


@dataclass
class Scenario:
    """OrbitZoo keyword arguments for one episode, plus a readable manifest."""

    seed: int
    epoch_utc: datetime
    orbitzoo_kwargs: dict[str, Any]
    situations: list[dict[str, Any]] = field(default_factory=list)
    encounters: list[PlannedEncounter] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "epoch_utc": self.epoch_utc.isoformat(),
            "agents": [body["name"] for body in self.orbitzoo_kwargs["spacecrafts"]],
            "other_objects": [body["name"] for body in self.orbitzoo_kwargs["drifters"]],
            "situations": self.situations,
            "encounters": [encounter.__dict__ for encounter in self.encounters],
        }


def _epoch_dict(epoch: datetime) -> dict[str, float]:
    return {
        "year": epoch.year,
        "month": epoch.month,
        "day": epoch.day,
        "hour": epoch.hour,
        "minute": epoch.minute,
        "second": epoch.second + epoch.microsecond / 1e6,
    }


def perigee_altitude(position: np.ndarray, velocity: np.ndarray) -> float:
    """Perigee altitude of the osculating two-body orbit, or -inf if it is not bound."""
    radius, speed = np.linalg.norm(position), np.linalg.norm(velocity)
    energy = speed**2 / 2 - GRAVITATIONAL_PARAMETER / radius
    if energy >= 0:
        return -math.inf
    semi_major_axis = -GRAVITATIONAL_PARAMETER / (2 * energy)
    angular_momentum = np.linalg.norm(np.cross(position, velocity))
    eccentricity = math.sqrt(max(0.0, 1 - angular_momentum**2 / (GRAVITATIONAL_PARAMETER * semi_major_axis)))
    return semi_major_axis * (1 - eccentricity) - EARTH_RADIUS_M


def plan_situations(rng: np.random.Generator, agents: int, extras: int, weights: dict[str, float]) -> list[str]:
    """Draw situations until every agent slot is used, never exceeding the extra-object slots."""
    names = [name for name in SITUATIONS if weights.get(name, 0) > 0]
    plan: list[str] = []
    while agents > 0:
        fitting = [name for name in names if SITUATION_SLOTS[name][0] <= agents and SITUATION_SLOTS[name][1] <= extras]
        if fitting:
            probabilities = np.array([weights[name] for name in fitting], dtype=float)
            choice = fitting[int(rng.choice(len(fitting), p=probabilities / probabilities.sum()))]
        else:
            choice = "quiet"
        plan.append(choice)
        agents -= SITUATION_SLOTS[choice][0]
        extras -= SITUATION_SLOTS[choice][1]
    return plan


class ScenarioGenerator:
    """Deterministically turns a seed into one collision-avoidance episode."""

    def __init__(
        self,
        config: ScenarioGeneratorConfig,
        pools: ScenarioPools,
        num_agents: int,
        maneuver: ManeuverConfig,
        safety: SafetyConfig,
        decision_interval_seconds: float,
    ) -> None:
        config.validate()
        self.config = config
        self.pools = pools
        self.num_agents = num_agents
        self.extra_objects = num_agents if config.extra_objects is None else config.extra_objects
        self.maneuver = maneuver
        self.safety = safety
        self.decision_interval_seconds = decision_interval_seconds

    def generate(self, seed: int) -> Scenario:
        rng = np.random.default_rng(seed)
        for _ in range(self.config.maximum_attempts):
            scenario = self._attempt(seed, rng)
            if scenario is not None:
                return scenario
        raise ScenarioGenerationError(f"no valid scenario for seed {seed} in {self.config.maximum_attempts} attempts")

    def _spacecraft(self, name: str, position: np.ndarray, velocity: np.ndarray, radius: float) -> dict[str, Any]:
        return {
            "name": name,
            "initial_state": [*map(float, position), *map(float, velocity)],
            "dry_mass": self.config.dry_mass_kg,
            "initial_fuel_mass": self.config.initial_fuel_mass_kg,
            "isp": self.maneuver.specific_impulse_seconds,
            "radius": radius,
            "forces": FORCES,
        }

    @staticmethod
    def _drifter(name: str, position: np.ndarray, velocity: np.ndarray, radius: float) -> dict[str, Any]:
        return {
            "name": name,
            "initial_state": [*map(float, position), *map(float, velocity)],
            "radius": radius,
            "forces": FORCES,
        }

    def _threat_start(
        self,
        rng: np.random.Generator,
        agent_state: tuple[np.ndarray, np.ndarray],
        epoch: datetime,
        meeting_time: float,
        miss_range: tuple[float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, CloseCallShape, float] | None:
        """Place a threat on a real shape at the meeting point and fly it back to the episode start."""
        agent_at_meeting = propagate(*agent_state, epoch, meeting_time)
        basis = rsw_bases(agent_at_meeting[0][np.newaxis], agent_at_meeting[1][np.newaxis])[0]
        for _ in range(10):
            shape = self.pools.shapes[int(rng.integers(len(self.pools.shapes)))]
            miss_rsw = shape.miss_rsw_m
            if miss_range is not None:
                miss_rsw = miss_rsw / max(np.linalg.norm(miss_rsw), 1e-9) * rng.uniform(*miss_range)
            position = agent_at_meeting[0] + basis.T @ miss_rsw
            velocity = agent_at_meeting[1] + basis.T @ shape.relative_velocity_rsw_mps
            if perigee_altitude(position, velocity) < self.config.minimum_perigee_altitude_meters:
                continue
            start = propagate(position, velocity, epoch + timedelta(seconds=meeting_time), -meeting_time)
            return start[0], start[1], shape, float(np.linalg.norm(miss_rsw))
        return None

    def _attempt(self, seed: int, rng: np.random.Generator) -> Scenario | None:
        config = self.config
        epoch = self.pools.base_epoch_utc + timedelta(seconds=float(rng.uniform(0, config.epoch_window_seconds)))
        plan = plan_situations(rng, self.num_agents, self.extra_objects, config.situation_weights)
        chosen = [self.pools.agents[index] for index in rng.choice(len(self.pools.agents), len(plan), replace=False)]
        teme = SGP4Trajectory(chosen, epoch).states(np.zeros(1))
        positions, velocities = teme_to_inertial(teme[0][:, 0], teme[1][:, 0], epoch)

        spacecrafts: list[dict[str, Any]] = []
        drifters: list[dict[str, Any]] = []
        situations: list[dict[str, Any]] = []
        encounters: list[PlannedEncounter] = []
        low, high = config.meeting_time_seconds
        for index, (situation, satellite) in enumerate(zip(plan, chosen)):
            agent_name = f"{satellite.norad_id}"
            agent_state = (positions[index], velocities[index])
            spacecrafts.append(self._spacecraft(agent_name, *agent_state, satellite.radius_meters))
            meeting_times = [float(rng.uniform(low, high))]
            if situation == "double_threat":
                meeting_times.append(meeting_times[0] + float(rng.uniform(*config.second_threat_delay_seconds)))
            others = []
            if situation != "quiet":
                for order, meeting_time in enumerate(meeting_times):
                    miss_range = config.harmless_miss_meters if situation == "harmless" else None
                    placed = self._threat_start(rng, agent_state, epoch, meeting_time, miss_range)
                    if placed is None:
                        return None
                    position, velocity, shape, miss = placed
                    if situation == "satellite_pair":
                        other = f"pair-{index}"
                        spacecrafts.append(self._spacecraft(other, position, velocity, satellite.radius_meters))
                    else:
                        other = f"{situation}-{index}-{order}"
                        drifters.append(self._drifter(other, position, velocity, shape.threat_radius_m))
                    others.append(other)
                    encounters.append(PlannedEncounter(agent_name, other, meeting_time, miss, shape.event_id))
            situations.append({"situation": situation, "agent": agent_name, "others": others})

        filler_count = self.extra_objects - len(drifters)
        if filler_count:
            filler = [self.pools.background[i] for i in rng.choice(len(self.pools.background), filler_count, replace=False)]
            filler_teme = SGP4Trajectory(filler, epoch).states(np.zeros(1))
            filler_positions, filler_velocities = teme_to_inertial(filler_teme[0][:, 0], filler_teme[1][:, 0], epoch)
            drifters += [
                self._drifter(f"background-{item.norad_id}", filler_positions[i], filler_velocities[i], item.radius_meters)
                for i, item in enumerate(filler)
            ]

        if not self._starts_clear(spacecrafts + drifters, encounters):
            return None
        return Scenario(
            seed=seed,
            epoch_utc=epoch,
            orbitzoo_kwargs={
                "dynamics_library": "orekit",
                "step_size": float(self.decision_interval_seconds),
                "initial_epoch": _epoch_dict(epoch),
                "spacecrafts": spacecrafts,
                "drifters": drifters,
            },
            situations=situations,
            encounters=encounters,
        )

    def _starts_clear(self, bodies: list[dict[str, Any]], encounters: list[PlannedEncounter]) -> bool:
        """True if no unplanned pair starts closer than the configured separation."""

        points = [
            SimpleNamespace(
                name=body["name"],
                position=np.array(body["initial_state"][:3]),
                velocity=np.array(body["initial_state"][3:]),
                radius=body["radius"],
            )
            for body in bodies
        ]
        snapshot = safety_snapshot(points, self.safety)
        planned = {frozenset((encounter.agent, encounter.other)) for encounter in encounters}
        first, second = np.triu_indices(len(points), k=1)
        close = snapshot.separations[first, second] < self.config.minimum_unplanned_separation_meters
        return all(
            frozenset((points[i].name, points[j].name)) in planned for i, j in zip(first[close], second[close])
        )
