"""Delta-v needed to clear real reference conjunctions, by warning time.

See docs/MANEUVER_SIZING.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from orbitzoo.thesis.environments.vectorized_observations import rsw_bases
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.scalability.dynamics import clohessy_wiltshire_displacement, mean_motions

BURN_ACTIONS = tuple(action for action in ManeuverAction if action is not ManeuverAction.NO_OP)


@dataclass(frozen=True)
class EncounterGeometry:
    """One conjunction at closest approach, seen from the maneuvering satellite."""

    event_id: int
    maneuvering_norad_id: int
    threat_norad_id: int
    miss_vector_m: np.ndarray
    relative_velocity_mps: np.ndarray
    agent_position_m: np.ndarray
    agent_velocity_mps: np.ndarray

    @property
    def miss_distance_m(self) -> float:
        return float(np.linalg.norm(self.miss_vector_m))


@dataclass(frozen=True)
class Requirement:
    """Smallest per-burn delta-v that clears one encounter, and the direction that achieves it."""

    event_id: int
    lead_seconds: float
    burns: int
    delta_v_mps: float
    action: ManeuverAction


def encounter_from_states(
    event_id: int,
    maneuvering_norad_id: int,
    threat_norad_id: int,
    agent_state: tuple[np.ndarray, np.ndarray],
    threat_state: tuple[np.ndarray, np.ndarray],
) -> EncounterGeometry:
    """Build the geometry from both objects' states at closest approach."""
    agent_position, agent_velocity = (np.asarray(value, dtype=float) for value in agent_state)
    threat_position, threat_velocity = (np.asarray(value, dtype=float) for value in threat_state)
    return EncounterGeometry(
        event_id=event_id,
        maneuvering_norad_id=maneuvering_norad_id,
        threat_norad_id=threat_norad_id,
        miss_vector_m=threat_position - agent_position,
        relative_velocity_mps=threat_velocity - agent_velocity,
        agent_position_m=agent_position,
        agent_velocity_mps=agent_velocity,
    )


def miss_displacement_per_mps(
    encounter: EncounterGeometry, action: ManeuverAction, burn_leads_seconds: Sequence[float]
) -> np.ndarray:
    """Change in the miss vector per m/s of each burn, projected across the relative velocity."""
    basis = rsw_bases(encounter.agent_position_m[np.newaxis], encounter.agent_velocity_mps[np.newaxis])[0]
    motion = float(mean_motions(encounter.agent_position_m[np.newaxis])[0])
    direction = np.asarray(action.rsw_unit_vector, dtype=float)
    displacement = sum(clohessy_wiltshire_displacement(direction, motion, lead) for lead in burn_leads_seconds)
    inertial = basis.T @ displacement
    speed = np.linalg.norm(encounter.relative_velocity_mps)
    along_path = encounter.relative_velocity_mps / speed if speed > 1e-9 else np.zeros(3)
    across_path = inertial - along_path * float(inertial @ along_path)
    # The agent moving by d shifts the threat's relative miss vector by -d.
    return -across_path


def required_delta_v(miss_vector: np.ndarray, gain: np.ndarray, safe_separation_m: float) -> float:
    """Smallest delta-v with |miss + delta_v * gain| >= safe separation, or inf if unreachable."""
    miss_squared = float(miss_vector @ miss_vector)
    if miss_squared >= safe_separation_m**2:
        return 0.0
    gain_squared = float(gain @ gain)
    if gain_squared <= 1e-18:
        return math.inf
    alignment = float(miss_vector @ gain)
    discriminant = alignment**2 - gain_squared * (miss_squared - safe_separation_m**2)
    return (-alignment + math.sqrt(discriminant)) / gain_squared


def burn_leads(lead_seconds: float, decision_interval_seconds: float, burns: int) -> list[float]:
    """Times before closest approach of ``burns`` consecutive decisions starting at ``lead_seconds``."""
    return [lead_seconds - index * decision_interval_seconds for index in range(burns)]


def requirement_for(
    encounter: EncounterGeometry,
    lead_seconds: float,
    burns: int,
    decision_interval_seconds: float,
    safe_separation_m: float,
) -> Requirement:
    """Best direction and per-burn delta-v when ``burns`` equal burns start ``lead_seconds`` before closest approach."""
    leads = burn_leads(lead_seconds, decision_interval_seconds, burns)
    best_action, best_delta_v = ManeuverAction.NO_OP, math.inf
    for action in BURN_ACTIONS:
        gain = miss_displacement_per_mps(encounter, action, leads)
        delta_v = required_delta_v(encounter.miss_vector_m, gain, safe_separation_m)
        if delta_v < best_delta_v:
            best_action, best_delta_v = action, delta_v
    return Requirement(encounter.event_id, lead_seconds, burns, best_delta_v, best_action)


def resolved_fraction(requirements: Sequence[Requirement], delta_v_mps: float) -> float:
    """Fraction of encounters cleared by burns of at most ``delta_v_mps`` each."""
    if not requirements:
        return 0.0
    return sum(requirement.delta_v_mps <= delta_v_mps for requirement in requirements) / len(requirements)


def smallest_passing_delta_v(
    requirements: Sequence[Requirement], candidates: Sequence[float], target_fraction: float
) -> float | None:
    """Smallest candidate delta-v that clears at least ``target_fraction`` of encounters."""
    for candidate in sorted(candidates):
        if resolved_fraction(requirements, candidate) >= target_fraction:
            return candidate
    return None
