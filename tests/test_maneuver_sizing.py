import dataclasses
from datetime import datetime, timezone
import math

import numpy as np
import pytest

from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.sizing import (
    Requirement,
    burn_leads,
    encounter_from_states,
    miss_displacement_per_mps,
    required_delta_v,
    requirement_for,
    resolved_fraction,
    smallest_passing_delta_v,
)
from orbitzoo.thesis.maneuvers.sizing_study import DetectionLead, SizingConfig, _fly, detected_requirements

MU = 3.986004418e14
RADIUS = 6_878_136.3
SPEED = math.sqrt(MU / RADIUS)
AGENT = (np.array([RADIUS, 0.0, 0.0]), np.array([0.0, SPEED, 0.0]))


def crossing(miss: np.ndarray):
    """A threat crossing the agent's position on a polar orbit, offset by ``miss``."""
    threat = (AGENT[0] + miss, np.array([0.0, 0.0, SPEED]))
    return encounter_from_states(0, 1, 2, AGENT, threat)


def test_required_delta_v_reaches_exactly_the_safe_separation() -> None:
    miss, gain = np.array([300.0, 0.0, 0.0]), np.array([100.0, 50.0, 0.0])

    delta_v = required_delta_v(miss, gain, 1_000.0)

    assert np.linalg.norm(miss + delta_v * gain) == pytest.approx(1_000.0)
    assert required_delta_v(np.array([1_200.0, 0, 0]), gain, 1_000.0) == 0.0
    assert required_delta_v(miss, np.zeros(3), 1_000.0) == math.inf


def test_displacement_along_the_relative_velocity_does_not_change_the_miss() -> None:
    encounter = crossing(np.array([300.0, 0.0, 0.0]))

    gain = miss_displacement_per_mps(encounter, ManeuverAction.PROGRADE, [600.0])

    assert gain @ encounter.relative_velocity_mps == pytest.approx(0.0, abs=1e-6)


def test_requirement_picks_the_cheapest_direction() -> None:
    encounter = crossing(np.array([400.0, 0.0, 0.0]))

    requirement = requirement_for(encounter, 600.0, 1, 120.0, 1_000.0)

    for action in ManeuverAction:
        if action is ManeuverAction.NO_OP:
            continue
        gain = miss_displacement_per_mps(encounter, action, [600.0])
        assert required_delta_v(encounter.miss_vector_m, gain, 1_000.0) >= requirement.delta_v_mps


def test_more_warning_or_more_burns_needs_less_delta_v() -> None:
    encounter = crossing(np.array([400.0, 0.0, 0.0]))
    short = requirement_for(encounter, 360.0, 1, 120.0, 1_000.0).delta_v_mps

    assert requirement_for(encounter, 1_200.0, 1, 120.0, 1_000.0).delta_v_mps < short
    assert requirement_for(encounter, 360.0, 3, 120.0, 1_000.0).delta_v_mps < short
    assert burn_leads(360.0, 120.0, 3) == [360.0, 240.0, 120.0]


def test_selection_is_the_smallest_candidate_meeting_the_target() -> None:
    requirements = [Requirement(index, 360.0, 3, value, ManeuverAction.PROGRADE) for index, value in enumerate([0.1, 0.4, 0.9, 3.0])]

    assert resolved_fraction(requirements, 1.0) == 0.75
    assert smallest_passing_delta_v(requirements, [0.5, 1.0, 2.0, 5.0], 0.75) == 1.0
    assert smallest_passing_delta_v(requirements, [0.5, 1.0], 0.9) is None


@pytest.mark.parametrize(
    "overrides",
    [{"selection_burns": 4}, {"target_fraction": 0.0}, {"lead_times_seconds": (60.0,)}, {"delta_v_candidates_mps": (0.0,)}],
)
def test_invalid_sizing_config_is_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SizingConfig(), **overrides).validate()


def test_orekit_flight_confirms_a_computed_requirement() -> None:
    config = SizingConfig(decision_interval_seconds=120.0)
    lead = 360.0
    rate = SPEED / RADIUS
    # Both objects start ``lead`` seconds before crossing the x-axis, 400 m apart radially.
    agent_state = np.concatenate(
        (RADIUS * np.array([math.cos(-rate * lead), math.sin(-rate * lead), 0]), SPEED * np.array([-math.sin(-rate * lead), math.cos(-rate * lead), 0]))
    )
    threat_radius = RADIUS + 400.0
    threat_rate = math.sqrt(MU / threat_radius**3)
    threat_speed = threat_radius * threat_rate
    threat_state = np.concatenate(
        (
            threat_radius * np.array([math.cos(-threat_rate * lead), 0, math.sin(-threat_rate * lead)]),
            threat_speed * np.array([-math.sin(-threat_rate * lead), 0, math.cos(-threat_rate * lead)]),
        )
    )
    start = datetime(2026, 9, 15, tzinfo=timezone.utc)

    coasting_miss, agent_at_tca, threat_at_tca = _fly(agent_state, threat_state, start, lead, config, None, 300.0)
    requirement = requirement_for(encounter_from_states(0, 1, 2, agent_at_tca, threat_at_tca), lead, 3, 120.0, 1_000.0)
    maneuvered_miss, _, _ = _fly(
        agent_state, threat_state, start, lead, config, (requirement.action, requirement.delta_v_mps, 3), 300.0
    )

    assert coasting_miss == pytest.approx(400.0, abs=1.0)
    assert maneuvered_miss == pytest.approx(1_000.0, rel=0.05)


def test_detected_requirements_use_each_conjunction_warning() -> None:
    encounter = crossing(np.array([400.0, 0.0, 0.0]))
    config = SizingConfig()
    leads = [DetectionLead(0, 600.0, 5)]

    [every] = detected_requirements(config, [encounter], leads, "all")
    [single] = detected_requirements(config, [encounter], leads, "single")
    [missed] = detected_requirements(config, [encounter], [DetectionLead(0, None, 0)], "all")

    assert every.delta_v_mps == pytest.approx(requirement_for(encounter, 600.0, 5, 120.0, 1_000.0).delta_v_mps)
    assert single.delta_v_mps == pytest.approx(requirement_for(encounter, 600.0, 1, 120.0, 1_000.0).delta_v_mps)
    assert missed.delta_v_mps == math.inf
