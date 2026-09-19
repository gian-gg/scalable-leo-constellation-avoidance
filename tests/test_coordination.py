import copy

import numpy as np

from orbitzoo.thesis.environments.collision_avoidance import (
    CollisionAvoidanceEnv,
    development_environment_kwargs,
)
from orbitzoo.thesis.environments.episodes import play_episode
from orbitzoo.thesis.evaluation.coordination import COORDINATION_COLUMNS, CoordinationCounts
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig
from orbitzoo.thesis.scalability.runner import pair_coordination
from orbitzoo.thesis.scalability.screening import ConjunctionEvent
from orbitzoo.thesis.scalability.simulator import SimulationResult

MANEUVER = ManeuverConfig(1.0, 10.0, 300.0, 60.0)


def test_counts_group_by_maneuvering_members_and_add_up() -> None:
    counts = CoordinationCounts().add(0, False).add(1, True).add(2, True).add(2, False)

    assert (counts.none_maneuvered, counts.one_maneuvered, counts.both_maneuvered) == (1, 1, 2)
    assert (counts.none_resolved, counts.one_resolved, counts.both_resolved) == (0, 1, 1)
    assert (counts + counts).total == 8
    assert tuple(counts.as_columns()) == COORDINATION_COLUMNS


def two_satellite_env() -> CollisionAvoidanceEnv:
    kwargs = copy.deepcopy(development_environment_kwargs())
    oncoming = dict(kwargs["spacecrafts"][0], name="oncoming", initial_state=kwargs["drifters"][0]["initial_state"])
    kwargs.update(spacecrafts=[kwargs["spacecrafts"][0], oncoming], drifters=[])
    return CollisionAvoidanceEnv(maneuver_config=MANEUVER, episode_horizon=4, **kwargs)


def scripted(first_step: list[int]):
    steps = iter([first_step])
    return lambda local, _: np.asarray(next(steps, [0, 0]))


def test_coasting_satellites_that_fly_past_are_not_resolved() -> None:
    summary = play_episode(two_satellite_env(), 0, scripted([0, 0]))

    assert summary.coordination == CoordinationCounts(none_maneuvered=1)


def test_maneuvers_during_the_conjunction_are_attributed_to_the_pair() -> None:
    one = play_episode(two_satellite_env(), 0, scripted([ManeuverAction.RADIAL_OUT, 0]))
    both = play_episode(two_satellite_env(), 0, scripted([ManeuverAction.RADIAL_OUT, ManeuverAction.RADIAL_IN]))

    assert one.coordination.one_maneuvered == 1
    assert both.coordination.both_maneuvered == 1


def result_with_burns(indices: list[int], times: list[float]) -> SimulationResult:
    return SimulationResult(
        events=[],
        delta_v_mps=np.zeros(3),
        maneuvers=len(indices),
        rejected_actions=0,
        decisions=15,
        wall_seconds=0.0,
        burn_indices=np.asarray(indices, dtype=np.intp),
        burn_times_seconds=np.asarray(times, dtype=float),
    )


def test_scale_classifies_agent_pair_conjunctions_by_burns_before_closest_approach() -> None:
    reference = result_with_burns([], [])
    reference.events = [
        ConjunctionEvent(0, 1, 900.0, 500.0, False),
        ConjunctionEvent(0, 2, 900.0, 500.0, False),
        ConjunctionEvent(1, 3, 3_000.0, 400.0, False),
    ]
    is_agent = np.array([True, True, False, True])
    policy = result_with_burns([0, 1, 1], [600.0, 720.0, 100.0])

    counts = pair_coordination(reference, policy, [False, False, True], is_agent, window_seconds=1_800.0)

    assert counts == CoordinationCounts(both_maneuvered=1, both_resolved=1, none_maneuvered=1)
