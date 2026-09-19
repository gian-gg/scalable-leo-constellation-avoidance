import copy
import math

import numpy as np
import pytest

from orbitzoo.thesis.environments.collision_avoidance import (
    CollisionAvoidanceEnv,
    development_environment_kwargs,
)
from orbitzoo.thesis.environments.observations import GLOBAL_BODY_FEATURE_DIM, NEIGHBOR_FEATURE_DIM, OWN_FEATURE_DIM
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig, actual_delta_v_mps

MANEUVER = ManeuverConfig(1.0, 10.0, 300.0, 60.0)
GRAVITATIONAL_PARAMETER = 3.986004418e14
ORBIT_RADIUS = 6_378_136.3 + 500_000.0
MEAN_MOTION = math.sqrt(GRAVITATIONAL_PARAMETER / ORBIT_RADIUS**3)
AGENT = 1
NUM_AGENTS = 4


def make_env(episode_horizon: int = 20, **overrides) -> CollisionAvoidanceEnv:
    kwargs = copy.deepcopy(development_environment_kwargs())
    kwargs.update(overrides)
    env = CollisionAvoidanceEnv(maneuver_config=MANEUVER, episode_horizon=episode_horizon, **kwargs)
    env.reset(seed=0)
    return env


def actions(agent_action: int = 0) -> np.ndarray:
    ids = np.zeros(NUM_AGENTS, dtype=np.int64)
    ids[AGENT] = agent_action
    return ids


def rsw_basis(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    radial = position / np.linalg.norm(position)
    cross_track = np.cross(position, velocity)
    cross_track /= np.linalg.norm(cross_track)
    return np.vstack([radial, np.cross(cross_track, radial), cross_track])


def spacecraft(env: CollisionAvoidanceEnv, index: int = AGENT):
    return env.dynamics.spacecrafts[index]


def test_reset_and_step_shapes() -> None:
    env = make_env()
    local, global_state = env.reset(seed=0)

    assert local.shape == (NUM_AGENTS, OWN_FEATURE_DIM + NEIGHBOR_FEATURE_DIM)
    assert global_state.shape == (GLOBAL_BODY_FEATURE_DIM * (NUM_AGENTS + 1),)
    next_local, next_global, rewards, dones, _ = env.step(actions())
    assert next_local.shape == local.shape
    assert next_global.shape == global_state.shape
    assert rewards.shape == dones.shape == (NUM_AGENTS,)


def test_same_seed_and_actions_are_bit_identical() -> None:
    first, second = make_env(), make_env()
    sequence = [actions(ManeuverAction.PROGRADE), actions(), actions(ManeuverAction.RADIAL_IN)]

    for ids in sequence:
        first_outputs = first.step(ids)
        second_outputs = second.step(ids)
        for first_array, second_array in zip(first_outputs[:4], second_outputs[:4]):
            np.testing.assert_array_equal(first_array, second_array)
    np.testing.assert_array_equal(spacecraft(first).position, spacecraft(second).position)


def test_step_advances_one_decision_interval_on_the_analytic_orbit() -> None:
    env = make_env()
    start_epoch = env.dynamics.current_epoch

    env.step(actions())

    assert env.dynamics.current_epoch.durationFrom(start_epoch) == env.decision_interval_seconds
    angle = math.pi / 2 + MEAN_MOTION * env.decision_interval_seconds
    expected = np.array([ORBIT_RADIUS * math.cos(angle), ORBIT_RADIUS * math.sin(angle), 0.0])
    assert np.linalg.norm(spacecraft(env).position - expected) < 1.0


@pytest.mark.parametrize("action", [action for action in ManeuverAction if action is not ManeuverAction.NO_OP])
def test_each_action_changes_velocity_along_its_rsw_direction(action: ManeuverAction) -> None:
    burned, coasted = make_env(), make_env()
    basis = rsw_basis(spacecraft(burned).position, spacecraft(burned).velocity)

    burned.step(actions(action))
    coasted.step(actions())

    delta_v = basis @ (spacecraft(burned).velocity - spacecraft(coasted).velocity)
    expected_direction = np.asarray(action.rsw_unit_vector)
    assert np.dot(delta_v, expected_direction) / np.linalg.norm(delta_v) > 0.99
    assert np.linalg.norm(delta_v) == pytest.approx(MANEUVER.commanded_delta_v_mps, rel=0.02)


def test_prograde_displacement_matches_clohessy_wiltshire() -> None:
    burned, coasted = make_env(), make_env()
    _, _, _, _, info = burned.step(actions(ManeuverAction.PROGRADE))
    coasted.step(actions())
    for _ in range(4):
        burned.step(actions())
        coasted.step(actions())

    reference = spacecraft(coasted)
    offset = rsw_basis(reference.position, reference.velocity) @ (spacecraft(burned).position - reference.position)
    burn_midpoint = 0.5 * info["maneuvers"][reference.name]["burn_duration_seconds"]
    elapsed = 5 * burned.decision_interval_seconds - burn_midpoint
    delta_v = MANEUVER.commanded_delta_v_mps
    expected_radial = 2 * delta_v / MEAN_MOTION * (1 - math.cos(MEAN_MOTION * elapsed))
    expected_along_track = delta_v / MEAN_MOTION * (4 * math.sin(MEAN_MOTION * elapsed) - 3 * MEAN_MOTION * elapsed)
    assert offset[0] == pytest.approx(expected_radial, rel=0.1)
    assert offset[1] == pytest.approx(expected_along_track, rel=0.1)


def test_reported_delta_v_and_fuel_match_orekit_mass_change() -> None:
    env = make_env()
    body = spacecraft(env)
    total_delta_v = total_fuel = 0.0

    for ids in (actions(ManeuverAction.PROGRADE), actions(), actions(ManeuverAction.CROSS_TRACK_NEGATIVE)):
        mass_before = body.get_mass()
        _, _, _, _, info = env.step(ids)
        maneuver = info["maneuvers"][body.name]
        assert maneuver["fuel_consumed_kg"] == pytest.approx(mass_before - body.get_mass())
        assert maneuver["actual_delta_v_mps"] == pytest.approx(
            actual_delta_v_mps(mass_before, body.get_mass(), body.isp)
        )
        expected_delta_v = 0.0 if ids[AGENT] == ManeuverAction.NO_OP else MANEUVER.commanded_delta_v_mps
        assert maneuver["actual_delta_v_mps"] == pytest.approx(expected_delta_v, abs=1e-6)
        total_delta_v += maneuver["actual_delta_v_mps"]
        total_fuel += maneuver["fuel_consumed_kg"]

    assert env.diagnostics.cumulative_delta_v_mps[body.name] == pytest.approx(total_delta_v)
    assert env.diagnostics.cumulative_fuel_kg[body.name] == pytest.approx(total_fuel)
    assert env.diagnostics.cumulative_delta_v_mps["satellite_3"] == 0.0


def test_action_without_enough_fuel_is_rejected_and_coasts() -> None:
    kwargs = development_environment_kwargs()
    kwargs["spacecrafts"][AGENT]["initial_fuel_mass"] = 0.01
    env = make_env(spacecrafts=kwargs["spacecrafts"])
    mass_before = spacecraft(env).get_mass()

    _, _, rewards, _, info = env.step(actions(ManeuverAction.PROGRADE))

    assert info["rejected_agents"] == [spacecraft(env).name]
    assert info["maneuvers"][spacecraft(env).name]["action"] == ManeuverAction.NO_OP
    assert spacecraft(env).get_mass() == mass_before
    assert rewards[AGENT] == pytest.approx(env.reward_config.infeasible_maneuver_penalty)


def test_horizon_terminates_every_agent_and_blocks_further_steps() -> None:
    env = make_env(episode_horizon=2)

    _, _, _, first_dones, first_info = env.step(actions())
    _, _, _, dones, info = env.step(actions())

    assert not first_dones.any() and first_info["termination_reason"] is None
    assert dones.all() and info["termination_reason"] == "horizon"
    with pytest.raises(RuntimeError):
        env.step(actions())


def test_collision_terminates_the_episode_and_records_the_pair() -> None:
    kwargs = development_environment_kwargs()
    shadow = dict(kwargs["drifters"][0], name="shadow", initial_state=list(kwargs["spacecrafts"][AGENT]["initial_state"]))
    env = make_env(drifters=[*kwargs["drifters"], shadow])

    _, _, rewards, dones, info = env.step(actions())

    assert dones.all() and info["termination_reason"] == "collision"
    assert {spacecraft(env).name, "shadow"} in [set(pair) for pair in info["collision_pairs"]]
    assert rewards[AGENT] == pytest.approx(env.reward_config.collision_penalty)


@pytest.mark.parametrize(
    "bad_actions",
    [np.zeros(NUM_AGENTS - 1, dtype=np.int64), np.full(NUM_AGENTS, 7), np.full(NUM_AGENTS, -1)],
)
def test_invalid_actions_are_rejected(bad_actions: np.ndarray) -> None:
    env = make_env()

    with pytest.raises(ValueError):
        env.step(bad_actions)


def test_passing_conjunction_is_reported_with_its_realized_miss() -> None:
    env = make_env()

    approaches = [info["close_approaches"] for *_, info in (env.step(actions()) for _ in range(3))]

    [passed] = [item for step in approaches for item in step]
    assert set(passed["pair"]) == {"satellite_1", "development_debris"}
    assert 0.0 <= passed["miss_distance_meters"] < env.safety_config.safe_separation_meters
    assert approaches[0] == []


def test_environment_observations_match_the_reference_encoder() -> None:
    env = make_env()
    env.step(actions(ManeuverAction.PROGRADE))

    local, global_state = env._state()
    reference = env.observation_encoder.encode(env._moving_bodies(), env.agent_names)

    np.testing.assert_array_equal(local, reference.local_observations)
    np.testing.assert_array_equal(global_state, reference.global_state)
