import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from orbitzoo.thesis.config import ExperimentConfig
from orbitzoo.thesis.environments.scenarios import GeneratedEpisodeSource, build_episode_source
from orbitzoo.thesis.evaluation.evaluator import held_out
from orbitzoo.thesis.scenarios.config import ScenarioGeneratorConfig
from orbitzoo.thesis.scenarios.generator import SITUATION_SLOTS, perigee_altitude, plan_situations
from orbitzoo.thesis.scenarios.pools import is_held_out
from orbitzoo.thesis.scenarios.propagation import propagate, teme_to_inertial

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "mappo_generated_smoke.json"
EPOCH = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
RADIUS = 6_878_136.3
SPEED = (3.986004418e14 / RADIUS) ** 0.5
REAL_DATA = pytest.mark.skipif(
    not (ROOT / "data" / "full" / "catalog.tle").exists()
    or not (ROOT / "runs" / "k_dt_calibration_20k_72h" / "reference_conjunctions.json").exists(),
    reason="needs the local TLE catalog and the calibration reference conjunctions",
)


@pytest.mark.parametrize("agents, extras", [(16, 16), (64, 64), (150, 150), (9, 0), (5, 3)])
def test_plans_fill_every_agent_slot_without_exceeding_extras(agents: int, extras: int) -> None:
    weights = ScenarioGeneratorConfig().situation_weights

    plan = plan_situations(np.random.default_rng(agents), agents, extras, weights)

    assert sum(SITUATION_SLOTS[name][0] for name in plan) == agents
    assert sum(SITUATION_SLOTS[name][1] for name in plan) <= extras


def test_plans_fall_back_to_quiet_when_nothing_else_fits() -> None:
    assert plan_situations(np.random.default_rng(0), 3, 0, {"debris": 1.0}) == ["quiet"] * 3


def test_held_out_split_is_deterministic_and_near_its_fraction() -> None:
    held = [is_held_out(identifier, 0.2) for identifier in range(1, 20_001)]

    assert held == [is_held_out(identifier, 0.2) for identifier in range(1, 20_001)]
    assert 0.18 < sum(held) / len(held) < 0.22


def test_perigee_of_a_circular_orbit_is_its_altitude() -> None:
    altitude = perigee_altitude(np.array([RADIUS, 0.0, 0.0]), np.array([0.0, SPEED, 0.0]))

    assert altitude == pytest.approx(RADIUS - 6_378_137.0, abs=1.0)
    assert perigee_altitude(np.array([RADIUS, 0.0, 0.0]), np.array([0.0, 2 * SPEED, 0.0])) == -np.inf


@pytest.mark.parametrize(
    "overrides",
    [
        {"split": "validation"},
        {"held_out_fraction": 1.0},
        {"situation_weights": {"unknown": 1.0}},
        {"situation_weights": {"debris": 0.0}},
        {"meeting_time_seconds": (900.0, 300.0)},
        {"extra_objects": -1},
    ],
)
def test_invalid_generator_config_is_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(ScenarioGeneratorConfig(), **overrides).validate()


def test_backward_then_forward_propagation_returns_to_the_start() -> None:
    position, velocity = np.array([RADIUS, 1_000.0, 5_000.0]), np.array([10.0, 7_000.0, 2_500.0])

    earlier = propagate(position, velocity, EPOCH, -900.0)
    again = propagate(*earlier, EPOCH - timedelta(seconds=900), 900.0)

    assert np.linalg.norm(again[0] - position) < 1.0


def test_teme_conversion_is_a_rotation() -> None:
    position, velocity = np.array([[RADIUS, 0.0, 0.0]]), np.array([[0.0, SPEED, 0.0]])

    converted = teme_to_inertial(position, velocity, EPOCH)

    assert np.linalg.norm(converted[0]) == pytest.approx(RADIUS, rel=1e-9)
    assert np.linalg.norm(converted[1]) == pytest.approx(SPEED, rel=1e-6)
    assert np.linalg.norm(converted[0] - position) > 1_000.0


def test_evaluation_draws_from_the_held_out_split() -> None:
    assert held_out(ExperimentConfig.load(SMOKE)).scenario_generator.split == "test"


@pytest.fixture(scope="module")
def source() -> GeneratedEpisodeSource:
    config = ExperimentConfig.load(SMOKE)
    config = dataclasses.replace(
        config, environment=dataclasses.replace(config.environment, num_agents=16, episode_horizon=12)
    )
    return build_episode_source(config)


@REAL_DATA
def test_same_seed_builds_the_same_scenario(source: GeneratedEpisodeSource) -> None:
    first, second = source.generator.generate(1_207), source.generator.generate(1_207)

    assert first.manifest() == second.manifest()
    assert first.orbitzoo_kwargs == second.orbitzoo_kwargs
    assert source.generator.generate(1_208).manifest() != first.manifest()


@REAL_DATA
def test_every_episode_has_the_same_size(source: GeneratedEpisodeSource) -> None:
    for seed in range(5):
        env = source.environment(seed)
        assert len(env.dynamics.spacecraft_names) == 16
        assert len(env._moving_bodies()) == 32


@REAL_DATA
def test_planned_close_calls_happen_as_planned_when_everyone_coasts(source: GeneratedEpisodeSource) -> None:
    env = source.environment(1_207)
    env.reset(1_207)
    realized = {}
    for _ in range(12):
        *_, info = env.step(np.zeros(len(env.dynamics.spacecraft_names), dtype=np.int64))
        realized.update({frozenset(item["pair"]): item["miss_distance_meters"] for item in info["close_approaches"]})

    planned = [item for item in env.scenario.encounters if item.planned_miss_meters < 990.0]
    assert planned
    for encounter in planned:
        assert realized[frozenset((encounter.agent, encounter.other))] == pytest.approx(encounter.planned_miss_meters, abs=10.0)
    assert set(realized) <= {frozenset((item.agent, item.other)) for item in env.scenario.encounters}


@REAL_DATA
def test_train_and_test_splits_share_no_satellites_or_shapes(source: GeneratedEpisodeSource) -> None:
    config = ExperimentConfig.load(SMOKE)
    test = GeneratedEpisodeSource(held_out(config)).generator.pools
    train = source.generator.pools

    assert not {item.norad_id for item in train.agents} & {item.norad_id for item in test.agents}
    assert not {shape.event_id for shape in train.shapes} & {shape.event_id for shape in test.shapes}
