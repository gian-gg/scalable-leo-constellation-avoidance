import csv
import dataclasses
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import pytest

from orbitzoo.cli import build_parser
from orbitzoo.rl_algorithms.mappo import MAPPO
from orbitzoo.thesis.config import ExperimentConfig
from orbitzoo.thesis.environments.observations import (
    NEIGHBOR_FEATURE_DIM,
    OWN_FEATURE_DIM,
    POSITION_SCALE_METERS,
    VELOCITY_SCALE_MPS,
)
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.environments.scenarios import FixedEpisodeSource, build_environment
from orbitzoo.thesis.evaluation.evaluator import (
    build_policy,
    evaluate,
    evaluate_policy,
    evaluation_seeds,
)
from orbitzoo.thesis.evaluation.policies import (
    ClohessyWiltshireAvoidancePolicy,
    NoOpPolicy,
)
from orbitzoo.thesis.environments.prediction import predict_closest_approach
from orbitzoo.thesis.environments.vectorized_observations import CatalogState, encode_local_observations, rsw_bases
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.sizing import encounter_from_states, miss_displacement_per_mps
from orbitzoo.thesis.scenarios.propagation import propagate
from orbitzoo.thesis.scalability.dynamics import clohessy_wiltshire_displacement
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig
from orbitzoo.thesis.runtime import initialize_run_directory
from orbitzoo.thesis.training.trainer import train

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mappo_smoke.json"
LARGE_MANEUVER = ManeuverConfig(1.0, 10.0, 300.0, 60.0)
SAFETY = SafetyConfig(safe_separation_meters=1_000.0, screening_horizon_seconds=1_800.0)
ORBIT_RADIUS = 6_878_136.3
MEAN_MOTION = math.sqrt(3.986004418e14 / ORBIT_RADIUS**3)


def observation(
    relative_position: list[float],
    relative_velocity: list[float],
    *,
    tca_seconds: float = 600.0,
    normalized_miss: float = 0.1,
    valid: bool = True,
) -> np.ndarray:
    row = np.zeros(OWN_FEATURE_DIM + NEIGHBOR_FEATURE_DIM, dtype=np.float32)
    row[0] = ORBIT_RADIUS / POSITION_SCALE_METERS
    block = row[OWN_FEATURE_DIM:]
    block[0:3] = np.asarray(relative_position) / POSITION_SCALE_METERS
    block[3:6] = np.asarray(relative_velocity) / VELOCITY_SCALE_MPS
    block[6] = tca_seconds / SAFETY.screening_horizon_seconds
    block[7] = normalized_miss
    block[11] = float(valid)
    return row


def rule_policy() -> ClohessyWiltshireAvoidancePolicy:
    return ClohessyWiltshireAvoidancePolicy(LARGE_MANEUVER, SAFETY)


def test_clohessy_wiltshire_prograde_burn_drifts_behind_and_rises() -> None:
    elapsed = 0.25 * 2 * math.pi / MEAN_MOTION

    radial, along_track, cross_track = clohessy_wiltshire_displacement(
        np.array([0.0, 1.0, 0.0]), MEAN_MOTION, elapsed
    )

    assert radial == pytest.approx(2 / MEAN_MOTION)
    assert along_track == pytest.approx((4 - 3 * math.pi / 2) / MEAN_MOTION)
    assert cross_track == 0.0


def test_rule_policy_coasts_when_threat_is_safe_absent_or_past() -> None:
    policy = rule_policy()
    rows = np.stack(
        [
            observation([0, 5_000, 0], [0, -10, 0], normalized_miss=2.0),
            observation([0, 5_000, 0], [0, -10, 0], valid=False),
            observation([0, 5_000, 0], [0, -10, 0], tca_seconds=0.0),
        ]
    )

    assert policy.choose(rows).tolist() == [ManeuverAction.NO_OP] * 3


def crossing_observation(miss_offset: list[float], meeting_seconds: float = 600.0) -> np.ndarray:
    """The agent's encoded view of a polar-orbit object crossing its equatorial orbit, built backwards in J2 physics."""
    epoch = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
    speed = ORBIT_RADIUS * MEAN_MOTION
    at_meeting = {
        "agent": (np.array([ORBIT_RADIUS, 0.0, 0.0]), np.array([0.0, speed, 0.0])),
        "threat": (np.array([ORBIT_RADIUS, 0.0, 0.0]) + np.array(miss_offset), np.array([0.0, 0.0, speed])),
    }
    start = {name: propagate(*state, epoch, -meeting_seconds) for name, state in at_meeting.items()}
    state = CatalogState(
        np.array([start["agent"][0], start["threat"][0]]),
        np.array([start["agent"][1], start["threat"][1]]),
        np.ones(2),
        np.array([True, False]),
        np.zeros(2),
    )
    return encode_local_observations(state, np.array([0]), 1, SAFETY)


def test_curved_features_see_the_crossing_minutes_ahead() -> None:
    observation = crossing_observation([300.0, 0.0, 0.0])

    assert observation[0, OWN_FEATURE_DIM + 6] * SAFETY.screening_horizon_seconds == pytest.approx(600.0, abs=2.0)
    assert observation[0, OWN_FEATURE_DIM + 7] * SAFETY.safe_separation_meters == pytest.approx(300.0, abs=20.0)


def test_rule_policy_choice_maximizes_the_predicted_miss() -> None:
    policy = rule_policy()
    observation = crossing_observation([300.0, 0.0, 0.0])

    chosen = ManeuverAction(policy.choose(observation)[0])

    row = observation[0].astype(float)
    own = (row[0:3] * POSITION_SCALE_METERS, row[3:6] * VELOCITY_SCALE_MPS)
    basis = rsw_bases(own[0][np.newaxis], own[1][np.newaxis])[0]
    threat = (own[0] + basis.T @ (row[7:10] * POSITION_SCALE_METERS), own[1] + basis.T @ (row[10:13] * VELOCITY_SCALE_MPS))
    approach = predict_closest_approach(own[0][None], own[1][None], threat[0][None], threat[1][None], 1_800.0)
    encounter = encounter_from_states(0, 0, 0, (approach.first_positions_m[0], approach.first_velocities_mps[0]),
                                      (approach.first_positions_m[0] + approach.miss_vectors_m[0],
                                       approach.first_velocities_mps[0] + approach.relative_velocities_mps[0]))

    def predicted_miss(action: ManeuverAction) -> float:
        if action is ManeuverAction.NO_OP:
            return encounter.miss_distance_m
        gain = miss_displacement_per_mps(encounter, action, [float(approach.time_seconds[0])])
        return float(np.linalg.norm(encounter.miss_vector_m + LARGE_MANEUVER.commanded_delta_v_mps * gain))

    assert chosen is not ManeuverAction.NO_OP
    assert predicted_miss(chosen) == max(predicted_miss(action) for action in ManeuverAction)


def test_noop_policy_never_burns() -> None:
    assert NoOpPolicy().choose(np.zeros((3, 19))).tolist() == [0, 0, 0]


def test_evaluation_seeds_do_not_overlap_training_seeds() -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    training_seeds = range(config.seed, config.seed + 100_000)

    assert not set(evaluation_seeds(config, 50)) & set(training_seeds)


def test_rule_widens_the_development_conjunction_compared_with_noop() -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    config = dataclasses.replace(config, maneuver=LARGE_MANEUVER)
    env = build_environment(config)
    source = FixedEpisodeSource(env)

    [coasting] = evaluate_policy(build_policy("noop", config, env.local_observation_dim), source, [0])
    [avoiding] = evaluate_policy(build_policy("rule", config, env.local_observation_dim), source, [0])

    assert coasting.mean_delta_v_per_agent_mps == 0.0
    assert avoiding.mean_delta_v_per_agent_mps > 0.0
    assert avoiding.minimum_separation_meters > coasting.minimum_separation_meters + 50.0


@pytest.fixture(scope="module")
def smoke_checkpoint(tmp_path_factory) -> Path:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    run_directory = initialize_run_directory(tmp_path_factory.mktemp("evaluation") / "run", config)
    train(config, run_directory)
    return run_directory / "checkpoints" / "latest.pt"


def test_checkpoint_policy_rebuilds_its_architecture(smoke_checkpoint: Path) -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    env = build_environment(config)

    policy = build_policy(f"smoke={smoke_checkpoint}", config, env.local_observation_dim)

    assert policy.name == "smoke"
    assert isinstance(policy.policy, MAPPO)
    assert policy.policy.actor_hidden_dims == config.training.actor_hidden_dims


def test_unknown_policy_is_rejected_before_writing_output(tmp_path: Path) -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    output = tmp_path / "evaluation"

    with pytest.raises(FileNotFoundError):
        evaluate(config, ["noop", "missing.pt"], 1, output)
    assert not output.exists()


def test_evaluate_writes_comparable_tables(tmp_path: Path, smoke_checkpoint: Path) -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    output = tmp_path / "evaluation"

    summaries = evaluate(config, ["noop", "rule", str(smoke_checkpoint)], 2, output)

    assert [summary.policy for summary in summaries] == ["noop", "rule", "mappo"]
    with (output / "episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == 6
    seeds_by_policy = {
        name: [row["seed"] for row in episodes if row["policy"] == name] for name in ("noop", "rule", "mappo")
    }
    assert seeds_by_policy["noop"] == seeds_by_policy["rule"] == seeds_by_policy["mappo"]
    assert (output / "summary.csv").is_file() and (output / "config.json").is_file()


def test_evaluate_command_prints_a_summary(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(
        ["evaluate", "--config", str(SMOKE_CONFIG), "--episodes", "1", "--output", str(tmp_path / "cli")]
    )

    args.handler(args)

    output = capsys.readouterr().out
    assert "noop" in output and "rule" in output
