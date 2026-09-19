import copy
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from orbitzoo.cli import build_parser
from orbitzoo.thesis.config import EnvironmentConfig, ExperimentConfig
from orbitzoo.thesis.environments.collision_avoidance import (
    CollisionAvoidanceEnv,
    development_environment_kwargs,
)
from orbitzoo.thesis.environments.scenarios import build_environment
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig
from orbitzoo.thesis.runtime import initialize_run_directory
from orbitzoo.thesis.training.trainer import _build_policy, run_episode, train

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mappo_smoke.json"
NEGLIGIBLE_MANEUVER = ManeuverConfig(0.01, 0.1, 300.0, 60.0)


def smoke_config(**training_overrides) -> ExperimentConfig:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    return dataclasses.replace(config, training=dataclasses.replace(config.training, **training_overrides))


def read_metrics(run_directory: Path) -> list[dict[str, str]]:
    with (run_directory / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("update_seconds")
    return rows


def run(tmp_path: Path, name: str, config: ExperimentConfig) -> Path:
    run_directory = initialize_run_directory(tmp_path / name, config)
    train(config, run_directory)
    return run_directory


@pytest.fixture(scope="module")
def two_update_run(tmp_path_factory) -> Path:
    return run(tmp_path_factory.mktemp("training"), "full", smoke_config())


class RecordingEnv:
    """Wraps an environment and records the raw rewards it returns."""

    def __init__(self, env: CollisionAvoidanceEnv) -> None:
        self.env = env
        self.raw_rewards: list[np.ndarray] = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        outputs = self.env.step(actions)
        self.raw_rewards.append(outputs[2].copy())
        self.last_outputs = outputs
        return outputs


def test_unknown_scenario_is_rejected() -> None:
    config = dataclasses.replace(ExperimentConfig.load(SMOKE_CONFIG), environment=EnvironmentConfig(num_agents=4, scenario="missing"))

    with pytest.raises(ValueError, match="unknown scenario"):
        build_environment(config)


def test_agent_count_must_match_the_scenario() -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    config = dataclasses.replace(config, environment=dataclasses.replace(config.environment, num_agents=16))

    with pytest.raises(ValueError, match="4 agents"):
        build_environment(config)


def test_horizon_steps_are_bootstrapped_from_the_critic() -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    env = RecordingEnv(build_environment(config))
    policy = _build_policy(config, env.env)

    run_episode(policy, env, seed=0)

    final_local, final_global = env.last_outputs[:2]
    expected = env.raw_rewards[-1] + policy.gamma * policy.values(final_local, final_global).numpy()
    np.testing.assert_allclose(policy.rollout.rewards[-1].numpy(), expected, rtol=1e-6)
    for raw, stored in zip(env.raw_rewards[:-1], policy.rollout.rewards[:-1]):
        np.testing.assert_allclose(stored.numpy(), raw, rtol=1e-6)


def test_collision_steps_are_not_bootstrapped() -> None:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    kwargs = copy.deepcopy(development_environment_kwargs())
    shadow = dict(kwargs["drifters"][0], name="shadow", initial_state=list(kwargs["spacecrafts"][1]["initial_state"]))
    kwargs["drifters"].append(shadow)
    env = RecordingEnv(
        CollisionAvoidanceEnv(maneuver_config=NEGLIGIBLE_MANEUVER, episode_horizon=10, **kwargs)
    )
    policy = _build_policy(config, env.env)

    summary = run_episode(policy, env, seed=0)

    assert summary.ended_in_collision and summary.length == 1
    np.testing.assert_allclose(policy.rollout.rewards[-1].numpy(), env.raw_rewards[-1], rtol=1e-6)


def test_training_writes_metrics_checkpoints_and_state(two_update_run: Path) -> None:
    rows = read_metrics(two_update_run)

    assert [row["update"] for row in rows] == ["1", "2"]
    assert all(np.isfinite(float(value)) for row in rows for value in row.values())
    assert (two_update_run / "checkpoints" / "latest.pt").is_file()
    state = json.loads((two_update_run / "training_state.json").read_text())
    assert state["completed_updates"] == 2
    assert any((two_update_run / "tensorboard").iterdir())


def test_same_seed_gives_identical_metrics(tmp_path: Path, two_update_run: Path) -> None:
    repeat = run(tmp_path, "repeat", smoke_config())

    assert read_metrics(repeat) == read_metrics(two_update_run)


def test_resume_matches_an_uninterrupted_run(tmp_path: Path, two_update_run: Path) -> None:
    interrupted = run(tmp_path, "interrupted", smoke_config(total_updates=1))

    result = train(smoke_config(), interrupted, resume=True)

    assert result.completed_updates == 2
    assert read_metrics(interrupted) == read_metrics(two_update_run)


def test_resume_requires_a_checkpoint(tmp_path: Path) -> None:
    config = smoke_config()
    empty = initialize_run_directory(tmp_path / "empty", config)

    with pytest.raises(FileNotFoundError):
        train(config, empty, resume=True)


def test_train_command_runs_the_smoke_config(tmp_path: Path, capsys) -> None:
    output = tmp_path / "cli_run"
    args = build_parser().parse_args(["train", "--config", str(SMOKE_CONFIG), "--output", str(output)])

    args.handler(args)

    assert "Completed updates: 2" in capsys.readouterr().out
    assert (output / "metrics.csv").is_file()
