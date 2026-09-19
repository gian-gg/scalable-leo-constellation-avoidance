import json
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitzoo.thesis.config import EnvironmentConfig, ExperimentConfig, TrainingConfig
from orbitzoo.thesis.runtime import create_run_directory, select_device


requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is not installed in this Python environment",
)


def test_config_round_trip(tmp_path):
    config = ExperimentConfig(seed=7)
    config_path = tmp_path / "config.json"

    config.save(config_path)

    assert ExperimentConfig.load(config_path) == config
    assert config.to_dict()["maneuver"]["commanded_delta_v_mps"] == 0.5


def test_invalid_neighborhood_is_rejected():
    config = ExperimentConfig(environment=EnvironmentConfig(num_agents=4, neighborhood_size=0))

    with pytest.raises(ValueError, match="neighborhood_size"):
        config.validate()


def test_neighborhood_may_exceed_training_agent_count_for_padding():
    config = ExperimentConfig(environment=EnvironmentConfig(num_agents=4, neighborhood_size=8))

    config.validate()


@requires_torch
def test_run_directory_contains_reproducibility_files(tmp_path):
    run_directory = create_run_directory(
        tmp_path,
        ExperimentConfig(seed=9),
        "mappo toy",
        now=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )

    assert run_directory.name == "2026-08-31_000000Z_mappo-toy_seed9"
    assert (run_directory / "config.json").is_file()
    assert (run_directory / "tensorboard").is_dir()
    info = json.loads((run_directory / "environment_info.json").read_text())
    assert info["device"] in {"cpu", "cuda", "mps"}


@requires_torch
def test_device_selection_returns_a_supported_torch_device():
    assert select_device().type in {"cpu", "cuda", "mps"}


def test_environment_defaults_match_calibration():
    config = EnvironmentConfig()

    assert config.neighborhood_size == 1
    assert config.decision_interval_seconds == 120.0


def test_toy_config_uses_calibrated_values():
    config = ExperimentConfig.load(Path(__file__).resolve().parents[1] / "configs" / "mappo_toy.json")

    assert config.environment.neighborhood_size == 1
    assert config.environment.decision_interval_seconds == 120.0


def test_training_config_round_trips_tuples_and_old_files_still_load(tmp_path):
    config = ExperimentConfig(training=TrainingConfig(actor_hidden_dims=[32, 16]))
    config_path = tmp_path / "config.json"
    config.save(config_path)
    assert ExperimentConfig.load(config_path) == config
    assert config.training.actor_hidden_dims == (32, 16)

    raw = json.loads(config_path.read_text())
    raw["training"] = {"rollout_steps": 64}
    del raw["environment"]["scenario"]
    config_path.write_text(json.dumps(raw))
    legacy = ExperimentConfig.load(config_path)
    assert legacy.training.total_updates == TrainingConfig().total_updates
    assert legacy.environment.scenario == "development"


@pytest.mark.parametrize(
    "overrides",
    [
        {"total_updates": 0},
        {"minibatch_size": 0},
        {"actor_hidden_dims": ()},
        {"critic_hidden_dims": (32, 0)},
        {"entropy_coefficient": -0.1},
        {"value_clip": 0.0},
        {"device": "tpu"},
    ],
)
def test_invalid_training_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        TrainingConfig(**overrides).validate()


def test_maneuver_defaults_match_the_sizing_decision():
    maneuver = ExperimentConfig().maneuver
    toy = ExperimentConfig.load(Path(__file__).resolve().parents[1] / "configs" / "mappo_toy.json").maneuver

    assert (maneuver.commanded_delta_v_mps, maneuver.maximum_thrust_newtons) == (0.5, 7.0)
    assert toy == maneuver
    heaviest_mass_kg = 800.0
    assert heaviest_mass_kg * maneuver.commanded_delta_v_mps / maneuver.maximum_thrust_newtons <= maneuver.maximum_burn_duration_seconds
