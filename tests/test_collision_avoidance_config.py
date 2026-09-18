import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitzoo.env import OrbitZoo
from orbitzoo.thesis.config import default_maneuver_config
from orbitzoo.thesis.environments.collision_avoidance import CollisionAvoidanceEnv


@pytest.fixture
def stub_orbitzoo(monkeypatch):
    monkeypatch.setattr(OrbitZoo, "__init__", lambda self, **kwargs: None)


def test_rejects_non_positive_decision_interval(stub_orbitzoo):
    with pytest.raises(ValueError, match="decision_interval_seconds"):
        CollisionAvoidanceEnv(maneuver_config=default_maneuver_config(), decision_interval_seconds=0.0)


def test_rejects_burn_longer_than_decision_interval(stub_orbitzoo):
    with pytest.raises(ValueError, match="maximum_burn_duration_seconds"):
        CollisionAvoidanceEnv(maneuver_config=default_maneuver_config(), decision_interval_seconds=30.0)


def test_defaults_use_calibrated_values(stub_orbitzoo):
    env = CollisionAvoidanceEnv(maneuver_config=default_maneuver_config())

    assert env.decision_interval_seconds == 120.0
    assert env.observation_encoder.neighborhood_size == 1
