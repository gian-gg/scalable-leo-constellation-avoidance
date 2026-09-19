from datetime import datetime, timezone

import numpy as np
import pytest

from orbitzoo.thesis.environments.prediction import predict_closest_approach
from orbitzoo.thesis.scenarios.propagation import propagate

RADIUS = 6_878_136.3
SPEED = (3.986004418e14 / RADIUS) ** 0.5
EPOCH = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)


def test_prediction_matches_a_crossing_built_in_environment_physics() -> None:
    meeting = 700.0
    agent = propagate(np.array([RADIUS, 0.0, 0.0]), np.array([0.0, SPEED, 0.0]), EPOCH, -meeting)
    threat = propagate(np.array([RADIUS + 400.0, 0.0, 0.0]), np.array([0.0, 0.0, SPEED]), EPOCH, -meeting)

    approach = predict_closest_approach(agent[0][None], agent[1][None], threat[0][None], threat[1][None], 1_800.0)

    assert approach.time_seconds[0] == pytest.approx(meeting, abs=1.0)
    assert approach.miss_distance_m[0] == pytest.approx(400.0, abs=40.0)
    np.testing.assert_allclose(approach.miss_vectors_m[0] @ approach.relative_velocities_mps[0], 0.0, atol=1e3)


def test_receding_pair_is_closest_now() -> None:
    position, velocity = np.array([[RADIUS, 0.0, 0.0]]), np.array([[0.0, SPEED, 0.0]])

    approach = predict_closest_approach(position, velocity, position + [[0, 2_000.0, 0]], velocity + [[0, 5.0, 0]], 1_800.0)

    assert approach.time_seconds[0] == 0.0
    assert approach.miss_distance_m[0] == pytest.approx(2_000.0)
