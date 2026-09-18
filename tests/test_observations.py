from dataclasses import dataclass

import numpy as np

from orbitzoo.thesis.environments.observations import (
    GLOBAL_BODY_FEATURE_DIM,
    NEIGHBOR_FEATURE_DIM,
    OWN_FEATURE_DIM,
    LocalObservationEncoder,
)
from orbitzoo.thesis.environments.safety import SafetyConfig, assess_all_pairs


@dataclass
class Body:
    name: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float = 1.0
    initial_fuel_mass: float = 0.0
    fuel: float = 0.0

    def get_fuel(self) -> float:
        return self.fuel


def body(
    name: str,
    position: list[float],
    velocity: list[float],
    *,
    fuel: float = 0.0,
    initial_fuel: float = 0.0,
) -> Body:
    return Body(
        name,
        np.asarray(position, dtype=float),
        np.asarray(velocity, dtype=float),
        initial_fuel_mass=initial_fuel,
        fuel=fuel,
    )


def test_local_width_is_fixed_and_missing_neighbors_are_masked() -> None:
    config = SafetyConfig(safe_separation_meters=1_000.0, screening_horizon_seconds=300.0)
    encoder = LocalObservationEncoder(neighborhood_size=3, safety_config=config)
    bodies = [
        body("agent", [7_000_000, 0, 0], [0, 7_500, 0], fuel=25, initial_fuel=50),
        body("debris", [7_002_000, 0, 0], [0, 7_490, 0]),
    ]

    state = encoder.encode(bodies, ["agent"])

    assert state.local_observations.shape == (1, OWN_FEATURE_DIM + 3 * NEIGHBOR_FEATURE_DIM)
    first_mask = OWN_FEATURE_DIM + NEIGHBOR_FEATURE_DIM - 1
    assert state.local_observations[0, first_mask] == 1.0
    assert np.all(state.local_observations[0, first_mask + 1 :] == 0.0)


def test_threat_ranking_beats_current_distance_and_uses_observer_rsw() -> None:
    config = SafetyConfig(safe_separation_meters=1_000.0, screening_horizon_seconds=300.0)
    encoder = LocalObservationEncoder(neighborhood_size=1, safety_config=config)
    observer = body("agent", [7_000_000, 0, 0], [0, 7_500, 0], fuel=50, initial_fuel=50)
    close_but_receding = body("close", [7_000_500, 0, 0], [10, 7_500, 0])
    collision_course = body("threat", [7_000_000, 2_000, 0], [0, 7_490, 0])
    bodies = [observer, close_but_receding, collision_course]

    state = encoder.encode(bodies, ["agent"], assess_all_pairs(bodies, config))

    relative_position = state.local_observations[0, OWN_FEATURE_DIM : OWN_FEATURE_DIM + 3]
    assert np.allclose(relative_position, [0.0, 2_000 / 10_000_000, 0.0])


def test_neighbor_block_identifies_maneuverable_bodies_and_fuel() -> None:
    config = SafetyConfig()
    encoder = LocalObservationEncoder(neighborhood_size=1, safety_config=config)
    bodies = [
        body("agent_1", [7_000_000, 0, 0], [0, 7_500, 0], fuel=10, initial_fuel=20),
        body("agent_2", [7_001_000, 0, 0], [0, 7_500, 0], fuel=15, initial_fuel=20),
        body("debris", [-7_000_000, 0, 0], [0, -7_500, 0]),
    ]

    state = encoder.encode(bodies, ["agent_1", "agent_2"])
    neighbor_tail = state.local_observations[0, -4:]

    assert np.allclose(neighbor_tail, [0.002, 1.0, 0.75, 1.0])


def test_global_state_uses_agent_then_non_agent_order() -> None:
    config = SafetyConfig()
    encoder = LocalObservationEncoder(neighborhood_size=1, safety_config=config)
    debris = body("debris", [3_000_000, 0, 0], [0, 3_000, 0])
    second = body("agent_2", [2_000_000, 0, 0], [0, 2_000, 0], fuel=2, initial_fuel=4)
    first = body("agent_1", [1_000_000, 0, 0], [0, 1_000, 0], fuel=3, initial_fuel=4)

    state = encoder.encode([debris, second, first], ["agent_1", "agent_2"])
    blocks = state.global_state.reshape(-1, GLOBAL_BODY_FEATURE_DIM)

    assert np.allclose(blocks[:, 0], [0.1, 0.2, 0.3])
    assert np.allclose(blocks[:, -1], [1.0, 1.0, 0.0])


def test_actor_width_does_not_depend_on_population_size() -> None:
    config = SafetyConfig()
    encoder = LocalObservationEncoder(neighborhood_size=2, safety_config=config)
    small = [
        body("agent", [7_000_000, 0, 0], [0, 7_500, 0], fuel=1, initial_fuel=1),
        body("debris_1", [7_001_000, 0, 0], [0, 7_500, 0]),
    ]
    large = small + [
        body("debris_2", [7_002_000, 0, 0], [0, 7_500, 0]),
        body("debris_3", [7_003_000, 0, 0], [0, 7_500, 0]),
    ]

    small_state = encoder.encode(small, ["agent"])
    large_state = encoder.encode(large, ["agent"])

    assert small_state.local_observations.shape[1] == encoder.local_observation_dim
    assert large_state.local_observations.shape[1] == encoder.local_observation_dim
    assert small_state.global_state.shape != large_state.global_state.shape


def test_local_rows_do_not_depend_on_body_input_order() -> None:
    config = SafetyConfig()
    encoder = LocalObservationEncoder(neighborhood_size=2, safety_config=config)
    bodies = [
        body("agent_1", [7_000_000, 0, 0], [0, 7_500, 0], fuel=10, initial_fuel=20),
        body("agent_2", [7_000_800, 0, 0], [0, 7_500, 0], fuel=15, initial_fuel=20),
        body("debris", [7_000_000, 3_000, 0], [0, 7_480, 0]),
    ]

    forward = encoder.encode(bodies, ["agent_1", "agent_2"])
    reversed_bodies = encoder.encode(bodies[::-1], ["agent_1", "agent_2"])
    swapped_agents = encoder.encode(bodies, ["agent_2", "agent_1"])

    np.testing.assert_array_equal(forward.local_observations, reversed_bodies.local_observations)
    np.testing.assert_array_equal(forward.global_state, reversed_bodies.global_state)
    np.testing.assert_array_equal(forward.local_observations[::-1], swapped_agents.local_observations)
