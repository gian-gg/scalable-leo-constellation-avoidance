import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from sgp4.io import fix_checksum

from orbitzoo.cli import build_parser
from orbitzoo.thesis.environments.observations import LocalObservationEncoder
from orbitzoo.thesis.environments.safety import SafetyConfig
from orbitzoo.thesis.evaluation.policies import (
    ClohessyWiltshireAvoidancePolicy,
    NoOpPolicy,
    clohessy_wiltshire_displacement,
)
from orbitzoo.thesis.maneuvers.contract import ManeuverConfig
from orbitzoo.thesis.scalability.config import ScalabilityConfig
from orbitzoo.thesis.scalability.dynamics import inertial_states, mean_motions, propagate_hill_states
from orbitzoo.thesis.environments.vectorized_observations import CatalogState, encode_local_observations, rsw_bases, top_neighbors
from orbitzoo.thesis.scalability.runner import build_scenarios, leo_objects, run_scalability
from orbitzoo.thesis.scalability.screening import (
    ConjunctionEvent,
    ConjunctionTracker,
    close_approaches,
    match_events,
)
from orbitzoo.thesis.scalability.simulator import SimulationSettings, simulate

MU = 3.986004418e14
ORBIT_RADIUS = 6_878_136.3
CROSSING_TIME = 900.0
SAFETY = SafetyConfig()
LARGE_MANEUVER = ManeuverConfig(1.0, 10.0, 300.0, 60.0)
X_AXIS, Y_AXIS, Z_AXIS = np.eye(3)
EXPERIMENT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mappo_smoke.json"


class CircularTrajectory:
    """Analytic circular orbits: each is (radius, in-plane unit u, unit v, phase at t=0)."""

    def __init__(self, orbits: list[tuple[float, np.ndarray, np.ndarray, float]]) -> None:
        self.orbits = orbits

    def states(self, times_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times_seconds, dtype=float)
        positions, velocities = [], []
        for radius, u, v, phase in self.orbits:
            rate = math.sqrt(MU / radius**3)
            angle = phase + rate * times
            positions.append(radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v))
            velocities.append(radius * rate * (-np.sin(angle)[:, None] * u + np.cos(angle)[:, None] * v))
        return np.array(positions), np.array(velocities)


def crossing_orbit(radius: float, u: np.ndarray, v: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    return radius, u, v, -math.sqrt(MU / radius**3) * CROSSING_TIME


def crossing_scene() -> CircularTrajectory:
    return CircularTrajectory(
        [
            crossing_orbit(ORBIT_RADIUS, X_AXIS, Y_AXIS),
            crossing_orbit(ORBIT_RADIUS + 500.0, X_AXIS, Z_AXIS),
            crossing_orbit(ORBIT_RADIUS + 30_000.0, X_AXIS, Y_AXIS),
        ]
    )


def settings(**overrides) -> SimulationSettings:
    values = dict(
        neighborhood_size=1,
        decision_interval_seconds=120,
        duration_seconds=1_800,
        fine_step_seconds=10,
        safety=SAFETY,
        maneuver=LARGE_MANEUVER,
        dry_mass_kg=200.0,
        initial_fuel_mass_kg=50.0,
        maximum_relative_speed_mps=20_000.0,
    )
    values.update(overrides)
    return SimulationSettings(**values)


@dataclass
class Body:
    name: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    initial_fuel_mass: float
    fuel: float

    def get_fuel(self) -> float:
        return self.fuel


@pytest.mark.parametrize("neighborhood_size", [1, 2, 3, 6])
@pytest.mark.parametrize("object_count", [4, 40])
def test_vectorized_observations_match_the_training_encoder(neighborhood_size: int, object_count: int) -> None:
    rng = np.random.default_rng(neighborhood_size * 100 + object_count)
    positions = np.array([7e6, 0, 0]) + rng.normal(0, 3_000, (object_count, 3))
    velocities = np.array([0, 7_500, 0]) + rng.normal(0, 20, (object_count, 3))
    radii = rng.uniform(0.5, 2.0, object_count)
    is_agent = rng.random(object_count) < 0.4
    is_agent[:2] = True
    fuel = rng.uniform(0, 50, object_count)
    bodies = [
        Body(f"b{index}", positions[index], velocities[index], radii[index], 50.0 if is_agent[index] else 0.0, fuel[index])
        for index in range(object_count)
    ]
    agents = np.flatnonzero(is_agent)

    expected = LocalObservationEncoder(neighborhood_size, SAFETY).encode(bodies, [f"b{index}" for index in agents])
    state = CatalogState(positions, velocities, radii, is_agent, np.where(is_agent, np.clip(fuel / 50, 0, 1), 0.0))
    actual = encode_local_observations(state, agents, neighborhood_size, SAFETY, agent_batch_size=3)

    np.testing.assert_array_equal(actual, expected.local_observations)


def test_hill_transition_matches_the_displacement_formula_and_composes() -> None:
    motion = mean_motions(np.array([[ORBIT_RADIUS, 0, 0]]))
    for direction in np.eye(3):
        offsets, _ = propagate_hill_states(np.zeros((1, 3)), direction[np.newaxis], motion, 587.5)
        np.testing.assert_allclose(offsets[0], clohessy_wiltshire_displacement(direction, motion[0], 587.5), atol=1e-9)

    start = (np.array([[100.0, -50.0, 20.0]]), np.array([[0.1, -0.2, 0.05]]))
    halves = propagate_hill_states(*propagate_hill_states(*start, motion, 300), motion, 300)
    whole = propagate_hill_states(*start, motion, 600)
    np.testing.assert_allclose(halves[0], whole[0], atol=1e-9)
    np.testing.assert_allclose(halves[1], whole[1], atol=1e-12)


def test_inertial_states_add_offsets_and_frame_rotation() -> None:
    position = np.array([[ORBIT_RADIUS, 0.0, 0.0]])
    velocity = np.array([[0.0, 7_600.0, 0.0]])
    motion = mean_motions(position)

    unchanged = inertial_states(position, velocity, np.zeros((1, 3)), np.zeros((1, 3)), motion)
    raised = inertial_states(position, velocity, np.array([[100.0, 0, 0]]), np.zeros((1, 3)), motion)

    np.testing.assert_array_equal(unchanged[0], position)
    np.testing.assert_array_equal(unchanged[1], velocity)
    np.testing.assert_allclose(raised[0], [[ORBIT_RADIUS + 100.0, 0, 0]])
    np.testing.assert_allclose(raised[1], velocity + [[0, motion[0] * 100.0, 0]])


def test_close_approaches_reports_agent_pairs_with_linear_minimum() -> None:
    positions = np.array([[0.0, 0, 0], [0, 400, 30_000], [0, 0, 50_000], [10.0, 0, 0]])
    velocities = np.array([[0.0, 0, 0], [0, 0, -7_000], [0, 0, 0], [0, 0, 0]])

    low, high, distances, offsets = close_approaches(positions, velocities, np.array([0]), 60_000.0, 5.0)

    pairs = dict(zip(zip(low.tolist(), high.tolist()), zip(distances, offsets)))
    assert set(pairs) == {(0, 1), (0, 2), (0, 3)}
    distance, offset = pairs[(0, 1)]
    assert offset == pytest.approx(30_000 / 7_000) and distance == pytest.approx(400.0)


def test_tracker_merges_samples_into_one_event_and_flags_collisions() -> None:
    tracker = ConjunctionTracker(1_000.0, np.array([1.0, 1.0, 5.0]), merge_gap_seconds=600.0)
    for time_seconds, distance in ((100.0, 900.0), (110.0, 300.0), (120.0, 800.0), (5_000.0, 700.0)):
        tracker.add(time_seconds, np.array([0]), np.array([1]), np.array([distance]), np.array([0.0]))
    tracker.add(200.0, np.array([0]), np.array([2]), np.array([4.0]), np.array([1.5]))

    events = tracker.events()

    assert [(event.first_index, event.second_index, event.tca_seconds, event.miss_distance_meters) for event in events] == [
        (0, 1, 110.0, 300.0),
        (0, 2, 201.5, 4.0),
        (0, 1, 5_000.0, 700.0),
    ]
    assert [event.is_collision for event in events] == [False, True, False]


def test_match_events_counts_shared_events_and_returns_new_ones() -> None:
    reference = [ConjunctionEvent(0, 1, 100.0, 500.0, False), ConjunctionEvent(0, 2, 900.0, 200.0, False)]
    shifted = ConjunctionEvent(0, 1, 102.0, 450.0, False)
    new = ConjunctionEvent(0, 3, 400.0, 800.0, False)

    recurs, unmatched = match_events(reference, [shifted, new], tolerance_seconds=300.0)

    assert recurs == [True, False]
    assert unmatched == [new]


def test_coasting_agent_detects_the_crossing_at_its_known_time() -> None:
    result = simulate(crossing_scene(), np.ones(3), np.array([0]), NoOpPolicy(), settings())

    [event] = result.events
    assert (event.first_index, event.second_index) == (0, 1)
    assert event.tca_seconds == pytest.approx(CROSSING_TIME, abs=0.5)
    assert event.miss_distance_meters == pytest.approx(500.0, abs=1.0)
    assert result.decisions == 15 and result.maneuvers == 0


def test_rule_policy_resolves_the_crossing_with_large_maneuvers() -> None:
    policy = ClohessyWiltshireAvoidancePolicy(LARGE_MANEUVER, SAFETY)

    result = simulate(crossing_scene(), np.ones(3), np.array([0]), policy, settings())

    assert result.events == []
    assert result.maneuvers >= 1
    assert result.delta_v_mps[0] == pytest.approx(result.maneuvers * LARGE_MANEUVER.commanded_delta_v_mps, rel=1e-9)


def test_agents_without_fuel_cannot_maneuver() -> None:
    policy = ClohessyWiltshireAvoidancePolicy(LARGE_MANEUVER, SAFETY)

    result = simulate(crossing_scene(), np.ones(3), np.array([0]), policy, settings(initial_fuel_mass_kg=0.0))

    assert result.maneuvers == 0 and result.rejected_actions > 0
    assert len(result.events) == 1


def test_pairs_without_an_agent_are_not_screened() -> None:
    result = simulate(crossing_scene(), np.ones(3), np.array([2]), NoOpPolicy(), settings())

    assert result.events == []


BASE_LINE_1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
BASE_LINE_2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"


def write_synthetic_catalog(directory: Path, object_count: int, candidate_count: int) -> None:
    tle_lines, metadata = [], ["norad_id,name,object_type,is_agent_candidate,radius_meters,constellation"]
    for index in range(object_count):
        norad_id = 40_000 + index
        field = f"{norad_id:05d}"
        line1 = fix_checksum((BASE_LINE_1[:2] + field + BASE_LINE_1[7:])[:68])
        line2 = BASE_LINE_2[:2] + field + BASE_LINE_2[7:17] + f"{(index * 7.3) % 360:8.4f}"
        line2 += BASE_LINE_2[25:43] + f"{(index * 11.9) % 360:8.4f}" + BASE_LINE_2[51:]
        tle_lines += [line1, fix_checksum(line2[:68])]
        object_type = "payload" if index < candidate_count * 2 else "debris"
        metadata.append(f"{norad_id},OBJECT {index},{object_type},{str(index < candidate_count).lower()},1.0,")
    (directory / "catalog.tle").write_text("\n".join(tle_lines) + "\n")
    (directory / "objects.csv").write_text("\n".join(metadata) + "\n")


@pytest.fixture
def synthetic_config(tmp_path: Path) -> Path:
    write_synthetic_catalog(tmp_path, object_count=30, candidate_count=5)
    config = {
        "catalog": {"tle_path": "catalog.tle", "metadata_path": "objects.csv", "maximum_tle_age_days": 14.0},
        "catalog_sizes": [8, 30],
        "hypothetical_agent_counts": [5, 10],
        "duration_seconds": 240,
        "experiment_config": str(EXPERIMENT_CONFIG),
    }
    path = tmp_path / "scalability.json"
    path.write_text(json.dumps(config))
    return path


def test_scenarios_are_nested_and_keep_every_agent_candidate(synthetic_config: Path) -> None:
    config = ScalabilityConfig.load(synthetic_config)
    objects, _ = leo_objects(config, synthetic_config)

    small, large, few_agents, more_agents = build_scenarios(config, objects, ("catalog", "agents"))

    assert [len(small.objects), len(large.objects)] == [8, 30]
    assert set(small.objects) <= set(large.objects)
    for scenario in (small, large):
        assert {scenario.objects[index].norad_id for index in scenario.agent_indices} == set(range(40_000, 40_005))
    assert set(few_agents.agent_indices) <= set(more_agents.agent_indices)
    assert (few_agents.agent_indices.size, more_agents.agent_indices.size) == (5, 10)


def test_catalog_smaller_than_the_candidate_pool_is_rejected(synthetic_config: Path) -> None:
    config = ScalabilityConfig.load(synthetic_config)
    objects, _ = leo_objects(config, synthetic_config)
    config = ScalabilityConfig(**{**config.__dict__, "catalog_sizes": (3,)})

    with pytest.raises(ValueError, match="agent candidates"):
        build_scenarios(config, objects, ("catalog",))


def test_run_writes_results_for_every_scenario_and_policy(synthetic_config: Path, tmp_path: Path) -> None:
    output = tmp_path / "scale"

    rows = run_scalability(synthetic_config, ["rule"], output)

    assert [(row["sweep"], row["objects"], row["agents"], row["policy"]) for row in rows] == [
        ("catalog", 8, 5, "noop"),
        ("catalog", 8, 5, "rule"),
        ("catalog", 30, 5, "noop"),
        ("catalog", 30, 5, "rule"),
        ("agents", 30, 5, "noop"),
        ("agents", 30, 5, "rule"),
        ("agents", 30, 10, "noop"),
        ("agents", 30, 10, "rule"),
    ]
    with (output / "results.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 8
    assert (output / "events.csv").is_file() and (output / "scalability_info.json").is_file()
    with pytest.raises(FileExistsError):
        run_scalability(synthetic_config, ["rule"], output)


def test_scale_command_runs_one_sweep(synthetic_config: Path, tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(
        ["scale", "--config", str(synthetic_config), "--sweep", "agents", "--output", str(tmp_path / "cli")]
    )

    args.handler(args)

    output = capsys.readouterr().out
    assert "agents" in output and "catalog" not in output.split("Artifacts")[1].split("\n", 2)[2]


def test_top_neighbors_match_the_encoded_first_neighbour() -> None:
    rng = np.random.default_rng(7)
    count = 30
    positions = np.array([7e6, 0, 0]) + rng.normal(0, 3_000, (count, 3))
    velocities = np.array([0, 7_500, 0]) + rng.normal(0, 20, (count, 3))
    state = CatalogState(positions, velocities, np.ones(count), np.zeros(count, bool), np.zeros(count))
    agents = np.arange(5)

    top = top_neighbors(state, agents, 1, SAFETY)[:, 0]
    encoded = encode_local_observations(state, agents, 1, SAFETY)

    relative = positions[top] - positions[agents]
    for row, agent in enumerate(agents):
        basis = rsw_bases(positions[[agent]], velocities[[agent]])[0]
        np.testing.assert_allclose(encoded[row, 7:10], (basis @ relative[row]).astype(np.float32) / 10_000_000.0)


def test_maneuvering_agents_report_their_final_slot_offset() -> None:
    policy = ClohessyWiltshireAvoidancePolicy(LARGE_MANEUVER, SAFETY)

    coasting = simulate(crossing_scene(), np.ones(3), np.array([0]), NoOpPolicy(), settings())
    avoiding = simulate(crossing_scene(), np.ones(3), np.array([0]), policy, settings())

    assert coasting.slot_deviation.distances_m[0] == 0.0
    assert avoiding.slot_deviation.distances_m[0] > 100.0
