from types import SimpleNamespace

import numpy as np

from orbitzoo.thesis.environments.safety import (
    SafetyConfig,
    assess_all_pairs,
    assess_pair,
    close_approaches_since,
    involved_agents,
    recent_closest_approach,
)


def body(name: str, position: list[float], velocity: list[float], radius: float = 1.0):
    return SimpleNamespace(name=name, position=np.array(position), velocity=np.array(velocity), radius=radius)


def test_linear_screen_detects_future_unsafe_conjunction() -> None:
    assessment = assess_pair(
        body("satellite", [0, 0, 0], [0, 0, 0]),
        body("debris", [2_000, 0, 0], [-10, 0, 0]),
        SafetyConfig(safe_separation_meters=1_000, screening_horizon_seconds=300),
    )
    assert assessment.current_separation_meters == 2_000
    assert assessment.time_to_closest_approach_seconds == 200
    assert assessment.predicted_miss_distance_meters == 0
    assert assessment.is_unsafe
    assert not assessment.is_collision


def test_recent_closest_approach_is_found_only_inside_the_lookback() -> None:
    config = SafetyConfig(safe_separation_meters=1_000)
    passed = (body("satellite", [0, 0, 0], [0, 0, 0]), body("debris", [300, 500, 0], [0, 10, 0]))
    approaching = (body("satellite", [0, 0, 0], [0, 0, 0]), body("debris", [300, -500, 0], [0, 10, 0]))
    long_ago = (body("satellite", [0, 0, 0], [0, 0, 0]), body("debris", [300, 5_000, 0], [0, 10, 0]))

    assert recent_closest_approach(*passed, lookback_seconds=120) == 300
    assert recent_closest_approach(*approaching, lookback_seconds=120) is None
    assert recent_closest_approach(*long_ago, lookback_seconds=120) is None
    assert close_approaches_since(passed, 120, config) == {("satellite", "debris"): 300}
    assert close_approaches_since(passed, 120, SafetyConfig(safe_separation_meters=200)) == {}


def test_receding_pair_has_zero_tca_and_current_miss_distance() -> None:
    assessment = assess_pair(
        body("satellite", [0, 0, 0], [0, 0, 0]),
        body("debris", [2_000, 0, 0], [10, 0, 0]),
        SafetyConfig(safe_separation_meters=1_000, screening_horizon_seconds=300),
    )
    assert assessment.time_to_closest_approach_seconds == 0
    assert assessment.predicted_miss_distance_meters == 2_000
    assert not assessment.is_unsafe


def test_tca_beyond_horizon_is_clamped_to_horizon() -> None:
    assessment = assess_pair(
        body("satellite", [0, 0, 0], [0, 0, 0]),
        body("debris", [10_000, 0, 0], [-10, 0, 0]),
        SafetyConfig(safe_separation_meters=1_000, screening_horizon_seconds=300),
    )
    assert assessment.time_to_closest_approach_seconds == 300
    assert assessment.predicted_miss_distance_meters == 7_000
    assert not assessment.is_unsafe


def test_zero_relative_velocity_uses_current_separation() -> None:
    assessment = assess_pair(
        body("satellite", [0, 0, 0], [7_500, 0, 0]),
        body("debris", [500, 0, 0], [7_500, 0, 0]),
        SafetyConfig(),
    )
    assert assessment.time_to_closest_approach_seconds == 0
    assert assessment.predicted_miss_distance_meters == 500
    assert assessment.is_unsafe


def test_threshold_boundaries_are_inclusive() -> None:
    config = SafetyConfig(safe_separation_meters=1_000)
    at_safe_threshold = assess_pair(body("a", [0, 0, 0], [0, 0, 0]), body("b", [1_000, 0, 0], [0, 0, 0]), config)
    at_combined_radius = assess_pair(
        body("a", [0, 0, 0], [0, 0, 0], radius=1.5), body("b", [3, 0, 0], [0, 0, 0], radius=1.5), config
    )
    assert at_safe_threshold.is_unsafe
    assert not at_safe_threshold.is_collision
    assert at_combined_radius.is_collision


def test_all_pairs_are_unique_and_stably_ordered() -> None:
    bodies = [body(name, [index * 5_000, 0, 0], [0, 0, 0]) for index, name in enumerate("abcd")]

    pairs = [assessment.pair for assessment in assess_all_pairs(bodies, SafetyConfig())]

    assert pairs == [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]


def test_involved_agents_returns_both_members_of_flagged_pairs() -> None:
    bodies = [
        body("a", [0, 0, 0], [0, 0, 0]),
        body("b", [500, 0, 0], [0, 0, 0]),
        body("c", [50_000, 0, 0], [0, 0, 0]),
    ]

    unsafe = involved_agents(assess_all_pairs(bodies, SafetyConfig()), "is_unsafe")

    assert unsafe == {"a", "b"}
