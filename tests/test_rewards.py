import pytest

from orbitzoo.thesis.environments.rewards import (
    RewardConfig,
    calculate_rewards,
    shortfall,
    threat_potential,
)
from orbitzoo.thesis.environments.safety import PairSafetyAssessment
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import (
    ManeuverConfig,
    build_maneuver_command,
    measure_maneuver_result,
)

MASS_KG = 250.0
ISP_SECONDS = 300.0
MANEUVER = ManeuverConfig(0.5, 7.0, ISP_SECONDS, 60.0)
SAFE = 1_000.0
CONFIG = RewardConfig()


def assessment(
    first: str = "sat",
    second: str = "debris",
    *,
    miss: float = 5_000.0,
    tca: float = 600.0,
    collision: bool = False,
) -> PairSafetyAssessment:
    return PairSafetyAssessment(
        first_name=first,
        second_name=second,
        current_separation_meters=1.0 if collision else 20_000.0,
        combined_radius_meters=2.0,
        time_to_closest_approach_seconds=tca,
        predicted_miss_distance_meters=miss,
        is_collision=collision,
        is_unsafe=miss <= SAFE,
    )


def result(action: ManeuverAction):
    command = build_maneuver_command(action, MASS_KG, MANEUVER)
    return measure_maneuver_result(command, MASS_KG, MASS_KG - command.expected_propellant_kg, ISP_SECONDS)


def reward(
    before: list,
    after: list,
    *,
    action: ManeuverAction = ManeuverAction.NO_OP,
    approaches: dict | None = None,
    rejected: set[str] | None = None,
) -> float:
    return calculate_rewards(
        ["sat"], {"sat": result(action)}, before, after, approaches or {}, SAFE, CONFIG, rejected
    )["sat"]


def test_shortfall_is_zero_outside_and_one_at_contact() -> None:
    assert shortfall(1_500.0, SAFE) == 0.0
    assert shortfall(SAFE, SAFE) == 0.0
    assert shortfall(250.0, SAFE) == 0.75
    assert shortfall(0.0, SAFE) == 1.0


def test_potential_uses_the_worst_approaching_threat_only() -> None:
    assessments = [
        assessment(miss=600.0),
        assessment(second="other", miss=200.0, tca=0.0),
        assessment(second="third", miss=900.0),
    ]

    assert threat_potential("sat", assessments, SAFE) == pytest.approx(-0.4)
    assert threat_potential("nobody", assessments, SAFE) == 0.0


def test_quiet_coasting_earns_nothing() -> None:
    assert reward([assessment()], [assessment()]) == 0.0


def test_fuel_is_charged_per_metre_per_second() -> None:
    assert reward([assessment()], [assessment()], action=ManeuverAction.PROGRADE) == pytest.approx(-0.5)


def test_widening_a_predicted_miss_is_rewarded_and_narrowing_it_is_penalised() -> None:
    widened = reward([assessment(miss=400.0)], [assessment(miss=700.0)])
    narrowed = reward([assessment(miss=700.0)], [assessment(miss=400.0)])

    assert widened == pytest.approx(10.0 * (0.99 * -0.3 + 0.6))
    assert narrowed < 0 < widened


def test_realized_close_approach_is_charged_by_its_shortfall() -> None:
    passed = reward([assessment(miss=400.0)], [], approaches={("sat", "debris"): 400.0})

    assert passed == pytest.approx(-10.0 * 0.6 + 10.0 * 0.6)
    assert reward([], [], approaches={("sat", "debris"): 400.0}) == pytest.approx(-6.0)
    assert reward([], [], approaches={("other", "debris"): 10.0}) == 0.0


def test_collision_dominates_and_ends_the_potential() -> None:
    collided = reward([assessment(miss=100.0)], [assessment(miss=0.0, tca=0.0, collision=True)])

    assert collided == pytest.approx(-100.0 + 10.0 * 0.9)


def test_rejected_action_is_penalised() -> None:
    assert reward([assessment()], [assessment()], rejected={"sat"}) == -2.0


def test_shaping_cancels_over_a_cycle_that_returns_to_the_same_state() -> None:
    config = RewardConfig(shaping_discount=1.0)
    states = [[assessment(miss=400.0)], [assessment(miss=800.0)], [assessment(miss=400.0)]]

    total = sum(
        calculate_rewards(["sat"], {"sat": result(ManeuverAction.NO_OP)}, before, after, {}, SAFE, config)["sat"]
        for before, after in zip(states, states[1:])
    )

    assert total == pytest.approx(0.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"collision_penalty": 0.0},
        {"close_approach_penalty": 1.0},
        {"shaping_weight": -1.0},
        {"shaping_discount": 0.0},
        {"delta_v_penalty_per_mps": -1.0},
        {"infeasible_maneuver_penalty": 1.0},
    ],
)
def test_reward_config_rejects_wrong_signs(overrides) -> None:
    with pytest.raises(ValueError):
        RewardConfig(**overrides).validate()
