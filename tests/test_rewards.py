import pytest

from orbitzoo.thesis.environments.rewards import RewardConfig, calculate_rewards
from orbitzoo.thesis.environments.safety import PairSafetyAssessment
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import (
    ManeuverConfig,
    build_maneuver_command,
    measure_maneuver_result,
)

MASS_KG = 250.0
ISP_SECONDS = 300.0
MANEUVER = ManeuverConfig(0.5, 10.0, ISP_SECONDS, 60.0)
BURN_COST = 0.5


def assessment(first: str, second: str, *, unsafe: bool = False, collision: bool = False) -> PairSafetyAssessment:
    return PairSafetyAssessment(
        first_name=first,
        second_name=second,
        current_separation_meters=1.0 if collision else 5_000.0,
        combined_radius_meters=2.0,
        time_to_closest_approach_seconds=60.0,
        predicted_miss_distance_meters=100.0 if unsafe or collision else 5_000.0,
        is_collision=collision,
        is_unsafe=unsafe or collision,
    )


def maneuver(action: ManeuverAction):
    command = build_maneuver_command(action, MASS_KG, MANEUVER)
    result = measure_maneuver_result(command, MASS_KG, MASS_KG - command.expected_propellant_kg, ISP_SECONDS)
    return command, result


def reward_for(action: ManeuverAction, before: list, after: list, rejected: set[str] | None = None) -> float:
    command, result = maneuver(action)
    rewards = calculate_rewards(
        ["sat"], {"sat": command}, {"sat": result}, before, after, RewardConfig(), rejected
    )
    return rewards["sat"]


SAFE: list = []
UNSAFE = [assessment("sat", "debris", unsafe=True)]
COLLISION = [assessment("sat", "debris", collision=True)]
NO_OP = ManeuverAction.NO_OP
BURN = ManeuverAction.PROGRADE


@pytest.mark.parametrize(
    ("case", "action", "before", "after", "rejected", "expected"),
    [
        ("safe coast", NO_OP, SAFE, SAFE, None, 0.0),
        ("unnecessary burn", BURN, SAFE, SAFE, None, -BURN_COST - 0.1),
        ("unresolved while coasting", NO_OP, UNSAFE, UNSAFE, None, -10.0),
        ("unresolved despite burn", BURN, UNSAFE, UNSAFE, None, -10.0 - BURN_COST),
        ("resolved by burn", BURN, UNSAFE, SAFE, None, 5.0 - BURN_COST),
        ("collision after coast", NO_OP, UNSAFE, COLLISION, None, -100.0),
        ("collision after burn", BURN, UNSAFE, COLLISION, None, -100.0 - BURN_COST),
        ("rejected action while safe", NO_OP, SAFE, SAFE, {"sat"}, -2.0),
        ("rejected action while unsafe", NO_OP, UNSAFE, UNSAFE, {"sat"}, -12.0),
    ],
)
def test_reward_outcome_table(case, action, before, after, rejected, expected) -> None:
    assert reward_for(action, before, after, rejected) == pytest.approx(expected, abs=1e-9), case


def test_other_pairs_do_not_affect_an_uninvolved_agent() -> None:
    other_pair = [assessment("other", "debris", collision=True)]

    assert reward_for(NO_OP, other_pair, other_pair) == 0.0


def test_resolution_is_individual_per_pair_member() -> None:
    command, result = maneuver(NO_OP)
    names = ["sat", "other"]
    before = [assessment("sat", "debris", unsafe=True)]
    after = [assessment("other", "debris", unsafe=True)]

    rewards = calculate_rewards(
        names, dict.fromkeys(names, command), dict.fromkeys(names, result), before, after, RewardConfig()
    )

    assert rewards == {"sat": 5.0, "other": -10.0}


@pytest.mark.parametrize(
    "overrides",
    [
        {"collision_penalty": 0.0},
        {"unsafe_penalty": 1.0},
        {"resolution_reward": -1.0},
        {"delta_v_penalty_per_mps": -1.0},
        {"unnecessary_maneuver_penalty": 0.1},
        {"infeasible_maneuver_penalty": 1.0},
    ],
)
def test_reward_config_rejects_wrong_signs(overrides) -> None:
    with pytest.raises(ValueError):
        RewardConfig(**overrides).validate()
