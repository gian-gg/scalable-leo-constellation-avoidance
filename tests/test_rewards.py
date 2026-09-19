
import numpy as np
import pytest

from orbitzoo.thesis.environments.observations import NEIGHBOR_FEATURE_DIM, OWN_FEATURE_DIM
from orbitzoo.thesis.environments.rewards import (
    RewardConfig,
    calculate_rewards,
    shortfall,
    threat_potentials,
)
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


def result(action: ManeuverAction):
    command = build_maneuver_command(action, MASS_KG, MANEUVER)
    return measure_maneuver_result(command, MASS_KG, MASS_KG - command.expected_propellant_kg, ISP_SECONDS)


def reward(
    potential_before: float = 0.0,
    potential_after: float = 0.0,
    *,
    action: ManeuverAction = ManeuverAction.NO_OP,
    approaches: dict | None = None,
    rejected: set[str] | None = None,
    collided: set[str] | None = None,
    config: RewardConfig = CONFIG,
) -> float:
    return calculate_rewards(
        ["sat"],
        {"sat": result(action)},
        np.array([potential_before]),
        np.array([potential_after]),
        collided or set(),
        approaches or {},
        SAFE,
        config,
        rejected,
    )["sat"]


def observation(*threats: tuple[float, float, bool]) -> np.ndarray:
    """One agent row with (miss m, time to closest approach s, valid) per neighbour slot."""
    row = np.zeros(OWN_FEATURE_DIM + NEIGHBOR_FEATURE_DIM * len(threats), dtype=np.float32)
    for slot, (miss, tca, valid) in enumerate(threats):
        block = row[OWN_FEATURE_DIM + slot * NEIGHBOR_FEATURE_DIM :]
        block[6], block[7], block[11] = tca / 1_800.0, min(miss / SAFE, 10.0), float(valid)
    return row[np.newaxis]


def test_shortfall_is_zero_outside_and_one_at_contact() -> None:
    assert shortfall(1_500.0, SAFE) == 0.0
    assert shortfall(SAFE, SAFE) == 0.0
    assert shortfall(250.0, SAFE) == 0.75
    assert shortfall(0.0, SAFE) == 1.0


def test_potential_uses_the_worst_valid_approaching_threat() -> None:
    worst = observation((600.0, 300.0, True), (200.0, 0.0, True), (900.0, 300.0, True), (100.0, 300.0, False))

    assert threat_potentials(worst, SAFE)[0] == pytest.approx(-0.4, abs=1e-6)
    assert threat_potentials(observation((5_000.0, 300.0, True)), SAFE)[0] == 0.0
    assert threat_potentials(observation((0.0, 0.0, False)), SAFE)[0] == 0.0


def test_quiet_coasting_earns_nothing() -> None:
    assert reward() == 0.0


def test_fuel_is_charged_per_metre_per_second() -> None:
    assert reward(action=ManeuverAction.PROGRADE) == pytest.approx(-0.5)


def test_widening_a_predicted_miss_is_rewarded_and_narrowing_it_is_penalised() -> None:
    widened = reward(-0.6, -0.3)
    narrowed = reward(-0.3, -0.6)

    assert widened == pytest.approx(10.0 * (0.99 * -0.3 + 0.6))
    assert narrowed < 0 < widened


def test_realized_close_approach_is_charged_by_its_shortfall() -> None:
    assert reward(approaches={("sat", "debris"): 400.0}) == pytest.approx(-6.0)
    assert reward(approaches={("other", "debris"): 10.0}) == 0.0


def test_collision_dominates_and_ends_the_potential() -> None:
    assert reward(-0.9, -1.0, collided={"sat"}) == pytest.approx(-100.0 + 10.0 * 0.9)


def test_rejected_action_is_penalised() -> None:
    assert reward(rejected={"sat"}) == -2.0


def test_shaping_cancels_over_a_cycle_that_returns_to_the_same_state() -> None:
    config = RewardConfig(shaping_discount=1.0)
    potentials = [-0.6, -0.2, -0.6]

    total = sum(reward(before, after, config=config) for before, after in zip(potentials, potentials[1:]))

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
