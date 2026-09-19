import math

import numpy as np
import pytest

from orbitzoo.thesis.evaluation.drift import SlotDeviation, return_delta_v, slot_deviation
from orbitzoo.thesis.scalability.dynamics import inertial_states, mean_motions, propagate_hill_states

RADIUS = 6_878_136.3
MOTION = math.sqrt(3.986004418e14 / RADIUS**3)
NOMINAL = (np.array([[RADIUS, 0.0, 0.0]]), np.array([[0.0, RADIUS * MOTION, 0.0]]))


def after_burn(delta_v_rsw: list[float], elapsed: float = 3_000.0) -> SlotDeviation:
    offsets, rates = propagate_hill_states(np.zeros((1, 3)), np.array([delta_v_rsw]), np.array([MOTION]), elapsed)
    return SlotDeviation(offsets, rates, np.array([MOTION]))


def test_deviation_recovers_the_hill_state_used_to_offset_a_satellite() -> None:
    offsets, rates = np.array([[120.0, -3_000.0, 40.0]]), np.array([[0.2, -1.5, 0.05]])
    positions, velocities = inertial_states(*NOMINAL, offsets, rates, mean_motions(NOMINAL[0]))

    deviation = slot_deviation(positions, velocities, *NOMINAL)

    np.testing.assert_allclose(deviation.offsets_m, offsets, atol=1e-6)
    np.testing.assert_allclose(deviation.rates_mps, rates, atol=1e-9)


def test_a_satellite_on_its_slot_needs_no_return() -> None:
    assert return_delta_v(SlotDeviation(np.zeros((1, 3)), np.zeros((1, 3)), np.array([MOTION])))[0] == 0.0


def test_return_cost_scales_with_the_deviation() -> None:
    single = return_delta_v(after_burn([0.0, 0.5, 0.0]))[0]
    double = return_delta_v(after_burn([0.0, 1.0, 0.0]))[0]

    assert double == pytest.approx(2 * single, rel=1e-9)


@pytest.mark.parametrize(
    "burn, low, high",
    [([0.0, 0.5, 0.0], 0.5, 1.5), ([0.5, 0.0, 0.0], 0.1, 0.5), ([0.0, 0.0, 0.5], 0.45, 0.55)],
)
def test_returning_from_one_burn_costs_a_plausible_amount(burn, low, high) -> None:
    assert low <= return_delta_v(after_burn(burn))[0] <= high


def test_waiting_before_returning_is_never_more_expensive() -> None:
    deviation = after_burn([0.0, 0.5, 0.0])

    immediate = return_delta_v(deviation, wait_periods=np.array([0.0]))[0]

    assert return_delta_v(deviation)[0] <= immediate
