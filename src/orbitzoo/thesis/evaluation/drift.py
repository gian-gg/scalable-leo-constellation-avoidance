"""How far maneuvers moved each satellite from its nominal slot, and what a return would cost.

Returning is out of scope for the policy; these quantities are only measured. See docs/EVALUATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitzoo.thesis.environments.vectorized_observations import rsw_bases
from orbitzoo.thesis.scalability.dynamics import mean_motions, propagate_hill_states

RETURN_TRANSFER_PERIODS = np.linspace(0.05, 2.0, 40)
RETURN_WAIT_PERIODS = np.linspace(0.0, 1.0, 13)
SINGULAR_CONDITION = 1e8


@dataclass(frozen=True)
class SlotDeviation:
    """Offsets from the nominal slot in its RSW frame, their rotating-frame rates, and the mean motion."""

    offsets_m: np.ndarray
    rates_mps: np.ndarray
    mean_motion: np.ndarray

    @property
    def distances_m(self) -> np.ndarray:
        return np.linalg.norm(self.offsets_m, axis=1)


def slot_deviation(
    positions: np.ndarray, velocities: np.ndarray, nominal_positions: np.ndarray, nominal_velocities: np.ndarray
) -> SlotDeviation:
    """Hill-frame state of each satellite relative to where it would be had it never maneuvered."""
    bases = rsw_bases(nominal_positions, nominal_velocities)
    motion = mean_motions(nominal_positions)
    offsets = np.einsum("aij,aj->ai", bases, positions - nominal_positions)
    rotation = np.stack((-motion * offsets[:, 1], motion * offsets[:, 0], np.zeros(len(offsets))), axis=1)
    rates = np.einsum("aij,aj->ai", bases, velocities - nominal_velocities) - rotation
    return SlotDeviation(offsets, rates, motion)


def _transition(motion: np.ndarray, elapsed_seconds: np.ndarray) -> np.ndarray:
    """Clohessy–Wiltshire state-transition matrix per satellite, shape [n, 6, 6]."""
    columns = []
    for unit in np.eye(6):
        offsets = np.tile(unit[:3], (motion.size, 1))
        rates = np.tile(unit[3:], (motion.size, 1))
        position, velocity = propagate_hill_states(offsets, rates, motion, elapsed_seconds)
        columns.append(np.concatenate((position, velocity), axis=1))
    return np.stack(columns, axis=2)


def _two_burn_delta_v(offsets: np.ndarray, rates: np.ndarray, motion: np.ndarray, transfer_periods: np.ndarray) -> np.ndarray:
    """Cheapest two-burn return starting now, over the given transfer times."""
    displaced = (np.abs(offsets).sum(axis=1) + np.abs(rates).sum(axis=1)) > 0
    best = np.where(displaced, np.inf, 0.0)
    for fraction in transfer_periods:
        transitions = _transition(motion, fraction * 2 * np.pi / motion)
        position_block, rate_block = transitions[:, :3, :3], transitions[:, :3, 3:]
        solvable = displaced & (np.linalg.cond(rate_block) < SINGULAR_CONDITION)
        if not solvable.any():
            continue
        departure = np.zeros_like(rates)
        departure[solvable] = -np.linalg.solve(
            rate_block[solvable], np.einsum("aij,aj->ai", position_block[solvable], offsets[solvable])[..., None]
        )[..., 0]
        arrival = np.einsum("aij,aj->ai", transitions[:, 3:, :3], offsets) + np.einsum(
            "aij,aj->ai", transitions[:, 3:, 3:], departure
        )
        total = np.linalg.norm(departure - rates, axis=1) + np.linalg.norm(arrival, axis=1)
        best = np.where(solvable & (total < best), total, best)
    return best


def return_delta_v(
    deviation: SlotDeviation,
    transfer_periods: np.ndarray = RETURN_TRANSFER_PERIODS,
    wait_periods: np.ndarray = RETURN_WAIT_PERIODS,
) -> np.ndarray:
    """Cheapest two-burn return to the slot, allowing a coast of up to ``wait_periods`` before the first burn."""
    motion = deviation.mean_motion
    best = np.full(motion.size, np.inf)
    for wait in wait_periods:
        offsets, rates = propagate_hill_states(deviation.offsets_m, deviation.rates_mps, motion, wait * 2 * np.pi / motion)
        best = np.minimum(best, _two_burn_delta_v(offsets, rates, motion, transfer_periods))
    return best
