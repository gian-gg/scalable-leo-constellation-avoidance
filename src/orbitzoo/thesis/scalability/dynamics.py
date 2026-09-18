"""Maneuver offsets from each agent's reference trajectory, propagated with Clohessy–Wiltshire."""

from __future__ import annotations

import numpy as np

from orbitzoo.thesis.scalability.observations import rsw_bases

EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14


def mean_motions(positions: np.ndarray) -> np.ndarray:
    """Circular-orbit mean motion for each position, in rad/s."""
    return np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / np.linalg.norm(positions, axis=1) ** 3)


def propagate_hill_states(
    offsets: np.ndarray, rates: np.ndarray, mean_motion: np.ndarray, elapsed_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance RSW offsets and rotating-frame rates with the CW state-transition matrix."""
    x, y, z = offsets.T
    vx, vy, vz = rates.T
    n = mean_motion
    angle = n * elapsed_seconds
    s, c = np.sin(angle), np.cos(angle)
    new_offsets = np.stack(
        (
            (4 - 3 * c) * x + s / n * vx + 2 / n * (1 - c) * vy,
            6 * (s - angle) * x + y - 2 / n * (1 - c) * vx + (4 * s - 3 * angle) / n * vy,
            c * z + s / n * vz,
        ),
        axis=1,
    )
    new_rates = np.stack(
        (
            3 * n * s * x + c * vx + 2 * s * vy,
            -6 * n * (1 - c) * x - 2 * s * vx + (4 * c - 3) * vy,
            -n * s * z + c * vz,
        ),
        axis=1,
    )
    return new_offsets, new_rates


class HillOffsets:
    """Per-agent offset from its reference orbit, held in the reference RSW frame."""

    def __init__(self, agent_count: int) -> None:
        self.offsets = np.zeros((agent_count, 3))
        self.rates = np.zeros((agent_count, 3))

    @property
    def is_zero(self) -> bool:
        return not (self.offsets.any() or self.rates.any())

    def apply_impulse(self, delta_v_rsw: np.ndarray) -> None:
        self.rates = self.rates + delta_v_rsw

    def advanced(self, mean_motion: np.ndarray, elapsed_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        """Offsets and rates after ``elapsed_seconds`` without changing the stored state."""
        if self.is_zero:
            return self.offsets.copy(), self.rates.copy()
        return propagate_hill_states(self.offsets, self.rates, mean_motion, elapsed_seconds)

    def advance(self, mean_motion: np.ndarray, elapsed_seconds: float) -> None:
        self.offsets, self.rates = self.advanced(mean_motion, elapsed_seconds)


def inertial_states(
    reference_positions: np.ndarray,
    reference_velocities: np.ndarray,
    offsets: np.ndarray,
    rates: np.ndarray,
    mean_motion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Add RSW offsets to reference states, including the frame-rotation velocity term."""
    bases = rsw_bases(reference_positions, reference_velocities)
    rotation = np.stack((-mean_motion * offsets[:, 1], mean_motion * offsets[:, 0], np.zeros(len(offsets))), axis=1)
    positions = reference_positions + np.einsum("aji,aj->ai", bases, offsets)
    velocities = reference_velocities + np.einsum("aji,aj->ai", bases, rates + rotation)
    return positions, velocities
