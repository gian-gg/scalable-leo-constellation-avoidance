"""Orekit helpers that match CollisionAvoidanceEnv's physics without OrbitZoo's covariance setup."""

from __future__ import annotations

from datetime import datetime

import numpy as np

import orbitzoo.env  # noqa: F401  (boots the JVM and loads Orekit data)
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate
from org.orekit.utils import PVCoordinates

from orbitzoo.dynamics.constants import EARTH_FLATTENING, EARTH_RADIUS, INERTIAL_FRAME, ITRF, MU
from orbitzoo.dynamics.constants import UTC

GRAVITY_DEGREE = 8
POSITION_TOLERANCE_METERS = 60.0

_earth = OneAxisEllipsoid(EARTH_RADIUS, EARTH_FLATTENING, ITRF)
_gravity = HolmesFeatherstoneAttractionModel(
    _earth.getBodyFrame(), GravityFieldFactory.getNormalizedProvider(GRAVITY_DEGREE, GRAVITY_DEGREE)
)


def absolute_date(epoch: datetime) -> AbsoluteDate:
    return AbsoluteDate(
        epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6, UTC
    )


def teme_to_inertial(positions: np.ndarray, velocities: np.ndarray, epoch: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Convert SGP4 TEME states to the EME2000 frame OrbitZoo propagates in."""
    transform = FramesFactory.getTEME().getTransformTo(INERTIAL_FRAME, absolute_date(epoch))
    converted = [
        transform.transformPVCoordinates(PVCoordinates(Vector3D(*map(float, position)), Vector3D(*map(float, velocity))))
        for position, velocity in zip(positions, velocities)
    ]
    return (
        np.array([[pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ()] for pv in converted]),
        np.array([[pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()] for pv in converted]),
    )


SUPPORTED_FORCES = ("gravity_newton", "gravity_hf")


def propagate_from(
    position: np.ndarray,
    velocity: np.ndarray,
    date: AbsoluteDate,
    seconds: float,
    forces: tuple[str, ...] = ("gravity_hf",),
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a coasting state by ``seconds`` (negative for backwards) with the environment's gravity model."""
    unsupported = set(forces) - set(SUPPORTED_FORCES)
    if unsupported:
        raise ValueError(f"unsupported forces {sorted(unsupported)}; supported: {SUPPORTED_FORCES}")
    orbit = CartesianOrbit(
        PVCoordinates(Vector3D(*map(float, position)), Vector3D(*map(float, velocity))), INERTIAL_FRAME, date, MU
    )
    tolerances = NumericalPropagator.tolerances(POSITION_TOLERANCE_METERS, orbit, OrbitType.CARTESIAN)
    integrator = DormandPrince853Integrator(1e-3, 500.0, tolerances[0], tolerances[1])
    integrator.setInitialStepSize(10.0)
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setMu(MU)
    if "gravity_hf" in forces:
        propagator.addForceModel(_gravity)
    propagator.setInitialState(SpacecraftState(orbit))
    pv = propagator.propagate(date.shiftedBy(float(seconds))).getPVCoordinates()
    return (
        np.array([pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ()]),
        np.array([pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()]),
    )


def propagate(
    position: np.ndarray,
    velocity: np.ndarray,
    epoch: datetime,
    seconds: float,
    forces: tuple[str, ...] = ("gravity_hf",),
) -> tuple[np.ndarray, np.ndarray]:
    """``propagate_from`` with a UTC datetime."""
    return propagate_from(position, velocity, absolute_date(epoch), seconds, forces)
