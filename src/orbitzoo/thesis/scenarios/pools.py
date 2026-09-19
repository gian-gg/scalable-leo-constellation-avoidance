"""Real satellites, background objects, and close-call shapes, each split into train and test."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sgp4.earth_gravity import wgs72

from orbitzoo.thesis.calibration.catalog import load_catalog
from orbitzoo.thesis.calibration.config import CalibrationConfig, CatalogConfig
from orbitzoo.thesis.calibration.models import CatalogObject, ObjectType
from orbitzoo.thesis.environments.vectorized_observations import rsw_bases
from orbitzoo.thesis.maneuvers.sizing_study import SizingConfig, load_encounters
from orbitzoo.thesis.scalability.simulator import SGP4Trajectory
from orbitzoo.thesis.scenarios.config import ScenarioGeneratorConfig

HASH_MULTIPLIER = 2_654_435_761


@dataclass(frozen=True)
class CloseCallShape:
    """A real conjunction's geometry at closest approach, in the maneuvering satellite's RSW frame."""

    event_id: int
    miss_rsw_m: np.ndarray
    relative_velocity_rsw_mps: np.ndarray
    threat_radius_m: float


@dataclass(frozen=True)
class ScenarioPools:
    """Everything a scenario draws from, restricted to one split."""

    base_epoch_utc: datetime
    agents: tuple[CatalogObject, ...]
    background: tuple[CatalogObject, ...]
    shapes: tuple[CloseCallShape, ...]


def is_held_out(identifier: int, held_out_fraction: float) -> bool:
    """Deterministic split by identifier, independent of any seed."""
    return (identifier * HASH_MULTIPLIER) % 2**32 < held_out_fraction * 2**32


def _in_split(identifier: int, config: ScenarioGeneratorConfig) -> bool:
    return is_held_out(identifier, config.held_out_fraction) == (config.split == "test")


def _catalog(config: ScenarioGeneratorConfig) -> tuple[tuple[CatalogObject, ...], datetime]:
    catalog_config = CatalogConfig(
        tle_path=str(Path(config.tle_path).resolve()),
        metadata_path=str(Path(config.metadata_path).resolve()),
        minimum_altitude_meters=config.minimum_altitude_meters,
        maximum_altitude_meters=config.maximum_altitude_meters,
        maximum_tle_age_days=config.maximum_tle_age_days,
    )
    catalog = load_catalog(CalibrationConfig(catalog=catalog_config), Path.cwd() / "scenario.json")
    objects = tuple(sorted(catalog.objects, key=lambda item: item.norad_id))
    positions, _ = SGP4Trajectory(objects, catalog.latest_epoch_utc).states(np.zeros(1))
    altitudes = np.linalg.norm(positions[:, 0], axis=1) - wgs72.radiusearthkm * 1_000.0
    in_band = (altitudes >= config.minimum_altitude_meters) & (altitudes <= config.maximum_altitude_meters)
    return tuple(item for item, keep in zip(objects, in_band) if keep), catalog.latest_epoch_utc


def _shapes(config: ScenarioGeneratorConfig) -> tuple[CloseCallShape, ...]:
    sizing_path = Path(config.sizing_config).resolve()
    encounters, objects, _, _, _ = load_encounters(SizingConfig.load(sizing_path), sizing_path)
    shapes = []
    for encounter in encounters:
        basis = rsw_bases(encounter.agent_position_m[np.newaxis], encounter.agent_velocity_mps[np.newaxis])[0]
        shapes.append(
            CloseCallShape(
                event_id=encounter.event_id,
                miss_rsw_m=basis @ encounter.miss_vector_m,
                relative_velocity_rsw_mps=basis @ encounter.relative_velocity_mps,
                threat_radius_m=objects[encounter.threat_norad_id].radius_meters,
            )
        )
    return tuple(shapes)


def load_pools(config: ScenarioGeneratorConfig) -> ScenarioPools:
    """Load the catalog and close-call shapes and keep only this config's split."""
    config.validate()
    objects, epoch = _catalog(config)
    agents = tuple(item for item in objects if item.is_agent_candidate and _in_split(item.norad_id, config))
    background = tuple(
        item for item in objects if item.object_type is not ObjectType.PAYLOAD and _in_split(item.norad_id, config)
    )
    shapes = tuple(shape for shape in _shapes(config) if _in_split(shape.event_id, config))
    if not agents or not shapes:
        raise ValueError(f"the {config.split} split has no agent satellites or no close-call shapes")
    return ScenarioPools(epoch, agents, background, shapes)
