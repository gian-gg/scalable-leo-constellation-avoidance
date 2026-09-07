"""Offline calibration utilities for selecting thesis environment parameters."""

from orbitzoo.thesis.calibration.config import (
    CalibrationConfig,
    CatalogConfig,
    PassingThresholds,
    PropagationConfig,
    SweepConfig,
)
from orbitzoo.thesis.calibration.catalog import (
    CatalogLoadError,
    load_catalog,
)
from orbitzoo.thesis.calibration.models import (
    AgentSelection,
    CalibrationRecommendation,
    CartesianStateFrame,
    CatalogObject,
    CombinationMetrics,
    EncounterWindow,
    EvaluationSplit,
    FineEncounterTrajectory,
    LoadedCatalog,
    ObjectType,
    RankedNeighbor,
    ReferenceConjunction,
)
from orbitzoo.thesis.calibration.propagation import (
    PropagationError,
    SGP4Propagation,
    propagate_catalog,
    propagate_objects,
)
from orbitzoo.thesis.calibration.two_resolution import (
    TwoResolutionPropagation,
    build_two_resolution_propagation,
)

__all__ = [
    "AgentSelection",
    "CalibrationConfig",
    "CalibrationRecommendation",
    "CatalogLoadError",
    "CatalogObject",
    "CatalogConfig",
    "CartesianStateFrame",
    "CombinationMetrics",
    "EncounterWindow",
    "EvaluationSplit",
    "FineEncounterTrajectory",
    "LoadedCatalog",
    "ObjectType",
    "PassingThresholds",
    "PropagationConfig",
    "PropagationError",
    "RankedNeighbor",
    "ReferenceConjunction",
    "SweepConfig",
    "SGP4Propagation",
    "TwoResolutionPropagation",
    "build_two_resolution_propagation",
    "load_catalog",
    "propagate_catalog",
    "propagate_objects",
]
