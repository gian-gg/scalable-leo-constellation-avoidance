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
    AgentSelectionManifest,
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
from orbitzoo.thesis.calibration.selection import (
    load_agent_selections,
    save_agent_selections,
    select_agent_populations,
)

__all__ = [
    "AgentSelection",
    "AgentSelectionManifest",
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
    "load_agent_selections",
    "propagate_catalog",
    "propagate_objects",
    "save_agent_selections",
    "select_agent_populations",
]
