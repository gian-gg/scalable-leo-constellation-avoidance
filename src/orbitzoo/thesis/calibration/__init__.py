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
    CandidateScreeningResult,
    CartesianStateFrame,
    CatalogObject,
    CombinationMetrics,
    EncounterWindow,
    EvaluationSplit,
    FineEncounterTrajectory,
    LoadedCatalog,
    ObjectType,
    PooledCombinationMetrics,
    RankedNeighbor,
    RankedNeighborFrame,
    ReferenceConjunction,
    ReferenceConjunctionManifest,
    ThreatDetection,
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
from orbitzoo.thesis.calibration.reference import (
    generate_reference_conjunctions,
    load_reference_conjunctions,
    save_reference_conjunctions,
)
from orbitzoo.thesis.calibration.ranking import (
    DecisionSchedule,
    build_decision_schedule,
    iter_ranked_neighbor_frames,
    rank_neighbors_at_epoch,
)
from orbitzoo.thesis.calibration.evaluation import evaluate_joint_combinations
from orbitzoo.thesis.calibration.metrics import (
    MetricCaseKey,
    aggregate_and_pool_detections,
    aggregate_detection_metrics,
    pool_combination_metrics,
)

__all__ = [
    "AgentSelection",
    "AgentSelectionManifest",
    "CalibrationConfig",
    "CalibrationRecommendation",
    "CandidateScreeningResult",
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
    "PooledCombinationMetrics",
    "PropagationConfig",
    "PropagationError",
    "RankedNeighbor",
    "RankedNeighborFrame",
    "ReferenceConjunction",
    "ReferenceConjunctionManifest",
    "DecisionSchedule",
    "SweepConfig",
    "ThreatDetection",
    "MetricCaseKey",
    "SGP4Propagation",
    "TwoResolutionPropagation",
    "build_decision_schedule",
    "build_two_resolution_propagation",
    "aggregate_and_pool_detections",
    "aggregate_detection_metrics",
    "evaluate_joint_combinations",
    "generate_reference_conjunctions",
    "iter_ranked_neighbor_frames",
    "load_agent_selections",
    "load_catalog",
    "load_reference_conjunctions",
    "propagate_catalog",
    "propagate_objects",
    "pool_combination_metrics",
    "rank_neighbors_at_epoch",
    "save_agent_selections",
    "save_reference_conjunctions",
    "select_agent_populations",
]
