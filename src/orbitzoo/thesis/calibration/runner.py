"""End-to-end orchestration and artifacts for offline k/delta-t calibration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Callable

from orbitzoo.thesis.calibration.catalog import load_catalog
from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.evaluation import evaluate_joint_combinations
from orbitzoo.thesis.calibration.metrics import aggregate_and_pool_detections
from orbitzoo.thesis.calibration.models import (
    CalibrationRecommendation,
    CombinationMetrics,
    LoadedCatalog,
    PooledCombinationMetrics,
    ReferenceConjunctionManifest,
)
from orbitzoo.thesis.calibration.ranking import (
    build_decision_schedule,
    iter_ranked_neighbor_frames,
)
from orbitzoo.thesis.calibration.recommendation import (
    NoPassingCombinationError,
    select_calibration_recommendation,
)
from orbitzoo.thesis.calibration.reference import (
    generate_reference_conjunctions,
    save_reference_conjunctions,
)
from orbitzoo.thesis.calibration.selection import (
    save_agent_selections,
    select_agent_populations,
)
from orbitzoo.thesis.calibration.two_resolution import (
    build_two_resolution_propagation,
)


class CalibrationRunStatus(str, Enum):
    """Terminal outcome of a completed calibration run."""

    ACCEPTED = "accepted"
    REJECTED_VALIDATION = "rejected_validation"
    NO_PASSING_CALIBRATION = "no_passing_calibration"


@dataclass(frozen=True)
class CalibrationRunResult:
    """Location and scientific outcome of a completed run."""

    output_directory: Path
    status: CalibrationRunStatus
    recommendation: CalibrationRecommendation | None
    total_runtime_seconds: float
    reference_conjunction_count: int
    candidate_window_count: int


ProgressCallback = Callable[[str], None]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_to_dict(metric: CombinationMetrics) -> dict[str, object]:
    return {
        "agent_count": metric.agent_count,
        "decision_interval_seconds": metric.decision_interval_seconds,
        "detected_conjunction_count": metric.detected_conjunction_count,
        "evaluation_split": metric.evaluation_split.value,
        "missed_conjunction_count": metric.missed_conjunction_count,
        "neighborhood_size": metric.neighborhood_size,
        "reference_conjunction_count": metric.reference_conjunction_count,
        "runtime_seconds": metric.runtime_seconds,
        "selection_seed": metric.selection_seed,
        "threat_recall": metric.threat_recall,
        "timely_detected_conjunction_count": (
            metric.timely_detected_conjunction_count
        ),
        "timely_detection_fraction": metric.timely_detection_fraction,
    }


def _pooled_metric_to_dict(metric: PooledCombinationMetrics) -> dict[str, object]:
    return {
        "decision_interval_seconds": metric.decision_interval_seconds,
        "detected_conjunction_count": metric.detected_conjunction_count,
        "evaluation_split": metric.evaluation_split.value,
        "minimum_threat_recall": metric.minimum_threat_recall,
        "minimum_timely_detection_fraction": (
            metric.minimum_timely_detection_fraction
        ),
        "missed_conjunction_count": metric.missed_conjunction_count,
        "neighborhood_size": metric.neighborhood_size,
        "passed": metric.passed,
        "reference_conjunction_count": metric.reference_conjunction_count,
        "runtime_seconds": metric.runtime_seconds,
        "sample_count": metric.sample_count,
        "threat_recall": metric.threat_recall,
        "timely_detected_conjunction_count": (
            metric.timely_detected_conjunction_count
        ),
        "timely_detection_fraction": metric.timely_detection_fraction,
    }


def _catalog_summary(
    catalog: LoadedCatalog,
    *,
    altitude_filtered_count: int,
    config: CalibrationConfig,
    config_path: Path,
) -> dict[str, object]:
    tle_path, metadata_path = config.resolve_catalog_paths(config_path)
    object_types = Counter(item.object_type.value for item in catalog.objects)
    return {
        "agent_candidate_count": sum(
            item.is_agent_candidate for item in catalog.objects
        ),
        "altitude_filtered_count": altitude_filtered_count,
        "catalog_epoch_utc": catalog.latest_epoch_utc.isoformat(),
        "metadata_path": str(metadata_path),
        "object_type_counts": dict(sorted(object_types.items())),
        "retained_record_count": len(catalog.objects),
        "schema_version": 1,
        "source_record_count": catalog.source_record_count,
        "stale_filtered_count": len(catalog.stale_filtered_norad_ids),
        "stale_filtered_norad_ids": list(catalog.stale_filtered_norad_ids),
        "tle_path": str(tle_path),
    }


def _recommendation_payload(
    *,
    status: CalibrationRunStatus,
    recommendation: CalibrationRecommendation | None,
    total_runtime_seconds: float,
    evaluation_runtime_seconds: float,
    catalog_epoch: str,
    error: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "catalog_epoch_utc": catalog_epoch,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
        "runtime_accounting": "shared across all k/delta-t combinations",
        "schema_version": 1,
        "status": status.value,
        "total_runtime_seconds": total_runtime_seconds,
    }
    if error is not None:
        payload["reason"] = error
    if recommendation is not None:
        payload.update(
            {
                "calibration_metrics": [
                    _metric_to_dict(item)
                    for item in recommendation.calibration_metrics
                ],
                "calibration_passed": recommendation.calibration_passed,
                "calibration_threat_recall": (
                    recommendation.calibration_threat_recall
                ),
                "calibration_timely_detection_fraction": (
                    recommendation.calibration_timely_detection_fraction
                ),
                "decision_interval_seconds": (
                    recommendation.decision_interval_seconds
                ),
                "minimum_decisions_before_tca": (
                    recommendation.minimum_decisions_before_tca
                ),
                "minimum_threat_recall": recommendation.minimum_threat_recall,
                "minimum_timely_detection_fraction": (
                    recommendation.minimum_timely_detection_fraction
                ),
                "neighborhood_size": recommendation.neighborhood_size,
                "validation_metrics": [
                    _metric_to_dict(item)
                    for item in recommendation.validation_metrics
                ],
                "validation_passed": recommendation.validation_passed,
                "validation_threat_recall": recommendation.validation_threat_recall,
                "validation_timely_detection_fraction": (
                    recommendation.validation_timely_detection_fraction
                ),
            }
        )
    return payload


def _summary_text(
    *,
    status: CalibrationRunStatus,
    recommendation: CalibrationRecommendation | None,
    catalog: LoadedCatalog,
    altitude_filtered_count: int,
    selected_agent_count: int,
    candidate_window_count: int,
    references: ReferenceConjunctionManifest,
    total_runtime_seconds: float,
    reason: str | None,
) -> str:
    lines = [
        "OrbitZoo k/delta-t calibration",
        f"Status: {status.value}",
        f"Catalog epoch UTC: {catalog.latest_epoch_utc.isoformat()}",
        f"Catalog objects retained: {len(catalog.objects)}",
        f"Catalog objects propagated: {altitude_filtered_count}",
        f"Unique selected agents: {selected_agent_count}",
        f"Candidate encounter windows: {candidate_window_count}",
        f"Reference conjunctions: {len(references.conjunctions)}",
        f"Reference collisions: {references.collision_count}",
        f"Total runtime seconds: {total_runtime_seconds:.6f}",
    ]
    if recommendation is not None:
        lines.extend(
            [
                f"Recommended k: {recommendation.neighborhood_size}",
                "Recommended delta-t seconds: "
                f"{recommendation.decision_interval_seconds}",
                "Calibration recall: "
                f"{recommendation.calibration_threat_recall:.6f}",
                "Calibration timely fraction: "
                f"{recommendation.calibration_timely_detection_fraction:.6f}",
                f"Validation recall: {recommendation.validation_threat_recall:.6f}",
                "Validation timely fraction: "
                f"{recommendation.validation_timely_detection_fraction:.6f}",
            ]
        )
    if reason is not None:
        lines.append(f"Reason: {reason}")
    return "\n".join(lines) + "\n"


def run_calibration(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> CalibrationRunResult:
    """Run every calibration phase once and atomically publish its artifacts."""
    source = Path(config_path).expanduser().resolve()
    requested_destination = Path(output_directory).expanduser()
    if requested_destination.exists() or requested_destination.is_symlink():
        raise FileExistsError(
            f"output directory already exists: {requested_destination.resolve()}"
        )
    destination = requested_destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda _message: None)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    started = perf_counter()

    try:
        report("[1/8] Loading configuration and catalog")
        config = CalibrationConfig.load(source)
        config.save(temporary / "resolved_config.json")
        catalog = load_catalog(config, source)

        report("[2/8] Propagating coarse catalog trajectory")
        propagation = build_two_resolution_propagation(catalog, config)
        propagated_count = len(propagation.coarse_propagation.objects)
        _write_json(
            temporary / "catalog_summary.json",
            _catalog_summary(
                catalog,
                altitude_filtered_count=propagated_count,
                config=config,
                config_path=source,
            ),
        )

        report("[3/8] Selecting deterministic agent populations")
        selections = select_agent_populations(
            propagation.coarse_propagation,
            config,
        )
        save_agent_selections(selections, temporary / "agent_selections.json")

        report("[4/8] Screening agent-to-catalog encounter windows")
        screening = propagation.screen_agent_selections(selections)

        report("[5/8] Fine-propagating candidate windows for reference truth")
        references = generate_reference_conjunctions(
            propagation.iter_fine_trajectories(screening.encounter_windows),
            catalog,
            config,
            screened_agent_norad_ids=screening.screened_agent_norad_ids,
        )
        save_reference_conjunctions(
            references,
            temporary / "reference_conjunctions.json",
        )

        report("[6/8] Ranking neighbors once at shared decision epochs")
        schedule = build_decision_schedule(
            propagation.coarse_propagation.start_epoch_utc,
            config,
        )
        ranking_frames = iter_ranked_neighbor_frames(
            propagation.coarse_propagation,
            config,
            agent_norad_ids=screening.screened_agent_norad_ids,
        )

        report("[7/8] Evaluating and aggregating all k/delta-t combinations")
        evaluation_started = perf_counter()
        detections = evaluate_joint_combinations(
            ranking_frames,
            references,
            selections,
            config,
            schedule=schedule,
        )
        metrics, pooled_metrics = aggregate_and_pool_detections(detections, config)
        evaluation_runtime = perf_counter() - evaluation_started
        _write_json(
            temporary / "combination_metrics.json",
            {
                "metrics": [_metric_to_dict(item) for item in metrics],
                "schema_version": 1,
            },
        )
        _write_json(
            temporary / "pooled_metrics.json",
            {
                "metrics": [
                    _pooled_metric_to_dict(item) for item in pooled_metrics
                ],
                "schema_version": 1,
            },
        )

        report("[8/8] Selecting recommendation and auditing validation")
        recommendation: CalibrationRecommendation | None
        reason: str | None = None
        try:
            recommendation = select_calibration_recommendation(
                metrics,
                pooled_metrics,
                config,
            )
        except NoPassingCombinationError as error:
            recommendation = None
            reason = str(error)
            status = CalibrationRunStatus.NO_PASSING_CALIBRATION
        else:
            status = (
                CalibrationRunStatus.ACCEPTED
                if recommendation.is_accepted
                else CalibrationRunStatus.REJECTED_VALIDATION
            )
            if not recommendation.validation_passed:
                reason = "selected calibration combination failed held-out validation"

        total_runtime = perf_counter() - started
        catalog_epoch = catalog.latest_epoch_utc.isoformat()
        _write_json(
            temporary / "recommendation.json",
            _recommendation_payload(
                status=status,
                recommendation=recommendation,
                total_runtime_seconds=total_runtime,
                evaluation_runtime_seconds=evaluation_runtime,
                catalog_epoch=catalog_epoch,
                error=reason,
            ),
        )
        (temporary / "summary.txt").write_text(
            _summary_text(
                status=status,
                recommendation=recommendation,
                catalog=catalog,
                altitude_filtered_count=propagated_count,
                selected_agent_count=len(screening.screened_agent_norad_ids),
                candidate_window_count=len(screening.encounter_windows),
                references=references,
                total_runtime_seconds=total_runtime,
                reason=reason,
            ),
            encoding="utf-8",
        )
        temporary.rename(destination)
        report(f"Complete: {status.value} ({total_runtime:.2f} seconds)")
        return CalibrationRunResult(
            output_directory=destination,
            status=status,
            recommendation=recommendation,
            total_runtime_seconds=total_runtime,
            reference_conjunction_count=len(references.conjunctions),
            candidate_window_count=len(screening.encounter_windows),
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
