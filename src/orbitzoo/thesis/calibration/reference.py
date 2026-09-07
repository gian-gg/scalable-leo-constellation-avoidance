"""Fine-resolution reference conjunction extraction and persistence."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import (
    FineEncounterTrajectory,
    LoadedCatalog,
    ReferenceConjunction,
    ReferenceConjunctionManifest,
)


def _trajectory_reference(
    trajectory: FineEncounterTrajectory,
    combined_radius_meters: float,
) -> ReferenceConjunction:
    """Find a bounded linear TCA inside every fine-state interval."""
    candidates: list[tuple[float, datetime, float]] = []
    for frame, next_frame in zip(trajectory.frames, trajectory.frames[1:]):
        interval_seconds = (next_frame.epoch_utc - frame.epoch_utc).total_seconds()
        relative_position = frame.positions_m[1] - frame.positions_m[0]
        relative_velocity = frame.velocities_mps[1] - frame.velocities_mps[0]
        relative_speed_squared = float(np.dot(relative_velocity, relative_velocity))
        if relative_speed_squared <= 1e-12:
            offset_seconds = 0.0
        else:
            offset_seconds = float(
                np.clip(
                    -np.dot(relative_position, relative_velocity)
                    / relative_speed_squared,
                    0.0,
                    interval_seconds,
                )
            )
        miss_distance_meters = float(
            np.linalg.norm(
                relative_position + relative_velocity * offset_seconds
            )
        )
        candidates.append(
            (
                miss_distance_meters,
                frame.epoch_utc + timedelta(seconds=offset_seconds),
                float(np.sqrt(relative_speed_squared)),
            )
        )

    final_frame = trajectory.frames[-1]
    final_relative_position = final_frame.positions_m[1] - final_frame.positions_m[0]
    final_relative_velocity = final_frame.velocities_mps[1] - final_frame.velocities_mps[0]
    candidates.append(
        (
            float(np.linalg.norm(final_relative_position)),
            final_frame.epoch_utc,
            float(np.linalg.norm(final_relative_velocity)),
        )
    )
    miss_distance_meters, tca_epoch_utc, relative_speed_mps = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    return ReferenceConjunction(
        first_norad_id=trajectory.window.first_norad_id,
        second_norad_id=trajectory.window.second_norad_id,
        tca_epoch_utc=tca_epoch_utc,
        miss_distance_meters=miss_distance_meters,
        relative_speed_mps=relative_speed_mps,
        combined_radius_meters=combined_radius_meters,
    )


def generate_reference_conjunctions(
    trajectories: Iterable[FineEncounterTrajectory],
    catalog: LoadedCatalog,
    config: CalibrationConfig,
    *,
    screened_agent_norad_ids: Iterable[int],
) -> ReferenceConjunctionManifest:
    """Convert fine candidate trajectories into unsafe conjunction truth."""
    config.validate()
    agent_ids = tuple(sorted(screened_agent_norad_ids))
    if not agent_ids:
        raise ValueError("at least one screened agent NORAD ID is required")
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("screened agent NORAD IDs must be unique")
    object_by_id = {item.norad_id: item for item in catalog.objects}
    absent_agent_ids = set(agent_ids).difference(object_by_id)
    if absent_agent_ids:
        raise ValueError(
            "screened agents are absent from the catalog: "
            f"{sorted(absent_agent_ids)}"
        )
    ineligible_agent_ids = sorted(
        identifier
        for identifier in agent_ids
        if not object_by_id[identifier].is_agent_candidate
    )
    if ineligible_agent_ids:
        raise ValueError(
            "screened agents are not metadata-approved agent candidates: "
            f"{ineligible_agent_ids}"
        )

    ordered_trajectories = sorted(
        tuple(trajectories),
        key=lambda item: (
            item.window.first_norad_id,
            item.window.second_norad_id,
            item.window.start_epoch_utc,
            item.window.end_epoch_utc,
        ),
    )
    references: list[ReferenceConjunction] = []
    cluster_pair: tuple[int, int] | None = None
    cluster_end: datetime | None = None
    cluster_reference: ReferenceConjunction | None = None

    def flush_cluster() -> None:
        if (
            cluster_reference is not None
            and cluster_reference.miss_distance_meters
            <= config.safety.safe_separation_meters
        ):
            references.append(cluster_reference)

    for trajectory in ordered_trajectories:
        pair = (
            trajectory.window.first_norad_id,
            trajectory.window.second_norad_id,
        )
        if pair[0] not in agent_ids and pair[1] not in agent_ids:
            raise ValueError("fine trajectory does not contain a screened agent")
        try:
            combined_radius_meters = (
                object_by_id[pair[0]].radius_meters
                + object_by_id[pair[1]].radius_meters
            )
        except KeyError as error:
            raise ValueError(
                f"fine trajectory references absent NORAD ID {error.args[0]}"
            ) from error
        reference = _trajectory_reference(trajectory, combined_radius_meters)
        overlaps_cluster = (
            cluster_pair == pair
            and cluster_end is not None
            and trajectory.window.start_epoch_utc <= cluster_end
        )
        if not overlaps_cluster:
            flush_cluster()
            cluster_pair = pair
            cluster_end = trajectory.window.end_epoch_utc
            cluster_reference = reference
            continue
        cluster_end = max(cluster_end, trajectory.window.end_epoch_utc)
        if cluster_reference is None or (
            reference.miss_distance_meters,
            reference.tca_epoch_utc,
        ) < (
            cluster_reference.miss_distance_meters,
            cluster_reference.tca_epoch_utc,
        ):
            cluster_reference = reference
    flush_cluster()

    references.sort(
        key=lambda item: (
            item.tca_epoch_utc,
            item.first_norad_id,
            item.second_norad_id,
        )
    )
    return ReferenceConjunctionManifest(
        catalog_epoch_utc=catalog.latest_epoch_utc,
        safe_separation_meters=config.safety.safe_separation_meters,
        reference_step_seconds=config.propagation.reference_step_seconds,
        screened_agent_norad_ids=agent_ids,
        conjunctions=tuple(references),
    )


def _conjunction_to_dict(conjunction: ReferenceConjunction) -> dict[str, object]:
    return {
        "combined_radius_meters": conjunction.combined_radius_meters,
        "first_norad_id": conjunction.first_norad_id,
        "is_collision": conjunction.is_collision,
        "miss_distance_meters": conjunction.miss_distance_meters,
        "relative_speed_mps": conjunction.relative_speed_mps,
        "second_norad_id": conjunction.second_norad_id,
        "tca_epoch_utc": conjunction.tca_epoch_utc.isoformat(),
    }


def save_reference_conjunctions(
    manifest: ReferenceConjunctionManifest,
    path: str | Path,
) -> None:
    """Save reference truth as deterministic, versioned JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "catalog_epoch_utc": manifest.catalog_epoch_utc.isoformat(),
        "conjunctions": [
            _conjunction_to_dict(conjunction)
            for conjunction in manifest.conjunctions
        ],
        "reference_step_seconds": manifest.reference_step_seconds,
        "safe_separation_meters": manifest.safe_separation_meters,
        "schema_version": manifest.schema_version,
        "screened_agent_norad_ids": list(manifest.screened_agent_norad_ids),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_reference_conjunctions(
    path: str | Path,
) -> ReferenceConjunctionManifest:
    """Load and validate a saved reference-conjunction manifest."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    conjunctions_list: list[ReferenceConjunction] = []
    for item in raw["conjunctions"]:
        conjunction = ReferenceConjunction(
            first_norad_id=int(item["first_norad_id"]),
            second_norad_id=int(item["second_norad_id"]),
            tca_epoch_utc=datetime.fromisoformat(item["tca_epoch_utc"]),
            miss_distance_meters=float(item["miss_distance_meters"]),
            relative_speed_mps=float(item["relative_speed_mps"]),
            combined_radius_meters=float(item["combined_radius_meters"]),
        )
        saved_collision = item.get("is_collision")
        if not isinstance(saved_collision, bool):
            raise ValueError("saved is_collision must be boolean")
        if saved_collision is not conjunction.is_collision:
            raise ValueError("saved collision classification is inconsistent")
        conjunctions_list.append(conjunction)
    return ReferenceConjunctionManifest(
        catalog_epoch_utc=datetime.fromisoformat(raw["catalog_epoch_utc"]),
        safe_separation_meters=float(raw["safe_separation_meters"]),
        reference_step_seconds=int(raw["reference_step_seconds"]),
        screened_agent_norad_ids=tuple(
            int(value) for value in raw["screened_agent_norad_ids"]
        ),
        conjunctions=tuple(conjunctions_list),
        schema_version=int(raw["schema_version"]),
    )
