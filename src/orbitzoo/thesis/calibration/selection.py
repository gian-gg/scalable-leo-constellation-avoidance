"""Deterministic nested agent selection and manifest persistence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np

from orbitzoo.thesis.calibration.config import CalibrationConfig
from orbitzoo.thesis.calibration.models import (
    AgentSelection,
    AgentSelectionManifest,
)
from orbitzoo.thesis.calibration.propagation import SGP4Propagation


def _selections_for_seeds(
    eligible_ids: tuple[int, ...],
    agent_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[AgentSelection, ...]:
    selections: list[AgentSelection] = []
    population = np.asarray(eligible_ids, dtype=np.int64)
    for seed in seeds:
        generator = np.random.Generator(np.random.PCG64(seed))
        shuffled_ids = tuple(int(value) for value in generator.permutation(population))
        selections.extend(
            AgentSelection(
                seed=seed,
                requested_agent_count=agent_count,
                agent_norad_ids=shuffled_ids[:agent_count],
            )
            for agent_count in agent_counts
        )
    return tuple(selections)


def select_agent_populations(
    propagation: SGP4Propagation,
    config: CalibrationConfig,
) -> AgentSelectionManifest:
    """Select nested populations from the propagated agent-candidate pool."""
    config.validate()
    candidate_ids = [
        item.norad_id for item in propagation.objects if item.is_agent_candidate
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("propagated agent-candidate NORAD IDs must be unique")
    eligible_ids = tuple(sorted(candidate_ids))
    required_count = max(config.sweep.agent_counts)
    if len(eligible_ids) < required_count:
        raise ValueError(
            f"agent selection requires {required_count} eligible propagated payloads; "
            f"found {len(eligible_ids)}"
        )
    return AgentSelectionManifest(
        catalog_epoch_utc=propagation.start_epoch_utc,
        eligible_norad_ids=eligible_ids,
        calibration_selections=_selections_for_seeds(
            eligible_ids,
            config.sweep.agent_counts,
            config.sweep.calibration_seeds,
        ),
        validation_selections=_selections_for_seeds(
            eligible_ids,
            config.sweep.agent_counts,
            config.sweep.validation_seeds,
        ),
    )


def _selection_to_dict(selection: AgentSelection) -> dict[str, object]:
    return {
        "agent_norad_ids": list(selection.agent_norad_ids),
        "requested_agent_count": selection.requested_agent_count,
        "seed": selection.seed,
    }


def save_agent_selections(
    manifest: AgentSelectionManifest,
    path: str | Path,
) -> None:
    """Save selected NORAD IDs as deterministic, versioned JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibration_selections": [
            _selection_to_dict(item) for item in manifest.calibration_selections
        ],
        "catalog_epoch_utc": manifest.catalog_epoch_utc.isoformat(),
        "eligible_norad_ids": list(manifest.eligible_norad_ids),
        "rng_algorithm": manifest.rng_algorithm,
        "schema_version": manifest.schema_version,
        "validation_selections": [
            _selection_to_dict(item) for item in manifest.validation_selections
        ],
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selection_from_dict(raw: dict[str, object]) -> AgentSelection:
    return AgentSelection(
        seed=int(raw["seed"]),
        requested_agent_count=int(raw["requested_agent_count"]),
        agent_norad_ids=tuple(int(value) for value in raw["agent_norad_ids"]),
    )


def load_agent_selections(path: str | Path) -> AgentSelectionManifest:
    """Load and validate a versioned agent-selection manifest."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AgentSelectionManifest(
        catalog_epoch_utc=datetime.fromisoformat(raw["catalog_epoch_utc"]),
        eligible_norad_ids=tuple(int(value) for value in raw["eligible_norad_ids"]),
        calibration_selections=tuple(
            _selection_from_dict(item) for item in raw["calibration_selections"]
        ),
        validation_selections=tuple(
            _selection_from_dict(item) for item in raw["validation_selections"]
        ),
        rng_algorithm=raw["rng_algorithm"],
        schema_version=int(raw["schema_version"]),
    )
