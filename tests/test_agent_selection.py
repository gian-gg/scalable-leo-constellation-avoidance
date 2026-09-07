from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitzoo.thesis.calibration import (
    CalibrationConfig,
    CatalogObject,
    ObjectType,
    load_agent_selections,
    save_agent_selections,
    select_agent_populations,
)


CATALOG_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _object(norad_id: int, candidate: bool = True) -> CatalogObject:
    return CatalogObject(
        norad_id=norad_id,
        name=f"OBJECT {norad_id}",
        tle_name=None,
        line1="test line 1",
        line2="test line 2",
        tle_epoch_utc=CATALOG_EPOCH,
        object_type=ObjectType.PAYLOAD if candidate else ObjectType.DEBRIS,
        is_agent_candidate=candidate,
        radius_meters=1.0,
        constellation="test",
        has_metadata=True,
    )


def _propagation(objects: list[CatalogObject]) -> SimpleNamespace:
    return SimpleNamespace(objects=tuple(objects), start_epoch_utc=CATALOG_EPOCH)


def _eligible_objects(count: int = 300) -> list[CatalogObject]:
    return [_object(10_000 + index) for index in range(count)]


def test_selections_are_nested_for_every_seed() -> None:
    manifest = select_agent_populations(
        _propagation(_eligible_objects()),
        CalibrationConfig(),
    )

    assert len(manifest.calibration_selections) == 30
    assert len(manifest.validation_selections) == 15
    for selections in (
        manifest.calibration_selections,
        manifest.validation_selections,
    ):
        by_seed: dict[int, list] = {}
        for selection in selections:
            by_seed.setdefault(selection.seed, []).append(selection)
        for seed_selections in by_seed.values():
            small, medium, large = seed_selections
            assert small.requested_agent_count == 16
            assert medium.requested_agent_count == 64
            assert large.requested_agent_count == 256
            assert medium.agent_norad_ids[:16] == small.agent_norad_ids
            assert large.agent_norad_ids[:64] == medium.agent_norad_ids


def test_same_seed_is_reproducible_and_source_order_independent() -> None:
    objects = _eligible_objects()
    forward = select_agent_populations(
        _propagation(objects),
        CalibrationConfig(),
    )
    reversed_order = select_agent_populations(
        _propagation(list(reversed(objects))),
        CalibrationConfig(),
    )

    assert forward == reversed_order
    assert (
        forward.calibration_selections[0].agent_norad_ids
        != forward.calibration_selections[3].agent_norad_ids
    )


def test_only_metadata_approved_payloads_enter_the_pool() -> None:
    eligible = _eligible_objects()
    debris_id = 99_999
    manifest = select_agent_populations(
        _propagation([_object(debris_id, candidate=False), *eligible]),
        CalibrationConfig(),
    )

    assert debris_id not in manifest.eligible_norad_ids
    assert all(
        debris_id not in selection.agent_norad_ids
        for selection in (
            *manifest.calibration_selections,
            *manifest.validation_selections,
        )
    )


def test_selection_manifest_round_trips_as_deterministic_json(tmp_path: Path) -> None:
    manifest = select_agent_populations(
        _propagation(_eligible_objects()),
        CalibrationConfig(),
    )
    path = tmp_path / "nested" / "agent_selections.json"

    save_agent_selections(manifest, path)
    first_serialization = path.read_text()
    loaded = load_agent_selections(path)
    save_agent_selections(loaded, path)

    assert loaded == manifest
    assert path.read_text() == first_serialization
    assert '"rng_algorithm": "PCG64"' in first_serialization
    assert '"requested_agent_count": 256' in first_serialization


def test_selection_fails_when_the_propagated_pool_is_too_small() -> None:
    with pytest.raises(ValueError, match="requires 256.*found 255"):
        select_agent_populations(
            _propagation(_eligible_objects(255)),
            CalibrationConfig(),
        )
