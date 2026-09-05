"""Strict loading and metadata enrichment for frozen TLE catalogs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Iterator

from sgp4.api import Satrec
from sgp4.conveniences import check_satrec, sat_epoch_datetime
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv, verify_checksum

from orbitzoo.thesis.calibration.config import CalibrationConfig, CatalogConfig
from orbitzoo.thesis.calibration.models import CatalogObject, LoadedCatalog, ObjectType


class CatalogLoadError(ValueError):
    """Raised when catalog input cannot be loaded without ambiguity."""


@dataclass(frozen=True)
class _TLERecord:
    norad_id: int
    name: str | None
    line1: str
    line2: str
    epoch_utc: datetime


@dataclass(frozen=True)
class _ObjectMetadata:
    norad_id: int
    name: str | None
    object_type: ObjectType
    is_agent_candidate: bool
    radius_meters: float
    constellation: str | None


def _error(path: Path, message: str, line_number: int | None = None) -> CatalogLoadError:
    location = str(path)
    if line_number is not None:
        location += f":{line_number}"
    return CatalogLoadError(f"{location}: {message}")


def _nonblank_lines(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise _error(path, str(error)) from error
    return [
        (line_number, line)
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]


def _normalize_tle_name(raw_name: str, path: Path, line_number: int) -> str:
    name = raw_name.strip()
    if name.startswith("0 "):
        name = name[2:].strip()
    if not name:
        raise _error(path, "TLE name cannot be empty", line_number)
    return name


def _group_tle_records(path: Path) -> Iterator[tuple[str | None, int, str, int, str]]:
    lines = _nonblank_lines(path)
    index = 0
    while index < len(lines):
        line_number, line = lines[index]
        name: str | None = None
        if line.startswith("2 "):
            raise _error(path, "unexpected TLE line 2", line_number)
        if not line.startswith("1 "):
            name = _normalize_tle_name(line, path, line_number)
            index += 1
            if index >= len(lines):
                raise _error(path, "TLE name is not followed by element lines", line_number)
            line_number, line = lines[index]
        if not line.startswith("1 "):
            raise _error(path, "expected TLE line 1", line_number)
        line1_number, line1 = line_number, line
        index += 1
        if index >= len(lines):
            raise _error(path, "TLE line 1 is not followed by line 2", line1_number)
        line2_number, line2 = lines[index]
        if not line2.startswith("2 "):
            raise _error(path, "expected TLE line 2", line2_number)
        yield name, line1_number, line1, line2_number, line2
        index += 1


def _validate_element_line(path: Path, line_number: int, line: str, prefix: str) -> None:
    if len(line) != 69:
        raise _error(path, f"TLE line {prefix} must contain exactly 69 characters", line_number)
    try:
        line.encode("ascii")
    except UnicodeEncodeError as error:
        raise _error(path, f"TLE line {prefix} must contain only ASCII characters", line_number) from error
    try:
        verify_checksum(line)
    except ValueError as error:
        raise _error(path, str(error), line_number) from error


def _parse_tle_file(path: Path) -> tuple[_TLERecord, ...]:
    records: list[_TLERecord] = []
    seen_ids: dict[int, int] = {}
    for name, line1_number, line1, line2_number, line2 in _group_tle_records(path):
        _validate_element_line(path, line1_number, line1, "1")
        _validate_element_line(path, line2_number, line2, "2")
        if line1[2:7] != line2[2:7]:
            raise _error(
                path,
                f"TLE lines have different catalog IDs {line1[2:7]!r} and {line2[2:7]!r}",
                line2_number,
            )
        try:
            # The pure-Python parser performs strict fixed-column validation.
            twoline2rv(line1, line2, wgs72)
            satellite = Satrec.twoline2rv(line1, line2)
            check_satrec(satellite)
            epoch = sat_epoch_datetime(satellite).astimezone(timezone.utc)
        except (TypeError, ValueError) as error:
            raise _error(path, f"invalid TLE: {error}", line1_number) from error
        norad_id = int(satellite.satnum)
        if norad_id in seen_ids:
            raise _error(
                path,
                f"duplicate NORAD ID {norad_id}; first seen on line {seen_ids[norad_id]}",
                line1_number,
            )
        seen_ids[norad_id] = line1_number
        records.append(_TLERecord(norad_id, name, line1, line2, epoch))
    if not records:
        raise _error(path, "TLE catalog contains no records")
    return tuple(records)


def _parse_object_type(value: str | None, path: Path, line_number: int) -> ObjectType:
    if value is None:
        raise _error(path, "object_type is required", line_number)
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return ObjectType(normalized)
    except ValueError as error:
        allowed = ", ".join(item.value for item in ObjectType)
        raise _error(path, f"invalid object_type {value!r}; expected one of {allowed}", line_number) from error


def _parse_boolean(value: str | None, path: Path, line_number: int) -> bool:
    if value is None:
        raise _error(path, "is_agent_candidate is required", line_number)
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise _error(path, f"invalid is_agent_candidate {value!r}; expected true or false", line_number)


def _parse_positive_id(value: str | None, path: Path, line_number: int) -> int:
    if value is None:
        raise _error(path, "norad_id is required", line_number)
    try:
        norad_id = int(value)
    except (TypeError, ValueError) as error:
        raise _error(path, f"invalid norad_id {value!r}", line_number) from error
    if norad_id <= 0:
        raise _error(path, "norad_id must be positive", line_number)
    return norad_id


def _parse_radius(
    value: str | None,
    default_radius_meters: float,
    path: Path,
    line_number: int,
) -> float:
    if value is None or not value.strip():
        return default_radius_meters
    try:
        radius = float(value)
    except (TypeError, ValueError) as error:
        raise _error(path, f"invalid radius_meters {value!r}", line_number) from error
    if not math.isfinite(radius) or radius <= 0:
        raise _error(path, "radius_meters must be finite and positive", line_number)
    return radius


def _load_metadata(path: Path, config: CatalogConfig) -> dict[int, _ObjectMetadata]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except (OSError, UnicodeError) as error:
        raise _error(path, str(error)) from error
    with stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"norad_id", "object_type", "is_agent_candidate"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise _error(path, f"metadata is missing required columns: {sorted(missing)}", 1)
        metadata: dict[int, _ObjectMetadata] = {}
        try:
            for row in reader:
                if not any(
                    str(value or "").strip()
                    for key, value in row.items()
                    if key is not None
                ):
                    continue
                line_number = reader.line_num
                if None in row:
                    raise _error(
                        path,
                        "metadata row has more values than the header",
                        line_number,
                    )
                norad_id = _parse_positive_id(row["norad_id"], path, line_number)
                if norad_id in metadata:
                    raise _error(path, f"duplicate metadata NORAD ID {norad_id}", line_number)
                object_type = _parse_object_type(row["object_type"], path, line_number)
                is_agent_candidate = _parse_boolean(
                    row["is_agent_candidate"], path, line_number
                )
                if is_agent_candidate and object_type is not ObjectType.PAYLOAD:
                    raise _error(
                        path,
                        "only payloads may be marked as agent candidates",
                        line_number,
                    )
                name = (row.get("name") or "").strip() or None
                constellation = (row.get("constellation") or "").strip() or None
                metadata[norad_id] = _ObjectMetadata(
                    norad_id=norad_id,
                    name=name,
                    object_type=object_type,
                    is_agent_candidate=is_agent_candidate,
                    radius_meters=_parse_radius(
                        row.get("radius_meters"),
                        config.default_radius_meters,
                        path,
                        line_number,
                    ),
                    constellation=constellation,
                )
        except csv.Error as error:
            raise _error(path, f"invalid metadata CSV: {error}", reader.line_num) from error
    return metadata


def load_catalog(
    config: CalibrationConfig,
    config_path: str | Path,
) -> LoadedCatalog:
    """Load, validate, enrich, and freshness-filter one frozen TLE catalog."""
    config.validate()
    tle_path, metadata_path = config.resolve_catalog_paths(config_path)
    records = _parse_tle_file(tle_path)
    metadata = _load_metadata(metadata_path, config.catalog)
    parsed_ids = {record.norad_id for record in records}
    orphan_ids = sorted(set(metadata).difference(parsed_ids))
    if orphan_ids:
        raise _error(
            metadata_path,
            f"metadata contains NORAD IDs absent from the TLE catalog: {orphan_ids}",
        )

    latest_epoch = max(record.epoch_utc for record in records)
    cutoff = latest_epoch - timedelta(days=config.catalog.maximum_tle_age_days)
    objects: list[CatalogObject] = []
    stale_ids: list[int] = []
    for record in records:
        if record.epoch_utc < cutoff:
            stale_ids.append(record.norad_id)
            continue
        details = metadata.get(record.norad_id)
        objects.append(
            CatalogObject(
                norad_id=record.norad_id,
                name=(details.name if details else None)
                or record.name
                or f"NORAD-{record.norad_id}",
                tle_name=record.name,
                line1=record.line1,
                line2=record.line2,
                tle_epoch_utc=record.epoch_utc,
                object_type=details.object_type if details else ObjectType.UNKNOWN,
                is_agent_candidate=details.is_agent_candidate if details else False,
                radius_meters=(
                    details.radius_meters
                    if details
                    else config.catalog.default_radius_meters
                ),
                constellation=details.constellation if details else None,
                has_metadata=details is not None,
            )
        )
    return LoadedCatalog(
        objects=tuple(objects),
        latest_epoch_utc=latest_epoch,
        source_record_count=len(records),
        stale_filtered_norad_ids=tuple(stale_ids),
    )
