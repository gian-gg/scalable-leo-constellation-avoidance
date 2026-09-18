"""Versioned configuration for the scalability evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from orbitzoo.thesis.calibration.config import CatalogConfig


@dataclass(frozen=True)
class ScalabilityConfig:
    """Catalog, timing, and sweep sizes; physics and policy settings come from ``experiment_config``."""

    experiment_config: str = "mappo_smoke.json"
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    duration_seconds: int = 21_600
    fine_step_seconds: int = 10
    maximum_relative_speed_mps: float = 20_000.0
    match_tolerance_seconds: float = 300.0
    dry_mass_kg: float = 200.0
    initial_fuel_mass_kg: float = 50.0
    seed: int = 0
    catalog_sizes: tuple[int, ...] = (1_000, 2_000, 5_000, 10_000, 20_000)
    hypothetical_agent_counts: tuple[int, ...] = (1_000, 2_000, 5_000, 10_000)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalog_sizes", tuple(self.catalog_sizes))
        object.__setattr__(self, "hypothetical_agent_counts", tuple(self.hypothetical_agent_counts))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported scalability schema version: {self.schema_version}")
        self.catalog.validate()
        if self.duration_seconds <= 0 or self.fine_step_seconds <= 0:
            raise ValueError("duration_seconds and fine_step_seconds must be positive")
        if self.maximum_relative_speed_mps <= 0 or self.match_tolerance_seconds <= 0:
            raise ValueError("maximum_relative_speed_mps and match_tolerance_seconds must be positive")
        if self.dry_mass_kg <= 0 or self.initial_fuel_mass_kg < 0:
            raise ValueError("dry_mass_kg must be positive and initial_fuel_mass_kg cannot be negative")
        for name in ("catalog_sizes", "hypothetical_agent_counts"):
            sizes = getattr(self, name)
            if any(size <= 0 for size in sizes) or list(sizes) != sorted(set(sizes)):
                raise ValueError(f"{name} must be strictly increasing positive integers")
        if not self.catalog_sizes and not self.hypothetical_agent_counts:
            raise ValueError("at least one sweep must have sizes")

    def resolve(self, config_path: str | Path, relative_path: str) -> Path:
        """Resolve a path relative to the directory holding this configuration."""
        path = Path(relative_path).expanduser()
        return path if path.is_absolute() else (Path(config_path).resolve().parent / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ScalabilityConfig":
        raw = json.loads(Path(path).read_text())
        raw["catalog"] = CatalogConfig(**raw.get("catalog", {}))
        config = cls(**raw)
        config.validate()
        return config
