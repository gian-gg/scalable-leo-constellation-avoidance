"""Settings for seeded scenario generation. Paths are relative to the working directory.

See docs/TRAINING_SCENARIOS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SITUATIONS = ("debris", "satellite_pair", "double_threat", "harmless", "quiet")
SPLITS = ("train", "test")


@dataclass(frozen=True)
class ScenarioGeneratorConfig:
    """Where real orbits and close-call shapes come from, and how episodes mix them."""

    tle_path: str = "data/full/catalog.tle"
    metadata_path: str = "data/full/objects.csv"
    sizing_config: str = "configs/maneuver_sizing.json"
    maximum_tle_age_days: float = 14.0
    minimum_altitude_meters: float = 200_000.0
    maximum_altitude_meters: float = 2_000_000.0
    split: str = "train"
    held_out_fraction: float = 0.2
    situation_weights: dict[str, float] = field(
        default_factory=lambda: {
            "debris": 0.30,
            "satellite_pair": 0.25,
            "double_threat": 0.15,
            "harmless": 0.15,
            "quiet": 0.15,
        }
    )
    extra_objects: int | None = None
    meeting_time_seconds: tuple[float, float] = (480.0, 990.0)
    second_threat_delay_seconds: tuple[float, float] = (60.0, 240.0)
    harmless_miss_meters: tuple[float, float] = (2_000.0, 10_000.0)
    epoch_window_seconds: float = 259_200.0
    minimum_unplanned_separation_meters: float = 10_000.0
    minimum_perigee_altitude_meters: float = 150_000.0
    dry_mass_kg: float = 250.0
    initial_fuel_mass_kg: float = 50.0
    maximum_attempts: int = 20

    def __post_init__(self) -> None:
        for name in ("meeting_time_seconds", "second_threat_delay_seconds", "harmless_miss_meters"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "situation_weights", dict(self.situation_weights))

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if not 0 < self.held_out_fraction < 1:
            raise ValueError("held_out_fraction must be in (0, 1)")
        unknown = set(self.situation_weights) - set(SITUATIONS)
        if unknown or not self.situation_weights:
            raise ValueError(f"situation_weights must use names from {SITUATIONS}")
        if any(weight < 0 for weight in self.situation_weights.values()) or sum(self.situation_weights.values()) <= 0:
            raise ValueError("situation weights must be non-negative with a positive total")
        for name in ("meeting_time_seconds", "second_threat_delay_seconds", "harmless_miss_meters"):
            low, high = getattr(self, name)
            if not 0 < low <= high:
                raise ValueError(f"{name} must be an increasing pair of positive values")
        if self.extra_objects is not None and self.extra_objects < 0:
            raise ValueError("extra_objects cannot be negative")
        if self.epoch_window_seconds < 0 or self.minimum_unplanned_separation_meters < 0:
            raise ValueError("epoch_window_seconds and minimum_unplanned_separation_meters cannot be negative")
        if self.dry_mass_kg <= 0 or self.initial_fuel_mass_kg < 0 or self.maximum_attempts <= 0:
            raise ValueError("masses and maximum_attempts must be valid")
