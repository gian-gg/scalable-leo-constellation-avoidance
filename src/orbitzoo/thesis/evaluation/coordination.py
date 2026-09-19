"""Counts of how many satellites in each satellite-to-satellite conjunction maneuvered."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

MANEUVER_GROUPS = ("none", "one", "both")


@dataclass(frozen=True)
class CoordinationCounts:
    """Agent-pair conjunctions grouped by how many members maneuvered, and how many were resolved."""

    none_maneuvered: int = 0
    one_maneuvered: int = 0
    both_maneuvered: int = 0
    none_resolved: int = 0
    one_resolved: int = 0
    both_resolved: int = 0

    def add(self, maneuvering_agents: int, resolved: bool) -> "CoordinationCounts":
        group = MANEUVER_GROUPS[maneuvering_agents]
        updates = {f"{group}_maneuvered": getattr(self, f"{group}_maneuvered") + 1}
        if resolved:
            updates[f"{group}_resolved"] = getattr(self, f"{group}_resolved") + 1
        return replace(self, **updates)

    def __add__(self, other: "CoordinationCounts") -> "CoordinationCounts":
        return CoordinationCounts(**{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)})

    @property
    def total(self) -> int:
        return self.none_maneuvered + self.one_maneuvered + self.both_maneuvered

    def as_columns(self, prefix: str = "pair_") -> dict[str, int]:
        return {f"{prefix}{field.name}": getattr(self, field.name) for field in fields(self)}


COORDINATION_COLUMNS = tuple(CoordinationCounts().as_columns())
