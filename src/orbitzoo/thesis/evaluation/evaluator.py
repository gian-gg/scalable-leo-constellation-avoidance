"""Evaluate decentralized policies on identical held-out episodes.

See docs/EVALUATION.md.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from orbitzoo.rl_algorithms.mappo import MAPPO
from orbitzoo.thesis.config import ExperimentConfig
from orbitzoo.thesis.environments.episodes import EpisodeSummary, play_episode
from orbitzoo.thesis.environments.scenarios import EpisodeSource, build_episode_source
from orbitzoo.thesis.evaluation.coordination import COORDINATION_COLUMNS, CoordinationCounts
from orbitzoo.thesis.evaluation.policies import (
    ClohessyWiltshireAvoidancePolicy,
    EvaluationPolicy,
    MAPPOActorPolicy,
    NoOpPolicy,
)
from orbitzoo.thesis.runtime import environment_info

EVALUATION_SEED_OFFSET = 1_000_000
EPISODE_FIELDS = (
    *(name for name in EpisodeSummary.__dataclass_fields__ if name != "coordination"),
    *COORDINATION_COLUMNS,
)


@dataclass(frozen=True)
class PolicySummary:
    """Aggregate evaluation metrics for one policy."""

    policy: str
    episodes: int
    collision_rate: float
    mean_return: float
    mean_unsafe_agent_steps: float
    mean_final_unsafe_agents: float
    mean_rejected_actions: float
    mean_delta_v_per_agent_mps: float
    mean_minimum_separation_meters: float
    minimum_separation_meters: float
    coordination: CoordinationCounts

    def as_row(self) -> dict[str, object]:
        row = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != "coordination"}
        return {**row, **self.coordination.as_columns()}


SUMMARY_FIELDS = (
    *(name for name in PolicySummary.__dataclass_fields__ if name != "coordination"),
    *COORDINATION_COLUMNS,
)


def _episode_row(policy_name: str, episode: EpisodeSummary) -> dict[str, object]:
    row = {name: getattr(episode, name) for name in EpisodeSummary.__dataclass_fields__ if name != "coordination"}
    return {"policy": policy_name, **row, **episode.coordination.as_columns()}


def evaluation_seeds(config: ExperimentConfig, episodes: int) -> list[int]:
    """Seeds disjoint from training episodes, which start at ``config.seed``."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    start = config.seed + EVALUATION_SEED_OFFSET
    return list(range(start, start + episodes))


def build_policy(spec: str, config: ExperimentConfig, local_observation_dim: int) -> EvaluationPolicy:
    """Create a policy from ``noop``, ``rule``, ``PATH`` or ``NAME=PATH`` to a MAPPO checkpoint."""
    if spec == "noop":
        return NoOpPolicy()
    if spec == "rule":
        return ClohessyWiltshireAvoidancePolicy(config.maneuver, config.safety)
    name, _, path = spec.rpartition("=")
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"policy {spec!r} is not noop, rule, or an existing checkpoint")
    policy = MAPPO.from_checkpoint(checkpoint)
    if policy.local_observation_dim != local_observation_dim:
        raise ValueError(
            f"checkpoint expects local observations of width {policy.local_observation_dim}, "
            f"but the environment produces {local_observation_dim}"
        )
    return MAPPOActorPolicy(policy, name or "mappo")


def summarize(policy_name: str, episodes: Sequence[EpisodeSummary]) -> PolicySummary:
    def mean(field: str) -> float:
        return float(np.mean([getattr(episode, field) for episode in episodes]))

    return PolicySummary(
        policy=policy_name,
        episodes=len(episodes),
        collision_rate=mean("ended_in_collision"),
        mean_return=mean("mean_agent_return"),
        mean_unsafe_agent_steps=mean("unsafe_agent_steps"),
        mean_final_unsafe_agents=mean("final_unsafe_agents"),
        mean_rejected_actions=mean("rejected_actions"),
        mean_delta_v_per_agent_mps=mean("mean_delta_v_per_agent_mps"),
        mean_minimum_separation_meters=mean("minimum_separation_meters"),
        minimum_separation_meters=min(episode.minimum_separation_meters for episode in episodes),
        coordination=sum((episode.coordination for episode in episodes), CoordinationCounts()),
    )


def evaluate_policy(policy: EvaluationPolicy, source: EpisodeSource, seeds: Sequence[int]) -> list[EpisodeSummary]:
    """Play every seed with actions chosen from local observations alone."""
    return [play_episode(source.environment(seed), seed, lambda local, _: policy.choose(local)) for seed in seeds]


def held_out(config: ExperimentConfig) -> ExperimentConfig:
    """The same config, drawing generated scenarios from the held-out test split."""
    return dataclasses.replace(
        config, scenario_generator=dataclasses.replace(config.scenario_generator, split="test")
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    config: ExperimentConfig,
    policy_specs: Sequence[str],
    episodes: int,
    output_directory: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[PolicySummary]:
    """Evaluate each policy on the same seeds and write episode and summary tables."""
    if not policy_specs:
        raise ValueError("at least one policy is required")
    seeds = evaluation_seeds(config, episodes)
    source = build_episode_source(held_out(config))
    width = source.environment(seeds[0]).local_observation_dim
    policies = [build_policy(spec, config, width) for spec in policy_specs]
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"policy names must be unique: {names}")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    config.save(output_directory / "config.json")
    (output_directory / "evaluation_info.json").write_text(
        json.dumps(
            {"policies": list(policy_specs), "seeds": seeds, **environment_info()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    episode_rows: list[dict] = []
    summaries: list[PolicySummary] = []
    for policy in policies:
        results = evaluate_policy(policy, source, seeds)
        episode_rows.extend(_episode_row(policy.name, result) for result in results)
        summaries.append(summarize(policy.name, results))
        if progress:
            summary = summaries[-1]
            progress(
                f"{policy.name}: collisions {summary.collision_rate:.2%}, "
                f"unsafe agent-steps {summary.mean_unsafe_agent_steps:.2f}, "
                f"delta-v {summary.mean_delta_v_per_agent_mps:.4f} m/s"
            )

    _write_csv(output_directory / "episodes.csv", ("policy", *EPISODE_FIELDS), episode_rows)
    _write_csv(output_directory / "summary.csv", SUMMARY_FIELDS, [summary.as_row() for summary in summaries])
    return summaries
