"""Train the shared MAPPO policy on the collision-avoidance environment.

See docs/TRAINING.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from orbitzoo.rl_algorithms.mappo import MAPPO
from orbitzoo.thesis.config import ExperimentConfig
from orbitzoo.thesis.environments.collision_avoidance import CollisionAvoidanceEnv
from orbitzoo.thesis.environments.scenarios import build_environment
from orbitzoo.thesis.runtime import select_device

METRICS_FILENAME = "metrics.csv"
STATE_FILENAME = "training_state.json"
CHECKPOINT_DIRECTORY = "checkpoints"
POLICY_CHECKPOINT = "latest.pt"
RNG_CHECKPOINT = "rng_state.pt"
LOSS_FIELDS = ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction")
METRIC_FIELDS = (
    "update",
    "environment_steps",
    "episodes",
    "mean_episode_return",
    "mean_episode_length",
    "collision_rate",
    "mean_unsafe_agent_steps",
    "mean_delta_v_per_agent_mps",
    "minimum_separation_meters",
    *LOSS_FIELDS,
    "update_seconds",
)


@dataclass(frozen=True)
class EpisodeSummary:
    """Outcome of one complete training episode."""

    mean_agent_return: float
    length: int
    ended_in_collision: bool
    unsafe_agent_steps: int
    mean_delta_v_per_agent_mps: float
    minimum_separation_meters: float


@dataclass(frozen=True)
class TrainingResult:
    """Where a training run was written and how far it got."""

    run_directory: Path
    completed_updates: int
    environment_steps: int
    final_metrics: dict[str, float]


def _device(name: str) -> torch.device:
    return select_device() if name == "auto" else torch.device(name)


def _build_policy(config: ExperimentConfig, env: CollisionAvoidanceEnv) -> MAPPO:
    training = config.training
    return MAPPO(
        local_observation_dim=env.local_observation_dim,
        global_state_dim=env.global_state_dim,
        num_actions=config.policy.num_actions,
        actor_hidden_dims=training.actor_hidden_dims,
        critic_hidden_dims=training.critic_hidden_dims,
        actor_learning_rate=training.actor_learning_rate,
        critic_learning_rate=training.critic_learning_rate,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        ppo_clip=training.ppo_clip,
        value_clip=training.value_clip,
        entropy_coefficient=training.entropy_coefficient,
        value_coefficient=training.value_coefficient,
        max_gradient_norm=training.max_gradient_norm,
        update_epochs=training.update_epochs,
        minibatch_size=training.minibatch_size,
        device=_device(training.device),
    )


def _reset(env: CollisionAvoidanceEnv, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # OrbitZoo's reset reseeds torch globally; keep the policy's sampling stream intact.
    rng_state = torch.get_rng_state()
    observations = env.reset(seed=seed)
    torch.set_rng_state(rng_state)
    return observations


def run_episode(policy: MAPPO, env: CollisionAvoidanceEnv, seed: int) -> EpisodeSummary:
    """Play one episode with the stochastic policy and store every transition."""
    local, global_state = _reset(env, seed)
    agent_names = set(env.agent_names)
    total_reward = np.zeros(env.num_agents, dtype=np.float64)
    unsafe_agent_steps = 0
    while True:
        actions, log_probabilities, values = policy.act(local, global_state)
        next_local, next_global, rewards, dones, info = env.step(actions.numpy())
        total_reward += rewards
        unsafe_agent_steps += len(agent_names & {name for pair in info["unsafe_pairs"] for name in pair})
        stored_rewards = rewards
        if info["termination_reason"] == "horizon":
            stored_rewards = rewards + policy.gamma * policy.values(next_local, next_global).numpy()
        policy.store_step(local, global_state, actions, log_probabilities, stored_rewards, dones, values)
        local, global_state = next_local, next_global
        if dones.all():
            break
    diagnostics = env.diagnostics
    return EpisodeSummary(
        mean_agent_return=float(total_reward.mean()),
        length=env.step_index,
        ended_in_collision=info["termination_reason"] == "collision",
        unsafe_agent_steps=unsafe_agent_steps,
        mean_delta_v_per_agent_mps=float(np.mean(list(diagnostics.cumulative_delta_v_mps.values()))),
        minimum_separation_meters=diagnostics.minimum_separation_meters,
    )


def _update_metrics(
    update: int,
    policy: MAPPO,
    episodes: list[EpisodeSummary],
    losses: dict[str, float],
    update_seconds: float,
) -> dict[str, float]:
    return {
        "update": update,
        "environment_steps": policy.environment_steps,
        "episodes": len(episodes),
        "mean_episode_return": float(np.mean([episode.mean_agent_return for episode in episodes])),
        "mean_episode_length": float(np.mean([episode.length for episode in episodes])),
        "collision_rate": float(np.mean([episode.ended_in_collision for episode in episodes])),
        "mean_unsafe_agent_steps": float(np.mean([episode.unsafe_agent_steps for episode in episodes])),
        "mean_delta_v_per_agent_mps": float(
            np.mean([episode.mean_delta_v_per_agent_mps for episode in episodes])
        ),
        "minimum_separation_meters": min(episode.minimum_separation_meters for episode in episodes),
        **{name: losses[name] for name in LOSS_FIELDS},
        "update_seconds": update_seconds,
    }


def _read_metrics(path: Path, completed_updates: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if int(row["update"]) <= completed_updates]


def _write_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(run_directory: Path, policy: MAPPO, episode_index: int) -> None:
    checkpoint_directory = run_directory / CHECKPOINT_DIRECTORY
    policy.save(checkpoint_directory / POLICY_CHECKPOINT)
    torch.save(
        {"numpy": np.random.get_state(), "torch": torch.get_rng_state()},
        checkpoint_directory / RNG_CHECKPOINT,
    )
    (run_directory / STATE_FILENAME).write_text(
        json.dumps(
            {
                "completed_updates": policy.update_count,
                "environment_steps": policy.environment_steps,
                "episode_index": episode_index,
            },
            indent=2,
        )
        + "\n"
    )


def _restore_checkpoint(run_directory: Path, policy: MAPPO) -> int:
    state_path = run_directory / STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"no {STATE_FILENAME} in {run_directory}; nothing to resume")
    state = json.loads(state_path.read_text())
    checkpoint_directory = run_directory / CHECKPOINT_DIRECTORY
    policy.load(checkpoint_directory / POLICY_CHECKPOINT)
    rng = torch.load(checkpoint_directory / RNG_CHECKPOINT, weights_only=False)
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    return int(state["episode_index"])


def train(
    config: ExperimentConfig,
    run_directory: str | Path,
    *,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TrainingResult:
    """Train until ``config.training.total_updates``, checkpointing along the way."""
    from tensorboardX import SummaryWriter

    config.validate()
    run_directory = Path(run_directory)
    training = config.training
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = build_environment(config)
    policy = _build_policy(config, env)
    episode_index = _restore_checkpoint(run_directory, policy) if resume else 0

    metrics_path = run_directory / METRICS_FILENAME
    rows: list[dict[str, float | str]] = list(_read_metrics(metrics_path, policy.update_count))
    writer = SummaryWriter(logdir=str(run_directory / "tensorboard"))
    try:
        while policy.update_count < training.total_updates:
            started = time.perf_counter()
            episodes: list[EpisodeSummary] = []
            while len(policy.rollout) < training.rollout_steps:
                episodes.append(run_episode(policy, env, config.seed + episode_index))
                episode_index += 1
            losses = policy.update(
                policy.rollout.local_observations[-1],
                policy.rollout.global_states[-1],
                np.ones(env.num_agents, dtype=bool),
            )
            metrics = _update_metrics(
                policy.update_count, policy, episodes, losses, time.perf_counter() - started
            )
            rows.append(metrics)
            _write_metrics(metrics_path, rows)
            for name, value in metrics.items():
                if name != "update" and math.isfinite(value):
                    writer.add_scalar(name, value, policy.update_count)
            if policy.update_count % training.checkpoint_interval == 0 or policy.update_count == training.total_updates:
                _save_checkpoint(run_directory, policy, episode_index)
            if progress:
                progress(
                    f"update {policy.update_count}/{training.total_updates}: "
                    f"return {metrics['mean_episode_return']:.3f}, "
                    f"collisions {metrics['collision_rate']:.2%}, "
                    f"{metrics['update_seconds']:.1f} s"
                )
    finally:
        writer.close()

    final = {name: float(value) for name, value in rows[-1].items()} if rows else {}
    return TrainingResult(run_directory, policy.update_count, policy.environment_steps, final)
