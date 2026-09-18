"""Play one collision-avoidance episode and summarize its outcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from orbitzoo.thesis.environments.collision_avoidance import CollisionAvoidanceEnv

ChooseActions = Callable[[np.ndarray, np.ndarray], np.ndarray]
StepOutputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
ObserveStep = Callable[[np.ndarray, np.ndarray, np.ndarray, StepOutputs], None]


@dataclass(frozen=True)
class EpisodeSummary:
    """Outcome of one complete episode."""

    seed: int
    mean_agent_return: float
    length: int
    ended_in_collision: bool
    unsafe_agent_steps: int
    final_unsafe_agents: int
    rejected_actions: int
    mean_delta_v_per_agent_mps: float
    minimum_separation_meters: float


def reset_environment(env: CollisionAvoidanceEnv, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reset the environment without disturbing the global torch random stream."""
    # OrbitZoo's reset reseeds torch globally.
    rng_state = torch.get_rng_state()
    observations = env.reset(seed=seed)
    torch.set_rng_state(rng_state)
    return observations


def _unsafe_agents(info: dict[str, Any], agent_names: set[str]) -> set[str]:
    return agent_names & {name for pair in info["unsafe_pairs"] for name in pair}


def play_episode(
    env: CollisionAvoidanceEnv,
    seed: int,
    choose_actions: ChooseActions,
    observe_step: ObserveStep | None = None,
) -> EpisodeSummary:
    """Run until termination; ``observe_step`` sees each pre-step state, actions, and outputs."""
    local, global_state = reset_environment(env, seed)
    agent_names = set(env.agent_names)
    total_reward = np.zeros(env.num_agents, dtype=np.float64)
    unsafe_agent_steps = rejected_actions = 0
    while True:
        actions = np.asarray(choose_actions(local, global_state), dtype=np.int64)
        outputs = env.step(actions)
        next_local, next_global, rewards, dones, info = outputs
        if observe_step:
            observe_step(local, global_state, actions, outputs)
        total_reward += rewards
        unsafe_agent_steps += len(_unsafe_agents(info, agent_names))
        rejected_actions += len(info["rejected_agents"])
        local, global_state = next_local, next_global
        if dones.all():
            break
    delta_v = env.diagnostics.cumulative_delta_v_mps
    return EpisodeSummary(
        seed=seed,
        mean_agent_return=float(total_reward.mean()),
        length=env.step_index,
        ended_in_collision=info["termination_reason"] == "collision",
        unsafe_agent_steps=unsafe_agent_steps,
        final_unsafe_agents=len(_unsafe_agents(info, agent_names)),
        rejected_actions=rejected_actions,
        mean_delta_v_per_agent_mps=float(np.mean(list(delta_v.values()))),
        minimum_separation_meters=env.diagnostics.minimum_separation_meters,
    )
