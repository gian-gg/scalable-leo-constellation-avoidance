"""Discrete Multi-Agent PPO with centralized training and decentralized execution.

The actor is shared by every homogeneous agent and consumes only that agent's
local observation. The critic is used only while training; it consumes the
global state plus the evaluated agent's local observation, allowing individual
agent rewards while retaining centralized context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from orbitzoo.rl_algorithms.main import RLAlgorithm


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    """Build a small Tanh MLP used by the actor and critic."""
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.Tanh()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class RolloutBatch:
    """One fixed-size, simultaneous multi-agent rollout stored on CPU."""

    local_observations: list[Tensor]
    global_states: list[Tensor]
    actions: list[Tensor]
    log_probabilities: list[Tensor]
    rewards: list[Tensor]
    dones: list[Tensor]
    values: list[Tensor]
    num_agents: int | None = None

    def __init__(self) -> None:
        self.local_observations = []
        self.global_states = []
        self.actions = []
        self.log_probabilities = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.num_agents = None

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        local_observations: Tensor,
        global_state: Tensor,
        actions: Tensor,
        log_probabilities: Tensor,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
    ) -> None:
        """Store all agents' data from one environment timestep."""
        local_observations = torch.as_tensor(local_observations, dtype=torch.float32).cpu()
        global_state = torch.as_tensor(global_state, dtype=torch.float32).cpu()
        actions = torch.as_tensor(actions, dtype=torch.long).cpu()
        log_probabilities = torch.as_tensor(log_probabilities, dtype=torch.float32).cpu()
        rewards = torch.as_tensor(rewards, dtype=torch.float32).cpu()
        dones = torch.as_tensor(dones, dtype=torch.bool).cpu()
        values = torch.as_tensor(values, dtype=torch.float32).cpu()

        if local_observations.ndim != 2:
            raise ValueError("local_observations must have shape [num_agents, local_observation_dim]")
        if global_state.ndim != 1:
            raise ValueError("global_state must have shape [global_state_dim]")

        num_agents = local_observations.shape[0]
        if self.num_agents is None:
            self.num_agents = num_agents
        elif num_agents != self.num_agents:
            raise ValueError("the number of agents must remain constant within one rollout")

        for name, tensor in (
            ("actions", actions),
            ("log_probabilities", log_probabilities),
            ("rewards", rewards),
            ("dones", dones),
            ("values", values),
        ):
            if tensor.shape != (num_agents,):
                raise ValueError(f"{name} must have shape [num_agents]")

        self.local_observations.append(local_observations.detach().clone())
        self.global_states.append(global_state.detach().clone())
        self.actions.append(actions.detach().clone())
        self.log_probabilities.append(log_probabilities.detach().clone())
        self.rewards.append(rewards.detach().clone())
        self.dones.append(dones.detach().clone())
        self.values.append(values.detach().clone())

    def tensors(self) -> dict[str, Tensor]:
        """Stack a non-empty rollout into [time, agents, ...] tensors."""
        if not self:
            raise RuntimeError("cannot train from an empty rollout")
        return {
            "local_observations": torch.stack(self.local_observations),
            "global_states": torch.stack(self.global_states),
            "actions": torch.stack(self.actions),
            "log_probabilities": torch.stack(self.log_probabilities),
            "rewards": torch.stack(self.rewards),
            "dones": torch.stack(self.dones),
            "values": torch.stack(self.values),
        }

    def clear(self) -> None:
        """Discard one completed rollout after a successful update."""
        self.local_observations.clear()
        self.global_states.clear()
        self.actions.clear()
        self.log_probabilities.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.num_agents = None


class MAPPO(RLAlgorithm):
    """Discrete shared-policy MAPPO for homogeneous agents.

    The actor receives a local observation and predicts a discrete action. The
    training-only critic receives the global state plus that agent's local
    observation, so it can estimate a separate value for each agent reward.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        local_observation_dim: int,
        global_state_dim: int,
        num_actions: int = 7,
        *,
        actor_hidden_dims: Sequence[int] = (128, 64),
        critic_hidden_dims: Sequence[int] = (256, 128),
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_clip: float = 0.2,
        value_clip: float = 0.2,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
        max_gradient_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        if local_observation_dim <= 0 or global_state_dim <= 0:
            raise ValueError("observation and global-state dimensions must be positive")
        if num_actions < 2:
            raise ValueError("num_actions must be at least 2")
        if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda must be in [0, 1]")
        if min(actor_learning_rate, critic_learning_rate, ppo_clip, value_clip, max_gradient_norm) <= 0:
            raise ValueError("learning rates, clip values, and max_gradient_norm must be positive")
        if entropy_coefficient < 0 or value_coefficient < 0:
            raise ValueError("entropy_coefficient and value_coefficient cannot be negative")
        if update_epochs <= 0 or minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive")

        selected_device = torch.device(device or "cpu")
        super().__init__(
            device=selected_device,
            has_continuous_action_space=False,
            action_space=None,
            action_to_thrust_fn=None,
        )
        self.local_observation_dim = local_observation_dim
        self.global_state_dim = global_state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.value_clip = value_clip
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.max_gradient_norm = max_gradient_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.critic_hidden_dims = tuple(critic_hidden_dims)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.update_count = 0
        self.environment_steps = 0

        self.actor = _mlp(local_observation_dim, actor_hidden_dims, num_actions).to(self.device)
        self.critic = _mlp(global_state_dim + local_observation_dim, critic_hidden_dims, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            (
                {"params": self.actor.parameters(), "lr": actor_learning_rate},
                {"params": self.critic.parameters(), "lr": critic_learning_rate},
            )
        )
        self.rollout = RolloutBatch()

    def _local_tensor(self, local_observations: Tensor | np.ndarray | Sequence[Sequence[float]]) -> Tensor:
        observations = torch.as_tensor(local_observations, dtype=torch.float32, device=self.device)
        if observations.ndim != 2 or observations.shape[1] != self.local_observation_dim:
            raise ValueError(
                "local_observations must have shape "
                f"[num_agents, {self.local_observation_dim}]"
            )
        return observations

    def _global_tensor(self, global_state: Tensor | np.ndarray | Sequence[float]) -> Tensor:
        state = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        if state.shape != (self.global_state_dim,):
            raise ValueError(f"global_state must have shape [{self.global_state_dim}]")
        return state

    @staticmethod
    def _critic_inputs(local_observations: Tensor, global_state: Tensor) -> Tensor:
        repeated_global_state = global_state.unsqueeze(0).expand(local_observations.shape[0], -1)
        return torch.cat((repeated_global_state, local_observations), dim=-1)

    def action_probabilities(
        self, local_observations: Tensor | np.ndarray | Sequence[Sequence[float]]
    ) -> Tensor:
        """Return one discrete-action distribution per local observation."""
        observations = self._local_tensor(local_observations)
        return torch.softmax(self.actor(observations), dim=-1)

    def act(
        self,
        local_observations: Tensor | np.ndarray | Sequence[Sequence[float]],
        global_state: Tensor | np.ndarray | Sequence[float],
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Choose actions and evaluate their training-only centralized values."""
        observations = self._local_tensor(local_observations)
        state = self._global_tensor(global_state)
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(observations))
            actions = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
            log_probabilities = distribution.log_prob(actions)
            values = self.critic(self._critic_inputs(observations, state)).squeeze(-1)
        return actions.cpu(), log_probabilities.cpu(), values.cpu()

    def values(
        self,
        local_observations: Tensor | np.ndarray | Sequence[Sequence[float]],
        global_state: Tensor | np.ndarray | Sequence[float],
    ) -> Tensor:
        """Return the critic's per-agent value estimates without tracking gradients."""
        observations = self._local_tensor(local_observations)
        state = self._global_tensor(global_state)
        self.critic.eval()
        with torch.no_grad():
            return self._values_for(observations, state).cpu()

    def select_actions(
        self,
        local_observations: Tensor | np.ndarray | Sequence[Sequence[float]],
        *,
        deterministic: bool = True,
    ) -> Tensor:
        """Choose decentralized actions without accessing the critic or global state."""
        observations = self._local_tensor(local_observations)
        self.actor.eval()
        with torch.no_grad():
            distribution = Categorical(logits=self.actor(observations))
            actions = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        return actions.cpu()

    def store_step(
        self,
        local_observations: Tensor | np.ndarray | Sequence[Sequence[float]],
        global_state: Tensor | np.ndarray | Sequence[float],
        actions: Tensor | np.ndarray | Sequence[int],
        log_probabilities: Tensor | np.ndarray | Sequence[float],
        rewards: Tensor | np.ndarray | Sequence[float],
        dones: Tensor | np.ndarray | Sequence[bool],
        values: Tensor | np.ndarray | Sequence[float],
    ) -> None:
        """Save one simultaneous multi-agent environment transition."""
        validated_observations = self._local_tensor(local_observations).detach().cpu()
        validated_global_state = self._global_tensor(global_state).detach().cpu()
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        if torch.any(action_tensor < 0) or torch.any(action_tensor >= self.num_actions):
            raise ValueError(f"actions must be integers in [0, {self.num_actions - 1}]")
        self.rollout.add(
            validated_observations,
            validated_global_state,
            action_tensor,
            log_probabilities,
            rewards,
            dones,
            values,
        )
        self.environment_steps += 1

    def _values_for(self, local_observations: Tensor, global_state: Tensor) -> Tensor:
        return self.critic(self._critic_inputs(local_observations, global_state)).squeeze(-1)

    def _compute_gae(
        self, rewards: Tensor, values: Tensor, dones: Tensor, final_values: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute GAE advantages and return targets for every timestep/agent."""
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(final_values)
        next_values = final_values
        for timestep in reversed(range(rewards.shape[0])):
            not_done = (~dones[timestep]).float()
            temporal_difference = rewards[timestep] + self.gamma * next_values * not_done - values[timestep]
            gae = temporal_difference + self.gamma * self.gae_lambda * not_done * gae
            advantages[timestep] = gae
            next_values = values[timestep]
        return advantages, advantages + values

    def update(
        self,
        final_local_observations: Tensor | np.ndarray | Sequence[Sequence[float]],
        final_global_state: Tensor | np.ndarray | Sequence[float],
        final_dones: Tensor | np.ndarray | Sequence[bool],
    ) -> dict[str, float]:
        """Optimize actor and critic from the collected on-policy rollout."""
        rollout = self.rollout.tensors()
        final_local = self._local_tensor(final_local_observations)
        final_global = self._global_tensor(final_global_state)
        final_done_tensor = torch.as_tensor(final_dones, dtype=torch.bool, device=self.device)
        if final_local.shape[0] != self.rollout.num_agents or final_done_tensor.shape != (self.rollout.num_agents,):
            raise ValueError("final observations and final_dones must match the rollout agent count")

        local_observations = rollout["local_observations"].to(self.device)
        global_states = rollout["global_states"].to(self.device)
        actions = rollout["actions"].to(self.device)
        old_log_probabilities = rollout["log_probabilities"].to(self.device)
        rewards = rollout["rewards"].to(self.device)
        dones = rollout["dones"].to(self.device)
        old_values = rollout["values"].to(self.device)

        self.critic.eval()
        with torch.no_grad():
            final_values = self._values_for(final_local, final_global)
            final_values = final_values * (~final_done_tensor).float()
            advantages, returns = self._compute_gae(rewards, old_values, dones, final_values)

        time_steps, num_agents, _ = local_observations.shape
        flat_local = local_observations.reshape(time_steps * num_agents, -1)
        flat_global = global_states.unsqueeze(1).expand(-1, num_agents, -1).reshape(time_steps * num_agents, -1)
        flat_actions = actions.reshape(-1)
        flat_old_log_probabilities = old_log_probabilities.reshape(-1)
        flat_old_values = old_values.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std(unbiased=False) + 1e-8)

        sample_count = flat_actions.shape[0]
        aggregate = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        optimization_steps = 0
        self.actor.train()
        self.critic.train()
        for _ in range(self.update_epochs):
            indices = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.minibatch_size):
                batch_indices = indices[start : start + self.minibatch_size]
                distribution = Categorical(logits=self.actor(flat_local[batch_indices]))
                new_log_probabilities = distribution.log_prob(flat_actions[batch_indices])
                ratios = torch.exp(new_log_probabilities - flat_old_log_probabilities[batch_indices])
                unclipped_actor_loss = ratios * flat_advantages[batch_indices]
                clipped_actor_loss = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * flat_advantages[batch_indices]
                actor_loss = -torch.minimum(unclipped_actor_loss, clipped_actor_loss).mean()

                critic_inputs = torch.cat((flat_global[batch_indices], flat_local[batch_indices]), dim=-1)
                new_values = self.critic(critic_inputs).squeeze(-1)
                clipped_values = flat_old_values[batch_indices] + (new_values - flat_old_values[batch_indices]).clamp(
                    -self.value_clip, self.value_clip
                )
                unclipped_value_error = (new_values - flat_returns[batch_indices]).square()
                clipped_value_error = (clipped_values - flat_returns[batch_indices]).square()
                critic_loss = 0.5 * torch.maximum(unclipped_value_error, clipped_value_error).mean()
                entropy = distribution.entropy().mean()
                total_loss = actor_loss + self.value_coefficient * critic_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), self.max_gradient_norm
                )
                self.optimizer.step()

                aggregate["actor_loss"] += actor_loss.item()
                aggregate["critic_loss"] += critic_loss.item()
                aggregate["entropy"] += entropy.item()
                aggregate["approx_kl"] += (flat_old_log_probabilities[batch_indices] - new_log_probabilities).mean().item()
                aggregate["clip_fraction"] += (torch.abs(ratios - 1.0) > self.ppo_clip).float().mean().item()
                optimization_steps += 1

        self.actor.eval()
        self.critic.eval()
        self.rollout.clear()
        self.update_count += 1
        return {name: value / optimization_steps for name, value in aggregate.items()}

    def checkpoint(self) -> dict[str, object]:
        """Return all state required to resume MAPPO training."""
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "local_observation_dim": self.local_observation_dim,
            "global_state_dim": self.global_state_dim,
            "num_actions": self.num_actions,
            "actor_hidden_dims": self.actor_hidden_dims,
            "critic_hidden_dims": self.critic_hidden_dims,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ppo_clip": self.ppo_clip,
            "value_clip": self.value_clip,
            "entropy_coefficient": self.entropy_coefficient,
            "value_coefficient": self.value_coefficient,
            "max_gradient_norm": self.max_gradient_norm,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "update_count": self.update_count,
            "environment_steps": self.environment_steps,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def save(self, checkpoint_path: str | Path) -> None:
        """Save model, optimizer, architecture, and training counters in one file."""
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), destination)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: torch.device | str | None = None) -> "MAPPO":
        """Build a MAPPO instance with the checkpoint's architecture and load its weights."""
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        policy = cls(
            local_observation_dim=checkpoint["local_observation_dim"],
            global_state_dim=checkpoint["global_state_dim"],
            num_actions=checkpoint["num_actions"],
            actor_hidden_dims=checkpoint["actor_hidden_dims"],
            critic_hidden_dims=checkpoint["critic_hidden_dims"],
            device=device,
        )
        policy.load(checkpoint_path)
        return policy

    def load(self, checkpoint_path: str | Path) -> None:
        """Restore a checkpoint into a MAPPO instance with matching dimensions."""
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported MAPPO checkpoint version")
        for name, expected_value in {
            "local_observation_dim": self.local_observation_dim,
            "global_state_dim": self.global_state_dim,
            "num_actions": self.num_actions,
        }.items():
            if checkpoint.get(name) != expected_value:
                raise ValueError(f"checkpoint {name} does not match this MAPPO instance")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint["update_count"]
        self.environment_steps = checkpoint["environment_steps"]
        self.actor.to(self.device).eval()
        self.critic.to(self.device).eval()
