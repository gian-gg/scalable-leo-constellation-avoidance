"""The first OrbitZoo-backed collision-avoidance task for shared MAPPO.

This module deliberately owns the RL task definition rather than extending
OrbitZoo's legacy single-agent training loop.  It returns dense arrays in the
same layout expected by :class:`orbitzoo.rl_algorithms.mappo.MAPPO`.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Sequence

import numpy as np

from orbitzoo.env import OrbitZoo
from orbitzoo.thesis.environments.diagnostics import EpisodeDiagnostics
from orbitzoo.thesis.environments.observations import LocalObservationEncoder
from orbitzoo.thesis.environments.rewards import RewardConfig, calculate_rewards
from orbitzoo.thesis.environments.safety import (
    PairSafetyAssessment,
    SafetyConfig,
    assess_all_pairs,
)
from orbitzoo.thesis.maneuvers.actions import ManeuverAction
from orbitzoo.thesis.maneuvers.contract import (
    ManeuverCommand,
    ManeuverConfig,
    ManeuverInfeasibleError,
    build_maneuver_command,
    measure_maneuver_result,
    orbitzoo_action_inputs,
)


class CollisionAvoidanceEnv(OrbitZoo):
    """A synchronous, fixed-agent collision-avoidance environment.

    Each actor sees its own state followed by a fixed number of threat-ranked
    relative-neighbor blocks. The centralized critic receives a separate full
    state whose width is allowed to depend on the training population.
    """

    def __init__(
        self,
        *,
        maneuver_config: ManeuverConfig,
        safety_config: SafetyConfig | None = None,
        reward_config: RewardConfig | None = None,
        neighborhood_size: int = 1,
        decision_interval_seconds: float = 120.0,
        episode_horizon: int = 100,
        **orbitzoo_kwargs: Any,
    ) -> None:
        if orbitzoo_kwargs.get("dynamics_library", "orekit") != "orekit":
            raise ValueError("CollisionAvoidanceEnv requires OrbitZoo's Orekit dynamics")
        if episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")
        if decision_interval_seconds <= 0:
            raise ValueError("decision_interval_seconds must be positive")
        maneuver_config.validate()
        if maneuver_config.maximum_burn_duration_seconds > decision_interval_seconds:
            raise ValueError("maximum_burn_duration_seconds cannot exceed decision_interval_seconds")
        self.decision_interval_seconds = decision_interval_seconds
        self.maneuver_config = maneuver_config
        self.safety_config = safety_config or SafetyConfig()
        self.reward_config = reward_config or RewardConfig()
        self.safety_config.validate()
        self.reward_config.validate()
        self.observation_encoder = LocalObservationEncoder(
            neighborhood_size, self.safety_config
        )
        self.episode_horizon = episode_horizon
        self.agent_names: list[str] = []
        self.step_index = 0
        self.is_terminated = True
        self.diagnostics = EpisodeDiagnostics([])
        super().__init__(**orbitzoo_kwargs)

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    @property
    def local_observation_dim(self) -> int:
        return self.observation_encoder.local_observation_dim

    @property
    def global_state_dim(self) -> int:
        return self.observation_encoder.global_state_dim(len(self._moving_bodies()))

    def _moving_bodies(self) -> list[Any]:
        return list(self.dynamics.get_moving_bodies())

    def _spacecraft_by_name(self) -> dict[str, Any]:
        return {spacecraft.name: spacecraft for spacecraft in self.dynamics.spacecrafts}

    def _state(
        self, assessments: Sequence[PairSafetyAssessment] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self.observation_encoder.encode(
            self._moving_bodies(), self.agent_names, assessments
        )
        return state.local_observations, state.global_state

    def _assessments(self) -> list[PairSafetyAssessment]:
        return assess_all_pairs(self._moving_bodies(), self.safety_config)

    def unsafe_assessments(self) -> list[PairSafetyAssessment]:
        """Assessments of body pairs currently predicted to pass within the safe separation."""
        return [assessment for assessment in self._assessments() if assessment.is_unsafe]

    @staticmethod
    def _minimum_separation(assessments: list[PairSafetyAssessment]) -> float:
        return min(
            (assessment.current_separation_meters for assessment in assessments),
            default=float("inf"),
        )

    def _validate_spacecraft_contract(self) -> None:
        if not self.agent_names:
            raise ValueError("CollisionAvoidanceEnv requires at least one maneuvering spacecraft")
        for spacecraft in self.dynamics.spacecrafts:
            if not math.isclose(
                spacecraft.isp,
                self.maneuver_config.specific_impulse_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"spacecraft {spacecraft.name!r} has isp={spacecraft.isp}, but the maneuver "
                    "contract must use the same specific impulse"
                )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Reset OrbitZoo and return local actor observations plus global critic state."""
        super().reset(seed)
        self.agent_names = list(self.dynamics.spacecraft_names)
        self._validate_spacecraft_contract()
        self.step_index = 0
        self.is_terminated = False
        self.diagnostics = EpisodeDiagnostics(self.agent_names)
        assessments = self._assessments()
        self.diagnostics.record_minimum_separation(self._minimum_separation(assessments))
        return self._state(assessments)

    def _commands_for_actions(
        self, action_ids: np.ndarray
    ) -> tuple[dict[str, ManeuverCommand], set[str]]:
        spacecraft = self._spacecraft_by_name()
        commands: dict[str, ManeuverCommand] = {}
        rejected_agents: set[str] = set()
        for name, action_id in zip(self.agent_names, action_ids, strict=True):
            body = spacecraft[name]
            try:
                commands[name] = build_maneuver_command(
                    int(action_id),
                    body.get_mass(),
                    self.maneuver_config,
                    available_propellant_kg=body.get_fuel(),
                )
            except ManeuverInfeasibleError:
                commands[name] = build_maneuver_command(
                    ManeuverAction.NO_OP, body.get_mass(), self.maneuver_config
                )
                rejected_agents.add(name)
        return commands, rejected_agents

    def step(
        self, action_ids: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply one simultaneous discrete action per satellite and propagate once."""
        if self.is_terminated:
            raise RuntimeError("call reset() before step(), or after an episode terminates")
        actions = np.asarray(action_ids, dtype=np.int64)
        if actions.shape != (self.num_agents,):
            raise ValueError("action_ids must have shape [num_agents]")
        if np.any(actions < int(ManeuverAction.NO_OP)) or np.any(actions > int(ManeuverAction.CROSS_TRACK_NEGATIVE)):
            raise ValueError("action IDs must be integers in [0, 6]")

        assessments_before = self._assessments()
        commands, rejected_agents = self._commands_for_actions(actions)
        thrusts, durations = orbitzoo_action_inputs(commands)
        spacecraft_before = self._spacecraft_by_name()
        masses_before = {name: spacecraft_before[name].get_mass() for name in self.agent_names}
        super().step(
            step_size=self.decision_interval_seconds,
            actions=thrusts,
            maneuver_durations=durations,
        )
        spacecraft_after = self._spacecraft_by_name()
        results = {
            name: measure_maneuver_result(
                commands[name],
                masses_before[name],
                spacecraft_after[name].get_mass(),
                spacecraft_after[name].isp,
            )
            for name in self.agent_names
        }
        assessments_after = self._assessments()
        rewards = calculate_rewards(
            self.agent_names,
            commands,
            results,
            assessments_before,
            assessments_after,
            self.reward_config,
            rejected_agents,
        )
        self.step_index += 1
        collision_pairs = [assessment.pair for assessment in assessments_after if assessment.is_collision]
        self.is_terminated = bool(collision_pairs) or self.step_index >= self.episode_horizon
        self.diagnostics.record_maneuvers(results)
        self.diagnostics.record_minimum_separation(self._minimum_separation(assessments_after))
        self.diagnostics.collision_pairs.extend(collision_pairs)
        local_observations, global_state = self._state(assessments_after)
        dones = np.full(self.num_agents, self.is_terminated, dtype=bool)
        info = {
            "step_index": self.step_index,
            "termination_reason": (
                "collision" if collision_pairs else "horizon" if self.is_terminated else None
            ),
            "rejected_agents": sorted(rejected_agents),
            "collision_pairs": collision_pairs,
            "unsafe_pairs": [assessment.pair for assessment in assessments_after if assessment.is_unsafe],
            "minimum_separation_meters": self.diagnostics.minimum_separation_meters,
            "assessments": [asdict(assessment) for assessment in assessments_after],
            "maneuvers": {
                name: {
                    "action": int(result.command.action),
                    "burn_duration_seconds": result.command.burn_duration_seconds,
                    "fuel_consumed_kg": result.fuel_consumed_kg,
                    "actual_delta_v_mps": result.actual_delta_v_mps,
                }
                for name, result in results.items()
            },
        }
        return local_observations, global_state, np.asarray(
            [rewards[name] for name in self.agent_names], dtype=np.float32
        ), dones, info


def development_environment_kwargs() -> dict[str, Any]:
    """Return a small deterministic conjunction fixture for integration work.

    It is intentionally *not* a final training-scenario generator.  Four
    spacecraft are spaced around a circular 500 km orbit; a non-maneuvering
    object begins 2 km from the first spacecraft with a closing along-track
    velocity so the safety screen has a known conjunction to report.
    """
    earth_radius = 6_378_136.3
    gravitational_parameter = 3.986004418e14
    orbit_radius = earth_radius + 500_000.0
    circular_speed = math.sqrt(gravitational_parameter / orbit_radius)

    def circular_state(angle: float) -> list[float]:
        return [
            orbit_radius * math.cos(angle),
            orbit_radius * math.sin(angle),
            0.0,
            -circular_speed * math.sin(angle),
            circular_speed * math.cos(angle),
            0.0,
        ]

    spacecrafts = [
        {
            "name": f"satellite_{index + 1}",
            "initial_state": circular_state(index * math.pi / 2),
            "dry_mass": 200.0,
            "initial_fuel_mass": 50.0,
            "isp": 300.0,
            "radius": 1.0,
            "forces": ["gravity_newton"],
        }
        for index in range(4)
    ]
    debris_state = circular_state(0.0)
    debris_state[1] += 2_000.0
    debris_state[4] -= 10.0
    return {
        "dynamics_library": "orekit",
        "step_size": 30.0,
        "initial_epoch": {"year": 2026, "month": 1, "day": 1, "hour": 0, "minute": 0, "second": 0},
        "spacecrafts": spacecrafts,
        "drifters": [
            {
                "name": "development_debris",
                "initial_state": debris_state,
                "dry_mass": 10.0,
                "radius": 1.0,
                "forces": ["gravity_newton"],
            }
        ],
    }
