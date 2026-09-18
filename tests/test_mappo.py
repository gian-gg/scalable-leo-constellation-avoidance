import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitzoo.rl_algorithms.mappo import MAPPO


@pytest.fixture
def algorithm():
    torch.manual_seed(4)
    return MAPPO(
        local_observation_dim=3,
        global_state_dim=5,
        num_actions=7,
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        update_epochs=2,
        minibatch_size=8,
    )


def sample_state():
    local_observations = torch.tensor(
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.4, 0.5, 0.6], [0.6, 0.5, 0.4]]
    )
    global_state = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    return local_observations, global_state


def test_actor_returns_a_shared_seven_action_distribution(algorithm):
    local_observations, _ = sample_state()

    probabilities = algorithm.action_probabilities(local_observations)

    assert probabilities.shape == (4, 7)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4))
    assert algorithm.actor[0].in_features == 3
    assert algorithm.actor[-1].out_features == 7


def test_critic_receives_global_state_and_local_agent_context(algorithm):
    local_observations, global_state = sample_state()
    _, _, values = algorithm.act(local_observations, global_state, deterministic=True)

    assert values.shape == (4,)
    assert algorithm.critic[0].in_features == 8


def test_decentralized_action_selection_does_not_require_global_state(algorithm):
    local_observations, _ = sample_state()

    actions = algorithm.select_actions(local_observations)

    assert actions.shape == (4,)
    assert torch.all((0 <= actions) & (actions < 7))


def test_update_changes_actor_weights_and_clears_rollout(algorithm):
    local_observations, global_state = sample_state()
    initial_weights = algorithm.actor[0].weight.detach().clone()
    for timestep in range(3):
        actions, log_probabilities, values = algorithm.act(local_observations, global_state)
        rewards = torch.tensor([1.0, 0.5, -0.25, 0.75]) + timestep * 0.1
        algorithm.store_step(
            local_observations,
            global_state,
            actions,
            log_probabilities,
            rewards,
            torch.zeros(4, dtype=torch.bool),
            values,
        )

    metrics = algorithm.update(local_observations, global_state, torch.ones(4, dtype=torch.bool))

    assert not torch.equal(initial_weights, algorithm.actor[0].weight)
    assert len(algorithm.rollout) == 0
    assert algorithm.update_count == 1
    assert set(metrics) == {"actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction"}


def test_save_and_load_restores_actor_output(algorithm, tmp_path):
    local_observations, _ = sample_state()
    before = algorithm.action_probabilities(local_observations).detach().clone()
    checkpoint = tmp_path / "mappo.pt"

    algorithm.save(checkpoint)
    restored = MAPPO(
        local_observation_dim=3,
        global_state_dim=5,
        num_actions=7,
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        update_epochs=2,
        minibatch_size=8,
    )
    restored.load(checkpoint)

    assert torch.allclose(before, restored.action_probabilities(local_observations))
    assert restored.environment_steps == 0


def test_rollout_rejects_agent_count_changes(algorithm):
    local_observations, global_state = sample_state()
    actions, log_probabilities, values = algorithm.act(local_observations, global_state)
    algorithm.store_step(
        local_observations,
        global_state,
        actions,
        log_probabilities,
        torch.zeros(4),
        torch.zeros(4, dtype=torch.bool),
        values,
    )

    with pytest.raises(ValueError, match="number of agents"):
        algorithm.store_step(
            local_observations[:3],
            global_state,
            actions[:3],
            log_probabilities[:3],
            torch.zeros(3),
            torch.zeros(3, dtype=torch.bool),
            values[:3],
        )


def test_values_match_the_critic_for_every_agent(algorithm):
    local_observations, global_state = sample_state()

    _, _, act_values = algorithm.act(local_observations, global_state)

    torch.testing.assert_close(algorithm.values(local_observations, global_state), act_values)
