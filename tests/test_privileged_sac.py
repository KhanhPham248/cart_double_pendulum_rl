"""A wider privileged critic input survives replay, updates and checkpoints."""
import pytest
import torch

from cart_double_pendulum.rl.sac import SACAgent, SACConfig, ReplayBuffer


@pytest.mark.parametrize('width', [8, 72])
def test_sac_with_privileged_input_updates_and_restores_without_changing_actor_interface(width):
    agent = SACAgent(width, 1, 'cpu', SACConfig(batch_size=4, hidden_dims=(16, 16)), critic_obs_dim=width+1)
    obs = torch.randn(4, width)
    extra = torch.tensor([[0.], [.2], [.6], [1.]])
    critic = torch.cat((obs, extra), dim=-1)
    agent.observe(obs, critic)
    replay = ReplayBuffer(8, width, 1, 'cpu', critic_obs_dim=width+1)
    replay.add(obs, torch.zeros(4, 1), torch.zeros(4), obs, torch.zeros(4), critic, critic)
    values = agent.update(replay.sample(4))
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    restored = SACAgent(width, 1, 'cpu', agent.config, critic_obs_dim=width+1)
    restored.load_state_dict(agent.state_dict())
    assert torch.equal(agent.act(obs, True), restored.act(obs, True))
    assert torch.equal(agent.normalize_critic(critic), restored.normalize_critic(critic))
    assert restored.act(obs).shape == (4, 1)
    with pytest.raises(ValueError, match='critic dimensions'):
        SACAgent(width, 1, 'cpu', agent.config).load_state_dict(agent.state_dict())


def test_legacy_symmetric_checkpoint_still_loads():
    agent = SACAgent(8, 1, 'cpu', SACConfig(hidden_dims=(16, 16)))
    state = {key: value for key, value in agent.state_dict().items() if not key.startswith('critic_obs')}
    restored = SACAgent(8, 1, 'cpu', agent.config)
    restored.load_state_dict(state)
    obs = torch.randn(2, 8)
    assert torch.equal(agent.act(obs, True), restored.act(obs, True))
