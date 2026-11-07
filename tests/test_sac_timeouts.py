"""Timeout replay must use the last state/history and still bootstrap its value."""
import pytest
import torch
from mjlab.managers import EventTermCfg, TerminationTermCfg

from cart_double_pendulum.tasks import make_task, mdp
from cart_double_pendulum.training.sac_env import SACEnvironment, replay_observations
from cart_double_pendulum.rl.sac import ReplayBuffer, SACAgent, SACConfig


@pytest.mark.parametrize('frames', [0, 4])
def test_mixed_timeout_failure_and_live_history(frames):
    cfg = make_task('swing_up').make_env_cfg(3, actor_history_frames=frames)
    # Keep actor and asymmetric critic histories the same width as full SAC.
    if frames:
        cfg.observations['critic'].terms['state'] = cfg.observations['actor'].terms['state'].__class__(
            func=mdp.actor_history_frame, params={'contract': cfg.plant_contract},
            history_length=frames, flatten_history_dim=True)
    def reset(env, env_ids):
        n = len(env_ids)
        q = torch.zeros(n, 3)
        q[:, 1] = .2
        env.scene[mdp.ASSET].write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=slice(0, 3), env_ids=env_ids)
    def failure(env):
        return (torch.arange(env.num_envs) == 1) & (env.common_step_counter >= 1)
    cfg.events = {'reset_system': EventTermCfg(func=reset, mode='reset')}
    cfg.terminations['failure'] = TerminationTermCfg(func=failure)
    cfg.rewards = {}
    cfg.episode_length_s = .06
    env = SACEnvironment(cfg=cfg, device='cpu')
    try:
        obs, _ = env.reset(seed=42)
        previous_history = obs['actor'].reshape(3, frames, 9).clone() if frames else None
        env.episode_length_buf[:] = torch.tensor([2, 2, 0])
        next_obs, reward, term, trunc, info = env.step(torch.tensor([[.1], [.2], [.3]]))
        assert term.tolist() == [False, True, False]
        assert trunc.tolist() == [True, True, False]
        successor = replay_observations(next_obs, term, trunc, info)
        assert not torch.equal(successor['actor'][0], next_obs['actor'][0])
        assert torch.equal(successor['actor'][2], next_obs['actor'][2])
        assert torch.equal(successor['critic'][0], info['final_observation']['critic'][0])
        if frames:
            terminal_window = successor['actor'][0].reshape(frames, 9)
            assert torch.equal(terminal_window[:-1], previous_history[0, 1:])
            assert terminal_window[-1, -1] == pytest.approx(.1)
            reset_window = next_obs['actor'][0].reshape(frames, 9)
            assert torch.equal(reset_window, reset_window[-1:].expand_as(reset_window))
            live_window = next_obs['actor'][2].reshape(frames, 9)
            assert torch.equal(live_window[:-1], previous_history[2, 1:])
        replay = ReplayBuffer(6, obs['actor'].shape[-1], 1, 'cpu', asymmetric_critic=True,
                              critic_obs_dim=obs['critic'].shape[-1])
        replay.add(obs['actor'], torch.zeros(3, 1), reward, successor['actor'], term,
                   obs['critic'], successor['critic'])
        assert replay.dones[:3, 0].tolist() == [0., 1., 0.]
        assert torch.equal(replay.next_observations[0], info['final_observation']['actor'][0])
        # A call with no ended worlds cannot reuse the previous final observation.
        env.common_step_counter = -10
        env.episode_length_buf[:] = 0
        _, _, _, _, second_info = env.step(torch.zeros(3, 1))
        assert 'final_observation' not in second_info
    finally:
        env.close()


def test_timeout_without_final_state_is_rejected():
    with pytest.raises(RuntimeError, match='pre-reset final observation'):
        replay_observations({'actor': torch.zeros(1, 8)}, torch.tensor([False]), torch.tensor([True]), {})


def test_sac_value_target_bootstraps_only_nonterminal_states(monkeypatch):
    from cart_double_pendulum.rl.sac import TransitionBatch
    agent = SACAgent(8, 1, 'cpu', SACConfig(batch_size=2))
    for target in agent.targets:
        for parameter in target.parameters():
            parameter.data.zero_()
        target.body[-1].bias.data.fill_(3.)
    monkeypatch.setattr(agent.actor, 'sample', lambda x, *args: (torch.zeros(len(x), 1, requires_grad=True), torch.zeros(len(x), 1)))
    observed = []
    original = torch.nn.functional.mse_loss
    def mse(prediction, target, *args, **kw):
        observed.append(target.detach().clone())
        return original(prediction, target, *args, **kw)
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', mse)
    agent.update(TransitionBatch(torch.zeros(2, 8), torch.zeros(2, 1), torch.ones(2, 1),
                                 torch.zeros(2, 8), torch.tensor([[0.], [1.]])))
    assert observed[0][:, 0].tolist() == pytest.approx([1 + agent.config.gamma * 3, 1])
