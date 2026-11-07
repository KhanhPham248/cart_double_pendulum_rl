"""Small CPU optimizer/export/observer integration checks, never used as trained policies."""
import json
import inspect
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import ROOT
from cart_double_pendulum.tasks import make_task
from cart_double_pendulum.training import task_directory
from cart_double_pendulum.runtime_policy import OnnxPolicy
from cart_double_pendulum.plant_metadata import plant_metadata


def metrics(*_):
    return dict(success_rate=0., stable_occupancy=0., mean_time_to_success_s=20.,
                mean_force_energy=0., nominal_gate=False)


def observer_update(model):
    """Check BPTT through the observer, including a PPO source's 20-frame history."""
    clean = torch.zeros(2, 32, 8)
    clean[:, :, 3] = -1
    clean[:, :, 5] = 1
    previous = torch.zeros(2, 32, 1)
    predicted, estimate, _, _ = model.sequence(clean + torch.randn_like(clean) * .001, previous)
    desired = model.clean_actions(clean, previous).detach()
    loss = (estimate - clean).square().mean() + (predicted - desired).square().mean()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0003)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()


def configured_task(name, bank, observer=None, short_episode=False):
    folder = ROOT / 'tasks/real' / name
    phase = 'balancing' if name == 'expert_balance_ppo' else 'swing_up'
    environment = task_directory.read_json(folder / 'environment.json')
    def configure(num_envs, **kwargs):
        if observer:
            kwargs['actor_observer_checkpoint'] = str(observer)
        cfg = make_task(phase).make_env_cfg(num_envs, **kwargs)
        cfg.rewards, cfg.scale_rewards_by_dt = task_directory.reward_terms(folder)
        if name in ('full_ppo', 'full_sac'):
            from cart_double_pendulum.training.full_task_curriculum import configure_environment
            configure_environment(cfg, folder, bank_path=str(bank), budget_control_steps=8)
        else:
            reset = environment['reset_system']
            cfg.events['reset_system'].func = task_directory.function(reset['function'])
            params = dict(reset['params'])
            if 'bank_path' in inspect.signature(cfg.events['reset_system'].func).parameters:
                params['bank_path'] = str(bank) if bank else None
            cfg.events['reset_system'].params = params
        if short_episode:
            cfg.episode_length_s = .04
        return cfg
    return SimpleNamespace(make_env_cfg=configure)


@pytest.fixture
def cpu_threads():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


@pytest.mark.parametrize('name', ['expert_swing_sac', 'tip_swing_sac', 'sac_ep0', 'sac_ep1', 'sac_ep2', 'sac_ep3', 'full_sac'])
def test_sac_optimizer_checkpoint_and_observer(name, synthetic_bank, tmp_path, monkeypatch, cpu_threads):
    import train_sac
    from cart_double_pendulum.training.model_temporal_expert import ModelTemporalExpert
    from cart_double_pendulum.training.temporal_expert import export_temporal
    monkeypatch.setattr(train_sac, 'evaluate_checkpoint', metrics)
    monkeypatch.setattr(train_sac, 'make_task', lambda _: configured_task(
        name, synthetic_bank if name == 'full_sac' else None, short_episode=True))
    captured = []
    original_add = train_sac.ReplayBuffer.add
    def recorded_add(buffer, obs, actions, rewards, following, dones, *extra):
        captured.append(dones.clone())
        return original_add(buffer, obs, actions, rewards, following, dones, *extra)
    monkeypatch.setattr(train_sac.ReplayBuffer, 'add', recorded_add)
    if name == 'full_sac':
        from cart_double_pendulum.training.full_task_optimization import FullSACAgent
        monkeypatch.setattr(train_sac, 'SACAgent', FullSACAgent)
    run = tmp_path / 'sac'
    monkeypatch.setattr(sys, 'argv', ['train_sac.py', '--task', 'swing_up', '--device', 'cpu',
        '--run-dir', str(run), '--num-envs', '4', '--total-transitions', '16', '--learning-starts', '4',
        '--replay-capacity', '32', '--batch-size', '4', '--checkpoint-interval', '8', '--robust-training'])
    sys.argv.extend(['--warmup-hold-steps', '5', '--target-entropy', '-.3', '--log-interval', '1'])
    train_sac.main()
    assert captured and not torch.stack(captured).any()  # short time limits bootstrap
    checkpoint = torch.load(run / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['transitions'] == 16
    assert checkpoint['agent']['actor_optimizer']['state']
    assert checkpoint['agent']['critic_obs_dim'] == checkpoint['agent']['obs_dim']  # no clock: the reward has no time
    assert checkpoint['agent']['config']['target_entropy'] == -.3
    model = ModelTemporalExpert(run / 'checkpoints/latest.pt', 'sac')
    observer_update(model)
    path = tmp_path / 'sac_observer.onnx'
    export_temporal(model, path)
    obs = np.array([0, 0, 0, -1, 0, 1, 0, 0], dtype=np.float32)
    policy = OnnxPolicy(path)
    assert np.isfinite(policy(obs))
    assert model.hidden_size == 71 + 9 * model.source_history_frames


@pytest.mark.parametrize('name', ['expert_balance_ppo', 'full_ppo'])
def test_ppo_optimizer_export_and_frozen_observer(name, synthetic_bank, tmp_path, monkeypatch, cpu_threads):
    import train_ppo
    import cart_double_pendulum.training.ppo_runner as runner
    from cart_double_pendulum.training.model_temporal_expert import ModelTemporalExpert
    from cart_double_pendulum.training.ppo_observer_adapter import load_observer, ObservedHistoryActor
    from cart_double_pendulum.training.temporal_expert import export_temporal
    monkeypatch.setattr(runner, 'evaluate_checkpoint', metrics)
    monkeypatch.setattr(train_ppo, 'make_task', lambda _: configured_task(name, synthetic_bank))
    run = tmp_path / name
    argv = ['train_ppo.py', '--task', 'balancing' if name == 'expert_balance_ppo' else 'swing_up',
            '--device', 'cpu', '--run-dir', str(run), '--num-envs', '4', '--max-iterations', '2',
            '--rollout-steps', '4', '--save-interval', '1', '--robust-training', '--require-fresh-start']
    if name == 'expert_balance_ppo':
        argv += ['--actor-history-frames', '20']
    else:
        argv += ['--update-diagnostics', '--reward-scale', '.1']
    monkeypatch.setattr(sys, 'argv', argv)
    train_ppo.main()
    assert (run / 'checkpoints/latest.pt').is_file()
    assert json.loads((run / 'FRESH_START_AUDIT.json').read_text())['fresh_actor']
    policy = OnnxPolicy(run / 'policy.onnx')
    obs = np.array([0, 0, 0, -1, 0, 1, 0, 0], dtype=np.float32)
    assert np.isfinite(policy(obs))
    if name == 'expert_balance_ppo':
        model = ModelTemporalExpert(run / 'checkpoints/latest.pt', 'ppo')
        observer_update(model)
        path = tmp_path / 'observer.pt'
        torch.save({'plant_metadata': plant_metadata(), 'model': model.state_dict(), 'config': {'algorithm': 'ppo', 'observer_model': 'model_gru'}}, path)
        restored = load_observer(path, 'cpu')
        assert restored.source_history_frames == 20
        from cart_double_pendulum.training.expert_actor import ExpertActor
        # Export the new actor with the frozen observer and native frame history.
        actor = ExpertActor(run / 'checkpoints/latest.pt', 'ppo')
        fused = ObservedHistoryActor(restored, actor)
        export_temporal(fused, tmp_path / 'observed_ppo.onnx')
        assert np.isfinite(OnnxPolicy(tmp_path / 'observed_ppo.onnx')(obs))
        # Exercise the production stage-2 path: filter -> 20-frame history -> new PPO actor.
        monkeypatch.setattr(train_ppo, 'make_task', lambda _: configured_task(name, synthetic_bank, path))
        stage2 = tmp_path / 'balance_stage2'
        argv[argv.index('--run-dir') + 1] = str(stage2)
        argv += ['--actor-observer-checkpoint', str(path)]
        monkeypatch.setattr(sys, 'argv', argv)
        train_ppo.main()
        assert np.isfinite(OnnxPolicy(stage2 / 'policy.onnx')(obs))


def test_full_sac_launcher_hook_reaches_the_trainer(monkeypatch):
    import cart_double_pendulum.rl.sac as sac
    from cart_double_pendulum.training.full_task_optimization import FullSACAgent
    monkeypatch.setattr(sac, 'SACAgent', FullSACAgent)
    namespace = runpy.run_path(str(ROOT / 'scripts/train_sac.py'), run_name='_smoke_import')
    assert namespace['SACAgent'] is FullSACAgent




def test_sac_resume_restores_best_and_adds_transitions(tmp_path, monkeypatch, cpu_threads):
    import train_sac
    evaluations = [0]
    def score(*_):
        evaluations[0] += 1
        return dict(metrics(), success_rate=evaluations[0] / 100)
    monkeypatch.setattr(train_sac, 'evaluate_checkpoint', score)
    monkeypatch.setattr(train_sac, 'make_task', lambda _: configured_task('sac_ep3', None, short_episode=True))
    def run(directory, checkpoint=None):
        argv = ['train_sac.py', '--task', 'swing_up', '--device', 'cpu', '--run-dir', str(directory),
            '--num-envs', '4', '--total-transitions', '16', '--learning-starts', '4',
            '--replay-capacity', '32', '--batch-size', '4', '--checkpoint-interval', '8',
            '--log-interval', '1', '--target-entropy', '-1']
        if checkpoint: argv.extend(['--resume-checkpoint', str(checkpoint)])
        monkeypatch.setattr(sys, 'argv', argv)
        train_sac.main()
    first = tmp_path / 'first'; run(first)
    source = first / 'checkpoints/best.pt'
    old = torch.load(source, map_location='cpu', weights_only=False)
    second = tmp_path / 'resumed'; run(second, source)
    audit = json.loads((second / 'resume.json').read_text())
    assert audit['source_transitions'] == old['transitions']
    assert audit['restored_update_count'] == old['agent']['update_count']
    assert audit['replay_restored'] is False and audit['additional_transitions'] == 16
    initial = torch.load(second / f"checkpoints/step_{old['transitions']:012d}.pt", map_location='cpu', weights_only=False)
    for key, value in old['agent']['actor'].items():
        assert torch.equal(initial['agent']['actor'][key], value)
    latest = torch.load(second / 'checkpoints/latest.pt', map_location='cpu', weights_only=False)
    assert latest['transitions'] == old['transitions'] + 16
    assert latest['agent']['update_count'] > old['agent']['update_count']
    assert initial['agent']['actor_optimizer']['state']
