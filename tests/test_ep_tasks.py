"""Two-link fixed EP contracts, reset subsets and target-aware evaluation."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from conftest import ROOT
from cart_double_pendulum.equilibria import absolute_target, joint_target, target_errors, target_mask, validate_ep
from cart_double_pendulum.tasks import tip
from cart_double_pendulum.training.task_directory import read_json, reward_terms


@pytest.mark.parametrize('ep', range(4))
def test_target_geometry_reward_and_mask(ep):
    target = absolute_target(ep)
    assert target == (math.pi if ep & 2 else 0, math.pi if ep & 1 else 0)
    q = torch.tensor([[0., *joint_target(ep)]])
    v = torch.zeros_like(q)
    reward = tip.reward_components(q, v, torch.zeros(1, 1), 5, ep)[0]
    assert reward.item() == pytest.approx(1)
    for other in range(4):
        if other != ep:
            assert tip.reward_components(q, v, torch.zeros(1, 1), 5, other)[0].item() < 1e-6
    assert all(np.allclose(error, 0) for error in target_errors(q.numpy(), np, ep))
    folder = ROOT / f'tasks/real/sac_ep{ep}'
    handoff = read_json(folder / 'evaluation.json')['handoff']
    assert target_mask(q.numpy(), v.numpy(), np, ep, **handoff).all()
    assert reward_terms(folder)[0]['tip_transition'].params['target_ep'] == ep


def test_default_reward_preserved():
    q, v, a = torch.randn(12, 3), torch.randn(12, 3), torch.randn(12, 1)
    old = (torch.exp(-.4 * a[:, 0].square()) * torch.exp(-.3 * q[:, 0].abs())
        * ((1 + torch.cos(torch.cumsum(q[:, 1:], -1) - math.pi)) * .5).prod(-1)
        * torch.exp(-torch.cumsum(v[:, 1:], -1).abs() * torch.tensor([.015, .009])).prod(-1))
    assert torch.equal(old, tip.reward_components(q, v, a)[0])


@pytest.mark.parametrize('ep', [-1,4,True,1.5])
def test_invalid_ep(ep):
    with pytest.raises(ValueError): validate_ep(ep)


@pytest.mark.parametrize('ep', range(4))
def test_subset_reset_stays_on_native_plant(ep, monkeypatch):
    from cart_double_pendulum.common import CONTRACT
    writes = []
    asset = SimpleNamespace(write_joint_state_to_sim=lambda q,v,**kw: writes.append((q,v,kw)))
    env = SimpleNamespace(device='cpu',num_envs=7,common_step_counter=0,scene={tip.mdp.ASSET:asset},cfg=SimpleNamespace(plant_contract=CONTRACT))
    monkeypatch.setattr(tip.mdp, '_ramp_progress', lambda _: 0.)
    tip.reset_ep_curriculum(env, torch.tensor([1,4]), target_ep=ep, near_probability=(1,1))
    q,v,kw = writes[0]
    assert q.shape == v.shape == (2,3)
    assert kw['env_ids'].tolist() == [1,4]
    assert torch.isfinite(q).all() and torch.isfinite(v).all()
    assert (abs(q[:,0]) < CONTRACT.rail_limit_m).all()


def test_evaluation_uses_target_not_upright():
    from cart_double_pendulum.training.ep_evaluation import evaluate_ep_controller
    definition = read_json(ROOT / 'tasks/real/sac_ep0/evaluation.json')
    definition['reset_cart_m'] = definition['reset_angle_rad'] = 0
    for case in definition['conditions']:
        case.update(initial_state='ep0', duration_s=.1, hold_s=.04, randomized=False)
    metrics = evaluate_ep_controller(lambda _: lambda observation: 0., definition, 1)
    assert metrics['success_rate'] == 1
    definition['target_ep'] = 3
    assert evaluate_ep_controller(lambda _: lambda observation: 0., definition, 1)['success_rate'] == 0


def test_tip_ep_shared_training_profile():
    """Keep optimizer/budget settings comparable; task reward/reset may differ."""
    import json
    names = ['tip_swing_sac', *(f'sac_ep{ep}' for ep in range(4))]
    configs = [json.loads((ROOT / 'tasks/real' / name / 'training.json').read_text()) for name in names]
    baseline = {key: value for key, value in configs[0].items() if key != 'seed'}
    for name, config in zip(names, configs):
        assert {key: value for key, value in config.items() if key != 'seed'} == baseline, name
        assert read_json(ROOT / 'tasks/real' / name / 'environment.json')['episode_length_s'] == 10
