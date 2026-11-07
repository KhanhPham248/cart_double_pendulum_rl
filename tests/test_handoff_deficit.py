"""Handoff deficit cost: a function of the state only, lower for near misses, zero inside handoff."""
import math
import json
from types import SimpleNamespace

import pytest
import torch

from conftest import ROOT, TASKS
from cart_double_pendulum.tasks import mdp
from cart_double_pendulum.training.task_directory import reward_terms


def world(name='expert_swing_sac', count=1, key='handoff_deficit'):
    term = reward_terms(ROOT / 'tasks/real' / name)[0][key]
    q, v = torch.zeros(count, 3), torch.zeros(count, 3)
    env = SimpleNamespace(num_envs=count, device='cpu', step_dt=.02, common_step_counter=0,
        scene={mdp.ASSET: SimpleNamespace(data=SimpleNamespace(joint_pos=q, joint_vel=v))},
        termination_manager=SimpleNamespace(terminated=torch.zeros(count, dtype=torch.bool)))
    return env, (lambda env: term.func(env, **term.params)), term, q, v


HANDOFF_TASKS = tuple(name for name in TASKS if 'handoff_deficit' in
    json.loads((ROOT / 'tasks/real' / name / 'reward.json').read_text())['terms'])


@pytest.mark.parametrize('name', HANDOFF_TASKS)
def test_handoff_tasks_have_one_bounded_cost_without_a_clock(name):
    env, cost, term, _, _ = world(name)
    samples = []
    for step in (0, 499, 500, 5000):
        env.common_step_counter = step
        samples.append(float(cost(env)[0]))
    assert 0. < samples[0] < 1. and len(set(samples)) == 1
    assert term.weight < 0 and 'discount' not in term.params


def test_near_misses_cost_less_and_losing_handoff_costs_again():
    env, cost, term, q, _ = world(count=3)
    hanging = cost(env).clone()
    q[0, 1] = math.pi - 1.5 * term.params['angle_threshold_rad']
    q[1, 1] = math.pi
    near_and_captured = cost(env)
    assert 0 < near_and_captured[0] < hanging[0]
    assert near_and_captured[1] == 0.
    q[1].zero_()
    assert cost(env)[1] == hanging[1]
    env.termination_manager.terminated[2] = True
    assert cost(env)[2] == 0.


def test_improving_only_one_pole_helps_even_when_the_other_is_still_hanging():
    env, cost, _, q, _ = world(count=2)
    # First world: link 1 upright, link 2 down. Second world: both down.
    q[0, 1], q[0, 2] = math.pi, -math.pi
    result = cost(env)
    assert 0 < result[0] < result[1]


@pytest.mark.parametrize('parameter,value', [('cart_safe_m', 0.), ('angle_threshold_rad', float('nan'))])
def test_invalid_threshold_is_rejected(parameter, value):
    env, _, term, _, _ = world()
    term.params[parameter] = value
    with pytest.raises(ValueError, match='handoff deficit'):
        term.func(env, **term.params)


def test_energy_reward_is_absolute_peaks_at_upright_energy_and_has_no_clock():
    env, energy, term, q, v = world(count=4, key='swing_energy')
    q[1, 1] = math.pi
    q[2, 1] = .3
    q[3, 1] = math.pi
    v[3, 1] = 12.0  # well above the upright energy: a spin
    values = energy(env)
    assert values[0] == 0. and values[1] == pytest.approx(1.)
    assert 0. < values[2] < .5 and values[3] < values[1]
    env.common_step_counter = 5000
    assert torch.equal(energy(env), values)


def test_real_environment_reward_does_not_change_with_time():
    from mjlab.managers import EventTermCfg
    from cart_double_pendulum.tasks import make_task
    from cart_double_pendulum.training.sac_env import SACEnvironment
    cfg = make_task('swing_up').make_env_cfg(2)
    def reset(env, env_ids):
        q = torch.zeros(len(env_ids), 3)
        env.scene[mdp.ASSET].write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=slice(0, 3), env_ids=env_ids)
    cfg.events = {'reset_system': EventTermCfg(func=reset, mode='reset')}
    cfg.rewards = {'handoff_deficit': reward_terms(ROOT / 'tasks/real/expert_swing_sac')[0]['handoff_deficit']}
    env = SACEnvironment(cfg=cfg, device='cpu')
    try:
        obs, _ = env.reset(seed=0)
        assert obs['actor'].shape == (2, 8) and obs['critic'].shape == (2, 8)
        _, early, _, _, _ = env.step(torch.zeros(2, 1))
        early = early.clone()  # mjlab reuses its reward buffer between steps.
        env.common_step_counter = 700
        _, late, terminated, _, _ = env.step(torch.zeros(2, 1))
        assert torch.allclose(late, early) and (late < 0).all() and not terminated.any()
    finally:
        env.close()
