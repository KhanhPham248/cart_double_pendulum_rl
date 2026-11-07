"""Five-millimeter centering is a physical task requirement, not a scaled rail fraction."""
import copy
import math
import pytest
import torch
from conftest import ROOT
from cart_double_pendulum.tasks import tip
from cart_double_pendulum.training.task_directory import read_json, reward_terms


@pytest.mark.parametrize('ep', range(4))
def test_center_bonus_has_far_gradient_and_respects_target(ep, monkeypatch):
    from cart_double_pendulum.equilibria import joint_target
    q = torch.tensor([[x, *joint_target(ep)] for x in [0., .005, .01, .05, .1]], requires_grad=True)
    v = torch.zeros_like(q)
    monkeypatch.setattr(tip.mdp, 'state', lambda _: (q, v))
    term = reward_terms(ROOT / f'tasks/real/sac_ep{ep}')[0]['precision_center']
    bonus = term.func(None, **term.params)
    assert bonus[0].item() == pytest.approx(1.)
    assert torch.all(bonus[:-1] > bonus[1:])
    bonus.sum().backward()
    assert q.grad[-1, 0] < 0  # a narrow Gaussian alone would starve this gradient
    assert torch.isfinite(q.grad).all()
    bad_q = q.detach().clone(); bad_q[:, 1] += math.pi
    monkeypatch.setattr(tip.mdp, 'state', lambda _: (bad_q, v))
    assert (term.func(None, **term.params) < bonus.detach()).all()


def test_ep_evaluator_requires_center_even_when_angle_target_is_held(monkeypatch):
    from cart_double_pendulum.training import ep_evaluation as ev
    from cart_double_pendulum.runtime import CartDoublePendulumRuntime
    definition = read_json(ROOT / 'tasks/real/sac_ep0/evaluation.json')
    definition.update(reset_cart_m=0, reset_angle_rad=0)
    for c in definition['conditions']:
        c.update(initial_state='ep0', duration_s=.1, hold_s=.04, randomized=False)
    original = CartDoublePendulumRuntime.step
    offset = [.006]
    def fixed(self, action):
        observation = original(self, action)
        self.data.qpos[self.qpos_indices[0]] = offset[0]
        self.data.qvel[:] = 0
        return observation
    monkeypatch.setattr(CartDoublePendulumRuntime, 'step', fixed)
    assert ev.evaluate_ep_controller(lambda _: lambda obs: 0., definition, 1)['success_rate'] == 0
    offset[0] = .005
    assert ev.evaluate_ep_controller(lambda _: lambda obs: 0., definition, 1)['success_rate'] == 1
    legacy = copy.deepcopy(definition); legacy.pop('centering'); legacy['protocol'] = 'fixed_ep_validation'
    offset[0] = .006
    assert ev.evaluate_ep_controller(lambda _: lambda obs: 0., legacy, 1)['success_rate'] == 1


def test_tip_robust_evaluator_center_requirement_is_opt_in(monkeypatch):
    from cart_double_pendulum.training import robust_eval as ev
    from cart_double_pendulum.runtime import CartDoublePendulumRuntime
    monkeypatch.setattr(ev, 'stable', lambda *_: True)
    original = CartDoublePendulumRuntime.step
    offset = [.006]
    def fixed(self, action):
        observation = original(self, action)
        self.data.qpos[self.qpos_indices[0]] = offset[0]
        return observation
    monkeypatch.setattr(CartDoublePendulumRuntime, 'step', fixed)
    case = ev.RobustCondition('center_probe', 'swing_up', 'hanging', .1, .04)
    factory = lambda _: lambda obs: 0.
    assert ev.evaluate_controller(factory, case, episodes=1)[0]['success_rate'] == 1
    assert ev.evaluate_controller(factory, case, episodes=1, cart_tolerance_m=.005)[0]['success_rate'] == 0
    offset[0] = .005
    assert ev.evaluate_controller(factory, case, episodes=1, cart_tolerance_m=.005)[0]['success_rate'] == 1


@pytest.mark.parametrize('name', ['tip_swing_sac', 'sac_ep0', 'sac_ep1', 'sac_ep2', 'sac_ep3'])
def test_task_reward_and_evaluation_agree_on_five_mm(name):
    folder = ROOT / 'tasks/real' / name
    assert read_json(folder / 'evaluation.json')['centering']['cart_tolerance_mm'] == 5.
    assert reward_terms(folder)[0]['precision_center'].params['cart_tolerance_mm'] == 5.
