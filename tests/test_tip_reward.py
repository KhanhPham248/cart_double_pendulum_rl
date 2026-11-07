import math
from types import SimpleNamespace
import pytest
import torch
from conftest import ROOT
from cart_double_pendulum.tasks import tip
from cart_double_pendulum.training.task_directory import reward_terms


def test_absolute_coordinates_and_rates():
    q = torch.tensor([[0., math.pi, 0.], [0., math.pi, math.pi], [0., 0., 0.]])
    v = torch.tensor([[0., 2., -2.], [0., 0., 0.], [0., 0., 0.]])
    reward, parts = tip.reward_components(q, v, torch.zeros(3, 1))
    assert reward[0].item() == pytest.approx(math.exp(-.03))
    assert parts['rates'][0, 1] == 1
    assert torch.allclose(reward[1:], torch.zeros(2), atol=1e-7)
    q[0, 1] += 2 * math.pi
    assert tip.reward_components(q, v, torch.zeros(3, 1))[0][0] == pytest.approx(reward[0])


def test_capture_braking_and_command(monkeypatch):
    q = torch.tensor([[0., math.pi, 0.], [0., math.pi, 0.]])
    v = torch.tensor([[0., 0., 0.], [1., 10., -10.]])
    env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.tensor([[0.], [1.]])))
    monkeypatch.setattr(tip.mdp, 'state', lambda _: (q, v))
    terms, scale = reward_terms(ROOT / 'tasks/real/tip_swing_sac')
    total = sum(term.weight * term.func(env, **term.params) for term in terms.values() if term.func.__name__ != 'rail_terminal_cost')
    assert not scale
    assert total[0].item() == pytest.approx(5.)
    assert 0 <= total[1] < total[0]
    reward, _ = tip.reward_components(q, torch.zeros_like(v), env.action_manager.action)
    assert reward[1].item() == pytest.approx(math.exp(-.4))


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_rate_scale(scale):
    with pytest.raises(ValueError):
        tip.reward_components(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 1), scale)
