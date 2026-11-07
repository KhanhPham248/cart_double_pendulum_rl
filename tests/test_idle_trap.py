"""Guard against preserving an idle expert or starving the replay of swing motion."""
import copy
import math

import pytest
import torch

from cart_double_pendulum.training.exploration import CenteredWarmup, CorrelatedNoise, SwingProgressGuard
from cart_double_pendulum.training.task_local_evaluation import swing_selection_key


def test_motion_is_selected_before_lower_effort_when_both_experts_fail():
    idle = dict(nominal_gate=False, success_rate=0., nominal_success_rate=0., stable_occupancy=0.,
        mean_time_to_success_s=20., mean_force_energy=0., cases={'nominal': {
            'never_handoff_rate':1.,'numerical_failure_rate':0.,'rail_failure_rate':0.,
            'mean_peak_height_fraction':.01,'mean_peak_pendulum_energy_fraction':.02}})
    moving = copy.deepcopy(idle)
    moving['mean_force_energy'] = 500.
    moving['cases']['nominal'].update(mean_peak_height_fraction=.8, mean_peak_pendulum_energy_fraction=1.)
    assert swing_selection_key(moving) > swing_selection_key(idle)
    # Success still outranks progress; progress must not override task success.
    successful = copy.deepcopy(idle)
    successful.update(nominal_gate=True, success_rate=1., nominal_success_rate=1.)
    assert swing_selection_key(successful) > swing_selection_key(moving)


def test_correlated_noise_is_standard_normal_slow_and_resets_per_world():
    torch.manual_seed(3)
    noise = CorrelatedNoise(4000, 1, 'cpu', correlation_s=.12, step_dt=.02)
    first = noise.sample().clone()
    second = noise.sample().clone()
    assert second.std().item() == pytest.approx(1., abs=.05)
    correlation = torch.corrcoef(torch.cat((first, second), dim=1).T)[0, 1].item()
    assert correlation == pytest.approx(math.exp(-.02 / .12), abs=.03)
    done = torch.zeros(4000, dtype=torch.bool)
    done[0] = True
    kept = noise.value[1:].clone()
    noise.reset(done)
    assert torch.equal(noise.value[1:], kept)


def test_warmup_holds_noise_but_centers_cart_and_resets_each_world():
    torch.manual_seed(17)
    warmup = CenteredWarmup(3, 'cpu', 5, kp=.5, kd=.2)
    zeros = torch.zeros(3)
    first = warmup.sample(zeros, zeros)
    for _ in range(4):
        assert torch.equal(warmup.sample(zeros, zeros), first)
    assert not torch.equal(warmup.sample(zeros, zeros), first)
    warmup.reset(torch.tensor([1]))
    assert warmup.remaining.tolist() == [4, 0, 4]
    warmup.noise.zero_()
    warmup.remaining.fill_(2)
    action = warmup.sample(torch.tensor([1., -1., 0.]), torch.tensor([1., -1., 0.]))
    assert action[:, 0].tolist() == pytest.approx([-.7, .7, 0.])
    assert action.abs().max() <= 1.


@pytest.mark.parametrize('hold,kp', [(0,.5),(1.5,.5),(5,float('nan'))])
def test_invalid_exploration_is_rejected(hold, kp):
    with pytest.raises(ValueError, match='Warmup'):
        CenteredWarmup(2, 'cpu', hold, kp=kp)


def test_idle_guard_waits_for_allowance_and_repeated_failures_then_recovers():
    metrics = {'nominal_success_rate':0., 'cases':{'nominal':{
        'never_handoff_rate':1., 'mean_peak_height_fraction':.01}}}
    guard = SwingProgressGuard(1000000)
    assert not guard.stalled(metrics, 500000)
    assert not guard.stalled(metrics, 1000000)
    assert guard.stalled(metrics, 1500000)
    metrics['cases']['nominal']['mean_peak_height_fraction'] = .5
    assert not guard.stalled(metrics, 2000000)
    assert guard.bad_evaluations == 0


def test_real_evaluator_reports_bounded_swing_progress():
    from cart_double_pendulum.training.robust_eval import RobustCondition, evaluate_controller
    condition = RobustCondition('progress_probe', 'swing_up', 'hanging', .2, .1)
    metrics, rows = evaluate_controller(lambda _: lambda observation: 0., condition,
        episodes=2, collect_rows=True, strict_protocol=True)
    for key in ('mean_peak_height_fraction', 'mean_peak_pendulum_energy_fraction'):
        assert 0 <= metrics[key] <= 1
    assert len(rows) == 2 and 'peak_height_fraction' in rows[0]
