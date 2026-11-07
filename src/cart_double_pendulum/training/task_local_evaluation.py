"""Evaluate the two expert tasks with their own evaluation.json."""
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from cart_double_pendulum.plant_scales import resolve
from cart_double_pendulum.tasks.config import BALANCING_TASK, PUSH_FORCE_N
from cart_double_pendulum.training import robust_eval as ev

PROTOCOL = 'task_local_expert_validation'


def policy_factory(name, path, condition):
    """Controller for ``robust_eval``: recurrent SAC experts run with the observer guard;
    ``push_*`` conditions push the cart 0.1 s after the first 1 s of stable balance.
    """
    from cart_double_pendulum.onnx_policy import BatchedOnnxPolicy
    from cart_double_pendulum.training.observer_guard import GuardedObserver
    def factory(runtime):
        policy=BatchedOnnxPolicy(path)
        if 'sac' in name and policy.recurrent:
            if policy.hidden_size!=71:raise ValueError('Unexpected SAC observer memory contract')
            policy=GuardedObserver(path)
        policy.reset()
        streak=step=0;first=None
        def act(observation):
            nonlocal streak,step,first
            if condition.key.startswith('push_') and name not in ('expert_swing_sac', 'tip_swing_sac'):
                stable=runtime.is_stable(cart_threshold_m=BALANCING_TASK.stable_cart_m,angle_threshold_rad=BALANCING_TASK.stable_angle_rad,
                                         velocity_threshold=BALANCING_TASK.stable_velocity)
                if first is None:
                    streak=streak+1 if stable else 0
                    if streak>=50:first=step
                sign=-1. if condition.key.endswith('neg') else 1.
                runtime.data.qfrc_applied[runtime.qvel_indices[0]]=sign*PUSH_FORCE_N if first is not None and first<=step<first+5 else 0.
            step+=1
            action=float(np.asarray(policy(np.asarray(observation)[None,:])).reshape(-1)[0])
            policy.record_applied_actions([[np.clip(action,-1,1)]])
            return action
        return act
    return factory


def evaluate_expert_task(name, policy_path, episodes, config_path, seed=None):
    if name not in ('expert_balance_ppo', 'expert_swing_sac', 'tip_swing_sac'):
        raise ValueError('Expected an expert task')
    definition = resolve(json.loads(Path(config_path).read_text()))
    centered_protocol = 'task_local_centered_expert_validation_v1'
    if definition['protocol'] not in (PROTOCOL, centered_protocol):
        raise ValueError(f'Expert evaluation requires protocol {PROTOCOL!r}')
    tolerance = definition.get('centering', {}).get('cart_tolerance_mm')
    if definition['protocol'] == centered_protocol and tolerance is None:
        raise ValueError('centered protocol requires cart tolerance')
    allowed = {f.name for f in fields(ev.RobustCondition)}
    cases = {}
    for i, value in enumerate(definition['conditions']):
        if set(value) != allowed:
            raise ValueError('Every condition must declare the complete evaluation contract')
        values = dict(value, random_axes=tuple(value['random_axes']))
        if seed is not None:
            values['seed'] = seed + 10000 * i
        condition = ev.RobustCondition(**values)
        # Expert conditions use fixed evaluator-owned pulses, never a second injector.
        factory = policy_factory(name, policy_path, condition)
        metrics, _ = ev.evaluate_controller(factory, condition, episodes=episodes,
            strict_protocol=True, numerical_failure_is_failure=True,
            cart_tolerance_m=None if tolerance is None else float(tolerance) * .001)
        cases[condition.key] = metrics
    if len(cases) < 2 or not all(key.startswith('push_') for key in list(cases)[2:]):
        raise ValueError('Expert validation declares nominal, randomized, then optional push_* conditions')
    nominal, randomized, *pushes = cases.values()
    robust = [randomized, *pushes]
    metrics = dict(success_rate=min(case['success_rate'] for case in robust),
        stable_occupancy=float(np.mean([case['stable_occupancy'] for case in robust])),
        mean_time_to_success_s=randomized['mean_time_to_success_s'] or 20.,
        mean_force_energy=randomized['mean_force_energy_n2s'],
        nominal_success_rate=nominal['success_rate'], nominal_stable_occupancy=nominal['stable_occupancy'],
        randomized_success_rate=randomized['success_rate'],
        nominal_gate=nominal['success_rate'] >= .99, cases=cases, protocol=definition['protocol'])
    if tolerance is not None:
        metrics['cart_tolerance_mm'] = float(tolerance)
    if name in ('expert_swing_sac', 'tip_swing_sac'):
        metrics['expert_progress_selection_key'] = swing_selection_key(metrics)
    return metrics


def swing_selection_key(metrics):
    """Keep real swing progress before effort when no checkpoint succeeds yet."""
    nominal = next(iter(metrics['cases'].values()))
    return [float(metrics['nominal_gate']), metrics['success_rate'],
        metrics['nominal_success_rate'], metrics['stable_occupancy'],
        1. - nominal['never_handoff_rate'],
        -nominal['numerical_failure_rate'], -nominal['rail_failure_rate'],
        nominal['mean_peak_height_fraction'], nominal['mean_peak_pendulum_energy_fraction'],
        -metrics['mean_time_to_success_s'], -metrics['mean_force_energy']]
