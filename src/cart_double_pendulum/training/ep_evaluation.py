"""Fixed-target EP evaluation on standalone MuJoCo, separate from upright handoff."""
import math
from pathlib import Path
import mujoco
import numpy as np
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.equilibria import joint_target, target_errors, target_mask, validate_ep
from cart_double_pendulum.runtime import CartDoublePendulumRuntime
from cart_double_pendulum.runtime_policy import OnnxPolicy
from cart_double_pendulum.training.task_directory import read_json


def evaluate_ep_controller(factory, definition, episodes):
  if episodes <= 0:
    raise ValueError('episodes must be positive')
  ep = validate_ep(definition['target_ep'])
  handoff = definition['handoff']
  tolerance = definition.get('centering', {}).get('cart_tolerance_mm')
  if definition['protocol'] == 'fixed_ep_centered_validation_v1' and tolerance is None:
    raise ValueError('centered protocol requires cart tolerance')
  tolerance = None if tolerance is None else float(tolerance) * .001
  if tolerance is not None and (not math.isfinite(tolerance) or not 0 < tolerance < CONTRACT.rail_limit_m):
    raise ValueError('cart tolerance must be finite, positive and inside rail')
  cases = {}
  dt = CONTRACT.control_timestep_s
  for condition in definition['conditions']:
    steps = round(condition['duration_s'] / dt)
    hold = round(condition['hold_s'] / dt)
    if not 0 < hold <= steps:
      raise ValueError('EP evaluation requires 0 < hold_s <= duration_s')
    rows = []
    for episode in range(episodes):
      seed = condition['seed'] + episode
      rng = np.random.default_rng(seed)
      randomization = definition['randomization'] if condition['randomized'] else {}
      scenario = {key: float(rng.uniform(*bounds)) for key, bounds in randomization.items()}
      runtime = CartDoublePendulumRuntime(CONTRACT.model_path, seed=seed, independent_observation_noise=True, **scenario)
      runtime.reset(mode='hanging', cart_position_m=0, angle_rad=0, velocity=0)
      source = condition['initial_state']
      q, v = np.zeros(3), np.zeros(3)
      if source == 'transition_random':
        q[1:] = rng.uniform(-math.pi, math.pi, 2)
        v[1:] = rng.uniform(-definition['transition_angular_velocity_rad_s'],
                              definition['transition_angular_velocity_rad_s'], 2)
        v[0] = rng.uniform(-definition['transition_cart_velocity_m_s'],
                             definition['transition_cart_velocity_m_s'])
      else:
        q[1:] = joint_target(int(source.removeprefix('ep')))
      q[0] = rng.uniform(-definition['reset_cart_m'], definition['reset_cart_m'])
      q[1:] += rng.uniform(-definition['reset_angle_rad'], definition['reset_angle_rad'], 2)
      runtime.data.qpos[list(runtime.qpos_indices)] = q
      runtime.data.qvel[list(runtime.qvel_indices)] = v
      mujoco.mj_forward(runtime.model, runtime.data)
      observation = runtime.observation().astype(np.float32)
      controller = factory(runtime)
      streak = stable_steps = 0
      success_time = None
      rail = numerical = False
      effort = alignment_sum = 0.
      sampled = 0
      for step in range(steps):
        try:
          action = float(np.clip(controller(observation), -1, 1))
          observation = runtime.step(action).astype(np.float32)
        except FloatingPointError:
          numerical = True
          break
        sampled += 1
        q, v = runtime.qpos, runtime.qvel
        if not np.isfinite(q).all() or not np.isfinite(v).all():
          numerical = True
          break
        rail = abs(q[0]) >= definition['rail_limit_m']
        numerical = bool(np.max(abs(v[1:])) > definition['overspeed_rad_s'])
        if rail or numerical:
          break
        stable = bool(target_mask(q, v, np, ep, **handoff))
        if tolerance is not None:
          stable = stable and abs(q[0]) <= tolerance
        stable_steps += stable
        streak = streak + 1 if stable else 0
        if streak >= hold and success_time is None:
          success_time = (step + 1) * dt
        errors = target_errors(q, np, ep)
        alignment_sum += float(np.prod((1 + np.cos(errors)) * .5))
        effort += (action * runtime.traction_force_limit_n) ** 2 * dt
      rows.append(dict(success=success_time is not None and not rail and not numerical,
          stable_occupancy=stable_steps / steps, time=success_time or condition['duration_s'],
          rail=rail, numerical=numerical, effort=effort, alignment=alignment_sum / max(1, sampled)))
    cases[condition['key']] = dict(success_rate=float(np.mean([r['success'] for r in rows])),
        stable_occupancy=float(np.mean([r['stable_occupancy'] for r in rows])),
        mean_time_to_success_s=float(np.mean([r['time'] for r in rows])),
        rail_failure_rate=float(np.mean([r['rail'] for r in rows])),
        numerical_failure_rate=float(np.mean([r['numerical'] for r in rows])),
        mean_force_energy=float(np.mean([r['effort'] for r in rows])),
        mean_target_alignment=float(np.mean([r['alignment'] for r in rows])))
  nominal = cases['ep_nominal']
  robust = [value for key, value in cases.items() if key != 'ep_nominal']
  metrics = dict(target_ep=ep, protocol=definition['protocol'], cases=cases,
      success_rate=min(value['success_rate'] for value in robust),
      stable_occupancy=float(np.mean([value['stable_occupancy'] for value in robust])),
      mean_time_to_success_s=float(np.mean([value['mean_time_to_success_s'] for value in robust])),
      mean_force_energy=float(np.mean([value['mean_force_energy'] for value in robust])),
      nominal_success_rate=nominal['success_rate'],
      randomized_success_rate=cases['ep_randomized']['success_rate'],
      nominal_gate=nominal['success_rate'] >= .99)
  if tolerance is not None:
    metrics['cart_tolerance_mm'] = tolerance * 1000
  metrics['expert_progress_selection_key'] = [float(metrics['nominal_gate']), metrics['success_rate'],
      metrics['nominal_success_rate'], metrics['stable_occupancy'], nominal['mean_target_alignment'],
      -metrics['mean_time_to_success_s'], -metrics['mean_force_energy']]
  return metrics


def evaluate_ep_task(name, path, episodes, config_path, seed=None):
  definition = read_json(Path(config_path))
  ep = validate_ep(definition['target_ep'])
  if name != f'sac_ep{ep}' or definition['protocol'] not in ('fixed_ep_validation', 'fixed_ep_centered_validation_v1'):
    raise ValueError('EP task name and evaluation target must agree')
  if seed is not None:
    for index, condition in enumerate(definition['conditions']):
      condition['seed'] = seed + 10000 * index
  def factory(runtime):
    policy = OnnxPolicy(path)
    policy.reset()
    return policy
  return evaluate_ep_controller(factory, definition, episodes)
