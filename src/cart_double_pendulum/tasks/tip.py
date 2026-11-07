"""Two-link adaptation of triple_tip_swingup TIP and capture rewards."""
from __future__ import annotations

import math
import torch
from . import mdp

# First two absolute-link coefficients of the triple TIP reward, in physical units.
RATE_COEFFICIENTS = (0.015, 0.009)


def reward_components(qpos, qvel, action, rate_scale=1.0, target_ep=3):
  if not math.isfinite(rate_scale) or rate_scale <= 0:
    raise ValueError("rate_scale must be finite and positive")
  from cart_double_pendulum.equilibria import absolute_target, validate_ep
  target_ep = validate_ep(target_ep)
  theta = torch.cumsum(qpos[:, 1:], dim=-1) - (
      math.pi if target_ep == 3 else qpos.new_tensor(absolute_target(target_ep)))
  omega = torch.cumsum(qvel[:, 1:], dim=-1)
  angles = (1.0 + torch.cos(theta)) * 0.5
  rates = torch.exp(-omega.abs() * omega.new_tensor(RATE_COEFFICIENTS) * rate_scale)
  force = torch.exp(-0.4 * action[:, 0].square())
  cart = torch.exp(-0.3 * qpos[:, 0].abs())
  reward = force * cart * angles.prod(-1) * rates.prod(-1)
  return reward, dict(force=force, cart=cart, angles=angles, rates=rates)


def tip_transition_reward(env, *, rate_scale=1.0, target_ep=3):
  qpos, qvel = mdp.state(env)
  reward, env._tip_reward_components = reward_components(
      qpos, qvel, env.action_manager.action, rate_scale, target_ep)
  return reward


def _capture_components(env, handoff, gate_multiplier):
  if not math.isfinite(gate_multiplier) or gate_multiplier <= 0:
    raise ValueError("gate_multiplier must be finite and positive")
  references = [handoff[key] for key in ("cart_safe_m", "angle_threshold_rad",
      "cart_velocity_threshold_m_s", "angular_velocity_threshold_rad_s")]
  if any(not math.isfinite(value) or value <= 0 for value in references):
    raise ValueError("capture references must be finite and positive")
  qpos, qvel = mdp.state(env)
  e1, e2, _ = mdp._double_pendulum_errors(qpos)
  sigma = gate_multiplier * handoff["angle_threshold_rad"]
  gate = torch.exp(-0.5 * ((e1 / sigma).square() + (e2 / sigma).square()))
  speeds = torch.cat((qvel[:, :1].abs() / handoff["cart_velocity_threshold_m_s"],
      torch.cumsum(qvel[:, 1:], -1).abs() / handoff["angular_velocity_threshold_rad_s"]), -1)
  settling = 1.0 / (1.0 + speeds.amax(-1).square())
  centered = torch.exp(-0.5 * (qpos[:, 0] / handoff["cart_safe_m"]).square())
  return gate, settling, centered


def tip_fast_top_cost(env, *, handoff, gate_multiplier=4.0):
  gate, settling, _ = _capture_components(env, handoff, gate_multiplier)
  qpos, qvel = mdp.state(env)
  return reward_components(qpos, qvel, env.action_manager.action)[0] * gate * (1 - settling)


def tip_capture_bonus(env, *, handoff, gate_multiplier=4.0):
  gate, settling, centered = _capture_components(env, handoff, gate_multiplier)
  return gate * settling * centered


def reset_ep_curriculum(env, env_ids=None, *, target_ep, curriculum_fraction=0.4,
                        near_probability=(0.6, 0.1), transition_probability=0.25):
  """Near-target curriculum, other EP starts and broad moving transitions."""
  from cart_double_pendulum.equilibria import joint_target, validate_ep
  target_ep = validate_ep(target_ep)
  if (not 0 < curriculum_fraction <= 1 or len(near_probability) != 2
      or any(not 0 <= p <= 1 for p in near_probability)
      or not 0 <= transition_probability <= 1):
    raise ValueError('invalid EP reset curriculum')
  ids = mdp._reset_ids(env, env_ids)
  n = len(ids)
  scales = mdp.env_scales(env)
  progress = mdp._ramp_progress(curriculum_fraction)
  near_p = near_probability[0] + progress * (near_probability[1] - near_probability[0])
  draw = torch.rand(n, device=env.device)
  near = draw < near_p
  moving = (~near) & (torch.rand(n, device=env.device) < transition_probability)
  # Uniform over the other three equilibria, never the target itself.
  source = torch.randint(3, (n,), device=env.device)
  source += (source >= target_ep).long()
  targets = torch.tensor([joint_target(ep) for ep in range(4)], device=env.device)
  q = torch.zeros((n, 3), device=env.device)
  v = torch.zeros_like(q)
  q[:, 0].uniform_(-.15 * scales.rail, .15 * scales.rail)
  q[:, 1:] = targets[source]
  q[:, 1:] += torch.empty((n, 2), device=env.device).uniform_(-.12, .12)
  v[:, 1:].uniform_(-.15 * scales.rate, .15 * scales.rate)
  v[:, 0].uniform_(-.02 * scales.cart_speed, .02 * scales.cart_speed)
  if near.any():
    k = int(near.sum())
    width = (.5 + 3.5 * progress) * scales.angle
    absolute_error = torch.empty((k, 2), device=env.device).uniform_(-width, width)
    q[near, 1:] = targets[target_ep]
    q[near, 1] += absolute_error[:, 0]
    q[near, 2] += absolute_error[:, 1] - absolute_error[:, 0]
    absolute_rate = torch.empty((k, 2), device=env.device).uniform_(
        -(.1 + 1.9 * progress) * scales.rate, (.1 + 1.9 * progress) * scales.rate)
    v[near, 1] = absolute_rate[:, 0]
    v[near, 2] = absolute_rate[:, 1] - absolute_rate[:, 0]
  if moving.any():
    k = int(moving.sum())
    q[moving, 0] = torch.empty(k, device=env.device).uniform_(-.5 * scales.rail, .5 * scales.rail)
    q[moving, 1:] = torch.empty((k, 2), device=env.device).uniform_(-math.pi, math.pi)
    v[moving, 1:] = torch.empty((k, 2), device=env.device).uniform_(
        -(2 + 6 * progress) * scales.rate, (2 + 6 * progress) * scales.rate)
    v[moving, 0] = torch.empty(k, device=env.device).uniform_(
        -(.1 + .15 * progress) * scales.cart_speed, (.1 + .15 * progress) * scales.cart_speed)
  env.scene[mdp.ASSET].write_joint_state_to_sim(q, v, joint_ids=slice(0, 3), env_ids=ids)


def tip_center_bonus(env, *, target_ep=3, cart_tolerance_mm=5.0,
                     fine_width_mm=5.0, broad_cart_m,
                     angle_sigma_rad, cart_velocity_reference_m_s,
                     angular_velocity_reference_rad_s):
  """Target-aware, bounded centering with a broad approach gradient and 5 mm capture."""
  from cart_double_pendulum.equilibria import target_errors
  values = (cart_tolerance_mm, fine_width_mm, broad_cart_m, angle_sigma_rad,
            cart_velocity_reference_m_s, angular_velocity_reference_rad_s)
  if any(not math.isfinite(value) or value <= 0 for value in values):
    raise ValueError('centering references must be finite and positive')
  q, v = mdp.state(env)
  errors = target_errors(q, torch, target_ep)
  gate = torch.exp(-0.5 * sum((error / angle_sigma_rad).square() for error in errors))
  speeds = torch.cat((v[:, :1].abs() / cart_velocity_reference_m_s,
      torch.cumsum(v[:, 1:], -1).abs() / angular_velocity_reference_rad_s), -1)
  settling = 1 / (1 + speeds.amax(-1).square())
  distance = q[:, 0].abs()
  broad = 1 / (1 + (distance / broad_cart_m).square())
  excess = (distance - cart_tolerance_mm * .001).clamp_min(0)
  fine = 1 / (1 + (excess / (fine_width_mm * .001)).square())
  return gate * settling * (.5 * broad + .5 * fine)
