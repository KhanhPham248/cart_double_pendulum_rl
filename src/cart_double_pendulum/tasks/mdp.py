"""Reset, observation, reward and termination terms for the cart double pendulum."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from cart_double_pendulum import curriculum
from cart_double_pendulum.common import (
  CONTRACT, PLANT_PARAMETERS, angle_error, noise_profile, pendulum_energy, policy_observation,
)
from cart_double_pendulum.plant_scales import plant_scales

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ASSET = "cart_double_pendulum"


def env_contract(env):
  return getattr(getattr(env, 'cfg', None), 'plant_contract', CONTRACT)


def env_scales(env):
  return plant_scales(env_contract(env))


def plant_term(*names: str):
  """Mark reward kwargs that the task loader fills from the MJCF contract."""
  unknown = set(names) - set(PLANT_PARAMETERS)
  if unknown:
    raise ValueError(f"unknown plant parameters: {sorted(unknown)}")

  def mark(func):
    func.plant_parameters = names
    return func
  return mark


def state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """qpos, qvel of cart_slide, base_joint and elbow_joint."""
  asset = env.scene[ASSET]
  return asset.data.joint_pos[:, :3], asset.data.joint_vel[:, :3]


def _reset_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
  curriculum.observe_step(env.common_step_counter)
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  return env_ids


def _uniform(env: ManagerBasedRlEnv, shape, limit: float) -> torch.Tensor:
  return torch.empty(shape, device=env.device).uniform_(-limit, limit)


def outward_states(qpos: np.ndarray, qvel: np.ndarray, contract=CONTRACT) -> np.ndarray:
  """Bank states with the cart off centre and moving outward."""
  S = plant_scales(contract)
  return (np.abs(qpos[:, 0]) >= .18 * S.rail) & (qpos[:, 0] * qvel[:, 0] > .0386 * S.rail * S.cart_speed)


# Domain randomization per world, relative to compiled defaults, scaled by the curriculum.

def _benchmark_scales(env, env_ids, scale_range):
  low, high = curriculum.scaled_range(scale_range, curriculum.levels().domain)
  if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
    raise ValueError("benchmark scale range must be finite, positive and ordered")
  ids = _reset_ids(env, env_ids).to(device=env.device, dtype=torch.long)
  return ids, torch.empty((len(ids), 1), device=env.device).uniform_(low, high)


def _scale_field(env, name: str, ids, columns, scales) -> None:
  default = env.sim.get_default_field(name)[columns][None]
  getattr(env.sim.model, name)[ids[:, None], columns] = default * scales.view(-1, 1, *[1] * (default.ndim - 2))


@requires_model_fields("body_mass", "body_inertia", "dof_armature", recompute=RecomputeLevel.set_const)
def randomize_benchmark_density(env, env_ids, *, scale_range=(0.85, 1.15)) -> None:
  """Scale body masses, inertias and the cart armature (reflected rotor inertia) together."""
  ids, scales = _benchmark_scales(env, env_ids, scale_range)
  asset = env.scene[ASSET]
  bodies = asset.indexing.body_ids[asset.find_bodies(("cart", "link1", "link2"))[0]]
  slide = asset.indexing.joint_v_adr[asset.find_joints(("cart_slide",))[0]]
  _scale_field(env, "body_mass", ids, bodies, scales)
  _scale_field(env, "body_inertia", ids, bodies, scales)
  _scale_field(env, "dof_armature", ids, slide, scales)


@requires_model_fields("dof_damping")
def randomize_benchmark_damping(env, env_ids, *, scale_range=(0.75, 1.25)) -> None:
  ids, scales = _benchmark_scales(env, env_ids, scale_range)
  asset = env.scene[ASSET]
  dofs = asset.indexing.joint_v_adr[asset.find_joints(("cart_slide", "base_joint", "elbow_joint"))[0]]
  _scale_field(env, "dof_damping", ids, dofs, scales)


@requires_model_fields("actuator_gainprm", "actuator_ctrlrange", "actuator_forcerange")
def randomize_benchmark_actuator_force(env, env_ids, *, scale_range=(0.90, 1.10)) -> None:
  """Scale motor gain and force range; the control range stays nominal."""
  ids, scales = _benchmark_scales(env, env_ids, scale_range)
  ctrl = env.scene[ASSET].actuators[0].global_ctrl_ids
  model = env.sim.model
  model.actuator_gainprm[ids[:, None], ctrl, 0] = env.sim.get_default_field("actuator_gainprm")[ctrl, 0][None] * scales
  model.actuator_ctrlrange[ids[:, None], ctrl] = env.sim.get_default_field("actuator_ctrlrange")[ctrl][None]
  _scale_field(env, "actuator_forcerange", ids, ctrl, scales)


# Resets.

def load_state_bank(path: str, device, contract=CONTRACT) -> tuple[torch.Tensor, torch.Tensor]:
  from cart_double_pendulum.plant_metadata import load_bank
  q, v = load_bank(path, contract)
  return torch.as_tensor(q, device=device), torch.as_tensor(v, device=device)


def _ramp_progress(fraction: float) -> float:
  """Reset-difficulty ramp over ``fraction`` of the budget on the pausable difficulty clock; full when idle."""
  if not 0.0 < fraction <= 1.0:
    raise ValueError("curriculum_fraction must lie in (0, 1]")
  return min(1.0, curriculum.clock_fraction() / fraction, curriculum.reset_difficulty_cap())


def reset_expert_swing(env, env_ids=None, *, bank_path: str | None = None, curriculum_fraction: float = .1536) -> None:
  """Hanging starts; 20% bank states (hanging when no bank) and 15% synthetic capture states."""
  S = env_scales(env)
  ids = _reset_ids(env, env_ids)
  n = len(ids)
  cache = getattr(env, "_expert_swing_bank", None)
  if cache is None and bank_path:
    cache = env._expert_swing_bank = load_state_bank(bank_path, env.device, env_contract(env))
  progress = _ramp_progress(curriculum_fraction)
  q = torch.zeros((n, 3), device=env.device)
  v = torch.zeros_like(q)
  draw = torch.rand(n, device=env.device)
  q[:, 0] = _uniform(env, n, .08 * S.rail)
  q[:, 1:].uniform_(-.12, .12)
  v[:] = _uniform(env, (n, 3), .1057 * S.rate)
  v[:, 0] *= S.cart_speed_per_rate
  real = draw < .20
  synthetic = (draw >= .20) & (draw < .35)
  if cache is not None and real.any():
    sample = torch.randint(len(cache[0]), (int(real.sum()),), device=env.device)
    q[real], v[real] = cache[0][sample], cache[1][sample]
  if synthetic.any():
    k = int(synthetic.sum())
    velocity = (.1057 + .2325 * progress) * S.rate
    e = _uniform(env, (k, 2), (.569 + .569 * progress) * S.angle)
    omega = _uniform(env, (k, 2), velocity)
    q[synthetic, 0] = _uniform(env, k, (.12 + .23 * progress) * S.rail)
    q[synthetic, 1] = math.pi + e[:, 0]
    q[synthetic, 2] = e[:, 1] - e[:, 0]
    v[synthetic, 0] = _uniform(env, k, velocity * S.cart_speed_per_rate)
    v[synthetic, 1] = omega[:, 0]
    v[synthetic, 2] = omega[:, 1] - omega[:, 0]
  env.scene[ASSET].write_joint_state_to_sim(q, v, joint_ids=slice(0, 3), env_ids=ids)


def reset_expert_balance(env, env_ids=None, *, bank_path: str, curriculum_fraction: float) -> None:
  """Near-upright starts; the share of bank states and hard outward bank states grows."""
  if not bank_path:
    raise ValueError("the balancing expert needs a handoff bank; see docs/BANKS.md")
  S = env_scales(env)
  ids = _reset_ids(env, env_ids)
  count = len(ids)
  cache = getattr(env, "_expert_balance_bank", None)
  if cache is None:
    bank_q, bank_v = (value.cpu().numpy() for value in load_state_bank(bank_path, "cpu", env_contract(env)))
    absolute = np.column_stack((bank_q[:, 1], bank_q[:, 1] + bank_q[:, 2])) - np.pi
    absolute_error = np.arctan2(np.sin(absolute), np.cos(absolute))
    difficult = outward_states(bank_q, bank_v, env_contract(env)) & (
      (np.max(np.abs(absolute_error), axis=1) >= .569 * S.angle)
      | (np.maximum(np.abs(bank_v[:, 1]), np.abs(bank_v[:, 1] + bank_v[:, 2])) >= .148 * S.rate)
    )
    cache = env._expert_balance_bank = (
      torch.as_tensor(bank_q, device=env.device),
      torch.as_tensor(bank_v, device=env.device),
      torch.as_tensor(np.flatnonzero(difficult), device=env.device, dtype=torch.long),
    )
  bank_q, bank_v, hard_ids = cache
  progress = _ramp_progress(curriculum_fraction)
  qpos = torch.zeros((count, 3), device=env.device)
  qvel = torch.zeros_like(qpos)
  qpos[:, 0] = _uniform(env, count, .04 * S.rail)
  qpos[:, 1] = math.pi + _uniform(env, count, .2845 * S.angle)
  qpos[:, 2] = _uniform(env, count, .2845 * S.angle)
  qvel[:] = _uniform(env, (count, 3), .04228 * S.rate)
  qvel[:, 0] *= S.cart_speed_per_rate
  nominal_fraction = .80 - .25 * progress
  bank_fraction = .15 + .10 * progress
  draw = torch.rand(count, device=env.device)
  bank_mask = (draw >= nominal_fraction) & (draw < nominal_fraction + bank_fraction)
  hard_mask = draw >= nominal_fraction + bank_fraction
  if bank_mask.any():
    index = torch.randint(len(bank_q), (int(bank_mask.sum()),), device=env.device)
    qpos[bank_mask], qvel[bank_mask] = bank_q[index], bank_v[index]
  if hard_mask.any():
    source = hard_ids if len(hard_ids) else torch.arange(len(bank_q), device=env.device)
    index = source[torch.randint(len(source), (int(hard_mask.sum()),), device=env.device)]
    qpos[hard_mask], qvel[hard_mask] = bank_q[index], bank_v[index]
  env.scene[ASSET].write_joint_state_to_sim(qpos, qvel, joint_ids=slice(0, 3), env_ids=ids)




# Observations.

def policy_state_with_contract(env: ManagerBasedRlEnv, *, contract=CONTRACT) -> torch.Tensor:
  result = policy_observation(*state(env), torch, contract)
  if not torch.isfinite(result).all():
    raise RuntimeError("cart double pendulum observation is non-finite")
  return result


def noisy_policy_state(env, *, max_noise_std: float = 0.02, contract=CONTRACT,
                       noise_per_episode: bool = False) -> torch.Tensor:
  if max_noise_std < 0.0:
    raise ValueError("max observation-noise standard deviation must be non-negative")
  observation = policy_state_with_contract(env, contract=contract)
  curriculum.observe_step(env.common_step_counter)
  if max_noise_std == 0.0:
    return observation
  noise_level = curriculum.levels().noise
  max_noise_std *= noise_level
  if noise_per_episode:
    # Shape probes run before the first reset; only the reset event samples the level.
    noise_std = getattr(env, "_actor_noise_std", None)
    if noise_std is None:
      env._actor_noise_std = noise_std = torch.zeros((env.num_envs, 1), device=env.device)
  else:
    noise_std = torch.rand((env.num_envs, 1), device=env.device) * max_noise_std
  profile = torch.tensor(noise_profile(contract, noise_level=noise_level), dtype=observation.dtype, device=observation.device)
  return observation + torch.randn_like(observation) * noise_std * profile


def reset_actor_noise_std(env, env_ids, *, max_noise_std: float = 0.02) -> None:
  """Sample one U(0, max_noise_std) sensor noise level per episode."""
  if max_noise_std < 0.0:
    raise ValueError("max observation-noise standard deviation must be non-negative")
  if not hasattr(env, "_actor_noise_std"):
    env._actor_noise_std = torch.zeros((env.num_envs, 1), device=env.device)
  ids = _reset_ids(env, env_ids)
  env._actor_noise_std[ids] = torch.rand((len(ids), 1), device=env.device) * max_noise_std * curriculum.levels().noise


class CurriculumImpulse(apply_body_impulse):
  """Cart pushes whose amplitude and frequency follow the curriculum."""

  def __call__(self, env, env_ids, force_range, torque_range, duration_s, cooldown_s, asset_cfg,
               body_point_offset=None):
    curriculum.observe_step(env.common_step_counter)
    levels = curriculum.levels()
    force_range = tuple(levels.push_amplitude * value for value in force_range)
    if levels.push_frequency > 0.0:
      cooldown_s = tuple(value / levels.push_frequency for value in cooldown_s)
    return super().__call__(env, env_ids, force_range, torque_range, duration_s, cooldown_s, asset_cfg,
                            body_point_offset)


def actor_history_frame(env, *, max_noise_std: float = 0.0, contract=CONTRACT,
                        noise_per_episode: bool = False, observer_checkpoint: str | None = None) -> torch.Tensor:
  """9-D frame: observation and the action applied over the last interval."""
  observation = noisy_policy_state(env, max_noise_std=max_noise_std, contract=contract,
                                   noise_per_episode=noise_per_episode)
  if observer_checkpoint is not None:
    from cart_double_pendulum.training.ppo_observer_adapter import filtered_observation
    observation = filtered_observation(env, observation, observer_checkpoint)
  return torch.cat((observation, env.action_manager.action), dim=-1)


# State tests shared by rewards, evaluators and terminations.

def _double_pendulum_errors(qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Upright errors of link 1, link 2 (absolute) and the elbow (relative)."""
  theta1 = qpos[:, 1]
  theta2_relative = qpos[:, 2]
  return (angle_error(theta1, math.pi, torch),
          angle_error(theta1 + theta2_relative, math.pi, torch),
          angle_error(theta2_relative, 0.0, torch))


def _capture(error: torch.Tensor, angle_sigma_rad: float) -> torch.Tensor:
  if angle_sigma_rad <= 0.0:
    raise ValueError("angle_sigma_rad must be positive")
  return torch.exp(-0.5 * (error / angle_sigma_rad).square())


def _upright(qpos: torch.Tensor, angle_sigma_rad: float) -> torch.Tensor:
  link1_error, link2_absolute_error, _ = _double_pendulum_errors(qpos)
  return _capture(link1_error, angle_sigma_rad) * _capture(link2_absolute_error, angle_sigma_rad)


def stable_mask(qpos, qvel, *, cart_threshold_m: float, angle_threshold_rad: float,
                velocity_threshold: float, contract=CONTRACT) -> torch.Tensor:
  """Centred, collinear and still upright state."""
  e1, e2_absolute, e2_relative = _double_pendulum_errors(qpos)
  return (
    (qpos[:, 0].abs() < cart_threshold_m)
    & (e1.abs() < angle_threshold_rad)
    & (e2_absolute.abs() < angle_threshold_rad)
    & (e2_relative.abs() < angle_threshold_rad)
    & (qvel[:, 0].abs() < velocity_threshold * plant_scales(contract).cart_speed_per_rate)
    & (qvel[:, 1:].abs().amax(dim=1) < velocity_threshold)
    & ((qvel[:, 1] + qvel[:, 2]).abs() < velocity_threshold)
  )


def swing_up_handoff_mask(qpos, qvel, *, cart_safe_m: float, angle_threshold_rad: float,
                          cart_velocity_threshold_m_s: float,
                          angular_velocity_threshold_rad_s: float) -> torch.Tensor:
  """Calm upright state, anywhere inside the safe rail band, ready for the balancing expert."""
  e1, e2_absolute, _ = _double_pendulum_errors(qpos)
  return (
    (qpos[:, 0].abs() < cart_safe_m)
    & (e1.abs() < angle_threshold_rad)
    & (e2_absolute.abs() < angle_threshold_rad)
    & (qvel[:, 0].abs() < cart_velocity_threshold_m_s)
    & (qvel[:, 1].abs() < angular_velocity_threshold_rad_s)
    & ((qvel[:, 1] + qvel[:, 2]).abs() < angular_velocity_threshold_rad_s)
  )


# Rewards.

def swing_handoff_ready(env, *, cart_safe_m: float, angle_threshold_rad: float,
                        cart_velocity_threshold_m_s: float, angular_velocity_threshold_rad_s: float):
  return swing_up_handoff_mask(*state(env), cart_safe_m=cart_safe_m, angle_threshold_rad=angle_threshold_rad,
                               cart_velocity_threshold_m_s=cart_velocity_threshold_m_s,
                               angular_velocity_threshold_rad_s=angular_velocity_threshold_rad_s).float()


def swing_time_to_handoff_cost(env, *, cart_safe_m: float, angle_threshold_rad: float,
                               cart_velocity_threshold_m_s: float, angular_velocity_threshold_rad_s: float):
  """Unit cost per step until the handoff state is reached."""
  return 1.0 - swing_handoff_ready(env, cart_safe_m=cart_safe_m, angle_threshold_rad=angle_threshold_rad,
                                   cart_velocity_threshold_m_s=cart_velocity_threshold_m_s,
                                   angular_velocity_threshold_rad_s=angular_velocity_threshold_rad_s)


def swing_up_handoff_bonus(env, *, cart_threshold_m: float, angle_threshold_rad: float,
                           velocity_threshold: float) -> torch.Tensor:
  return stable_mask(*state(env), cart_threshold_m=cart_threshold_m, angle_threshold_rad=angle_threshold_rad,
                     velocity_threshold=velocity_threshold, contract=env_contract(env)).float()


def balancing_stability_bonus(env, *, cart_threshold_m: float, angle_threshold_rad: float,
                              velocity_threshold: float) -> torch.Tensor:
  return swing_up_handoff_bonus(env, cart_threshold_m=cart_threshold_m, angle_threshold_rad=angle_threshold_rad,
                                velocity_threshold=velocity_threshold)


def double_upright_capture(env: ManagerBasedRlEnv, *, angle_sigma_rad: float) -> torch.Tensor:
  return _upright(state(env)[0], angle_sigma_rad)


def gated_link2_upright_capture(env, *, link2_angle_sigma_rad: float, link1_gate_sigma_rad: float) -> torch.Tensor:
  """Link-2 upright reward, gated on link 1 having risen first."""
  link1_error, link2_absolute_error, _ = _double_pendulum_errors(state(env)[0])
  return _capture(link2_absolute_error, link2_angle_sigma_rad) * _capture(link1_error, link1_gate_sigma_rad)


@plant_term("link1_length_m", "link1_com_m", "link2_com_m", "link1_mass_kg", "link2_mass_kg")
def double_com_height(env, *, link1_length_m: float, link1_com_m: float, link2_com_m: float,
                      link1_mass_kg: float, link2_mass_kg: float) -> torch.Tensor:
  """Combined link COM height: 0 both hanging, 1 both upright."""
  qpos, _ = state(env)
  height = -(link1_mass_kg * link1_com_m + link2_mass_kg * link1_length_m) * torch.cos(qpos[:, 1])
  height += -(link2_mass_kg * link2_com_m) * torch.cos(qpos[:, 1] + qpos[:, 2])
  scale = link1_mass_kg * link1_com_m + link2_mass_kg * (link1_length_m + link2_com_m)
  return ((height - -scale) / (scale - -scale)).clamp(0.0, 1.0)


def pendulum_energy_fraction(env: ManagerBasedRlEnv, **plant: float) -> torch.Tensor:
  energy, upright_energy = pendulum_energy(*state(env), torch, **plant)
  return energy / upright_energy


@plant_term(*PLANT_PARAMETERS)
def swing_energy_target(env, *, overshoot_width: float = .75, **plant: float) -> torch.Tensor:
  """Energy fraction up to the upright target, falling to zero over ``overshoot_width`` above it."""
  if overshoot_width <= 0.0:
    raise ValueError("overshoot_width must be positive")
  fraction = pendulum_energy_fraction(env, **plant)
  above = (1.0 - ((fraction - 1.0).clamp_min(0.0) / overshoot_width).square()).clamp_min(0.0)
  return torch.where(fraction <= 1.0, fraction.clamp_min(0.0), above)


def discounted(func):
  """Mark a reward whose ``discount`` the task loader fills with the trainer's gamma."""
  func.uses_discount = True
  return func


def handoff_deficit(env, *, cart_safe_m: float, angle_threshold_rad: float, cart_velocity_threshold_m_s: float,
                    angular_velocity_threshold_rad_s: float) -> torch.Tensor:
  """Distance to the handoff region in [0, 1): zero inside, a function of the state only (no clock)."""
  limits = (cart_safe_m, angle_threshold_rad, cart_velocity_threshold_m_s, angular_velocity_threshold_rad_s)
  if not all(math.isfinite(value) and value > 0.0 for value in limits):
    raise ValueError('handoff deficit thresholds must be finite and positive')
  q, v = state(env)
  e1, e2, _ = _double_pendulum_errors(q)
  ratio = torch.stack((q[:, 0].abs() / cart_safe_m, e1.abs() / angle_threshold_rad, e2.abs() / angle_threshold_rad,
    v[:, 0].abs() / cart_velocity_threshold_m_s, v[:, 1].abs() / angular_velocity_threshold_rad_s,
    (v[:, 1] + v[:, 2]).abs() / angular_velocity_threshold_rad_s), dim=1)
  # Product, not maximum: progress on one link still lowers the cost while the other is far.
  deficit = 1.0 - ratio.clamp_min(1.0).reciprocal().prod(dim=1)
  return torch.where(env.termination_manager.terminated, torch.zeros_like(deficit), deficit)


@plant_term(*PLANT_PARAMETERS)
def swing_excess_energy_cost(env, *, excess_scale: float, **plant: float) -> torch.Tensor:
  if excess_scale <= 0.0:
    raise ValueError("excess_scale must be positive")
  return ((pendulum_energy_fraction(env, **plant) - 1.0) / excess_scale).clamp(0.0, 1.0)


def _velocity_settling_quality(qvel, *, cart_velocity_reference_m_s: float,
                               angular_velocity_reference_rad_s: float) -> torch.Tensor:
  """exp(-s^2/2) of the largest normalized generalized speed."""
  if min(cart_velocity_reference_m_s, angular_velocity_reference_rad_s) <= 0.0:
    raise ValueError("velocity references must be positive")
  speed = torch.stack((
    qvel[:, 0].abs() / cart_velocity_reference_m_s,
    qvel[:, 1].abs() / angular_velocity_reference_rad_s,
    (qvel[:, 1] + qvel[:, 2]).abs() / angular_velocity_reference_rad_s,
    qvel[:, 2].abs() / angular_velocity_reference_rad_s,
  ), dim=1).amax(dim=1)
  return torch.exp(-0.5 * speed.square())


def swing_capture_quality(env, *, angle_sigma_rad: float, cart_sigma_m: float, cart_velocity_reference_m_s: float,
                          angular_velocity_reference_rad_s: float, center_floor: float = 0.25) -> torch.Tensor:
  """Quiet double-upright capture with a preference for the rail centre."""
  if min(angle_sigma_rad, cart_sigma_m) <= 0.0 or not 0.0 <= center_floor <= 1.0:
    raise ValueError("capture parameters are invalid")
  qpos, qvel = state(env)
  settling = _velocity_settling_quality(qvel, cart_velocity_reference_m_s=cart_velocity_reference_m_s,
                                        angular_velocity_reference_rad_s=angular_velocity_reference_rad_s)
  centered = torch.exp(-0.5 * (qpos[:, 0] / cart_sigma_m).square())
  return _upright(qpos, angle_sigma_rad) * settling * (center_floor + (1.0 - center_floor) * centered)


def balancing_settling_quality(env, *, capture_sigma_rad: float, cart_velocity_reference_m_s: float,
                               angular_velocity_reference_rad_s: float) -> torch.Tensor:
  """Settling reward near upright; no cart-position term, so recovery travel is not punished."""
  qpos, qvel = state(env)
  settling = _velocity_settling_quality(qvel, cart_velocity_reference_m_s=cart_velocity_reference_m_s,
                                        angular_velocity_reference_rad_s=angular_velocity_reference_rad_s)
  return _upright(qpos, capture_sigma_rad) * settling


def captured_cart_center(env, *, cart_sigma_m: float, capture_sigma_rad: float) -> torch.Tensor:
  """Cart centring, active only once both links are near upright."""
  if cart_sigma_m <= 0.0:
    raise ValueError("cart_sigma_m must be positive")
  qpos, _ = state(env)
  return _upright(qpos, capture_sigma_rad) * torch.exp(-0.5 * (qpos[:, 0] / cart_sigma_m).square())


def cart_center_cost(env, *, cart_sigma_m: float) -> torch.Tensor:
  if cart_sigma_m <= 0.0:
    raise ValueError("cart_sigma_m must be positive")
  return 1.0 - torch.exp(-0.5 * (state(env)[0][:, 0] / cart_sigma_m).square())


def cart_center_quadratic(env, *, reference_m: float | None = None) -> torch.Tensor:
  reference_m = .5 * env_scales(env).rail if reference_m is None else reference_m
  if reference_m <= 0:
    raise ValueError("centering reference must be positive")
  return (state(env)[0][:, 0] / reference_m).square()


def cart_velocity_smooth_l1(env, *, reference_m_s: float | None = None, max_cost: float = 8.0) -> torch.Tensor:
  """Pseudo-Huber cart speed cost, capped so its peak does not depend on the plant."""
  reference_m_s = .3215 * env_scales(env).cart_speed if reference_m_s is None else reference_m_s
  if reference_m_s <= 0.0 or max_cost <= 0.0:
    raise ValueError("cart velocity reference and max_cost must be positive")
  return (2.0 * (torch.sqrt(1.0 + (state(env)[1][:, 0] / reference_m_s).square()) - 1.0)).clamp(max=max_cost)


def rail_safety_barrier(env, *, free_cart_m: float, rail_limit_m: float) -> torch.Tensor:
  """0 inside ``free_cart_m``, rising quadratically to 1 at ``rail_limit_m``."""
  if not 0.0 < free_cart_m < rail_limit_m:
    raise ValueError("rail safety barrier parameters are invalid")
  excess = (state(env)[0][:, 0].abs() - free_cart_m) / (rail_limit_m - free_cart_m)
  return excess.clamp(0.0, 1.0).square()


def _forever(failed: torch.Tensor, discount: float) -> torch.Tensor:
  """A failure costs as much as the weight paid every step forever, so ending an episode never pays."""
  if not 0.0 < discount < 1.0:
    raise ValueError("terminal-cost discount must lie in (0, 1)")
  return failed.float() / (1.0 - discount)


@discounted
def rail_terminal_cost(env, *, rail_limit_m: float, discount: float) -> torch.Tensor:
  """Rail failure, or a diverged simulation: spinning the sim apart must not end an episode for free."""
  if rail_limit_m <= 0.0:
    raise ValueError("rail_limit_m must be positive")
  return _forever(rail_failure(env, rail_limit_m=rail_limit_m) | numerical_failure(env), discount)


def _rate_cost(env, value: torch.Tensor, reference: float, capture_sigma_rad: float | None,
               max_cost: float = 1.0) -> torch.Tensor:
  """Squared rate capped at ``max_cost``, optionally gated to the upright neighbourhood."""
  if reference <= 0.0 or max_cost <= 0.0:
    raise ValueError("velocity reference and max_cost must be positive")
  cost = (value / reference).square().clamp(max=max_cost)
  return cost if capture_sigma_rad is None else _upright(state(env)[0], capture_sigma_rad) * cost


def link1_velocity_cost(env, *, reference_rad_s: float, capture_sigma_rad: float | None = None,
                        max_cost: float = 1.0) -> torch.Tensor:
  return _rate_cost(env, state(env)[1][:, 1], reference_rad_s, capture_sigma_rad, max_cost)


def link2_absolute_velocity_cost(env, *, reference_rad_s: float, capture_sigma_rad: float | None = None,
                                 max_cost: float = 1.0) -> torch.Tensor:
  qvel = state(env)[1]
  return _rate_cost(env, qvel[:, 1] + qvel[:, 2], reference_rad_s, capture_sigma_rad, max_cost)


def force_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.action.square().sum(dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=1)


@discounted
def balancing_fall_cost(env, *, angle_limit_rad: float, discount: float) -> torch.Tensor:
  return _forever(balancing_fall(env, angle_limit_rad=angle_limit_rad), discount)


# Terminations.

def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def rail_failure(env: ManagerBasedRlEnv, *, rail_limit_m: float) -> torch.Tensor:
  return state(env)[0][:, 0].abs() > rail_limit_m


def balancing_fall(env: ManagerBasedRlEnv, *, angle_limit_rad: float) -> torch.Tensor:
  if angle_limit_rad <= 0.0:
    raise ValueError("angle_limit_rad must be positive")
  link1_error, link2_absolute_error, _ = _double_pendulum_errors(state(env)[0])
  return (link1_error.abs() > angle_limit_rad) | (link2_absolute_error.abs() > angle_limit_rad)


def numerical_failure(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Non-finite state, or link rates far beyond any swing (4x the observation scale): a diverged simulation."""
  qpos, qvel = state(env)
  diverged = (qvel[:, 1:].abs() > 4.0 * env_contract(env).joint_velocity_scale).any(dim=1)
  return ~(torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1)) | diverged


class full_fall_penalty:
  """One-time cost when a captured pendulum falls (the full evaluator's fall rule); the episode goes on.

  Captured = inside the evaluator's capture box; the next capture re-arms it.
  """

  def __init__(self, cfg, env) -> None:
    self.armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None) -> None:
    self.armed[slice(None) if env_ids is None else env_ids] = False

  def __call__(self, env, *, capture_cart_m: float, angle_rad: float, velocity: float,
               fall_angle_rad: float) -> torch.Tensor:
    q, v = state(env)
    e1, e2_absolute, e2_relative = _double_pendulum_errors(q)
    tilt = torch.maximum(e1.abs(), e2_absolute.abs())
    captured = ((tilt < angle_rad) & (e2_relative.abs() < angle_rad) & (q[:, 0].abs() < capture_cart_m)
                & (v[:, 0].abs() < velocity * env_scales(env).cart_speed_per_rate)
                & (v[:, 1:].abs().amax(dim=1) < velocity) & ((v[:, 1] + v[:, 2]).abs() < velocity))
    fallen = self.armed & (tilt > fall_angle_rad)
    self.armed = (self.armed & ~fallen) | captured
    return fallen.float()
