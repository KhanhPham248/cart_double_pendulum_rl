"""Headless MuJoCo runtime used by every evaluator and by observer training."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from cart_double_pendulum.common import (
  CONTRACT,
  CartDoublePendulumContract,
  angle_error,
  noise_profile,
  policy_observation,
  rail_guard,
)
from cart_double_pendulum.plant_scales import plant_scales


class CartDoublePendulumRuntime:
  def __init__(
    self,
    model_path: str | Path,
    *,
    mass_scale: float = 1.0,
    damping_scale: float = 1.0,
    force_scale: float = 1.0,
    observation_noise_std: float = 0.0,
    noise_curriculum_level: float = 1.0,
    seed: int = 0,
    contract: CartDoublePendulumContract = CONTRACT,
    independent_observation_noise: bool = False,
  ) -> None:
    if min(mass_scale, damping_scale, force_scale) <= 0.0:
      raise ValueError("runtime scales must be positive")
    if observation_noise_std < 0.0:
      raise ValueError("observation noise must be non-negative")
    self.rng = np.random.default_rng(seed)
    # Opt-in for paired evaluation: sensor sampling must not change reset draws.
    self.noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 9187])) if independent_observation_noise else self.rng
    self.contract = contract
    self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    self.data = mujoco.MjData(self.model)
    self._validate_model()
    for body_name in ("cart", "link1", "link2"):
      body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
      if body_id < 0:
        raise ValueError(f"runtime model is missing body {body_name!r}")
      self.model.body_mass[body_id] *= mass_scale
      # Density scale: mass and inertia together.
      self.model.body_inertia[body_id] *= mass_scale
    # The cart armature (reflected rotor inertia) belongs to the cart mass.
    self.model.dof_armature[self._joint_qvel_index(contract.actuated_joint)] *= mass_scale
    self.model.dof_damping[:] *= damping_scale
    self.traction_force_limit_n = contract.actuator_force_limit_n * force_scale
    self.model.actuator_ctrlrange[0] = (
      -self.traction_force_limit_n,
      self.traction_force_limit_n,
    )
    self.model.actuator_forcerange[0] = (
      -self.traction_force_limit_n,
      self.traction_force_limit_n,
    )
    self.observation_noise_std = observation_noise_std
    self.sensor_noise_profile = np.asarray(noise_profile(contract, noise_level=noise_curriculum_level))
    self.applied_action, self.guard_active = 0.0, False
    self.qpos_indices = tuple(self._joint_qpos_index(name) for name in contract.joint_names)
    self.qvel_indices = tuple(self._joint_qvel_index(name) for name in contract.joint_names)

  def _validate_model(self) -> None:
    if (self.model.nq, self.model.nv, self.model.nu) != (3, 3, 1):
      raise ValueError(
        "runtime model must have the three policy joints and one actuator; "
        f"got {(self.model.nq, self.model.nv, self.model.nu)}"
      )
    if abs(self.model.opt.timestep - self.contract.physics_timestep_s) > 1.0e-12:
      raise ValueError("MuJoCo timestep does not match contract")
    if self.model.actuator_trntype[0] != mujoco.mjtTrn.mjTRN_JOINT:
      raise ValueError("the only actuator must drive cart_slide traction")
    actuator_joint = int(self.model.actuator_trnid[0, 0])
    cart_slide = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_JOINT, self.contract.actuated_joint
    )
    if actuator_joint != cart_slide:
      raise ValueError("actuator is not attached to cart_slide")

  def _joint_qpos_index(self, name: str) -> int:
    joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
      raise ValueError(f"model is missing joint {name!r}")
    return int(self.model.jnt_qposadr[joint])

  def _joint_qvel_index(self, name: str) -> int:
    joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
      raise ValueError(f"model is missing joint {name!r}")
    return int(self.model.jnt_dofadr[joint])

  @property
  def qpos(self) -> np.ndarray:
    return self.data.qpos[list(self.qpos_indices)].copy()

  @property
  def qvel(self) -> np.ndarray:
    return self.data.qvel[list(self.qvel_indices)].copy()

  def observation(self) -> np.ndarray:
    result = policy_observation(self.qpos, self.qvel, np, self.contract).astype(np.float32)
    if self.observation_noise_std:
      result += self.noise_rng.normal(0.0, self.observation_noise_std, result.shape) * self.sensor_noise_profile
    if not np.isfinite(result).all():
      raise FloatingPointError("runtime observation is non-finite")
    return result

  def reset(
    self,
    *,
    mode: str = "hanging",
    cart_position_m: float | None = None,
    angle_rad: float | None = None,
    velocity: float | None = None,
  ) -> np.ndarray:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[:] = 0.0
    self.data.qvel[:] = 0.0
    # Plant-scale defaults; hanging angles in radians.
    scales = plant_scales(self.contract)
    cart_range = 0.18 * scales.rail if cart_position_m is None else cart_position_m
    angle_range = (0.853 * scales.angle if mode == "upright" else 0.18) if angle_rad is None else angle_rad
    velocity_range = 0.148 * scales.rate if velocity is None else velocity
    if min(cart_range, angle_range) < 0.0 or velocity_range < 0.0:
      raise ValueError("reset ranges must be non-negative")
    if mode == "hanging":
      policy_qpos = self.rng.uniform(-1.0, 1.0, size=3) * (cart_range, angle_range, angle_range)
      policy_qvel = self.rng.uniform(-velocity_range, velocity_range, size=3)
    elif mode == "upright":
      policy_qpos = np.array(
        [self.rng.uniform(-cart_range, cart_range), math.pi, 0.0], dtype=np.float64
      )
      policy_qpos[1:] += self.rng.uniform(-angle_range, angle_range, size=2)
      policy_qvel = self.rng.uniform(-velocity_range, velocity_range, size=3)
    elif mode == "random":
      policy_qpos = self.rng.uniform(
      [-0.9 * self.contract.rail_limit_m, -math.pi, -math.pi],
      [0.9 * self.contract.rail_limit_m, math.pi, math.pi],
      )
      policy_qvel = self.rng.uniform(-0.634, 0.634, size=3) * scales.rate
    else:
      raise ValueError("reset mode must be hanging, upright or random")
    policy_qvel[0] *= scales.cart_speed_per_rate
    self.data.qpos[list(self.qpos_indices)] = policy_qpos
    self.data.qvel[list(self.qvel_indices)] = policy_qvel
    mujoco.mj_forward(self.model, self.data)
    return self.observation()

  def step(self, normalized_force: float) -> np.ndarray:
    action = float(np.asarray(normalized_force).reshape(-1)[0])
    if not math.isfinite(action):
      raise FloatingPointError("action is non-finite")
    # The rail guard is part of the plant interface; deployment applies the same limit.
    commanded = float(np.clip(action, -1.0, 1.0))
    self.applied_action = float(rail_guard(commanded, self.qpos[0], self.qvel[0], self.contract))
    self.guard_active = self.applied_action != commanded
    self.data.ctrl[0] = self.applied_action * self.traction_force_limit_n
    for _ in range(self.contract.decimation):
      mujoco.mj_step(self.model, self.data)
    return self.observation()

  def is_stable(
    self,
    *,
    cart_threshold_m: float | None = None,
    angle_threshold_rad: float | None = None,
    velocity_threshold: float | None = None,
  ) -> bool:
    """Still upright state; ``velocity_threshold`` is rad/s, its cart part follows the plant scales."""
    scales = plant_scales(self.contract)
    cart_threshold_m = 0.3 * scales.rail if cart_threshold_m is None else cart_threshold_m
    angle_threshold_rad = 0.711 * scales.angle if angle_threshold_rad is None else angle_threshold_rad
    velocity_threshold = 0.2114 * scales.rate if velocity_threshold is None else velocity_threshold
    qpos, qvel = self.qpos, self.qvel
    link2_absolute_error = angle_error(qpos[1] + qpos[2], math.pi, np)
    link2_relative_error = angle_error(qpos[2], 0.0, np)
    link2_absolute_velocity = qvel[1] + qvel[2]
    return bool(
      abs(qpos[0]) < cart_threshold_m
      and abs(angle_error(qpos[1], math.pi, np)) < angle_threshold_rad
      and abs(link2_absolute_error) < angle_threshold_rad
      and abs(link2_relative_error) < angle_threshold_rad
      and abs(qvel[0]) < velocity_threshold * scales.cart_speed_per_rate
      and np.max(np.abs(qvel[1:])) < velocity_threshold
      and abs(link2_absolute_velocity) < velocity_threshold
    )

  def is_swing_up_handoff(
    self,
    *,
    cart_safe_m: float,
    angle_threshold_rad: float,
    cart_velocity_threshold_m_s: float,
    angular_velocity_threshold_rad_s: float,
  ) -> bool:
    """Whether swing-up has reached a calm upright state safe for handoff."""

    qpos, qvel = self.qpos, self.qvel
    link2_absolute_error = angle_error(qpos[1] + qpos[2], math.pi, np)
    link2_absolute_velocity = qvel[1] + qvel[2]
    return bool(
      abs(qpos[0]) < cart_safe_m
      and abs(angle_error(qpos[1], math.pi, np)) < angle_threshold_rad
      and abs(link2_absolute_error) < angle_threshold_rad
      and abs(qvel[0]) < cart_velocity_threshold_m_s
      and abs(qvel[1]) < angular_velocity_threshold_rad_s
      and abs(link2_absolute_velocity) < angular_velocity_threshold_rad_s
    )
