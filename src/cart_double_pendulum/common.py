"""Plant contract compiled from the MJCF file, and the policy observation.

Every physical constant comes from the MJCF; the control period and the sensor noise are hardware choices.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "assets" / "cart_double_pendulum_real.xml"

CONTROL_TIMESTEP_S = 0.02
# Sensor noise standard deviation in physical units; provisional, calibrate from encoder logs.
SENSOR_NOISE = {"cart_position_m": 0.00423, "cart_velocity_m_s": 0.02, "angle_rad": 0.002, "joint_rate_rad_s": 0.06}
NON_NEGATIVE = ("cart_armature_kg", "base_joint_damping", "elbow_joint_damping",
                "cart_frictionloss_n", "base_joint_frictionloss", "elbow_joint_frictionloss")

CART_BODY, LINK1_BODY, LINK2_BODY, TIP_SITE = "cart", "link1", "link2", "tip"
JOINT_NAMES = ("cart_slide", "base_joint", "elbow_joint")

# Plant constants that reward terms may receive (see tasks.mdp.plant_term).
PLANT_PARAMETERS = (
  "link1_length_m",
  "link2_length_m",
  "link1_com_m",
  "link2_com_m",
  "link1_mass_kg",
  "link2_mass_kg",
  "link1_com_inertia_kg_m2",
  "link2_com_inertia_kg_m2",
  "gravity_m_s2",
)


@dataclass(frozen=True)
class CartDoublePendulumContract:
  model_path: str
  actuator_force_limit_n: float
  rail_limit_m: float
  cart_mass_kg: float
  link1_length_m: float
  link2_length_m: float
  link1_com_m: float
  link2_com_m: float
  link1_mass_kg: float
  link2_mass_kg: float
  link1_com_inertia_kg_m2: float
  link2_com_inertia_kg_m2: float
  cart_armature_kg: float
  cart_damping_n_s_m: float
  base_joint_damping: float
  elbow_joint_damping: float
  cart_frictionloss_n: float
  base_joint_frictionloss: float
  elbow_joint_frictionloss: float
  gravity_m_s2: float
  physics_timestep_s: float
  cart_velocity_scale: float
  joint_velocity_scale: float
  control_timestep_s: float = CONTROL_TIMESTEP_S
  observation_names: tuple[str, ...] = (
    "cart_position_norm",
    "cart_velocity_norm",
    "sin_theta1",
    "cos_theta1",
    "sin_theta2",
    "cos_theta2",
    "theta1_velocity_norm",
    "theta2_velocity_norm",
  )
  action_names: tuple[str, ...] = ("cart_force",)
  joint_names: tuple[str, ...] = JOINT_NAMES
  actuated_joint: str = JOINT_NAMES[0]

  def __post_init__(self) -> None:
    values = {key: value for key, value in asdict(self).items() if isinstance(value, float)}
    invalid = sorted(key for key, value in values.items()
                     if not np.isfinite(value) or value < 0.0 or (value == 0.0 and key not in NON_NEGATIVE))
    if invalid:
      raise ValueError(f"contract values must be finite and positive (or non-negative): {invalid}")
    ratio = self.control_timestep_s / self.physics_timestep_s
    if abs(ratio - round(ratio)) > 1e-9:
      raise ValueError("control timestep must be an integer multiple of the MJCF timestep")

  @property
  def observation_dim(self) -> int:
    return len(self.observation_names)

  @property
  def decimation(self) -> int:
    return round(self.control_timestep_s / self.physics_timestep_s)

  def plant_parameters(self) -> dict[str, float]:
    return {name: getattr(self, name) for name in PLANT_PARAMETERS}


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
  index = mujoco.mj_name2id(model, kind, name)
  if index < 0:
    raise ValueError(f"MJCF model is missing {kind.name.removeprefix('mjOBJ_').lower()} {name!r}")
  return index


def _axis_inertia(model: mujoco.MjModel, body: int, joint: int) -> float:
  """Inertia of ``body`` about its COM, around the axis of ``joint``."""
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
  axis = rotation.reshape(3, 3).T @ model.jnt_axis[joint]
  return float(np.sum(model.body_inertia[body] * axis * axis))


def _link_geometry(
  model: mujoco.MjModel, body: int, joint: int, end: np.ndarray, name: str
) -> tuple[float, float]:
  """Link length and joint-to-COM distance in the swing plane.

  The energy rewards need the COM on the line between the link's joints.
  """
  axis = model.jnt_axis[joint] / np.linalg.norm(model.jnt_axis[joint])
  planar = lambda vector: vector - np.dot(vector, axis) * axis
  span = planar(end)
  length = float(np.linalg.norm(span))
  if length <= 0.0:
    raise ValueError(f"{name} has zero length in the MJCF model")
  direction = span / length
  com = planar(model.body_ipos[body])
  distance = float(np.dot(com, direction))
  if np.linalg.norm(com - distance * direction) > 1e-9:
    raise ValueError(
      f"{name} centre of mass {model.body_ipos[body].tolist()} is off the line between its "
      "joints; the reward model assumes it lies on that line"
    )
  return length, distance


def load_contract(model_path: str | Path = MODEL_PATH) -> CartDoublePendulumContract:
  path = Path(model_path).resolve()
  model = mujoco.MjModel.from_xml_path(str(path))
  cart = _id(model, mujoco.mjtObj.mjOBJ_BODY, CART_BODY)
  link1 = _id(model, mujoco.mjtObj.mjOBJ_BODY, LINK1_BODY)
  link2 = _id(model, mujoco.mjtObj.mjOBJ_BODY, LINK2_BODY)
  tip = _id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
  slide, base, elbow = (_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES)
  if model.site_bodyid[tip] != link2 or model.body_parentid[link2] != link1:
    raise ValueError("MJCF must nest link2 inside link1 and place the tip site on link2")
  if model.nu != 1 or model.actuator_trnid[0, 0] != slide:
    raise ValueError("MJCF must define exactly one actuator, driving cart_slide")
  force = model.actuator_forcerange[0]
  if not model.actuator_forcelimited[0] or force[0] != -force[1]:
    raise ValueError("cart actuator needs a symmetric forcerange")
  if model.actuator_gear[0, 0] != 1.0:
    raise ValueError("cart actuator gear must be 1 so that control equals force")
  rail = model.jnt_range[slide]
  if not model.jnt_limited[slide] or rail[0] != -rail[1]:
    raise ValueError("cart_slide needs a symmetric range")
  if model.opt.gravity[0] != 0.0 or model.opt.gravity[1] != 0.0 or model.opt.gravity[2] >= 0.0:
    raise ValueError("gravity must point along -z")
  link1_length, link1_com = _link_geometry(model, link1, base, model.body_pos[link2], LINK1_BODY)
  link2_length, link2_com = _link_geometry(model, link2, elbow, model.site_pos[tip], LINK2_BODY)
  dofs = [model.jnt_dofadr[joint] for joint in (slide, base, elbow)]
  gravity = float(-model.opt.gravity[2])
  masses = model.body_mass[[link1, link2]]
  link_inertia = (_axis_inertia(model, link1, base) + masses[0] * link1_com ** 2
                  + _axis_inertia(model, link2, elbow) + masses[1] * (link1_length + link2_com) ** 2)
  upright_energy = 2 * gravity * (masses[0] * link1_com + masses[1] * (link1_length + link2_com))
  # Observation scales: the fastest the cart can move on this rail, and twice the link rate of an upright-energy swing.
  acceleration = force[1] / (model.body_mass[cart] + model.dof_armature[dofs[0]] + masses.sum())
  cart_speed = min(float(np.sqrt(2 * acceleration * rail[1])),
                   float(force[1] / model.dof_damping[dofs[0]]) if model.dof_damping[dofs[0]] > 0 else np.inf)
  return CartDoublePendulumContract(
    model_path=str(path),
    actuator_force_limit_n=float(force[1]),
    rail_limit_m=float(rail[1]),
    cart_mass_kg=float(model.body_mass[cart]),
    link1_length_m=link1_length,
    link2_length_m=link2_length,
    link1_com_m=link1_com,
    link2_com_m=link2_com,
    link1_mass_kg=float(model.body_mass[link1]),
    link2_mass_kg=float(model.body_mass[link2]),
    link1_com_inertia_kg_m2=_axis_inertia(model, link1, base),
    link2_com_inertia_kg_m2=_axis_inertia(model, link2, elbow),
    cart_armature_kg=float(model.dof_armature[dofs[0]]),
    cart_damping_n_s_m=float(model.dof_damping[dofs[0]]),
    base_joint_damping=float(model.dof_damping[dofs[1]]),
    elbow_joint_damping=float(model.dof_damping[dofs[2]]),
    cart_frictionloss_n=float(model.dof_frictionloss[dofs[0]]),
    base_joint_frictionloss=float(model.dof_frictionloss[dofs[1]]),
    elbow_joint_frictionloss=float(model.dof_frictionloss[dofs[2]]),
    gravity_m_s2=gravity,
    physics_timestep_s=float(model.opt.timestep),
    cart_velocity_scale=cart_speed,
    joint_velocity_scale=float(2 * np.sqrt(2 * upright_energy / link_inertia)),
  )


CONTRACT = load_contract()
NOISE_REFERENCE = 0.02  # noise setting that means SENSOR_NOISE (link channels: see noise_profile)


@cache
def noise_profile(contract: CartDoublePendulumContract = CONTRACT, *, noise_level: float = 1.0) -> tuple[float, ...]:
  """Per-observation noise factors; link channels follow curriculum.link_noise_level (0.75 x SENSOR_NOISE at level 1)."""
  from cart_double_pendulum import curriculum
  link_level = curriculum.link_noise_level(noise_level)
  link_factor = link_level / noise_level if noise_level > 0.0 else 1.0
  n = SENSOR_NOISE
  physical = (n["cart_position_m"] / contract.rail_limit_m, n["cart_velocity_m_s"] / contract.cart_velocity_scale,
              *(n["angle_rad"] * link_factor,) * 4,
              *(n["joint_rate_rad_s"] * link_factor / contract.joint_velocity_scale,) * 2)
  return tuple(value / NOISE_REFERENCE for value in physical)


# Rail guard: the worst randomized cart (mass x1.15, damping x0.75, force x0.81) must still stop inside 0.9 x rail.
RAIL_GUARD_FRACTION, GUARD_MASS, GUARD_DAMPING, GUARD_FORCE = 0.9, 1.15, 0.75, 0.81


def _guard_upper(x, v, contract, xp):
  """Largest normalized force toward +rail after which the worst cart can still brake inside the guard limit."""
  mass = contract.cart_mass_kg + contract.cart_armature_kg + contract.link1_mass_kg + contract.link2_mass_kg
  force, damping, dt = contract.actuator_force_limit_n, contract.cart_damping_n_s_m, contract.control_timestep_s
  # One control period under action u (nominal plant): x1 = x0 + xu*u, v1 = v0 + vu*u.
  if damping > 1e-3:
    tau = mass / damping
    decay = math.exp(-dt / tau)
    x0, v0 = x + v * tau * (1. - decay), v * decay
    xu, vu = force / damping * (dt - tau * (1. - decay)), force / damping * (1. - decay)
  else:
    x0, v0, xu, vu = x + v * dt, v, .5 * force / mass * dt * dt, force / mass * dt
  brake, worst_mass, worst_damping = GUARD_FORCE * force, GUARD_MASS * mass, GUARD_DAMPING * damping
  def stop(u):
    speed = xp.clip(v0 + vu * u, 0., None)
    if worst_damping > 1e-3:
      tau_w = worst_mass / worst_damping
      return x0 + xu * u + tau_w * (speed - brake / worst_damping * xp.log1p(speed * worst_damping / brake))
    return x0 + xu * u + .5 * worst_mass * speed * speed / brake
  limit = RAIL_GUARD_FRACTION * contract.rail_limit_m
  low, high = x * 0. - 1., x * 0. + 1.
  # Bisection on the monotone stop point; stays at -1 (full brake) when nothing is safe.
  for _ in range(16):
    middle = .5 * (low + high)
    safe = stop(middle) <= limit
    low, high = xp.where(safe, middle, low), xp.where(safe, high, middle)
  return xp.where(stop(x * 0. + 1.) <= limit, x * 0. + 1., low)


def rail_guard(action, x, v, contract: CartDoublePendulumContract = CONTRACT, xp: Any = np):
  """Normalized action limited so the cart can always stop inside RAIL_GUARD_FRACTION of the rail."""
  upper, lower = _guard_upper(x, v, contract, xp), -_guard_upper(-x, -v, contract, xp)
  return xp.minimum(xp.maximum(action, lower), upper)


def pendulum_energy(qpos, qvel, xp, *, link1_length_m, link1_com_m, link2_com_m, link1_mass_kg, link2_mass_kg,
                    link1_com_inertia_kg_m2, link2_com_inertia_kg_m2, gravity_m_s2, **_):
  """Link energy in the cart frame (cart kinetic energy excluded) and its value at rest upright."""
  theta1, theta2 = qpos[..., 1], qpos[..., 1] + qpos[..., 2]
  omega1, omega2 = qvel[..., 1], qvel[..., 1] + qvel[..., 2]
  link2_vx = link1_length_m * xp.cos(theta1) * omega1 + link2_com_m * xp.cos(theta2) * omega2
  link2_vz = link1_length_m * xp.sin(theta1) * omega1 + link2_com_m * xp.sin(theta2) * omega2
  kinetic = 0.5 * (link1_mass_kg * link1_com_m ** 2 + link1_com_inertia_kg_m2) * omega1 ** 2
  kinetic = kinetic + 0.5 * link2_mass_kg * (link2_vx ** 2 + link2_vz ** 2) + 0.5 * link2_com_inertia_kg_m2 * omega2 ** 2
  potential = gravity_m_s2 * ((link1_mass_kg * link1_com_m + link2_mass_kg * link1_length_m) * (1.0 - xp.cos(theta1))
                              + link2_mass_kg * link2_com_m * (1.0 - xp.cos(theta2)))
  upright = 2.0 * gravity_m_s2 * (link1_mass_kg * link1_com_m + link2_mass_kg * (link1_length_m + link2_com_m))
  return kinetic + potential, upright


def angle_error(angle: Any, target: float, array_api: Any) -> Any:
  difference = angle - target
  return array_api.arctan2(array_api.sin(difference), array_api.cos(difference))


def policy_observation(
  qpos: Any,
  qvel: Any,
  array_api: Any,
  contract: CartDoublePendulumContract = CONTRACT,
) -> Any:
  """The eight policy inputs, in ``observation_names`` order."""
  return array_api.stack(
    (
      qpos[..., 0] / contract.rail_limit_m,
      qvel[..., 0] / contract.cart_velocity_scale,
      array_api.sin(qpos[..., 1]),
      array_api.cos(qpos[..., 1]),
      array_api.sin(qpos[..., 2]),
      array_api.cos(qpos[..., 2]),
      qvel[..., 1] / contract.joint_velocity_scale,
      qvel[..., 2] / contract.joint_velocity_scale,
    ),
    -1,
  )
