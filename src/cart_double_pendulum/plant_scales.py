"""Plant scales from the MJCF; task parameters are multiples of them."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math
from typing import Any
import warnings

import mujoco
import numpy as np

from cart_double_pendulum.common import CONTRACT, NOISE_REFERENCE, noise_profile, CartDoublePendulumContract, pendulum_energy

UNITS = ("rail", "force", "angle", "rate", "cart_speed")
SHARED_CART_SPEED_PER_RATE = 1.521  # cart part of a shared velocity range, in basins


@dataclass(frozen=True)
class PlantScales:
  """Rail (m), force (N), upright capture basin (rad, rad/s, m/s) and upright pole (1/s)."""
  rail: float
  force: float
  angle: float
  rate: float
  cart_speed: float
  upright_pole: float

  def value(self, unit: str, multiple: float) -> float:
    if unit not in UNITS:
      raise ValueError(f"unknown plant scale {unit!r}; use one of {UNITS}")
    return float(multiple) * getattr(self, unit)

  @property
  def cart_speed_per_rate(self) -> float:
    """m/s of cart per rad/s of a shared velocity range."""
    return SHARED_CART_SPEED_PER_RATE * self.cart_speed / self.rate


def _lqr_gain(model: mujoco.MjModel, decimation: int, rail: float, force: float) -> tuple[np.ndarray, float]:
  data = mujoco.MjData(model)
  data.qpos[:] = (0.0, math.pi, 0.0)
  mujoco.mj_forward(model, data)
  a = np.zeros((6, 6))
  b = np.zeros((6, 1))
  mujoco.mjd_transitionFD(model, data, 1e-6, True, a, b, None, None)
  pole = float(np.log(np.abs(np.linalg.eigvals(a))).max() / model.opt.timestep)
  ad = np.linalg.matrix_power(a, decimation)
  bd = sum(np.linalg.matrix_power(a, i) @ b for i in range(decimation))
  gravity = -float(model.opt.gravity[2])
  # Natural-unit weights: rail, radian, g/pole, pole, force limit.
  q = np.diag(1.0 / np.array([rail, 1.0, 1.0, gravity / pole, pole, pole]) ** 2)
  r = np.array([[1.0 / force**2]])
  p = q.copy()
  for _ in range(20000):
    gain = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
    p_next = q + ad.T @ p @ (ad - bd @ gain)
    if np.max(np.abs(p_next - p)) <= 1e-9 * np.max(np.abs(p_next)):
      break
    p = p_next
  else:
    raise ValueError("the upright LQR design did not converge at this control period")
  gain = np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)
  if np.abs(np.linalg.eigvals(ad - bd @ gain)).max() >= 1.0:
    raise ValueError("the upright LQR closed loop is unstable at this control period")
  return gain, pole


def _caught(model, data, gain, decimation, rail, qpos, qvel, seconds: float = 2.0) -> bool:
  mujoco.mj_resetData(model, data)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  low, high = model.actuator_ctrlrange[0]
  for _ in range(round(seconds / (decimation * model.opt.timestep))):
    state = np.r_[data.qpos[0], data.qpos[1] - math.pi, data.qpos[2], data.qvel]
    data.ctrl[0] = np.clip(-(gain @ state)[0], low, high)
    for _ in range(decimation):
      mujoco.mj_step(model, data)
    if abs(data.qpos[0]) > 0.95 * rail or not np.isfinite(data.qpos).all():
      return False
  errors = (data.qpos[1] - math.pi, data.qpos[1] + data.qpos[2] - math.pi)
  return max(abs(math.remainder(e, 2 * math.pi)) for e in errors) < 0.05


def _largest(test, high: float, steps: int = 18) -> float:
  low = 0.0
  for _ in range(steps):
    middle = 0.5 * (low + high)
    low, high = (middle, high) if test(middle) else (low, middle)
  return low


def measure_plant_scales(model: mujoco.MjModel, control_timestep_s: float) -> PlantScales:
  """Rail, force and the largest upright offsets a saturated LQR at the control period still catches."""
  if model.nq != 3 or model.nu != 1:
    raise ValueError("expected a cart with two links and one actuator")
  decimation = round(control_timestep_s / model.opt.timestep)
  rail = float(model.jnt_range[0, 1])
  force = float(model.actuator_forcerange[0, 1] if model.actuator_forcelimited[0]
                else model.actuator_ctrlrange[0, 1] * model.actuator_gear[0, 0])
  gain, pole = _lqr_gain(model, decimation, rail, force)
  data = mujoco.MjData(model)
  caught = lambda qpos, qvel: _caught(model, data, gain, decimation, rail, qpos, qvel)
  directions = [(math.cos(a), math.sin(a)) for a in np.linspace(0.0, math.pi, 4, endpoint=False)]
  angle = np.mean([_largest(lambda s: caught((0.0, math.pi + s * u, s * (w - u)), (0.0, 0.0, 0.0)), 1.5)
                   for u, w in directions])
  rate = np.mean([_largest(lambda s: caught((0.0, math.pi, 0.0), (0.0, s * u, s * (w - u))), 20.0)
                  for u, w in directions])
  cart_speed = _largest(lambda s: caught((0.0, math.pi, 0.0), (s, 0.0, 0.0)), 20.0)
  scales = PlantScales(rail=rail, force=force, angle=float(angle), rate=float(rate),
                       cart_speed=float(cart_speed), upright_pole=pole)
  if min(scales.angle, scales.rate, scales.cart_speed) <= 0.0:
    raise ValueError(f"the plant cannot be balanced at a {control_timestep_s} s control period: {scales}")
  return scales


@cache
def plant_scales(contract: CartDoublePendulumContract = CONTRACT) -> PlantScales:
  model = mujoco.MjModel.from_xml_path(contract.model_path)
  return measure_plant_scales(model, contract.control_timestep_s)


def _pump_time(model: mujoco.MjModel, contract: CartDoublePendulumContract, seconds: float = 20.0) -> float | None:
  """Fastest time a bang-bang energy pump (cart kept inside the rail) reaches the upright energy; None if never."""
  data = mujoco.MjData(model)
  decimation, force, rail = contract.decimation, contract.actuator_force_limit_n, contract.rail_limit_m
  plant = contract.plant_parameters()
  best = None
  for sign in (1.0, -1.0):
    for edge in (0.3, 0.5, 0.7):
      mujoco.mj_resetData(model, data)
      data.qpos[1] = 0.05
      for step in range(round(seconds / contract.control_timestep_s)):
        outward = abs(data.qpos[0]) > edge * rail and data.qpos[0] * data.qvel[0] > 0
        data.ctrl[0] = -force * np.sign(data.qpos[0]) if outward else sign * force * np.sign(
          data.qvel[1] * math.cos(data.qpos[1]))
        for _ in range(decimation):
          mujoco.mj_step(model, data)
        if abs(data.qpos[0]) > 0.95 * rail or not np.isfinite(data.qpos).all():
          break
        energy, upright = pendulum_energy(data.qpos, data.qvel, np, **plant)
        if energy >= upright:
          best = min(best or math.inf, (step + 1) * contract.control_timestep_s)
          break
  return best


@cache
def plant_report(contract: CartDoublePendulumContract = CONTRACT) -> dict:
  """Feasibility of the plant: control margin, capture basin against sensor noise and swing-up pump time."""
  scales = plant_scales(contract)
  sensor_profile = noise_profile(contract)
  report = {"pole_times_control_period": scales.upright_pole * contract.control_timestep_s,
            "basin_angle_over_noise": scales.angle / (NOISE_REFERENCE * sensor_profile[2]),
            "basin_rate_over_noise": scales.rate / (NOISE_REFERENCE * sensor_profile[6] * contract.joint_velocity_scale),
            "pump_to_upright_energy_s": _pump_time(mujoco.MjModel.from_xml_path(contract.model_path), contract)}
  checks = ((report["pole_times_control_period"] > 0.5, "the upright pole grows more than 1.65x per control period"),
            (report["basin_angle_over_noise"] < 3.0, "angle noise exceeds a third of the capture basin"),
            (report["basin_rate_over_noise"] < 1.0, "joint-rate noise exceeds the capture basin"),
            (report["pump_to_upright_energy_s"] is None, "a bang-bang pump cannot reach the upright energy in 20 s"))
  report["warnings"] = [message for failed, message in checks if failed]
  for message in report["warnings"]:
    warnings.warn(f"plant feasibility: {message}", stacklevel=2)
  return report


def scale_tag(value: Any) -> tuple[str, float] | None:
  """``(unit, multiple)`` for a ``{"<unit>": number}`` value, else None."""
  if isinstance(value, dict) and len(value) == 1:
    unit, multiple = next(iter(value.items()))
    if unit in UNITS and isinstance(multiple, (int, float)) and not isinstance(multiple, bool):
      return unit, float(multiple)
  return None


def resolve(value: Any, scales: PlantScales | None = None) -> Any:
  """Replace every ``{"<unit>": multiple}`` in loaded task JSON with a physical value."""
  tag = scale_tag(value)
  if tag is not None:
    return (scales or plant_scales()).value(*tag)
  if isinstance(value, dict):
    return {key: resolve(item, scales) for key, item in value.items()}
  if isinstance(value, list):
    return [resolve(item, scales) for item in value]
  return value
