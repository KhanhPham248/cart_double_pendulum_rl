"""Robustness evaluation on the standalone MuJoCo runtime, shared by all evaluators.

Starts are independent of training resets: hanging (runtime defaults) or exactly upright.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import mujoco
import numpy as np

from cart_double_pendulum import curriculum
from cart_double_pendulum.common import CONTRACT, MODEL_PATH, pendulum_energy
from cart_double_pendulum.runtime import CartDoublePendulumRuntime
from cart_double_pendulum.tasks import SWING_UP_TASK, make_task
from cart_double_pendulum.tasks.config import (
  BALANCE_FALL_RAD, DAMPING_SCALE_RANGE, FORCE_SCALE_RANGE, INERTIA_SCALE_RANGE, PUSH_FORCE_N,
)


MODEL = MODEL_PATH
DT = CONTRACT.control_timestep_s
# An episode fails when the cart leaves this range or a link falls this far.
RAIL_FAILURE_M = SWING_UP_TASK.rail_failure_m
FALL_ANGLE_RAD = BALANCE_FALL_RAD


@dataclass(frozen=True)
class RobustCondition:
  key: str
  task: str
  initial_state: str
  duration_s: float
  hold_s: float
  random_axes: tuple[str, ...] = ()
  push_force_n: float = 0.0
  push_start_s: float = 2.0
  push_duration_s: float = 0.10
  seed: int = 74_101
  sensor_noise_scale: float = 1.0
  stability_task: str | None = None
  fall_check_after_stable_entry: bool = False


SWING_NOMINAL = RobustCondition("swing_nominal", "swing_up", "hanging", 20.0, 1.0, seed=74_101)
SWING_RANDOMIZED = RobustCondition(
  "swing_randomized", "swing_up", "hanging", 20.0, 1.0,
  ("mass", "damping", "force", "noise"), seed=75_101,
)
BALANCE_NOMINAL = RobustCondition("balance_nominal", "balancing", "upright_exact", 20.0, 10.0, seed=76_101)
BALANCE_PUSH = RobustCondition(
  "balance_push", "balancing", "upright_exact", 20.0, 10.0,
  ("mass", "damping", "force", "noise"), PUSH_FORCE_N, 2.0, .10, 77_101,
)


def scenarios(condition: RobustCondition, episodes: int) -> list[dict[str, float]]:
  if episodes <= 0:
    raise ValueError("episodes must be positive")
  if not math.isfinite(condition.sensor_noise_scale) or condition.sensor_noise_scale < 0:
    raise ValueError('sensor noise scale must be finite and nonnegative')
  rng = np.random.default_rng(condition.seed)
  values: list[dict[str, float]] = []
  axes = set(condition.random_axes)
  # Randomized cases follow the curriculum stage.
  level = curriculum.evaluation_level()
  inertia, damping, force = (curriculum.scaled_range(r, level)
                             for r in (INERTIA_SCALE_RANGE, DAMPING_SCALE_RANGE, FORCE_SCALE_RANGE))
  for _ in range(episodes):
    values.append({
      "mass_scale": float(rng.uniform(*inertia)) if "mass" in axes else 1.0,
      "damping_scale": float(rng.uniform(*damping)) if "damping" in axes else 1.0,
      "force_scale": float(rng.uniform(*force)) if "force" in axes else 1.0,
      "observation_noise_std": float(rng.uniform(0., .02)) * level * condition.sensor_noise_scale if "noise" in axes else 0.0,
      "noise_curriculum_level": level,
    })
  return values


def reset_runtime(runtime: CartDoublePendulumRuntime, initial_state: str) -> np.ndarray:
  if initial_state == "hanging":
    return runtime.reset(mode="hanging").astype(np.float32)
  if initial_state != "upright_exact":
    raise ValueError(f"unsupported initial state {initial_state!r}")
  mujoco.mj_resetData(runtime.model, runtime.data)
  runtime.data.qpos[:] = 0.0
  runtime.data.qvel[:] = 0.0
  runtime.data.qpos[list(runtime.qpos_indices)] = (0.0, math.pi, 0.0)
  mujoco.mj_forward(runtime.model, runtime.data)
  return runtime.observation().astype(np.float32)


def stable(runtime: CartDoublePendulumRuntime, condition: RobustCondition) -> bool:
  stability_task = condition.stability_task or condition.task
  cfg = make_task(stability_task).config
  if stability_task == "swing_up":
    return runtime.is_swing_up_handoff(
      cart_safe_m=float(cfg.handoff_cart_safe_m),
      angle_threshold_rad=float(cfg.handoff_angle_rad),
      cart_velocity_threshold_m_s=float(cfg.handoff_cart_velocity_m_s),
      angular_velocity_threshold_rad_s=float(cfg.handoff_angular_velocity_rad_s),
    )
  return runtime.is_stable(
    cart_threshold_m=cfg.stable_cart_m,
    angle_threshold_rad=cfg.stable_angle_rad,
    velocity_threshold=cfg.stable_velocity,
  )


def evaluate_controller(
  controller_factory: Callable[[CartDoublePendulumRuntime], Callable[[np.ndarray], float]],
  condition: RobustCondition,
  *,
  episodes: int,
  collect_rows: bool = False,
  strict_protocol: bool = False,
  numerical_failure_is_failure: bool = False,
  cart_tolerance_m: float | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
  """Evaluate a normalized-action controller on a fixed paired scenario set."""

  if cart_tolerance_m is not None and (not math.isfinite(cart_tolerance_m) or not 0 < cart_tolerance_m < CONTRACT.rail_limit_m):
    raise ValueError('cart_tolerance_m must be finite, positive and inside rail')
  hold_steps = round(condition.hold_s / DT)
  push_first = round(condition.push_start_s / DT)
  push_last = push_first + round(condition.push_duration_s / DT)
  total_steps = round(condition.duration_s / DT)
  successes = rail_failures = never_handoff = lost_after_handoff = fall_failures = 0
  guarded_steps = 0
  numerical_failures = 0
  stable_samples = total_samples = saturated = 0
  success_times: list[float] = []
  excursions: list[float] = []
  force_energy: list[float] = []
  peak_heights, peak_pendulum_energies = [], []
  a_height = CONTRACT.link1_mass_kg * CONTRACT.link1_com_m + CONTRACT.link2_mass_kg * CONTRACT.link1_length_m
  b_height = CONTRACT.link2_mass_kg * CONTRACT.link2_com_m
  rows: list[dict[str, float | int | str]] = []
  for episode, scenario in enumerate(scenarios(condition, episodes)):
    runtime = CartDoublePendulumRuntime(
      MODEL, seed=condition.seed + episode, contract=CONTRACT,
      independent_observation_noise=strict_protocol, **scenario
    )
    observation = reset_runtime(runtime, condition.initial_state)
    controller = controller_factory(runtime)
    if hasattr(controller,'reset'):
      controller.reset()
    streak = 0
    ever_handoff = False
    success_time: float | None = None
    episode_stable = episode_saturated = 0
    max_cart = energy = 0.0
    peak_height = peak_pendulum_energy = 0.0
    rail = False
    fallen = False
    numerical_failure = False
    numerical_error = ''
    for step in range(total_steps):
      pushing = condition.push_force_n != 0 and push_first <= step < push_last
      runtime.data.qfrc_applied[runtime.qvel_indices[0]] = condition.push_force_n if pushing else 0.0
      try:
        action = float(np.clip(controller(observation), -1.0, 1.0))
        observation = runtime.step(action).astype(np.float32)
      except FloatingPointError as error:
        if not numerical_failure_is_failure:
          raise
        numerical_failure = True
        numerical_error = str(error)
        runtime.data.qfrc_applied[runtime.qvel_indices[0]] = 0.
        break
      runtime.data.qfrc_applied[runtime.qvel_indices[0]] = 0.0
      force = abs(action) * runtime.traction_force_limit_n
      energy += force * force * DT
      episode_saturated += int(abs(action) >= .95)
      guarded_steps += int(runtime.guard_active)
      max_cart = max(max_cart, abs(float(runtime.qpos[0])))
      q, v = runtime.qpos, runtime.qvel
      height = .5 * (1. - (a_height*np.cos(q[1]) + b_height*np.cos(q[1]+q[2])) / (a_height+b_height))
      pendulum, upright = pendulum_energy(q, v, np, **CONTRACT.plant_parameters())
      peak_height = max(peak_height, float(height))
      peak_pendulum_energy = max(peak_pendulum_energy, float(np.clip(pendulum/upright, 0., 1.)))
      if abs(float(runtime.qpos[0])) > RAIL_FAILURE_M:
        rail = True
        break
      if (strict_protocol and (condition.task == 'balancing' or condition.fall_check_after_stable_entry)
          and (not condition.fall_check_after_stable_entry or ever_handoff)):
        q = runtime.qpos
        errors = np.arctan2(np.sin(np.array([q[1], q[1] + q[2]]) - math.pi),
                            np.cos(np.array([q[1], q[1] + q[2]]) - math.pi))
        if np.max(np.abs(errors)) > FALL_ANGLE_RAD:
          fallen = True
          break
      if condition.push_force_n and step < push_last:
        continue
      is_stable = stable(runtime, condition)
      if cart_tolerance_m is not None:
        is_stable = is_stable and abs(runtime.qpos[0]) <= cart_tolerance_m
      ever_handoff |= is_stable
      episode_stable += int(is_stable)
      total_samples += 1
      streak = streak + 1 if is_stable else 0
      if streak >= hold_steps and success_time is None:
        success_time = (step + 1) * DT - (condition.push_start_s if condition.push_force_n else 0.0)
    success = not numerical_failure and success_time is not None and (not strict_protocol or not (rail or fallen))
    numerical_failures += int(numerical_failure)
    successes += int(success)
    rail_failures += int(rail)
    fall_failures += int(fallen)
    never_handoff += int(not ever_handoff)
    lost_after_handoff += int(ever_handoff and not success)
    stable_samples += episode_stable
    saturated += episode_saturated
    excursions.append(max_cart); force_energy.append(energy)
    peak_heights.append(peak_height); peak_pendulum_energies.append(peak_pendulum_energy)
    if success:
      success_times.append(success_time)
    if collect_rows:
      rows.append({
        "condition": condition.key, "episode": episode, "success": int(success),
        "success_time_s": -1.0 if success_time is None else success_time,
        "rail_failure": int(rail), "ever_handoff": int(ever_handoff),
        "fall_failure": int(fallen),
        "numerical_failure": int(numerical_failure), "numerical_error": numerical_error,
        "max_cart_excursion_m": max_cart, "force_energy_n2s": energy,
        "peak_height_fraction": peak_height, "peak_pendulum_energy_fraction": peak_pendulum_energy,
        "action_saturation_fraction": episode_saturated / max(1, step + 1), **scenario,
      })
  return ({
    "episodes": episodes,
    "success_rate": successes / episodes,
    "stable_occupancy": stable_samples / max(1, (
      episodes * (total_steps - (push_last if condition.push_force_n else 0))
      if strict_protocol else total_samples)),
    "mean_time_to_success_s": float(np.mean(success_times)) if success_times else None,
    "rail_failure_rate": rail_failures / episodes,
    "rail_guard_fraction": guarded_steps / max(1, total_steps * episodes),
    "fall_failure_rate": fall_failures / episodes,
    "numerical_failure_rate": numerical_failures / episodes,
    "never_handoff_rate": never_handoff / episodes,
    "lost_after_handoff_rate": lost_after_handoff / episodes,
    "mean_max_cart_excursion_m": float(np.mean(excursions)),
    "mean_force_energy_n2s": float(np.mean(force_energy)),
    "mean_peak_height_fraction": float(np.mean(peak_heights)),
    "mean_peak_pendulum_energy_fraction": float(np.mean(peak_pendulum_energies)),
    "action_saturation_fraction": saturated / max(1, total_steps * episodes),
  }, rows)
