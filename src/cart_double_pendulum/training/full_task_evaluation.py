"""Capture and centring evaluation shared by full_ppo and full_sac.

Scores use clean state; controllers see noisy observations.  Push trials share
seeds across signs and score from the end of the pulse; any later failure
voids the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cart_double_pendulum import curriculum
from cart_double_pendulum.common import CONTRACT, MODEL_PATH
from cart_double_pendulum.plant_scales import plant_scales, resolve
from cart_double_pendulum.runtime import CartDoublePendulumRuntime


PROTOCOL = "full_episode_capture_center_push"
DT = CONTRACT.control_timestep_s
MODEL = MODEL_PATH


def _load_config(path):
    config = resolve(json.loads(Path(path).read_text()))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"full-task evaluation requires protocol {PROTOCOL!r}")
    for key in ("duration_s", "capture_hold_s", "center_hold_s", "capture_cart_m",
                "precision_cart_m", "angle_rad", "velocity", "rail_limit_m", "fall_angle_rad"):
        value = config[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if (round(config["duration_s"] / DT) < 1
            or not math.isclose(round(config["duration_s"] / DT) * DT, config["duration_s"], abs_tol=1e-10)):
        raise ValueError("duration_s must be a multiple of the control timestep")
    if not 0 < config["precision_cart_m"] <= config["capture_cart_m"] < config["rail_limit_m"]:
        raise ValueError("cart thresholds must satisfy precision <= capture < rail")
    if not config["angle_rad"] < config["fall_angle_rad"] <= math.pi:
        raise ValueError("angle thresholds must satisfy angle < fall <= pi")
    for key in ("mass_scale", "damping_scale", "force_scale", "observation_noise_std"):
        bounds = config["randomization"][key]
        if (len(bounds) != 2 or any(isinstance(v, bool) or not math.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1] or bounds[0] < 0
                or (key != "observation_noise_std" and bounds[0] == 0)):
            raise ValueError(f"invalid randomization range {key}")
    for key in ("force_n", "duration_s", "after_stable_s"):
        value = config["push"][key]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"push.{key} must be finite and nonnegative")
    if config["push"]["duration_s"] <= 0:
        raise ValueError("push.duration_s must be positive")
    if (round(config["push"]["duration_s"] / DT) < 1
            or not math.isclose(round(config["push"]["duration_s"] / DT) * DT,
                                config["push"]["duration_s"], abs_tol=1e-10)):
        raise ValueError("push.duration_s must be a multiple of the control timestep")
    return config


def _hold_steps(seconds):
    return math.ceil(seconds / DT - 1e-10)


def _controller_factory(name, policy_path, case_key):
    # Pushes are applied here, not by the factory ("full_task_" keys disable it).
    from cart_double_pendulum.training.task_local_evaluation import policy_factory

    condition = SimpleNamespace(key=f"full_task_{case_key}")
    return policy_factory(name, policy_path, condition)


def _clean_state(runtime, config):
    q = np.asarray(runtime.qpos, dtype=np.float64)
    v = np.asarray(runtime.qvel, dtype=np.float64)
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise FloatingPointError("non-finite clean simulator state")
    with np.errstate(over="raise", invalid="raise"):
        absolute = np.array([q[1] - math.pi, q[1] + q[2] - math.pi])
        errors = np.arctan2(np.sin(absolute), np.cos(absolute))
        relative_error = np.arctan2(np.sin(q[2]), np.cos(q[2]))
        calm = (np.max(np.abs(errors)) < config["angle_rad"]
                and abs(relative_error) < config["angle_rad"]
                and abs(v[0]) < config["velocity"] * plant_scales().cart_speed_per_rate
                and np.max(np.abs(v[1:])) < config["velocity"]
                and abs(v[1] + v[2]) < config["velocity"])
    return (float(q[0]), bool(calm and abs(q[0]) < config["capture_cart_m"]),
            bool(calm and abs(q[0]) < config["precision_cart_m"]),
            bool(np.max(np.abs(errors)) > config["fall_angle_rad"]))


def _finite_observation(observation):
    observation = np.asarray(observation)
    if not np.isfinite(observation).all():
        raise FloatingPointError("non-finite policy observation")
    return observation


@dataclass
class _Episode:
    capture_success: bool = False
    center_success: bool = False
    success: bool = False
    rail: bool = False
    fall: bool = False
    numerical: bool = False
    handoff: bool = False
    eligible: bool = False
    push_started: bool = False
    pushed: bool = False
    success_time_s: float | None = None
    capture_occupancy: float = 0.0
    center_occupancy: float = 0.0
    longest_capture_hold_s: float = 0.0
    longest_center_hold_s: float = 0.0
    cart_rms_m: float = 0.0
    max_cart_excursion_m: float = 0.0
    force_energy_n2s: float = 0.0
    saturation_fraction: float = 0.0


def _evaluate_episode(runtime, controller_factory, config, *, is_push, push_force):
    result = _Episode()
    total_steps = round(config["duration_s"] / DT)
    capture_required = _hold_steps(config["capture_hold_s"])
    center_required = _hold_steps(config["center_hold_s"])
    stable_required = _hold_steps(config["push"]["after_stable_s"])
    pulse_steps = round(config["push"]["duration_s"] / DT)
    capture_streak = center_streak = stable_streak = 0
    longest_capture = longest_center = capture_samples = center_samples = 0
    cart_square_sum = state_samples = saturated = 0
    capture_time = center_time = None
    push_first = push_end = None
    completed_steps = 0
    try:
        observation = _finite_observation(runtime.reset(mode="hanging"))
        cart, captured, _, _ = _clean_state(runtime, config)
        result.handoff = captured
        result.max_cart_excursion_m = abs(cart)
        result.rail = abs(cart) > config["rail_limit_m"]
        controller = controller_factory(runtime)
        if hasattr(controller, "reset"):
            controller.reset()
        for step in range(total_steps):
            if result.rail:
                break
            pushing = push_first is not None and push_first <= step < push_end
            runtime.data.qfrc_applied[runtime.qvel_indices[0]] = push_force if pushing else 0.0
            try:
                # Inspect the raw output first: clipping Inf would hide failure.
                raw_action = float(controller(observation))
                if not math.isfinite(raw_action):
                    raise FloatingPointError("non-finite raw policy action")
                action = float(np.clip(raw_action, -1.0, 1.0))
                result.push_started |= bool(pushing and push_force)
                next_observation = runtime.step(action)
                completed_steps = step + 1
                if pushing and step + 1 == push_end and push_force:
                    result.pushed = True
                force = action * runtime.traction_force_limit_n
                result.force_energy_n2s += force * force * DT
                saturated += int(abs(action) >= .95)
                observation = _finite_observation(next_observation)
                cart, captured, centered, outside_fall = _clean_state(runtime, config)
            finally:
                runtime.data.qfrc_applied[runtime.qvel_indices[0]] = 0.0
            result.max_cart_excursion_m = max(result.max_cart_excursion_m, abs(cart))
            cart_square_sum += cart * cart
            state_samples += 1
            result.rail = abs(cart) > config["rail_limit_m"]
            # Hanging swing-up is allowed; a fall is armed only on capture entry.
            result.fall = result.handoff and outside_fall
            result.handoff |= captured
            if result.rail or result.fall:
                break
            if is_push and push_first is None:
                stable_streak = stable_streak + 1 if captured else 0
                if captured and stable_streak >= stable_required:
                    result.eligible = True
                    push_first = step + 1
                    push_end = push_first + pulse_steps
            # Exclude both the pre-push interval and the final pulse sample.
            if is_push and (not result.pushed or step < push_end):
                continue
            capture_samples += int(captured)
            center_samples += int(centered)
            capture_streak = capture_streak + 1 if captured else 0
            center_streak = center_streak + 1 if centered else 0
            longest_capture = max(longest_capture, capture_streak)
            longest_center = max(longest_center, center_streak)
            clock_origin = push_end * DT if is_push else 0.0
            if capture_streak >= capture_required and capture_time is None:
                capture_time = (step + 1) * DT - clock_origin
            if center_streak >= center_required and center_time is None:
                center_time = (step + 1) * DT - clock_origin
    except FloatingPointError:
        result.numerical = True
    finally:
        runtime.data.qfrc_applied[runtime.qvel_indices[0]] = 0.0

    valid = (completed_steps == total_steps and not (result.rail or result.fall or result.numerical)
             and (not is_push or result.pushed))
    result.capture_success = valid and capture_time is not None
    result.center_success = valid and center_time is not None
    result.success = result.capture_success and result.center_success
    if result.success:
        result.success_time_s = max(capture_time, center_time)
    # Unobserved samples after a failure remain in the occupancy denominator.
    score_steps = max(0, total_steps - push_end) if is_push and result.pushed else total_steps
    result.capture_occupancy = capture_samples / max(1, score_steps)
    result.center_occupancy = center_samples / max(1, score_steps)
    result.longest_capture_hold_s = longest_capture * DT
    result.longest_center_hold_s = longest_center * DT
    result.cart_rms_m = math.sqrt(cart_square_sum / max(1, state_samples))
    result.saturation_fraction = saturated / total_steps
    return result


def _case_metrics(rows, duration_s):
    def mean(field):
        return float(np.mean([getattr(row, field) for row in rows]))

    times = [row.success_time_s for row in rows if row.success]
    pushed = sum(row.pushed for row in rows)
    recovered = sum(row.pushed and row.success for row in rows)
    return {
        "episodes": len(rows),
        "full_process_success_rate": mean("capture_success"),
        "precision_center_success_rate": mean("center_success"),
        "success_rate": mean("success"),
        "stable_occupancy": mean("capture_occupancy"),
        "capture_occupancy": mean("capture_occupancy"),
        "center_occupancy": mean("center_occupancy"),
        "mean_time_to_success_s": float(np.mean(times)) if times else None,
        "rail_failure_rate": mean("rail"),
        "fall_failure_rate": mean("fall"),
        "numerical_failure_rate": mean("numerical"),
        "handoff_rate": mean("handoff"),
        "never_handoff_rate": 1.0 - mean("handoff"),
        "lost_after_handoff_rate": sum(r.handoff and not r.success for r in rows) / len(rows),
        "mean_cart_rms_m": mean("cart_rms_m"),
        "mean_max_cart_excursion_m": mean("max_cart_excursion_m"),
        "mean_longest_capture_hold_s": mean("longest_capture_hold_s"),
        "mean_longest_center_hold_s": mean("longest_center_hold_s"),
        "mean_force_energy_n2s": mean("force_energy_n2s"),
        "mean_force_energy": mean("force_energy_n2s") / duration_s,
        "action_saturation_fraction": mean("saturation_fraction"),
        "eligible_count": sum(r.eligible for r in rows),
        "push_started_count": sum(r.push_started for r in rows),
        "pushed_count": pushed,
        "recovered_count": recovered,
        "conditional_recovery_rate": recovered / pushed if pushed else None,
    }


def full_gate(metrics):
    """Curriculum gate inputs shared by full_ppo and full_sac: capture holds, then centring."""
    nominal, randomized = metrics["cases"]["nominal"], metrics["cases"]["randomized"]
    return (nominal["full_process_success_rate"], randomized["full_process_success_rate"],
            (nominal["precision_center_success_rate"], randomized["precision_center_success_rate"]))


def evaluate_full_task(name, policy_path, episodes, config_path, seed=None):
    """Full PPO/SAC policy on the shared full-process protocol."""
    if name not in ("full_ppo", "full_sac"):
        raise ValueError("full-task evaluation accepts only full_ppo and full_sac")
    return evaluate_full_protocol(lambda key: _controller_factory(name, policy_path, key),
                                  episodes, config_path, seed)


def evaluate_full_protocol(case_factory, episodes, config_path, seed=None):
    """Shared benchmark for every full-process method: ``case_factory(case_key)`` builds the controller factory.

    ``success_rate`` = worst randomized or signed-push case; ``nominal_gate`` = nominal capture hold.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    config = _load_config(config_path)
    seed = config["seed"] if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases = {}
    robust_rows = []
    # Randomized cases follow the curriculum stage.
    level = curriculum.evaluation_level()
    bounds = {axis: (tuple(level * value for value in config["randomization"][axis]) if axis == "observation_noise_std"
                     else curriculum.scaled_range(config["randomization"][axis], level))
              for axis in ("mass_scale", "damping_scale", "force_scale", "observation_noise_std")}
    for key, offset, sign in (("nominal", 0, 0), ("randomized", 10000, 0),
                              ("push_neg", 20000, -1), ("push_pos", 20000, 1)):
        case_seed = seed + offset
        rng = np.random.default_rng(case_seed)
        factory = case_factory(key)
        rows = []
        for episode in range(episodes):
            scenario = {axis: (float(rng.uniform(*bounds[axis])) if offset else
                               (0.0 if axis == "observation_noise_std" else 1.0))
                        for axis in ("mass_scale", "damping_scale", "force_scale", "observation_noise_std")}
            runtime = CartDoublePendulumRuntime(
                MODEL, seed=case_seed + episode, contract=CONTRACT,
                independent_observation_noise=True, noise_curriculum_level=level, **scenario,
            )
            rows.append(_evaluate_episode(runtime, factory, config, is_push=bool(sign),
                                          push_force=sign * config["push"]["force_n"]))
        cases[key] = _case_metrics(rows, config["duration_s"])
        if key != "nominal":
            robust_rows.extend(rows)
    robust = [cases[key] for key in ("randomized", "push_neg", "push_pos")]
    times = [row.success_time_s for row in robust_rows if row.success]
    return {
        "success_rate": min(case["success_rate"] for case in robust),
        "nominal_gate": cases["nominal"]["full_process_success_rate"] >= .99,
        "nominal_success_rate": cases["nominal"]["success_rate"],
        "randomized_success_rate": cases["randomized"]["success_rate"],
        "stable_occupancy": float(np.mean([case["stable_occupancy"] for case in robust])),
        "mean_time_to_success_s": float(np.mean(times)) if times else config["duration_s"],
        "mean_force_energy": float(np.mean([case["mean_force_energy"] for case in robust])),
        "mean_max_cart_excursion_m": float(np.mean([case["mean_max_cart_excursion_m"] for case in robust])),
        "cases": cases,
        "protocol": PROTOCOL,
    }
