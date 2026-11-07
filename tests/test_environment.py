"""Build and step real mjlab environments through the launcher configuration."""
import json
import math

import mjlab.envs
import pytest
import torch

from conftest import ROOT
from cart_double_pendulum.tasks import config, make_task, mdp
from cart_double_pendulum.training import task_directory


def task_env(name, bank, **kwargs):
  """Environment with the task directory's reward and reset, like control_tasks.train."""
  folder = ROOT / "tasks/real" / name
  environment = task_directory.read_json(folder / "environment.json")
  phase = "balancing" if name == "expert_balance_ppo" else "swing_up"
  cfg = make_task(phase).make_env_cfg(2, **kwargs)
  reset = environment["reset_system"]
  cfg.events["reset_system"].func = task_directory.function(reset["function"])
  cfg.events["reset_system"].params = {**reset["params"], "bank_path": bank}
  cfg.rewards, cfg.scale_rewards_by_dt = task_directory.reward_terms(folder)
  cfg.episode_length_s = environment["episode_length_s"]
  cfg.seed = 0
  return mjlab.envs.ManagerBasedRlEnv(cfg=cfg, device="cpu")


def run(env, steps=10):
  observations, _ = env.reset(seed=0)
  for step in range(steps):
    action = torch.full((env.num_envs, 1), math.sin(step))
    observations, reward, *_ = env.step(action)
    assert torch.isfinite(observations["actor"]).all() and torch.isfinite(reward).all()
  env.close()


def test_simulation_uses_mjcf_gravity_and_timestep():
  cfg = make_task("swing_up").make_env_cfg(2)
  assert cfg.sim.mujoco.gravity == (0.0, 0.0, -config.CONTRACT.gravity_m_s2)
  assert cfg.sim.mujoco.timestep == config.CONTRACT.physics_timestep_s


def test_swing_expert_trains_without_a_bank():
  run(task_env("expert_swing_sac", None))


def test_swing_expert_uses_a_bank(synthetic_bank):
  run(task_env("expert_swing_sac", str(synthetic_bank)))


def test_balance_expert_requires_a_bank():
  with pytest.raises(ValueError, match="handoff bank"):
    run(task_env("expert_balance_ppo", None, actor_history_frames=20))


def test_balance_expert_with_bank(synthetic_bank):
  run(task_env("expert_balance_ppo", str(synthetic_bank), actor_history_frames=20, robust_training=True,
               benchmark_randomization=True, actor_noise_per_episode=True))


def test_invalid_bank_is_rejected(tmp_path):
  import numpy as np
  path = tmp_path / "bad.npz"
  np.savez(path, qpos=np.zeros((4, 2)), qvel=np.zeros((4, 2)))
  with pytest.raises(ValueError, match="N x 3"):
    mdp.load_state_bank(str(path), "cpu")


def test_reset_velocities_follow_the_plant_scales():
  """Shared reset velocity ranges are rad/s for the links; the cart part follows the plant scales."""
  import numpy as np
  from cart_double_pendulum.common import MODEL_PATH
  from cart_double_pendulum.plant_scales import plant_scales
  from cart_double_pendulum.runtime import CartDoublePendulumRuntime
  scales = plant_scales()
  runtime = CartDoublePendulumRuntime(MODEL_PATH, seed=0)
  speeds = []
  for _ in range(200):
    runtime.reset(mode="hanging")
    speeds.append(np.abs(runtime.qvel[:2]))
  speeds = np.array(speeds)
  assert speeds[:, 0].max() <= .148 * scales.rate * scales.cart_speed_per_rate
  assert speeds[:, 1].max() <= .148 * scales.rate < 1.1 * speeds[:, 1].max()
  env = task_env("expert_swing_sac", None)
  env.reset(seed=0)
  assert env.scene[mdp.ASSET].data.joint_vel[:, 0].abs().max() <= .338 * scales.rate * scales.cart_speed_per_rate
  env.close()


def test_reward_term_log_adds_up_to_the_reward():
  from cart_double_pendulum.training.reward_log import RewardTermLog

  class Writer:
    def __init__(self):
      self.values = {}

    def add_scalar(self, name, value, step):
      self.values[name] = value

  env = task_env("expert_swing_sac", None)
  log = RewardTermLog(env)
  env.reset(seed=0)
  total = 0.0
  for step in range(10):
    _, reward, *_ = env.step(torch.full((env.num_envs, 1), math.sin(step)))
    total += float(reward.sum())
  writer = Writer()
  log.write(writer, 10)
  assert set(writer.values) == {f"reward_terms/{name}" for name in env.reward_manager.active_terms}
  assert sum(writer.values.values()) == pytest.approx(total / (10 * env.num_envs), rel=1e-5)
  env.close()


def test_pumping_to_upright_energy_beats_idling_and_spinning_without_spikes():
  plant = config.CONTRACT.plant_parameters()
  env = task_env("expert_swing_sac", None)
  mean = {}
  for target in (0.0, 1.0, 2.2):
    env.reset(seed=0)
    env.scene[mdp.ASSET].write_joint_state_to_sim(torch.tensor([[0., .02, 0.]] * 2), torch.zeros(2, 3),
                                                  joint_ids=slice(0, 3))
    rewards = []
    for _ in range(400):
      q, v = mdp.state(env)
      fraction = mdp.pendulum_energy_fraction(env, **plant)
      # Energy pump towards the target fraction with weak cart centring; target 0 means no force.
      pump = (target - fraction).clamp(-1, 1) * torch.sign(v[:, 1] * torch.cos(q[:, 1]) + 1e-6)
      action = pump - .45 * q[:, 0] / config.CONTRACT.rail_limit_m - .1 * v[:, 0] if target else 0 * pump
      _, reward, terminated, _, _ = env.step(action[:, None].clamp(-1, 1))
      assert not terminated.any()
      rewards.append(reward.clone())
    rewards = torch.stack(rewards)
    assert rewards.abs().max() < 60.0
    mean[target] = float(rewards.mean())
  env.close()
  assert mean[1.0] > mean[2.2] and mean[1.0] > mean[0.0] + 0.5, mean


def test_reward_discounts_come_from_the_trainer(tmp_path):
  from cart_double_pendulum.rl.sac import DEFAULT_DISCOUNT

  for folder in sorted((ROOT / "tasks/real").iterdir()):
    gamma = task_directory.read_json(folder / "training.json").get("gamma", DEFAULT_DISCOUNT)
    terms, _ = task_directory.reward_terms(folder)
    assert {t.params["discount"] for t in terms.values() if "discount" in t.params} == {gamma}, folder.name
  folder = tmp_path / "task"
  folder.mkdir()
  (folder / "reward.json").write_text(json.dumps({"scale_rewards_by_dt": False, "terms": {"rail": {
    "function": "rail_terminal_cost", "weight": -25.0, "params": {"rail_limit_m": {"rail": 0.95}, "discount": 0.99}}}}))
  with pytest.raises(ValueError, match="trainer's gamma"):
    task_directory.reward_terms(folder)


def test_failure_costs_more_than_any_cost_it_could_escape():
  import mujoco
  from cart_double_pendulum.common import CONTRACT, MODEL_PATH

  x_max = config.SWING_UP_TASK.rail_failure_m
  v_max = CONTRACT.actuator_force_limit_n / mujoco.MjModel.from_xml_path(str(MODEL_PATH)).dof_damping[0]
  peak = {"tip_fast_top_cost": lambda p: 1.0, "rail_safety_barrier": lambda p: 1.0, "swing_excess_energy_cost": lambda p: 1.0,
          "swing_time_to_handoff_cost": lambda p: 1.0, "force_l2": lambda p: 1.0, "action_rate_l2": lambda p: 4.0,
          "handoff_deficit": lambda p: 1.0,
          "full_fall_penalty": lambda p: 0.0,  # once per capture, checked below
          "cart_center_cost": lambda p: 1.0,
          "cart_center_quadratic": lambda p: (x_max / p.get("reference_m", .5 * CONTRACT.rail_limit_m)) ** 2,
          "full_center_penalty": lambda p: 1.0,
          "cart_velocity_smooth_l1": lambda p: min(2 * (math.sqrt(1 + (v_max / p["reference_m_s"]) ** 2) - 1),
                                                   p.get("max_cost", 8.0)),
          "link1_velocity_cost": lambda p: p.get("max_cost", 1.0),
          "link2_absolute_velocity_cost": lambda p: p.get("max_cost", 1.0)}
  terminal = ("rail_terminal_cost", "balancing_fall_cost")
  for folder in sorted((ROOT / "tasks/real").iterdir()):
    terms, _ = task_directory.reward_terms(folder)
    costs = [t for t in terms.values() if t.weight < 0 and getattr(t.func, "__name__", "") not in terminal]
    escapable = sum(-t.weight * peak[t.func.__name__](t.params) for t in costs)
    failures = [t for t in terms.values() if getattr(t.func, "__name__", "") in terminal]
    assert failures and all(-t.weight >= 1.2 * escapable for t in failures), (folder.name, escapable)
    # A one-time fall cost must stay below a rail failure, so crashing never dodges it.
    once = [-t.weight for t in terms.values() if getattr(t.func, "__name__", "") == "full_fall_penalty"]
    assert all(cost < min(-t.weight / (1 - t.params["discount"]) for t in failures) for cost in once)


def test_success_anywhere_on_the_rail_beats_failure():
  """Upright and still beats hanging still at every cart position and schedule point."""
  for name in ("expert_swing_sac", "full_ppo", "full_sac"):
    folder = ROOT / "tasks/real" / name
    cfg = make_task("swing_up").make_env_cfg(1)
    cfg.rewards, cfg.scale_rewards_by_dt = task_directory.reward_terms(folder)
    cfg.events = {"reset_system": cfg.events["reset_system"]}
    cfg.events["reset_system"].func, cfg.events["reset_system"].params = mdp.reset_expert_swing, {}
    env = mjlab.envs.ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset(seed=0)
    env._full_budget_control_steps = 100
    schedule = (folder / "curriculum.json")
    fractions = [point["fraction"] for point in json.loads(schedule.read_text())["schedule"]] if schedule.is_file() else [1.0]
    for fraction in fractions:
      env.common_step_counter = int(100 * fraction)
      for x in torch.linspace(-0.94, 0.94, 9).tolist():
        rewards = []
        for theta in (math.pi, 0.0):
          env.scene[mdp.ASSET].write_joint_state_to_sim(torch.tensor([[x * config.CONTRACT.rail_limit_m, theta, 0.0]]),
                                                        torch.zeros(1, 3), joint_ids=slice(0, 3))
          env.termination_manager.compute()
          env.reward_manager.compute(dt=env.step_dt)
          rewards.append(float(env.reward_manager._step_reward[0].sum()))
        assert rewards[0] > rewards[1], (name, fraction, x, rewards)
    env.close()
