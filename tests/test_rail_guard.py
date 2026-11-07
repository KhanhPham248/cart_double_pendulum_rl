"""Rail guard: no rail failure under any command, identical in the CPU runtime and the mjlab env."""
import numpy as np
import pytest
import torch

from cart_double_pendulum.common import CONTRACT, MODEL_PATH, RAIL_GUARD_FRACTION, rail_guard
from cart_double_pendulum.runtime import CartDoublePendulumRuntime
from cart_double_pendulum.tasks import mdp
from test_environment import task_env

RAIL = CONTRACT.rail_limit_m


def test_guard_leaves_the_centre_free_and_brakes_at_the_ends():
  assert rail_guard(1.0, 0.0, CONTRACT.cart_velocity_scale) == 1.0
  assert rail_guard(1.0, RAIL_GUARD_FRACTION * RAIL, 0.5) == -1.0
  assert rail_guard(-1.0, -RAIL_GUARD_FRACTION * RAIL, -0.5) == 1.0


def test_numpy_and_torch_guards_agree():
  x, v = np.meshgrid(np.linspace(-RAIL, RAIL, 31), np.linspace(-1.2, 1.2, 31))
  action = np.random.default_rng(0).uniform(-1, 1, x.shape)
  expected = rail_guard(action, x, v)
  actual = rail_guard(*(torch.as_tensor(value) for value in (action, x, v)), CONTRACT, torch)
  assert np.allclose(actual.numpy(), expected)


def test_worst_randomized_plant_never_reaches_the_rail_failure():
  worst = 0.0
  for seed in range(24):
    for mass, damping, force in ((1.15, .75, .9), (.85, .75, 1.1), (1.15, 1.25, .9)):
      runtime = CartDoublePendulumRuntime(MODEL_PATH, mass_scale=mass, damping_scale=damping,
                                          force_scale=force, seed=seed)
      runtime.reset(mode="hanging")
      rng = np.random.default_rng(seed)
      # Bang-bang commands with random switching: the hardest case for the guard.
      for step in range(300):
        if step % 5 == 0:
          command = float(rng.choice((-1.0, 1.0)))
        runtime.step(command)
        if np.abs(runtime.qvel[1:]).max() > 60.0:
          break
        worst = max(worst, abs(runtime.qpos[0]))
  assert worst < 0.95 * RAIL


def test_environment_applies_the_guard():
  env = task_env("expert_swing_sac", None)
  env.reset(seed=0)
  term = env.action_manager.get_term(CONTRACT.action_names[0])
  worst = 0.0
  for _ in range(150):
    *_, terminated, _, _ = env.step(torch.ones((2, 1)))
    worst = max(worst, float(mdp.state(env)[0][:, 0].abs().max()))
    assert not terminated.any()
  assert term.guard_active.any() and worst < 0.95 * RAIL
  # The policy's own action stays visible to observations and replay.
  assert torch.equal(env.action_manager.action, torch.ones((2, 1)))
  env.close()


def test_diverged_simulation_terminates_and_costs_like_a_rail_crash():
  from types import SimpleNamespace
  q, v = torch.zeros(3, 3), torch.zeros(3, 3)
  v[1, 1] = 3.0 * CONTRACT.joint_velocity_scale  # a fast but physical spin
  v[2, 2] = 5.0 * CONTRACT.joint_velocity_scale  # a diverged integrator
  env = SimpleNamespace(scene={mdp.ASSET: SimpleNamespace(data=SimpleNamespace(joint_pos=q, joint_vel=v))})
  assert mdp.numerical_failure(env).tolist() == [False, False, True]
  cost = mdp.rail_terminal_cost(env, rail_limit_m=.95 * RAIL, discount=.995)
  assert cost.tolist() == [0., 0., pytest.approx(200.)]


def test_full_fall_is_charged_once_per_capture_like_the_evaluator():
  import math
  from types import SimpleNamespace
  from cart_double_pendulum.tasks import mdp
  q, v = torch.zeros(2, 3), torch.zeros(2, 3)
  env = SimpleNamespace(num_envs=2, device='cpu', scene={mdp.ASSET: SimpleNamespace(data=SimpleNamespace(joint_pos=q, joint_vel=v))})
  fall = mdp.full_fall_penalty(None, env)
  params = dict(capture_cart_m=.025, angle_rad=.08, velocity=.27, fall_angle_rad=.47)
  q[:, 1] = math.pi
  q[1, 0] = .1  # upright but outside the capture box: never armed
  assert fall(env, **params).sum() == 0
  q[:, 1] = math.pi + .6  # both fall; only the captured one pays, once
  assert fall(env, **params).tolist() == [1., 0.]
  assert fall(env, **params).sum() == 0
  q[0, 1] = math.pi  # captured again: armed again
  fall(env, **params)
  fall.reset(torch.tensor([0]))
  q[0, 1] = math.pi + .6
  assert fall(env, **params).sum() == 0
