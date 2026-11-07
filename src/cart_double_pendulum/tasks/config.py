"""Shared mjlab environment: plant, observations, actions, randomization, terminations.

Rewards and resets come from each task directory (scripts/control_tasks.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortAction, JointEffortActionCfg
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
import mujoco
import torch

from cart_double_pendulum.common import CONTRACT, CartDoublePendulumContract, rail_guard
from cart_double_pendulum.plant_scales import plant_scales

from . import mdp


# Domain-randomization scales; training/robust_eval.py samples the same ranges.
INERTIA_SCALE_RANGE = (0.85, 1.15)
DAMPING_SCALE_RANGE = (0.75, 1.25)
FORCE_SCALE_RANGE = (0.90, 1.10)
# Defaults below are for the CLI's default XML; explicit contracts get their own task thresholds.
PUSH_FORCE_N = 0.25 * plant_scales(CONTRACT).force
BALANCE_FALL_RAD = 3.319 * plant_scales(CONTRACT).angle


@dataclass(kw_only=True)
class CartEnvironmentCfg(ManagerBasedRlEnvCfg):
  plant_contract: CartDoublePendulumContract = CONTRACT


class RailGuardAction(JointEffortAction):
  """Cart force limited by the rail guard each control step; replay keeps the policy's own action."""

  def process_actions(self, actions):
    super().process_actions(actions)
    q, v = mdp.state(self._env)
    commanded = self._raw_actions.clamp(-1., 1.)
    guarded = rail_guard(commanded[:, 0], q[:, 0], v[:, 0], mdp.env_contract(self._env), torch)[:, None]
    self.guard_active = (guarded != commanded)[:, 0]
    self._processed_actions = guarded * self._scale + self._offset


@dataclass(kw_only=True)
class RailGuardActionCfg(JointEffortActionCfg):
  def build(self, env):
    return RailGuardAction(self, env)


@dataclass(frozen=True)
class CartDoublePendulumTaskCfg:
  """Success thresholds shared by training and evaluation, resolved for one plant."""
  phase: str
  episode_length_s: float
  handoff_cart_safe_m: float | None
  handoff_angle_rad: float | None
  handoff_cart_velocity_m_s: float | None
  handoff_angular_velocity_rad_s: float | None
  stable_cart_m: float
  stable_angle_rad: float
  stable_velocity: float
  rail_failure_m: float


def task_thresholds(phase, contract=CONTRACT):
  scales = plant_scales(contract)
  handoff = dict(handoff_cart_safe_m=.65 * scales.rail, handoff_angle_rad=1.138 * scales.angle,
                 handoff_cart_velocity_m_s=.418 * scales.cart_speed,
                 handoff_angular_velocity_rad_s=.5285 * scales.rate)
  if phase == 'balancing':
    handoff = dict.fromkeys(handoff)
  swing = phase == 'swing_up'
  return CartDoublePendulumTaskCfg(
    phase=phase, episode_length_s=25. if phase != 'balancing' else 20., **handoff,
    stable_cart_m=(.16 if swing else .12) * scales.rail,
    stable_angle_rad=(.6638 if swing else .569) * scales.angle,
    stable_velocity=(.2114 if swing else .1691) * scales.rate,
    rail_failure_m=.95 * scales.rail)


SWING_UP_TASK = task_thresholds('swing_up')
BALANCING_TASK = task_thresholds('balancing')


def mujoco_options(contract):
  """Copy every option mjlab would otherwise override from its own defaults."""
  opt = mujoco.MjModel.from_xml_path(contract.model_path).opt
  return MujocoCfg(
    timestep=float(opt.timestep), gravity=tuple(float(x) for x in opt.gravity),
    integrator={0: 'euler', 3: 'implicitfast'}[int(opt.integrator)],
    solver={0: 'pgs', 1: 'cg', 2: 'newton'}[int(opt.solver)],
    jacobian={0: 'dense', 1: 'sparse', 2: 'auto'}[int(opt.jacobian)],
    cone={0: 'pyramidal', 1: 'elliptic'}[int(opt.cone)],
    impratio=float(opt.impratio), iterations=int(opt.iterations), tolerance=float(opt.tolerance),
    ls_iterations=int(opt.ls_iterations), ls_tolerance=float(opt.ls_tolerance),
    ccd_iterations=int(opt.ccd_iterations),
    multiccd=bool(opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD))


def _model_spec(model_path: Path) -> mujoco.MjSpec:
  # mjlab adds the actuator itself; drop the MJCF actuator and keyframes.
  spec = mujoco.MjSpec.from_file(str(model_path))
  for key in list(spec.keys):
    spec.delete(key)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


def _reset_from_task_directory(env, env_ids=None, **_) -> None:
  raise RuntimeError("reset_system must be installed from the task's environment.json")


def make_env_cfg(
  *,
  num_envs: int,
  task_cfg: CartDoublePendulumTaskCfg,
  contract: CartDoublePendulumContract = CONTRACT,
  robust_training: bool = False,
  benchmark_randomization: bool = False,
  actor_noise_max_std: float = 0.02,
  actor_history_frames: int = 0,
  actor_noise_per_episode: bool = False,
  actor_observer_checkpoint: str | None = None,
  cart_impulses: bool = True,
) -> ManagerBasedRlEnvCfg:
  """``robust_training``: actor sensor noise and cart pushes.  ``benchmark_randomization``:
  mass, damping and force scales per reset.  ``actor_history_frames``: stacked 9-D actor
  frames; the critic always sees the clean 8-D state.
  """
  if num_envs <= 0:
    raise ValueError("num_envs must be positive")
  if actor_observer_checkpoint is not None and actor_history_frames < 2:
    raise ValueError("an actor observer requires native history frames")
  if not isinstance(actor_history_frames, int) or isinstance(actor_history_frames, bool) or actor_history_frames < 0:
    raise ValueError("actor_history_frames must be a non-negative integer")
  if actor_noise_max_std < 0.0:
    raise ValueError("actor observation-noise standard deviation must be non-negative")
  scales = plant_scales(contract)
  model_path = Path(contract.model_path)
  system = EntityCfg(
    spec_fn=lambda: _model_spec(model_path),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=(contract.actuated_joint,),
          effort_limit=contract.actuator_force_limit_n,
          armature=contract.cart_armature_kg,
          frictionloss=contract.cart_frictionloss_n,  # mjlab would otherwise reset the MJCF value to 0
        ),
      )
    ),
  )
  observations = {}
  for name in ("actor", "critic"):
    # Asymmetric actor-critic: only the actor sees sensor noise.
    noisy_actor = robust_training and name == "actor" and actor_noise_max_std > 0.0
    observations[name] = ObservationGroupCfg(
      {
        "state": ObservationTermCfg(
          func=mdp.noisy_policy_state if noisy_actor else mdp.policy_state_with_contract,
          params=(
            {"max_noise_std": actor_noise_max_std, "contract": contract}
            if noisy_actor else {"contract": contract}
          ),
        )
      },
      enable_corruption=False,
    )
    term = observations[name].terms["state"]
    if noisy_actor and actor_noise_per_episode:
      term.params["noise_per_episode"] = True
    if name == "actor" and actor_history_frames > 0:
      term.func = mdp.actor_history_frame
      term.params = {"contract": contract, "max_noise_std": actor_noise_max_std if noisy_actor else 0.0}
      if actor_noise_per_episode:
        term.params["noise_per_episode"] = True
      if actor_observer_checkpoint is not None:
        term.params["observer_checkpoint"] = actor_observer_checkpoint
      term.history_length = actor_history_frames
      term.flatten_history_dim = True
  actions = {
    contract.action_names[0]: RailGuardActionCfg(
      entity_name="cart_double_pendulum",
      actuator_names=(contract.actuated_joint,),
      scale=contract.actuator_force_limit_n,
    )
  }
  events = {
    "reset_system": EventTermCfg(func=_reset_from_task_directory, mode="reset", params={}),
  }
  if actor_observer_checkpoint is not None:
    from cart_double_pendulum.training.ppo_observer_adapter import reset_actor_observer
    events["reset_actor_observer"] = EventTermCfg(func=reset_actor_observer, mode="reset")
  if actor_noise_per_episode and robust_training:
    events["reset_actor_noise_std"] = EventTermCfg(
      func=mdp.reset_actor_noise_std,
      mode="reset",
      params={"max_noise_std": actor_noise_max_std},
    )
  if benchmark_randomization:
    for name, func, scale_range in (
      ("randomize_inertia", mdp.randomize_benchmark_density, INERTIA_SCALE_RANGE),
      ("randomize_damping", mdp.randomize_benchmark_damping, DAMPING_SCALE_RANGE),
      ("randomize_actuation", mdp.randomize_benchmark_actuator_force, FORCE_SCALE_RANGE),
    ):
      events[name] = EventTermCfg(func=func, mode="reset", params={"scale_range": scale_range})
  if robust_training and cart_impulses:
    events["robust_cart_impulse"] = EventTermCfg(
      func=mdp.CurriculumImpulse,
      mode="step",
      params={
        "force_range": (-.25 * scales.force, .25 * scales.force),
        "torque_range": (0.0, 0.0),
        "duration_s": (0.10, 0.10),
        "cooldown_s": (4.0, 8.0),
        "asset_cfg": SceneEntityCfg("cart_double_pendulum", body_names=("cart",)),
      },
    )
  terminations = {
    "time_limit": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "rail_failure": TerminationTermCfg(
      func=mdp.rail_failure,
      params={"rail_limit_m": task_cfg.rail_failure_m},
    ),
    "numerical_failure": TerminationTermCfg(func=mdp.numerical_failure),
  }
  if task_cfg.phase == "balancing":
    terminations["balancing_fall"] = TerminationTermCfg(
      func=mdp.balancing_fall,
      params={"angle_limit_rad": 3.319 * scales.angle},
    )
  return CartEnvironmentCfg(
    plant_contract=contract,
    decimation=contract.decimation,
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=5.0,
      entities={"cart_double_pendulum": system},
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards={},
    terminations=terminations,
    sim=SimulationCfg(
      mujoco=mujoco_options(contract)
    ),
    viewer=ViewerConfig(lookat=(0.0, -0.06, 0.08), distance=0.9, elevation=-8.0),
    episode_length_s=task_cfg.episode_length_s,
    is_finite_horizon=True,
    scale_rewards_by_dt=False,
  )


@dataclass(frozen=True)
class Task:
  name: str
  config: CartDoublePendulumTaskCfg
  contract: CartDoublePendulumContract = CONTRACT

  def make_env_cfg(self, num_envs: int = 512, **options) -> ManagerBasedRlEnvCfg:
    return make_env_cfg(num_envs=num_envs, task_cfg=self.config, contract=self.contract, **options)


_TASKS = {"swing_up": SWING_UP_TASK, "balancing": BALANCING_TASK}


def list_tasks() -> tuple[str, ...]:
  return tuple(_TASKS)


def make_task(name: str, contract=CONTRACT) -> Task:
  try:
    return Task(name, task_thresholds(_TASKS[name].phase, contract), contract)
  except KeyError as error:
    raise ValueError(f"unknown task {name!r}; choose from {list_tasks()}") from error
