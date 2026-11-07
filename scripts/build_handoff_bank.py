#!/usr/bin/env python3
"""Record a handoff bank (reset states near the swing-to-balance handoff); see docs/BANKS.md.

python scripts/build_handoff_bank.py --policy results/real/expert_swing_sac/runs/<run>/train/policy.onnx
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cart_double_pendulum.common import CONTRACT, MODEL_PATH, angle_error
from cart_double_pendulum.plant_metadata import plant_metadata
from cart_double_pendulum.runtime import CartDoublePendulumRuntime
from cart_double_pendulum.tasks import SWING_UP_TASK
from cart_double_pendulum.plant_scales import plant_scales
from cart_double_pendulum.tasks.mdp import outward_states
from cart_double_pendulum.training.robust_eval import RAIL_FAILURE_M, RobustCondition, scenarios
from cart_double_pendulum.training.task_local_evaluation import policy_factory

DEFAULT_OUTPUT = ROOT / 'data' / 'handoff_bank.npz'


def capture_error(qpos: np.ndarray) -> float:
  """Largest absolute angle error of the two links from upright, in radians."""
  return float(max(abs(angle_error(qpos[1], math.pi, np)), abs(angle_error(qpos[1] + qpos[2], math.pi, np))))


def capture_rate(qvel: np.ndarray) -> float:
  """Largest absolute link rate (link 1 and absolute link 2), in rad/s."""
  return float(max(abs(qvel[1]), abs(qvel[1] + qvel[2])))


def is_handoff(runtime: CartDoublePendulumRuntime) -> bool:
  task = SWING_UP_TASK
  return runtime.is_swing_up_handoff(
    cart_safe_m=task.handoff_cart_safe_m, angle_threshold_rad=task.handoff_angle_rad,
    cart_velocity_threshold_m_s=task.handoff_cart_velocity_m_s,
    angular_velocity_threshold_rad_s=task.handoff_angular_velocity_rad_s)


def record_episode(policy_path, scenario, seed, args):
  runtime = CartDoublePendulumRuntime(MODEL_PATH, seed=seed, contract=CONTRACT,
                                      independent_observation_noise=True, **scenario)
  observation = runtime.reset(mode='hanging').astype(np.float32)
  controller = policy_factory('expert_swing_sac', policy_path, RobustCondition(
    key='bank', task='swing_up', initial_state='hanging', duration_s=args.duration_s, hold_s=0.))(runtime)
  rows, handoff_step = [], None
  steps = round(args.duration_s / CONTRACT.control_timestep_s)
  after = round(args.after_handoff_s / CONTRACT.control_timestep_s)
  for step in range(steps):
    observation = runtime.step(float(np.clip(controller(observation), -1.0, 1.0))).astype(np.float32)
    qpos, qvel = runtime.qpos, runtime.qvel
    if abs(qpos[0]) > RAIL_FAILURE_M or not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
      break
    # Record only states inside the capture window; a pass that leaves it is not a takeover state.
    inside = capture_error(qpos) < args.capture_angle_rad and capture_rate(qvel) < args.capture_rate_rad_s
    handoff = is_handoff(runtime)
    if handoff and handoff_step is None:
      handoff_step = step
    if inside and step % args.stride == 0:
      rows.append((qpos.copy(), qvel.copy(), int(handoff), (step + 1) * CONTRACT.control_timestep_s))
    if handoff_step is not None and step - handoff_step >= after:
      break
  return rows, handoff_step is not None


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--policy', type=Path, required=True, help='swing-up ONNX policy (expert_swing_sac run)')
  parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument('--episodes', type=int, default=512)
  parser.add_argument('--seed', type=int, default=930001)
  parser.add_argument('--duration-s', type=float, default=20.0)
  parser.add_argument('--capture-angle-rad', type=float, default=2.845 * plant_scales().angle,
                      help='record states with both links within this angle of upright')
  parser.add_argument('--capture-rate-rad-s', type=float, default=2.845 * plant_scales().rate,
                      help='and with both link rates below this')
  parser.add_argument('--min-handoff-fraction', type=float, default=0.5,
                      help='refuse the bank unless this fraction of episodes reaches the handoff state')
  parser.add_argument('--after-handoff-s', type=float, default=2.0,
                      help='keep recording this long after the first handoff state')
  parser.add_argument('--stride', type=int, default=2, help='record every n-th control step')
  parser.add_argument('--force', action='store_true', help='overwrite an existing bank')
  args = parser.parse_args()
  if min(args.episodes, args.stride, args.duration_s, args.capture_angle_rad, args.capture_rate_rad_s) <= 0:
    parser.error('episodes, stride, duration and capture window must be positive')
  if not 0 < args.min_handoff_fraction <= 1:
    parser.error('min handoff fraction must lie in (0, 1]')
  if not args.policy.is_file():
    parser.error(f'{args.policy} does not exist')
  output = args.output.resolve()
  if output.exists() and not args.force:
    parser.error(f'{output} exists; pass --force to replace it')

  condition = RobustCondition('bank', 'swing_up', 'hanging', args.duration_s, 0.,
                              ('mass', 'damping', 'force', 'noise'), seed=args.seed)
  qpos, qvel, stage, episode_seed, time_s = [], [], [], [], []
  reached = 0
  for episode, scenario in enumerate(scenarios(condition, args.episodes)):
    rows, handoff = record_episode(args.policy, scenario, args.seed + episode, args)
    reached += int(handoff)
    for q, v, s, t in rows:
      qpos.append(q); qvel.append(v); stage.append(s); episode_seed.append(args.seed + episode); time_s.append(t)
    if (episode + 1) % 64 == 0:
      print(f'BANK episodes={episode + 1} reached_handoff={reached} states={len(qpos)}', flush=True)
  if not qpos or reached < args.min_handoff_fraction * args.episodes:
    sys.exit(f'{reached}/{args.episodes} episodes reached the handoff state (need {args.min_handoff_fraction:.0%}); '
             'train the swing expert further before building a bank')
  qpos, qvel = np.asarray(qpos), np.asarray(qvel)
  summary = {
    'policy': str(args.policy.resolve()),
    'policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
    'model': str(MODEL_PATH), 'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    'plant': CONTRACT.plant_parameters(), 'plant_metadata': plant_metadata(),
    'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    'swing_task': asdict(SWING_UP_TASK),
    'episodes': args.episodes, 'episodes_reaching_handoff': reached,
    'states': len(qpos), 'handoff_states': int(np.sum(stage)), 'outward_states': int(outward_states(qpos, qvel).sum()),
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, plant_metadata=json.dumps(plant_metadata()), qpos=qpos, qvel=qvel, stage=np.asarray(stage, dtype=np.int8),
                      episode_seed=np.asarray(episode_seed, dtype=np.int64), time_s=np.asarray(time_s))
  output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
