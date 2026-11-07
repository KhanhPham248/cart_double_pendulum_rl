#!/usr/bin/env python3
"""Train one cart-double-pendulum policy with SAC. Started by tasks/real/<name>/run.py train."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from cart_double_pendulum.training.sac_env import SACEnvironment as ManagerBasedRlEnv, replay_observations
from cart_double_pendulum.plant_metadata import plant_metadata

from cart_double_pendulum import curriculum
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.rl import ReplayBuffer
from cart_double_pendulum.rl.sac import SACAgent, SACConfig
from cart_double_pendulum.rl.sac import export_deterministic_onnx
from cart_double_pendulum.tasks import list_tasks, make_task
from cart_double_pendulum.tasks.mdp import state
from cart_double_pendulum.training.checkpoint_evaluation import evaluate_checkpoint
from cart_double_pendulum.training.reward_log import RewardTermLog


def _dump(path: Path, values: dict) -> None:
  path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _save(
  run_dir: Path,
  agent: SACAgent,
  transitions: int,
  task_name: str,
  train_args: dict,
  metrics: dict[str, float],
  best: bool,
) -> Path:
  checkpoint_dir = run_dir / "checkpoints"
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  checkpoint = checkpoint_dir / f"step_{transitions:012d}.pt"
  payload = {
    "format": "cart_double_pendulum_sac_v1",
    "plant_metadata": plant_metadata(getattr(agent, "plant_contract", CONTRACT)),
    "task": task_name,
    "algorithm": "sac",
    "observation_dim": agent.obs_dim,
    "action_dim": agent.action_dim,
    "transitions": transitions,
    "train_args": train_args,
    "evaluation": metrics,
    "agent": agent.state_dict(),
  }
  torch.save(payload, checkpoint)
  # policy.onnx follows best.pt; policy_latest.onnx follows latest.pt.
  export_deterministic_onnx(agent, str(run_dir / "policy_latest.onnx"))
  shutil.copy2(checkpoint, checkpoint_dir / "latest.pt")
  (run_dir / "evaluation_latest.json").write_text(
    json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
  )
  if best:
    shutil.copy2(checkpoint, checkpoint_dir / "best.pt")
    export_deterministic_onnx(agent, str(run_dir / "policy.onnx"))
    (run_dir / "evaluation_best.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
  return checkpoint


def _selection_key(metrics: dict) -> tuple:
  """The task's own ranking of the fixed full-level benchmark."""
  if 'full_task_selection_key' in metrics:
    key = tuple(metrics['full_task_selection_key'])
  elif 'expert_progress_selection_key' in metrics:
    key = tuple(metrics['expert_progress_selection_key'])
  else:
    key = (float(metrics.get('nominal_gate', True)), metrics['success_rate'], metrics['stable_occupancy'],
           -metrics['mean_time_to_success_s'], -metrics['mean_force_energy'])
  return key


def train(args: argparse.Namespace) -> Path:
  if not (args.reward_scale > 0.0 and args.reward_scale < float('inf')):
    raise ValueError('reward scale must be finite and positive')
  if (args.warmup_hold_steps < 1 or any(not math.isfinite(v) or v < 0 for v in
      (args.warmup_centering_kp, args.warmup_centering_kd))):
    raise ValueError('Warmup requires a positive hold and finite nonnegative centering gains')
  progress_guard = None
  if args.stagnation_after_transitions:
    from cart_double_pendulum.training.exploration import SwingProgressGuard
    progress_guard = SwingProgressGuard(args.stagnation_after_transitions, args.stagnation_patience,
                                       args.stagnation_minimum_height)
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but PyTorch cannot see a CUDA device")
  random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
  task = make_task(args.task)
  run_dir = Path(args.run_dir).resolve()
  run_dir.mkdir(parents=True, exist_ok=True)
  _dump(run_dir / "train_config.yaml", vars(args))
  env_cfg=task.make_env_cfg(
      args.num_envs,
      robust_training=args.robust_training,
      benchmark_randomization=args.benchmark_randomization,
      actor_noise_per_episode=args.actor_noise_per_episode,
      actor_noise_max_std=args.actor_noise_max_std,
      cart_impulses=not args.disable_cart_impulses,
    )
  env_cfg.seed=args.seed
  env = ManagerBasedRlEnv(cfg=env_cfg,device=args.device)
  reward_log = RewardTermLog(env)
  observations, _ = env.reset(seed=args.seed)
  obs = observations["actor"]
  asymmetric = args.robust_training or observations['critic'].shape[-1] != obs.shape[-1]
  critic_obs = observations['critic'] if asymmetric else None
  agent = SACAgent(
    obs_dim=obs.shape[-1],
    action_dim=env.action_manager.total_action_dim,
    device=args.device,
    config=SACConfig(batch_size=args.batch_size, learning_rate=args.learning_rate,
                     target_entropy=args.target_entropy),
    critic_obs_dim=observations['critic'].shape[-1],
  )
  agent.plant_contract = env_cfg.plant_contract
  resume = None
  if args.resume_checkpoint:
    from cart_double_pendulum.training.sac_resume import restore
    resume = restore(agent, args.resume_checkpoint, args.task, env_cfg.plant_contract)
    resume['additional_transitions'] = args.total_transitions
    (run_dir / 'resume.json').write_text(json.dumps(resume, indent=2))
    print('SAC_RESUME', json.dumps({key: value for key, value in resume.items() if key != 'source_launch'}), flush=True)
  replay = ReplayBuffer(args.replay_capacity, obs.shape[-1], 1, args.device,
                        asymmetric_critic=asymmetric, critic_obs_dim=observations['critic'].shape[-1])
  agent.observe(obs, critic_obs)
  warmup = None
  if args.warmup_hold_steps > 1:
    from cart_double_pendulum.training.exploration import CenteredWarmup
    warmup = CenteredWarmup(args.num_envs, args.device, args.warmup_hold_steps,
                           kp=args.warmup_centering_kp, kd=args.warmup_centering_kd)
  from cart_double_pendulum.training.exploration import CorrelatedNoise, NOISE_CORRELATION_S
  exploration = CorrelatedNoise(args.num_envs, agent.action_dim, args.device,
                                correlation_s=NOISE_CORRELATION_S, step_dt=env.step_dt)
  print(f'SAC_EXPLORATION warmup_seconds_per_env={args.learning_starts / args.num_envs * env.step_dt:.3f} '
        f'hold_steps={args.warmup_hold_steps} target_entropy={agent.target_entropy} '
        f'noise_correlation_s={NOISE_CORRELATION_S}', flush=True)
  writer = SummaryWriter(run_dir / "tensorboard")
  start_transitions = resume['source_transitions'] if resume else 0
  transitions = start_transitions
  phase_transitions = 0
  args.next_checkpoint = start_transitions + args.checkpoint_interval
  collects = 0
  def evaluate_current():
    export_deterministic_onnx(agent, str(run_dir / 'policy_candidate.onnx'))
    return evaluate_checkpoint(run_dir / 'policy_candidate.onnx', args.task, args.evaluation_episodes)
  # Rank the initial/restored policy under the current objective, not its historical score.
  initial_metrics = evaluate_current()
  best_key = _selection_key(initial_metrics)
  _save(run_dir, agent, transitions, args.task, vars(args), initial_metrics, True)
  total_collects = (args.total_transitions + args.num_envs - 1) // args.num_envs
  try:
    for collect in range(1, total_collects + 1):
      if not resume and phase_transitions < args.learning_starts:
        if warmup is None:
          actions = torch.empty((args.num_envs, 1), device=agent.device).uniform_(-1., 1.)
        else:
          q, v = state(env)
          actions = warmup.sample(q[:, 0] / agent.plant_contract.rail_limit_m,
                                  v[:, 0] / agent.plant_contract.cart_velocity_scale)
      else:
        actions = agent.act(obs, noise=exploration.sample())
      next_observations, rewards, terminated, truncated, extras = env.step(actions)
      successor = replay_observations(next_observations, terminated, truncated, extras)
      next_obs = next_observations["actor"]
      next_critic_obs = next_observations['critic'] if asymmetric else None
      replay.add(obs, actions, rewards * args.reward_scale, successor["actor"], terminated,
                 critic_obs, successor["critic"] if asymmetric else None)
      if warmup is not None:
        warmup.reset(terminated | truncated)
      exploration.reset(terminated | truncated)
      agent.observe(next_obs, next_critic_obs)
      obs = next_obs
      critic_obs = next_critic_obs
      transitions += args.num_envs
      phase_transitions += args.num_envs
      collects += 1
      if phase_transitions >= args.learning_starts and len(replay) >= args.batch_size:
        updates = max(1, round(args.num_envs * args.utd_ratio))
        update_values: dict[str, float] = {}
        for _ in range(updates):
          for key, value in agent.update(replay.sample(args.batch_size)).items():
            update_values[key] = update_values.get(key, 0.0) + value / updates
        for key, value in update_values.items():
          writer.add_scalar(key, value, transitions)
      if collect % args.log_interval == 0:
        writer.add_scalar("train/mean_reward", float(rewards.mean()), transitions)
        writer.add_scalar('train/scaled_reward_mean', float(rewards.mean()) * args.reward_scale, transitions)
        writer.add_scalar('train/action_saturation', float((actions.abs() >= .95).float().mean()), transitions)
        writer.add_scalar("train/replay_size", len(replay), transitions)
        guard = env.action_manager.get_term(agent.plant_contract.action_names[0]).guard_active
        writer.add_scalar('train/rail_guard_fraction', float(guard.float().mean()), transitions)
        reward_log.write(writer, transitions)
      if transitions >= args.next_checkpoint:
        metrics = evaluate_current()
        writer.add_scalar("eval/success_rate", metrics["success_rate"], transitions)
        writer.add_scalar("eval/stable_occupancy", metrics["stable_occupancy"], transitions)
        writer.add_scalar("eval/mean_force_energy", metrics["mean_force_energy"], transitions)
        key = _selection_key(metrics)
        best = key > best_key
        if best:
          best_key = key
        checkpoint = _save(run_dir, agent, transitions, args.task, vars(args), metrics, best)
        print(f"SAC_CHECKPOINT {checkpoint} {json.dumps(metrics, sort_keys=True)}", flush=True)
        if progress_guard is not None and progress_guard.stalled(metrics, transitions):
          reason = (f'Swing learning stagnated: {progress_guard.bad_evaluations} evaluations with no handoff '
                    f'and peak height below {progress_guard.minimum_height:.0%} after {transitions} transitions')
          _dump(run_dir / 'learning_stagnation.yaml', {'reason': reason, 'checkpoint': str(checkpoint), 'metrics': metrics})
          raise RuntimeError(reason)
        restore = curriculum.skill_checkpoint(metrics.get('curriculum', {}), checkpoint)
        if restore:
          agent.load_state_dict(torch.load(restore, map_location=args.device, weights_only=False)['agent'])
          print(f"SAC_ROLLBACK {restore}", flush=True)
        args.next_checkpoint += args.checkpoint_interval
    writer.flush()
  finally:
    writer.close()
    env.close()
  return run_dir


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=list_tasks(), required=True)
  parser.add_argument('--benchmark-randomization', action='store_true')
  parser.add_argument('--actor-noise-per-episode',action='store_true')
  parser.add_argument('--reward-scale', type=float, default=1., help='Scale reward stored in replay; native evaluation criteria stay unchanged')
  parser.add_argument('--learning-rate', type=float, default=3.e-4)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--total-transitions", type=int, default=2_000_000,
                      help="new transitions in this run, additional to source when resuming")
  parser.add_argument("--resume-checkpoint", help="restore complete SAC state; recollect replay under current reward")
  parser.add_argument("--learning-starts", type=int, default=25_000)
  parser.add_argument('--warmup-hold-steps', type=int, default=1)
  parser.add_argument('--warmup-centering-kp', type=float, default=.5)
  parser.add_argument('--warmup-centering-kd', type=float, default=.2)
  parser.add_argument('--target-entropy', type=float, default=None)
  parser.add_argument('--stagnation-after-transitions', type=int, default=0)
  parser.add_argument('--stagnation-patience', type=int, default=2)
  parser.add_argument('--stagnation-minimum-height', type=float, default=.15)
  parser.add_argument("--replay-capacity", type=int, default=1_000_000)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--utd-ratio", type=float, default=0.25)
  parser.add_argument("--checkpoint-interval", type=int, default=200_000)
  parser.add_argument("--evaluation-episodes", type=int, default=32)
  parser.add_argument("--log-interval", type=int, default=25)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--run-dir", required=True)
  parser.add_argument(
    "--robust-training",
    action="store_true",
    help="add actor observation noise (asymmetric critic) and random cart impulses",
  )
  parser.add_argument("--actor-noise-max-std", type=float, default=0.02)
  parser.add_argument(
    "--disable-cart-impulses", action="store_true",
    help="omit random external cart-force pulses",
  )
  args = parser.parse_args()
  args.next_checkpoint = args.checkpoint_interval
  print(f"TRAIN_COMPLETE run_dir={train(args)}", flush=True)


if __name__ == "__main__":
  main()
