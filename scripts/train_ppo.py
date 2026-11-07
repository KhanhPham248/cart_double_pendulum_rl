#!/usr/bin/env python3
"""Train one cart-double-pendulum policy with mjlab/RSL-RL PPO. Started by tasks/real/<name>/run.py train."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from cart_double_pendulum.tasks import list_tasks, make_task
from cart_double_pendulum.rl.sac import DEFAULT_DISCOUNT
from cart_double_pendulum.training.ppo_runner import CartPpoRunner


def _runner_cfg(args: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
  initial_action_std = 0.25 if args.task == "balancing" else 0.60
  if args.initial_action_std is not None:
    initial_action_std = args.initial_action_std
  entropy_coef = args.entropy_coef if args.entropy_coef is not None else (0.001 if args.task == "balancing" else 0.003)
  cfg=RslRlOnPolicyRunnerCfg(
    seed=args.seed,
    actor=RslRlModelCfg(
      hidden_dims=(128, 128, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": initial_action_std,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(128, 128, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=False,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=args.learning_rate,
      schedule=args.ppo_schedule,
      gamma=args.gamma,
      lam=args.gae_lambda,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    num_steps_per_env=args.rollout_steps,
    max_iterations=args.max_iterations,
    save_interval=args.save_interval,
    experiment_name=f"cart_double_pendulum_{args.task}",
    run_name="",
    logger="tensorboard",
    wandb_project="cart-double-pendulum",
    upload_model=False,
    clip_actions=1.0,
  )
  if args.update_diagnostics:
    cfg.algorithm.class_name='cart_double_pendulum.training.diagnostic_ppo:DiagnosticPPO'
  return cfg


def train(args: argparse.Namespace) -> Path:
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but PyTorch cannot see a GPU")
  configure_torch_backends()
  task = make_task(args.task)
  run_dir = Path(args.run_dir).resolve()
  run_dir.mkdir(parents=True, exist_ok=True)
  cfg = _runner_cfg(args)
  (run_dir / "train_config.yaml").write_text(
    yaml.safe_dump(vars(args), sort_keys=False), encoding="utf-8"
  )
  (run_dir / "ppo_config.yaml").write_text(
    yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), sort_keys=False),
    encoding="utf-8",
  )
  env_cfg = task.make_env_cfg(
      args.num_envs,
      robust_training=args.robust_training,
      actor_history_frames=args.actor_history_frames,
      actor_noise_per_episode=args.actor_noise_per_episode,
      actor_observer_checkpoint=args.actor_observer_checkpoint,
      benchmark_randomization=args.benchmark_randomization,
      actor_noise_max_std=args.actor_noise_max_std,
      cart_impulses=not args.disable_cart_impulses,
    )
  # Seed before environment startup randomization and the wrapper's first reset.
  env_cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  if args.reward_scale!=1.:
    from cart_double_pendulum.training.reward_scale import RewardScaledVecEnv
    wrapped=RewardScaledVecEnv(env,reward_scale=args.reward_scale,clip_actions=1.)
  else:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
  runner = CartPpoRunner(
    wrapped,
    asdict(cfg),
    str(run_dir),
    args.device,
    task_name=args.task,
    evaluation_episodes=args.evaluation_episodes,
    max_action_std=args.max_action_std,
    min_action_std=args.min_action_std,
  )
  if args.require_fresh_start:
    def digest(model):
      signature=hashlib.sha256()
      for key,value in sorted(model.state_dict().items()):
        signature.update(key.encode());signature.update(value.detach().cpu().contiguous().numpy().tobytes())
      return signature.hexdigest()
    counts={name:float(model.obs_normalizer.count.item())for name,model in (
      ('actor',runner.alg.actor),('critic',runner.alg.critic))}
    if any(value!=0. for value in counts.values())or env.common_step_counter!=0:
      raise RuntimeError('Fresh start unexpectedly has learned normalization or environment counter')
    (run_dir/'FRESH_START_AUDIT.json').write_text(json.dumps({'fresh_actor':True,'fresh_critic':True,
      'normalizer_initial_counts':counts,'optimizer_initial_states':len(runner.alg.optimizer.state),
      'initial_actor_sha256':digest(runner.alg.actor),'initial_critic_sha256':digest(runner.alg.critic),
      'environment_counter':int(env.common_step_counter),'observer_is_fixed_preprocessor':bool(args.actor_observer_checkpoint),
      'source_actor_checkpoint_loaded':False},indent=2)+'\n')
  try:
    if args.evaluate_initial_policy:
      # Evaluates and keeps the untrained policy as iteration 0.
      runner.save(str(run_dir / 'initial.pt'), {'stage': 'initial_policy'})
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    runner.save(str(run_dir / "final.pt"), {"task": args.task, "algorithm": "ppo"})
  finally:
    wrapped.close()
  return run_dir


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=list_tasks(), required=True)
  parser.add_argument('--benchmark-randomization', action='store_true')
  parser.add_argument('--update-diagnostics',action='store_true',
                      help='Log minibatch analytic KL, clip fraction and explained variance without changing PPO updates')
  parser.add_argument('--actor-history-frames',type=int,default=0)
  parser.add_argument('--actor-observer-checkpoint',type=str,default=None)
  parser.add_argument('--reward-scale',type=float,default=1.,help='Positive PPO reward-unit conversion')
  parser.add_argument('--actor-noise-per-episode',action='store_true')
  parser.add_argument('--require-fresh-start',action='store_true',
                      help='Audit that actor, critic and normalizers start untrained')
  parser.add_argument('--evaluate-initial-policy',action='store_true',
                      help='Evaluate and checkpoint the untrained policy before learning')
  parser.add_argument('--initial-action-std', type=float)
  parser.add_argument('--max-action-std', type=float, default=1.0, help='upper bound of the Gaussian action std')
  parser.add_argument('--min-action-std', type=float, default=0.0, help='lower bound of the Gaussian action std')
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--max-iterations", type=int, default=2_000)
  parser.add_argument("--save-interval", type=int, default=50)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--learning-rate", type=float, default=3.0e-4)
  parser.add_argument('--rollout-steps',type=int,default=24)
  parser.add_argument('--gamma',type=float,default=DEFAULT_DISCOUNT)
  parser.add_argument('--gae-lambda',type=float,default=.95)
  parser.add_argument('--ppo-schedule', choices=('fixed', 'adaptive'), default='fixed')
  parser.add_argument('--entropy-coef', type=float)
  parser.add_argument("--run-dir", required=True)
  parser.add_argument(
    "--robust-training",
    action="store_true",
    help="add actor observation noise and random cart impulses",
  )
  parser.add_argument("--actor-noise-max-std", type=float, default=0.02)
  parser.add_argument(
    "--disable-cart-impulses", action="store_true",
    help="omit random external cart-force pulses",
  )
  parser.add_argument("--evaluation-episodes", type=int, default=16)
  args = parser.parse_args()
  if args.rollout_steps<1 or not 0.<args.gamma<1. or not 0.<=args.gae_lambda<=1.:
    parser.error('rollout steps must be positive; gamma in(0,1), GAE lambda in[0,1]')
  if args.max_action_std <= 0. or not 0. <= args.min_action_std < args.max_action_std:
    parser.error('action std bounds must satisfy 0 <= min < max')
  if not math.isfinite(args.reward_scale) or args.reward_scale<=0.:
    parser.error('--reward-scale must be finite and positive')
  if args.actor_history_frames==1 or args.actor_history_frames<0:
    parser.error('--actor-history-frames must be zero or at least two')
  print(f"TRAIN_COMPLETE run_dir={train(args)}", flush=True)


if __name__ == "__main__":
  main()
