"""PPO runner with checkpoint evaluation and lexicographic best selection."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from mjlab.rl import MjlabOnPolicyRunner

from cart_double_pendulum import curriculum
from cart_double_pendulum.policy_metadata import write_policy_metadata
from cart_double_pendulum.plant_metadata import plant_metadata

from cart_double_pendulum.training.checkpoint_evaluation import evaluate_checkpoint
from cart_double_pendulum.training.reward_log import RewardTermLog


class CartPpoRunner(MjlabOnPolicyRunner):
  def export_policy_to_onnx(self,path,filename='policy.onnx',verbose=False):
    observer=getattr(self.env.unwrapped,'_actor_observer_model',None)
    if observer is None:
      super().export_policy_to_onnx(path,filename,verbose)
      write_policy_metadata(Path(path)/filename, self.env.unwrapped.cfg.plant_contract)
      return
    import copy
    from cart_double_pendulum.training.ppo_observer_adapter import ObservedHistoryActor
    from cart_double_pendulum.training.temporal_expert import export_temporal
    actor=self.alg.get_policy().as_onnx(verbose=verbose).cpu().eval()
    fused=ObservedHistoryActor(copy.deepcopy(observer).cpu().eval(),actor)
    export_temporal(fused,Path(path)/filename)

  def __init__(
    self,
    *args,
    task_name: str,
    evaluation_episodes: int,
    max_action_std: float = 1.0,
    min_action_std: float = 0.0,
    **kwargs,
  ) -> None:
    self.task_name = task_name
    self.evaluation_episodes = evaluation_episodes
    self._last_save_iteration = -1
    self._best_key = (-1.0, -1.0, -1.0, -1.0, float("inf"), float("inf"))
    super().__init__(*args, **kwargs)
    # Per-term reward means next to rsl_rl's own logs, once per iteration.
    reward_log = RewardTermLog(self.env.unwrapped)
    log = self.logger.log

    def logged(**values):
      log(**values)
      reward_log.write(self.logger.writer, values["it"])

    self.logger.log = logged
    # Actions are clipped to [-1, 1]: cap the std above, and floor it so exploration cannot collapse.
    update = self.alg.update

    def capped():
      result = update()
      self.alg.actor.distribution.std_param.data.clamp_(min=min_action_std, max=max_action_std)
      return result

    self.alg.update = capped

  def _restore_training_checkpoint(self, path) -> None:
    payload = torch.load(path, map_location=self.device, weights_only=False)
    # Rollout normalization creates inference buffers (notably _std). Copying
    # restored state into those buffers must also run under inference mode.
    # Deserialize outside that context so optimizer state remains normal tensors.
    with torch.inference_mode():
      self.alg.load(payload, {"actor": True, "critic": True, "optimizer": True}, strict=True)

  def save(self, path: str, infos=None) -> None:
    iteration = self.current_learning_iteration
    if iteration == self._last_save_iteration:
      return
    run_dir = Path(self.logger.log_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"iteration_{iteration:08d}.pt"
    temporary = checkpoint.with_suffix(".tmp")
    infos={**(infos or {}),'reward_scale':float(getattr(self.env,'reward_scale',1.))}
    super().save(str(temporary), infos)
    payload = torch.load(temporary, map_location="cpu", weights_only=False)
    payload["plant_metadata"] = plant_metadata(self.env.unwrapped.cfg.plant_contract)
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    shutil.copy2(checkpoint, checkpoint_dir / "latest.pt")
    candidate_name = "policy_candidate.onnx"
    self.export_policy_to_onnx(str(run_dir), candidate_name)
    metrics = evaluate_checkpoint(run_dir / candidate_name, self.task_name, self.evaluation_episodes)
    shutil.copy2(run_dir / candidate_name, run_dir / "policy_latest.onnx")
    # Keep every evaluated policy for the final selection.
    candidate_archive = checkpoint_dir / f"iteration_{iteration:08d}.onnx"
    shutil.copy2(run_dir / candidate_name, candidate_archive)
    (run_dir / "evaluation_latest.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / f"iteration_{iteration:08d}.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    key = (
      float(metrics.get('nominal_gate', True)),
      float(metrics["success_rate"]),
      float(metrics["stable_occupancy"]),
      float(metrics.get("both_upright_calm_occupancy", 0.0)),
      float(metrics.get("both_upright_occupancy", 0.0)),
      -float(metrics["mean_time_to_success_s"]),
      -float(metrics["mean_force_energy"]),
    )
    restore = curriculum.skill_checkpoint(metrics.get('curriculum', {}), checkpoint)
    if restore:
      self._restore_training_checkpoint(restore)
      print(f"PPO_ROLLBACK {restore}", flush=True)
    if 'full_task_selection_key' in metrics:
      key = tuple(metrics['full_task_selection_key'])
    if key > self._best_key:
      self._best_key = key
      shutil.copy2(checkpoint, checkpoint_dir / "best.pt")
      shutil.copy2(run_dir / candidate_name, run_dir / "policy.onnx")
      (run_dir / "evaluation_best.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
      )
    self._last_save_iteration = iteration
    print(
      f"PPO_CHECKPOINT_EVALUATED iteration={iteration} "
      f"success={metrics['success_rate']:.3f} "
      f"stable={metrics['stable_occupancy']:.3f}",
      flush=True,
    )
