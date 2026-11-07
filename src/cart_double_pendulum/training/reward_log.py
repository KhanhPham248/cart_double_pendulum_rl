"""Mean weighted value of every reward term per control step, written to TensorBoard."""

import torch


class RewardTermLog:
  def __init__(self, env) -> None:
    manager = env.reward_manager
    self.names = list(manager.active_terms)
    self.sums = torch.zeros(len(self.names), device=env.device)
    self.samples = 0
    compute = manager.compute

    def logged(dt):
      reward = compute(dt)
      self.sums += manager._step_reward.sum(dim=0)
      self.samples += manager._step_reward.shape[0]
      return reward

    manager.compute = logged

  def write(self, writer, step: int) -> None:
    """Write ``reward_terms/<term>`` averaged since the last write, then restart."""
    if writer is None or not self.samples:
      return
    for name, value in zip(self.names, (self.sums / self.samples).tolist()):
      writer.add_scalar(f"reward_terms/{name}", value, step)
    self.sums.zero_()
    self.samples = 0
