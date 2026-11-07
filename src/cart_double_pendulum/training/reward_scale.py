"""Positive unit conversion for PPO rewards; physics and actor inputs unchanged."""
import math
from mjlab.rl import RslRlVecEnvWrapper


class RewardScaledVecEnv(RslRlVecEnvWrapper):
  def __init__(self,env,*,reward_scale,clip_actions=1.):
    if not math.isfinite(reward_scale) or reward_scale<=0.:
      raise ValueError('reward scale must be finite and positive')
    self.reward_scale=float(reward_scale)
    super().__init__(env,clip_actions=clip_actions)

  def step(self,actions):
    observations,rewards,dones,extras=super().step(actions)
    return observations,rewards*self.reward_scale,dones,extras
