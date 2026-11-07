"""Small GPU-resident SAC implementation for the cart task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Normal

from cart_double_pendulum.policy_metadata import write_policy_metadata
from cart_double_pendulum.common import CONTRACT

DEFAULT_DISCOUNT = 0.995  # SAC and PPO gamma; only PPO tasks may override it in training.json


class TransitionBatch(NamedTuple):
  observations: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  next_observations: torch.Tensor
  dones: torch.Tensor
  critic_observations: torch.Tensor | None = None
  next_critic_observations: torch.Tensor | None = None


class ReplayBuffer:
  def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str,
               *, asymmetric_critic: bool = False, critic_obs_dim: int | None = None):
    if min(capacity, obs_dim, action_dim) <= 0:
      raise ValueError("replay dimensions must be positive")
    self.capacity = capacity
    self.device = torch.device(device)
    self.observations = torch.empty((capacity, obs_dim), device=self.device)
    self.actions = torch.empty((capacity, action_dim), device=self.device)
    self.rewards = torch.empty((capacity, 1), device=self.device)
    self.next_observations = torch.empty((capacity, obs_dim), device=self.device)
    self.dones = torch.empty((capacity, 1), device=self.device)
    critic_obs_dim = obs_dim if critic_obs_dim is None else critic_obs_dim
    if critic_obs_dim <= 0:
      raise ValueError('critic replay dimension must be positive')
    asymmetric_critic = asymmetric_critic or critic_obs_dim != obs_dim
    self.critic_observations = torch.empty((capacity, critic_obs_dim), device=self.device) if asymmetric_critic else None
    self.next_critic_observations = torch.empty_like(self.critic_observations) if asymmetric_critic else None
    self.position = 0
    self.size = 0

  def add(
    self,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_observations: torch.Tensor,
    dones: torch.Tensor,
    critic_observations: torch.Tensor | None = None,
    next_critic_observations: torch.Tensor | None = None,
  ) -> None:
    count = observations.shape[0]
    if observations.ndim != 2 or actions.ndim != 2:
      raise ValueError("replay inputs must be batched")
    if count > self.capacity:
      observations = observations[-self.capacity :]
      actions = actions[-self.capacity :]
      rewards = rewards[-self.capacity :]
      next_observations = next_observations[-self.capacity :]
      dones = dones[-self.capacity :]
      if critic_observations is not None:
        critic_observations = critic_observations[-self.capacity :]
      if next_critic_observations is not None:
        next_critic_observations = next_critic_observations[-self.capacity :]
      count = self.capacity
    indices = (torch.arange(count, device=self.device) + self.position) % self.capacity
    self.observations[indices] = observations.detach()
    self.actions[indices] = actions.detach()
    self.rewards[indices, 0] = rewards.detach().reshape(-1)
    self.next_observations[indices] = next_observations.detach()
    self.dones[indices, 0] = dones.detach().float().reshape(-1)
    if self.critic_observations is not None:
      if critic_observations is None or next_critic_observations is None:
        raise ValueError('asymmetric replay requires clean current and next critic observations')
      self.critic_observations[indices] = critic_observations.detach()
      self.next_critic_observations[indices] = next_critic_observations.detach()
    self.position = (self.position + count) % self.capacity
    self.size = min(self.capacity, self.size + count)

  def sample(self, batch_size: int) -> TransitionBatch:
    if self.size < batch_size:
      raise ValueError("not enough replay samples")
    indices = torch.randint(0, self.size, (batch_size,), device=self.device)
    return TransitionBatch(
      self.observations[indices],
      self.actions[indices],
      self.rewards[indices],
      self.next_observations[indices],
      self.dones[indices],
      self.critic_observations[indices] if self.critic_observations is not None else None,
      self.next_critic_observations[indices] if self.next_critic_observations is not None else None,
    )

  def __len__(self) -> int:
    return self.size


def mlp(input_dim: int, output_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
  layers: list[nn.Module] = []
  previous = input_dim
  for width in hidden:
    layers.extend((nn.Linear(previous, width), nn.ReLU()))
    previous = width
  layers.append(nn.Linear(previous, output_dim))
  return nn.Sequential(*layers)


class Actor(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...]):
    super().__init__()
    self.body = mlp(obs_dim, 2 * action_dim, hidden)
    self.action_dim = action_dim

  def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = self.body(observations).chunk(2, dim=-1)
    return mean, log_std.clamp(-5.0, 2.0)

  def sample(
    self, observations: torch.Tensor, deterministic: bool = False, noise: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    mean, log_std = self(observations)
    if deterministic:
      return torch.tanh(mean), None
    distribution = Normal(mean, log_std.exp())
    # ``noise``: a standard-normal draw supplied by the caller (correlated in time during collection).
    pre_tanh = distribution.rsample() if noise is None else mean + log_std.exp() * noise
    action = torch.tanh(pre_tanh)
    log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
    log_prob -= torch.log(1.0 - action.square() + 1.0e-6).sum(dim=-1, keepdim=True)
    return action, log_prob


class Critic(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...]):
    super().__init__()
    self.body = mlp(obs_dim + action_dim, 1, hidden)

  def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return self.body(torch.cat((observations, actions), dim=-1))


@dataclass(frozen=True)
class SACConfig:
  gamma: float = DEFAULT_DISCOUNT
  tau: float = 0.005
  learning_rate: float = 3.0e-4
  init_alpha: float = 0.2
  batch_size: int = 256
  hidden_dims: tuple[int, ...] = (256, 256)
  target_entropy: float | None = None


class SACAgent:
  def __init__(self, obs_dim: int, action_dim: int, device: str, config: SACConfig | None = None,
               *, critic_obs_dim: int | None = None):
    self.obs_dim = obs_dim
    self.critic_obs_dim = obs_dim if critic_obs_dim is None else critic_obs_dim
    if self.critic_obs_dim <= 0:
      raise ValueError('critic observation dimension must be positive')
    self.action_dim = action_dim
    self.device = torch.device(device)
    self.config = config or SACConfig()
    c = self.config
    self.actor = Actor(obs_dim, action_dim, c.hidden_dims).to(self.device)
    self.critics = nn.ModuleList(
      [Critic(self.critic_obs_dim, action_dim, c.hidden_dims).to(self.device) for _ in range(2)]
    )
    self.targets = nn.ModuleList(
      [Critic(self.critic_obs_dim, action_dim, c.hidden_dims).to(self.device) for _ in range(2)]
    )
    for target, critic in zip(self.targets, self.critics):
      target.load_state_dict(critic.state_dict())
      target.requires_grad_(False)
    self.log_alpha = torch.tensor(
      c.init_alpha, device=self.device, dtype=torch.float32
    ).log().requires_grad_()
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.learning_rate)
    self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=c.learning_rate)
    self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=c.learning_rate)
    self.target_entropy = -float(action_dim) if c.target_entropy is None else float(c.target_entropy)
    if not torch.isfinite(torch.tensor(self.target_entropy)):
      raise ValueError('SAC target entropy must be finite')
    self.update_count = 0
    self.obs_mean = torch.zeros(obs_dim, device=self.device)
    self.obs_var = torch.ones(obs_dim, device=self.device)
    self.obs_count = torch.tensor(1.0, device=self.device)
    self.critic_obs_mean = torch.zeros(self.critic_obs_dim, device=self.device)
    self.critic_obs_var = torch.ones(self.critic_obs_dim, device=self.device)
    self.critic_obs_count = torch.tensor(1.0, device=self.device)

  @property
  def alpha(self) -> torch.Tensor:
    return self.log_alpha.exp()

  @torch.no_grad()
  def observe(self, observations: torch.Tensor, critic_observations: torch.Tensor | None = None) -> None:
    mean = observations.mean(dim=0)
    var = observations.var(dim=0, unbiased=False).clamp_min(1.0e-6)
    n = float(observations.shape[0])
    total = self.obs_count + n
    delta = mean - self.obs_mean
    self.obs_mean += delta * (n / total)
    self.obs_var = (self.obs_var * self.obs_count + var * n + delta.square() * self.obs_count * n / total) / total
    self.obs_count = total
    if self.critic_obs_dim != self.obs_dim:
      if critic_observations is None:
        raise ValueError('SAC with a privileged critic requires its observations for normalization')
      mean = critic_observations.mean(dim=0)
      var = critic_observations.var(dim=0, unbiased=False).clamp_min(1.0e-6)
      n = float(critic_observations.shape[0])
      total = self.critic_obs_count + n
      delta = mean - self.critic_obs_mean
      self.critic_obs_mean += delta * (n / total)
      self.critic_obs_var = (self.critic_obs_var * self.critic_obs_count + var * n
                            + delta.square() * self.critic_obs_count * n / total) / total
      self.critic_obs_count = total

  def normalize(self, observations: torch.Tensor) -> torch.Tensor:
    return (observations - self.obs_mean) / self.obs_var.sqrt().clamp_min(1.0e-3)

  def normalize_critic(self, observations: torch.Tensor) -> torch.Tensor:
    if self.critic_obs_dim == self.obs_dim:
      return self.normalize(observations)
    return (observations - self.critic_obs_mean) / self.critic_obs_var.sqrt().clamp_min(1.0e-3)

  @torch.no_grad()
  def act(self, observations: torch.Tensor, deterministic: bool = False,
          noise: torch.Tensor | None = None) -> torch.Tensor:
    actions, _ = self.actor.sample(self.normalize(observations), deterministic, noise)
    return actions

  def update(self, batch: TransitionBatch) -> dict[str, float]:
    c = self.config
    obs = self.normalize(batch.observations)
    next_obs = self.normalize(batch.next_observations)
    if self.critic_obs_dim != self.obs_dim and (
        batch.critic_observations is None or batch.next_critic_observations is None):
      raise ValueError('SAC replay is missing privileged critic observations')
    critic_obs = self.normalize_critic(batch.critic_observations) if batch.critic_observations is not None else obs
    next_critic_obs = self.normalize_critic(batch.next_critic_observations) if batch.next_critic_observations is not None else next_obs
    with torch.no_grad():
      next_action, next_log_prob = self.actor.sample(next_obs)
      next_q = torch.minimum(
        *(target(next_critic_obs, next_action) for target in self.targets)
      ) - self.alpha.detach() * next_log_prob
      td_target = batch.rewards + c.gamma * (1.0 - batch.dones) * next_q
    q_values = [critic(critic_obs, batch.actions) for critic in self.critics]
    critic_loss = sum(torch.nn.functional.mse_loss(q, td_target) for q in q_values)
    self.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
    self.critic_optimizer.step()

    for critic in self.critics:
      critic.requires_grad_(False)
    action, log_prob = self.actor.sample(obs)
    q = torch.minimum(*(critic(critic_obs, action) for critic in self.critics))
    actor_loss = (self.alpha.detach() * log_prob - q).mean()
    self.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
    self.actor_optimizer.step()
    for critic in self.critics:
      critic.requires_grad_(True)

    alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
    self.alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    self.alpha_optimizer.step()
    with torch.no_grad():
      for target, critic in zip(self.targets, self.critics):
        for target_param, param in zip(target.parameters(), critic.parameters()):
          target_param.lerp_(param, c.tau)
    self.update_count += 1
    return {
      "loss/actor": float(actor_loss.detach()),
      "loss/critic": float(critic_loss.detach()),
      "loss/alpha": float(alpha_loss.detach()),
      "train/alpha": float(self.alpha.detach()),
      "train/entropy": float(-log_prob.mean().detach()),
      "train/q_mean": float(q.mean().detach()),
      "train/target_q_mean": float(td_target.mean()),
      "train/td_abs_mean": float(torch.stack([(v.detach() - td_target).abs().mean() for v in q_values]).mean()),
      # Extremes expose rare outlier transitions that the means hide.
      "train/td_abs_max": float(torch.stack([(v.detach() - td_target).abs().max() for v in q_values]).max()),
      "train/q_min": float(q_values[0].detach().min()),
      "train/q_max": float(q_values[0].detach().max()),
      "train/target_q_min": float(td_target.min()),
      "train/target_q_max": float(td_target.max()),
      "train/sampled_action_saturation": float((action.detach().abs() >= .95).float().mean()),
    }

  def state_dict(self) -> dict:
    return {
      "obs_dim": self.obs_dim,
      "critic_obs_dim": self.critic_obs_dim,
      "action_dim": self.action_dim,
      "config": asdict(self.config),
      "actor": self.actor.state_dict(),
      "critics": [critic.state_dict() for critic in self.critics],
      "targets": [target.state_dict() for target in self.targets],
      "log_alpha": self.log_alpha.detach(),
      "actor_optimizer": self.actor_optimizer.state_dict(),
      "critic_optimizer": self.critic_optimizer.state_dict(),
      "alpha_optimizer": self.alpha_optimizer.state_dict(),
      "update_count": self.update_count,
      "obs_mean": self.obs_mean,
      "obs_var": self.obs_var,
      "obs_count": self.obs_count,
      "critic_obs_mean": self.critic_obs_mean,
      "critic_obs_var": self.critic_obs_var,
      "critic_obs_count": self.critic_obs_count,
    }

  def load_state_dict(self, state: dict) -> None:
    if (int(state["obs_dim"]), int(state["action_dim"])) != (self.obs_dim, self.action_dim):
      raise ValueError("SAC checkpoint dimensions do not match")
    if int(state.get('critic_obs_dim', state['obs_dim'])) != self.critic_obs_dim:
      raise ValueError('SAC checkpoint critic dimensions do not match; restart with the new reward/critic')
    self.actor.load_state_dict(state["actor"])
    for models, values in ((self.critics, state["critics"]), (self.targets, state["targets"])):
      for model, value in zip(models, values):
        model.load_state_dict(value)
    self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
    self.actor_optimizer.load_state_dict(state["actor_optimizer"])
    self.critic_optimizer.load_state_dict(state["critic_optimizer"])
    self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    self.update_count = int(state["update_count"])
    self.obs_mean.copy_(state["obs_mean"])
    self.obs_var.copy_(state["obs_var"])
    self.obs_count.copy_(state["obs_count"])
    if 'critic_obs_mean' in state:
      self.critic_obs_mean.copy_(state['critic_obs_mean'])
      self.critic_obs_var.copy_(state['critic_obs_var'])
      self.critic_obs_count.copy_(state['critic_obs_count'])


class _DeterministicActor(nn.Module):
  def __init__(self, agent: SACAgent):
    super().__init__()
    self.actor = agent.actor
    self.register_buffer("obs_mean", agent.obs_mean.detach().clone())
    self.register_buffer("obs_std", agent.obs_var.detach().sqrt().clamp_min(1.0e-3))

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    mean, _ = self.actor((observations - self.obs_mean) / self.obs_std)
    return torch.tanh(mean)


def export_deterministic_onnx(agent: SACAgent, path: str) -> None:
  """Export SAC's deterministic actor with the exact eight-value contract."""

  wrapper = _DeterministicActor(agent).to(agent.device).eval()
  sample = torch.zeros((2, agent.obs_dim), device=agent.device)
  torch.onnx.export(
    wrapper,
    sample,
    path,
    input_names=["observations"],
    output_names=["actions"],
    dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=18,
    dynamo=False,
  )
  write_policy_metadata(path, getattr(agent, "plant_contract", CONTRACT))
