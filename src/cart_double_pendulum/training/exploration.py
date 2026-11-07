"""Coherent warmup exploration, centered on the rail; never a swing-up teacher."""
import math

import torch

NOISE_CORRELATION_S = 0.12  # SAC collection noise correlation time; about the cart time constant x2


class CenteredWarmup:
  """Hold random forcing briefly and add cart centering during replay warmup.

  No demonstrations, policy labels, energy-pump controller or cloning loss.
  The actual bounded action is what the environment executes and replay stores.
  """
  def __init__(self, num_envs, device, hold_steps, *, kp=.5, kd=.2):
    if (isinstance(hold_steps, bool) or hold_steps < 1 or int(hold_steps) != hold_steps
        or not all(math.isfinite(v) and v >= 0 for v in (kp, kd))):
      raise ValueError('Warmup needs a positive integer hold and finite nonnegative centering gains')
    self.hold_steps, self.kp, self.kd = int(hold_steps), kp, kd
    self.noise = torch.zeros(num_envs, device=device)
    self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)

  def reset(self, env_ids):
    self.remaining[env_ids] = 0

  def sample(self, cart_position_norm, cart_velocity_norm):
    refresh = self.remaining <= 0
    candidate = torch.empty_like(self.noise).uniform_(-1., 1.)
    self.noise = torch.where(refresh, candidate, self.noise)
    self.remaining = torch.where(refresh, self.hold_steps, self.remaining) - 1
    action = self.noise - self.kp * cart_position_norm - self.kd * cart_velocity_norm
    return action.clamp(-1., 1.)[:, None]


class CorrelatedNoise:
  """Standard-normal AR(1) noise per environment; the damped cart filters out white 50 Hz noise."""
  def __init__(self, num_envs, action_dim, device, *, correlation_s, step_dt):
    if not (math.isfinite(correlation_s) and correlation_s > 0. and step_dt > 0.):
      raise ValueError('Correlated noise needs a positive correlation time and step')
    self.rho = math.exp(-step_dt / correlation_s)
    self.value = torch.randn((num_envs, action_dim), device=device)

  def sample(self):
    self.value = self.rho * self.value + math.sqrt(1. - self.rho ** 2) * torch.randn_like(self.value)
    return self.value

  def reset(self, done):
    # A new episode starts from a fresh stationary draw.
    self.value[done] = torch.randn_like(self.value[done])


class SwingProgressGuard:
  """Reject repeated idle checkpoints after a configurable learning allowance."""
  def __init__(self, after_transitions, patience=2, minimum_height=.15):
    if after_transitions < 1 or patience < 1 or not 0. < minimum_height <= 1.:
      raise ValueError('Swing progress guard needs positive allowance/patience and a height fraction in (0,1]')
    self.after, self.patience, self.minimum_height = after_transitions, patience, minimum_height
    self.bad_evaluations = 0

  def stalled(self, metrics, transitions):
    if transitions < self.after:
      return False
    nominal = next(iter(metrics['cases'].values()))
    progress = (metrics['nominal_success_rate'] > 0. or nominal['never_handoff_rate'] < 1.
                or nominal['mean_peak_height_fraction'] >= self.minimum_height)
    self.bad_evaluations = 0 if progress else self.bad_evaluations + 1
    return self.bad_evaluations >= self.patience
