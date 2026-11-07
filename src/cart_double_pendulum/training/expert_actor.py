"""Frozen PPO/SAC actor restored from a training or observer checkpoint."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from cart_double_pendulum.rl.sac import Actor
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.plant_metadata import check_plant_metadata


class ExpertActor(nn.Module):
  @classmethod
  def from_frozen_state(cls,state,algorithm):
    """Restore the actor embedded in an observer checkpoint; saved std already includes eps."""
    actor=cls.__new__(cls);nn.Module.__init__(actor);actor.algorithm=algorithm
    mean,std=state['mean'],state['std']
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():
      raise ValueError('embedded actor normalization must be finite with positive std')
    network={key.removeprefix('network.'):value for key,value in state.items() if key.startswith('network.')}
    if algorithm=='sac':
      weights=sorted((key for key in network if key.startswith('body.') and key.endswith('.weight')),
                     key=lambda key:int(key.split('.')[1]))
      if not weights or network[weights[-1]].shape[0]%2:
        raise ValueError('invalid embedded SAC actor architecture')
      hidden=tuple(network[key].shape[0]for key in weights[:-1])
      actor.network=Actor(mean.shape[-1],network[weights[-1]].shape[0]//2,hidden)
    elif algorithm=='ppo':
      weights=sorted((key for key in network if key.endswith('.weight')),key=lambda key:int(key.split('.')[0]))
      if not weights:raise ValueError('embedded PPO actor has no linear layers')
      layers=[]
      for index,key in enumerate(weights):
        weight=network[key];layers.append(nn.Linear(weight.shape[1],weight.shape[0]))
        if index<len(weights)-1:layers.append(nn.ELU())
      actor.network=nn.Sequential(*layers)
    else:raise ValueError(f'unsupported embedded expert algorithm:{algorithm}')
    actor.network.load_state_dict(network)
    actor.register_buffer('mean',mean.reshape(1,-1).clone())
    actor.register_buffer('std',std.reshape(1,-1).clone())
    return actor

  def __init__(self, checkpoint: str | Path, algorithm: str, contract=CONTRACT):
    super().__init__()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    check_plant_metadata(payload.get('plant_metadata'), contract, where=str(checkpoint))
    self.algorithm = algorithm
    if algorithm == 'sac':
      state = payload['agent']
      self.network = Actor(int(state['obs_dim']), int(state['action_dim']),
                           tuple(state['config']['hidden_dims']))
      self.network.load_state_dict(state['actor'])
      mean, std = state['obs_mean'], state['obs_var'].sqrt().clamp_min(1.e-3)
    elif algorithm == 'ppo':
      state = payload['actor_state_dict']
      weights = sorted((k for k in state if k.startswith('mlp.') and k.endswith('.weight')),
                       key=lambda k: int(k.split('.')[1]))
      layers = []
      for i, key in enumerate(weights):
        weight = state[key]
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.load_state_dict({'weight': weight, 'bias': state[key.replace('.weight', '.bias')]})
        layers.append(layer)
        if i < len(weights) - 1:
          layers.append(nn.ELU())
      self.network = nn.Sequential(*layers)
      # RSL-RL EmpiricalNormalization divides by std + eps (default eps=.01).
      mean, std = state['obs_normalizer._mean'], state['obs_normalizer._std'] + .01
    else:
      raise ValueError(f'unsupported expert algorithm: {algorithm}')
    self.register_buffer('mean', mean.reshape(1, -1).clone())
    self.register_buffer('std', std.reshape(1, -1).clone())

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    normalized = (observation - self.mean) / self.std
    if self.algorithm == 'sac':
      mean, _ = self.network(normalized)
      return torch.tanh(mean)
    return self.network(normalized).clamp(-1., 1.)

  def frozen_copy(self):
    return copy.deepcopy(self).eval().requires_grad_(False)
