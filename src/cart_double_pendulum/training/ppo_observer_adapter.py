"""Frozen causal observer for PPO RL and its matching self-contained export."""
from pathlib import Path
import torch
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.plant_metadata import check_plant_metadata
from torch import nn
from cart_double_pendulum.training.model_temporal_expert import ModelTemporalExpert


def load_observer(path,device,contract=CONTRACT):
  payload=torch.load(path,map_location='cpu',weights_only=False)
  check_plant_metadata(payload.get('plant_metadata'), contract, where=str(path))
  cfg=payload['config']
  if cfg.get('observer_model')!='model_gru':
    raise ValueError('PPO observer adapter requires a constrained model_gru checkpoint')
  state=payload['model']
  embedded={key.removeprefix('expert.'):value for key,value in state.items() if key.startswith('expert.')}
  hidden_size=state['memory.weight_hh'].shape[1]
  if cfg.get('hidden_size',hidden_size)!=hidden_size:
    raise ValueError('observer config and stored recurrent state dimensions disagree')
  model=ModelTemporalExpert(None,cfg['algorithm'],hidden_size=hidden_size,
    full_cross_gain=cfg.get('full_cross_gain',False),frozen_expert_state=embedded,
    frozen_dynamics=(state['transition'],state['actuation']), contract=contract)
  model.load_state_dict(payload['model'])
  return model.to(device).eval().requires_grad_(False)


def reset_actor_observer(env,env_ids=None):
  if hasattr(env,'_actor_observer_hidden'):
    if env_ids is None:env._actor_observer_hidden.zero_()
    else:env._actor_observer_hidden[env_ids]=0.


def filtered_observation(env,observation,checkpoint):
  """Called once by the cached actor observation manager, never by critic."""
  path=str(Path(checkpoint).resolve())
  if getattr(env,'_actor_observer_path',None)!=path:
    env._actor_observer_model=load_observer(path,env.device,env.cfg.plant_contract)
    frames=env.cfg.observations['actor'].terms['state'].history_length
    if env._actor_observer_model.source_history_frames!=frames:
      raise ValueError('PPO and frozen observer history lengths must match')
    env._actor_observer_path=path
    env._actor_observer_hidden=torch.zeros(env.num_envs,env._actor_observer_model.hidden_size,device=env.device)
  model=env._actor_observer_model
  with torch.no_grad():
    _,estimated,hidden,confidence=model._step(observation,env.action_manager.action,env._actor_observer_hidden)
  env._actor_observer_hidden=hidden.detach()
  return confidence*estimated+(1.-confidence)*observation


class ObservedHistoryActor(nn.Module):
  """Same filter/history as training, with the *new* learned PPO actor."""
  def __init__(self,observer,actor):
    super().__init__()
    if not observer.source_history_frames:
      raise ValueError('observer must carry native9D frame history')
    self.contract=observer.contract
    self.observer=observer
    self.actor=actor
    self.hidden_size=observer.hidden_size

  def forward(self,observations,previous_action,hidden):
    _,_,next_hidden,_=self.observer._step(observations,previous_action,hidden[0])
    history=next_hidden[:,self.observer.memory_size+6:-1]
    action=self.actor(history).clamp(-1.,1.)
    return action,next_hidden[None]
