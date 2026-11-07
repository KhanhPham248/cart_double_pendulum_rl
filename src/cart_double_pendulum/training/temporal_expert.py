"""Recurrent observer in front of a frozen expert: noisy sensors, previous action, memory."""
from __future__ import annotations
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cart_double_pendulum.plant_scales import plant_scales
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.policy_metadata import policy_session, write_policy_metadata
from cart_double_pendulum.training.expert_actor import ExpertActor


class TemporalExpert(nn.Module):
  def __init__(self, checkpoint, algorithm, hidden_size=64, contract=CONTRACT):
    super().__init__()
    self.contract=contract
    self.scales=plant_scales(contract)
    self.expert=ExpertActor(checkpoint,algorithm,contract).requires_grad_(False).eval()
    self.hidden_size=hidden_size
    self.memory=nn.GRU(9,hidden_size,batch_first=True)
    self.correction=nn.Linear(hidden_size,8)
    nn.init.zeros_(self.correction.weight)
    nn.init.zeros_(self.correction.bias)
    self.register_buffer('input_mean',self.expert.mean.clone())
    self.register_buffer('input_std',self.expert.std.clamp_min(.05).clone())

  def sequence(self, observation, previous_action, hidden=None):
    normalized=(observation-self.input_mean)/self.input_std
    features=torch.cat((normalized,previous_action),dim=-1)
    latent,next_hidden=self.memory(features,hidden)
    estimate=observation+self.correction(latent)*self.input_std
    # Keep angle channels on the physical unit circles.
    parts=[estimate[...,:2]]
    for first in (2,4):
      pair=estimate[...,first:first+2]
      parts.append(pair/torch.linalg.vector_norm(pair,dim=-1,keepdim=True).clamp_min(1.e-5))
    parts.append(estimate[...,6:])
    estimate=torch.cat(parts,dim=-1)
    theta1=torch.atan2(-observation[...,2],-observation[...,3])
    sin2=observation[...,2]*observation[...,5]+observation[...,3]*observation[...,4]
    cos2=observation[...,3]*observation[...,5]-observation[...,2]*observation[...,4]
    theta2=torch.atan2(-sin2,-cos2)
    angle=self.scales.angle
    confidence=torch.sigmoid((3.082*angle-torch.maximum(theta1.abs(),theta2.abs()))/(.2371*angle)).unsqueeze(-1)
    mixed=confidence*estimate+(1.-confidence)*observation
    shape=observation.shape[:-1]
    action=self.expert(mixed.reshape(-1,8)).reshape(*shape,1)
    return action,estimate,next_hidden,confidence

  def forward(self, observation, previous_action, hidden):
    action,_,next_hidden,_=self.sequence(observation[:,None,:],previous_action[:,None,:],hidden)
    return action[:,0,:],next_hidden


def export_temporal(model,path):
  # Export a CPU copy: the CUDA GRUCell traces an operator ONNX cannot express.
  portable=copy.deepcopy(model).cpu().eval()
  torch.onnx.export(portable,(torch.zeros(2,8),torch.zeros(2,1),
    torch.zeros(1,2,model.hidden_size)),str(path),
    input_names=['observations','previous_action','hidden'],output_names=['action','next_hidden'],
    dynamic_axes={'observations':{0:'batch'},'previous_action':{0:'batch'},'hidden':{1:'batch'},
                  'action':{0:'batch'},'next_hidden':{1:'batch'}},opset_version=18,dynamo=False)
  write_policy_metadata(path, model.contract)


class TemporalOnnxPolicy:
  def __init__(self,path,session=None,contract=CONTRACT):
    self.path=Path(path).resolve()
    self.session=session if session is not None else policy_session(self.path, contract=contract)
    hidden=next(value for value in self.session.get_inputs() if value.name=='hidden')
    self.hidden_size=int(hidden.shape[2])
    self.reset()

  def reset(self):
    self.hidden=np.zeros((1,1,self.hidden_size),dtype=np.float32)
    self.previous_action=np.zeros((1,1),dtype=np.float32)

  def record_applied_action(self,action):
    self.previous_action[:]=float(np.clip(action,-1,1))

  def __call__(self,observation):
    action,self.hidden=self.session.run(None,{'observations':np.asarray(observation,dtype=np.float32).reshape(1,8),
      'previous_action':self.previous_action,'hidden':self.hidden})
    value=float(np.clip(action.reshape(-1)[0],-1,1))
    self.record_applied_action(value)
    return value
