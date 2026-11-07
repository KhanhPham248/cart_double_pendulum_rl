"""Model-based observer: nominal upright A/B model with GRU-learned measurement gains."""
import math

import mujoco
import numpy as np
import torch
from torch import nn

from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.plant_scales import plant_scales
from cart_double_pendulum.training.expert_actor import ExpertActor



class ModelTemporalExpert(nn.Module):
  def __init__(self,checkpoint,algorithm,hidden_size=64,full_cross_gain=False,
               frozen_expert_state=None,frozen_dynamics=None, contract=CONTRACT):
    super().__init__()
    self.contract = contract
    self.scales = S = plant_scales(contract)
    self.expert=(ExpertActor(checkpoint,algorithm,contract) if frozen_expert_state is None
                 else ExpertActor.from_frozen_state(frozen_expert_state,algorithm)).requires_grad_(False).eval()
    self.memory_size=hidden_size
    source_width=self.expert.mean.shape[-1]
    if source_width!=8 and (source_width<18 or source_width%9):
      raise ValueError('source must use8D or9*H native observation/action history')
    self.source_history_frames=0 if source_width==8 else source_width//9
    self.hidden_size=hidden_size+7+9*self.source_history_frames
    self.memory=nn.GRUCell(15,hidden_size)
    self.correction=nn.Linear(hidden_size,6)  # bounded sensor update gains
    nn.init.zeros_(self.correction.weight);nn.init.constant_(self.correction.bias,-1.5)
    self.cross_correction=nn.Linear(hidden_size,36) if full_cross_gain else None
    if self.cross_correction is not None:
      nn.init.zeros_(self.cross_correction.weight);nn.init.zeros_(self.cross_correction.bias)
    mean=self.expert.mean if not self.source_history_frames else self.expert.mean[:,-9:-1]
    std=self.expert.std if not self.source_history_frames else self.expert.std[:,-9:-1]
    self.register_buffer('input_mean',mean.clone())
    self.register_buffer('input_std',std.clamp_min(.05).clone())
    self.register_buffer('physical_scale',torch.tensor([[.5*S.rail,1.185*S.angle,1.185*S.angle,.643*S.cart_speed,.8456*S.rate,.8456*S.rate]],
                                                     dtype=torch.float32))
    if frozen_dynamics is None:
      # Nominal plant linearised at up-up, integrated over one control period.
      model=mujoco.MjModel.from_xml_path(contract.model_path);data=mujoco.MjData(model)
      data.qpos[:]=(0,math.pi,0);mujoco.mj_forward(model,data)
      a=np.empty((6,6));b=np.empty((6,1))
      mujoco.mjd_transitionFD(model,data,1.e-5,0,a,b,None,None)
      aa=np.linalg.matrix_power(a,contract.decimation)
      bb=sum(np.linalg.matrix_power(a,i)@b for i in range(contract.decimation))*contract.actuator_force_limit_n
      transition=torch.from_numpy(aa.astype(np.float32));actuation=torch.from_numpy(bb.astype(np.float32))
    else:
      transition,actuation=(value.detach().clone().float()for value in frozen_dynamics)
      if transition.shape!=(6,6) or actuation.shape!=(6,1) or not torch.isfinite(transition).all() or not torch.isfinite(actuation).all():
        raise ValueError('invalid stored nominal observer dynamics')
    self.register_buffer('transition',transition)
    self.register_buffer('actuation',actuation)

  def _step(self,observation,previous_action,hidden):
    memory=hidden[:,:self.memory_size]
    previous_state=hidden[:,self.memory_size:self.memory_size+6]
    initialized=hidden[:,-1:]
    theta1=torch.atan2(-observation[:,2],-observation[:,3])
    theta2r=torch.atan2(observation[:,4],observation[:,5])
    # Physical state (x, theta1 - pi, theta2, x_dot, omega1, omega2) decoded from the observation.
    c=self.contract
    S=self.scales
    measured=torch.stack((observation[:,0]*c.rail_limit_m,theta1,theta2r,c.cart_velocity_scale*observation[:,1],
                           c.joint_velocity_scale*observation[:,6],c.joint_velocity_scale*observation[:,7]),dim=1)
    predicted=previous_state@self.transition.T+previous_action@self.actuation.T
    # Reset: measured positions, zero velocity prior.
    initial=torch.cat((measured[:,:3],torch.zeros_like(measured[:,3:])),dim=1)
    predicted=initialized*predicted+(1.-initialized)*initial
    innovation=measured-predicted
    innovation=torch.cat((innovation[:,:1],torch.atan2(torch.sin(innovation[:,1:3]),
      torch.cos(innovation[:,1:3])),innovation[:,3:]),dim=1)
    features=torch.cat(((observation-self.input_mean)/self.input_std,previous_action,
                         innovation/self.physical_scale),dim=1)
    next_memory=self.memory(features,memory)
    estimated=self.measurement_update(predicted,innovation,next_memory)
    estimate_obs=torch.stack((estimated[:,0]/c.rail_limit_m,estimated[:,3]/c.cart_velocity_scale,
      torch.sin(math.pi+estimated[:,1]),torch.cos(math.pi+estimated[:,1]),torch.sin(estimated[:,2]),
      torch.cos(estimated[:,2]),estimated[:,4]/c.joint_velocity_scale,estimated[:,5]/c.joint_velocity_scale),dim=1)
    theta2a=torch.atan2(torch.sin(theta1+theta2r),torch.cos(theta1+theta2r))
    confidence=torch.sigmoid((2.134*S.angle-torch.maximum(theta1.abs(),theta2a.abs()))/(.1897*S.angle)).unsqueeze(1)
    mixed=confidence*estimate_obs+(1.-confidence)*observation
    next_state=confidence*estimated+(1.-confidence)*measured
    if self.source_history_frames:
      previous_history=hidden[:,self.memory_size+6:-1].reshape(-1,self.source_history_frames,9)
      frame=torch.cat((mixed,previous_action),dim=1)
      history=torch.cat((previous_history[:,1:],frame[:,None]),dim=1)
      history=torch.where((initialized<.5).unsqueeze(-1),frame[:,None].expand_as(history),history)
      action=self.expert(history.flatten(start_dim=1))
      next_hidden=torch.cat((next_memory,next_state,history.flatten(start_dim=1),torch.ones_like(initialized)),dim=1)
    else:
      action=self.expert(mixed)
      next_hidden=torch.cat((next_memory,next_state,torch.ones_like(initialized)),dim=1)
    return action,estimate_obs,next_hidden,confidence

  def measurement_update(self,predicted,innovation,memory):
    gain=.01+.98*torch.sigmoid(self.correction(memory))
    estimated=predicted+gain*innovation
    if self.cross_correction is not None:
      # Cross-channel correction has no free equilibrium offset.
      cross=.10*torch.tanh(self.cross_correction(memory)).reshape(-1,6,6)
      off_diagonal=1.-torch.eye(6,device=innovation.device)
      cross=cross*off_diagonal[None]
      estimated=estimated+torch.bmm(cross,(innovation/self.physical_scale).unsqueeze(-1)).squeeze(-1)*self.physical_scale
    return estimated

  def clean_actions(self,observation,previous_action):
    """Clean training labels with the source's causal history; never deploy input."""
    batch,steps,_=observation.shape
    if not self.source_history_frames:
      return self.expert(observation.reshape(-1,8)).reshape(batch,steps,1)
    frame=torch.cat((observation[:,0],previous_action[:,0]),dim=1)
    history=frame[:,None].expand(-1,self.source_history_frames,-1).clone()
    result=[]
    for index in range(steps):
      frame=torch.cat((observation[:,index],previous_action[:,index]),dim=1)
      history=torch.cat((history[:,1:],frame[:,None]),dim=1)
      result.append(self.expert(history.flatten(start_dim=1)))
    return torch.stack(result,dim=1)

  def sequence(self,observation,previous_action,hidden=None):
    batch=observation.shape[0]
    state=torch.zeros(batch,self.hidden_size,device=observation.device) if hidden is None else hidden[0]
    actions=[];estimates=[];confidences=[]
    for index in range(observation.shape[1]):
      action,estimate,state,confidence=self._step(observation[:,index],previous_action[:,index],state)
      actions.append(action);estimates.append(estimate);confidences.append(confidence)
    return torch.stack(actions,1),torch.stack(estimates,1),state[None],torch.stack(confidences,1)

  def forward(self,observation,previous_action,hidden):
    action,_,next_hidden,_=self._step(observation,previous_action,hidden[0])
    return action,next_hidden[None]
