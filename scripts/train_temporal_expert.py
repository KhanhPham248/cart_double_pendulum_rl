#!/usr/bin/env python3
"""DAgger adaptation of a recurrent sensor observer; started by run.py train-observer."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from cart_double_pendulum.plant_metadata import load_bank, plant_metadata
from torch.utils.tensorboard import SummaryWriter

from cart_double_pendulum.common import CONTRACT,policy_observation,noise_profile
from cart_double_pendulum.plant_scales import plant_scales
from cart_double_pendulum.runtime import CartDoublePendulumRuntime
from cart_double_pendulum.runtime_policy import OnnxPolicy
from cart_double_pendulum.tasks.config import DAMPING_SCALE_RANGE, FORCE_SCALE_RANGE, INERTIA_SCALE_RANGE, PUSH_FORCE_N
from cart_double_pendulum.training.temporal_expert import TemporalExpert,TemporalOnnxPolicy,export_temporal
from cart_double_pendulum.training.robust_eval import (
  MODEL,SWING_NOMINAL,SWING_RANDOMIZED,BALANCE_NOMINAL,BALANCE_PUSH,RAIL_FAILURE_M,FALL_ANGLE_RAD,evaluate_controller,
)


def qualifies(candidate, baseline, task, *, robust_case=None):
  nominal, robust = ('swing_nominal', 'swing_randomized') if task == 'swing_up' else ('balance_nominal', 'balance_push')
  robust=robust_case or robust
  n, n0, r, r0 = candidate[nominal], baseline[nominal], candidate[robust], baseline[robust]
  return (n['success_rate'] >= n0['success_rate'] - .01
          and n['stable_occupancy'] >= n0['stable_occupancy'] - .01
          and r['success_rate'] >= r0['success_rate'] + .03 - 1.e-9
          and r['stable_occupancy'] >= r0['stable_occupancy'] - .01
          and r['rail_failure_rate'] <= r0['rail_failure_rate'] + .01
          and r['fall_failure_rate'] <= r0['fall_failure_rate'] + .01)


def rank(metrics, task, *, robust_case=None):
  value = metrics[robust_case or ('swing_randomized' if task == 'swing_up' else 'balance_push')]
  return value['success_rate'], value['stable_occupancy'], -value['rail_failure_rate']


def evaluate_policy(policy,task,episodes,seed,*,robust_case=None):
  cases=(SWING_NOMINAL,SWING_RANDOMIZED) if task=='swing_up' else (BALANCE_NOMINAL,BALANCE_PUSH)
  if robust_case=='balance_randomized_no_push':
    if task!='balancing':raise ValueError('balance randomized case needs balancing task')
    cases=(BALANCE_NOMINAL,replace(BALANCE_PUSH,key=robust_case,push_force_n=0.))
  elif robust_case not in (None,'balance_push'):
    raise ValueError('unsupported robust case')
  def factory(runtime):
    if hasattr(policy,'reset'):policy.reset()
    return policy
  return {case.key:evaluate_controller(factory,replace(case,seed=seed+ordinal*10000),
    episodes=episodes,strict_protocol=True)[0] for ordinal,case in enumerate(cases)}


def collect(policy,task,bank_path,episodes,seed):
  rng=np.random.default_rng(seed)
  q,v=load_bank(bank_path, CONTRACT)
  trajectories=[]
  for episode in range(episodes):
    sigma=float(rng.uniform(0,.02))
    runtime=CartDoublePendulumRuntime(MODEL,contract=CONTRACT,seed=seed+episode,
      mass_scale=float(rng.uniform(*INERTIA_SCALE_RANGE)),damping_scale=float(rng.uniform(*DAMPING_SCALE_RANGE)),
      force_scale=float(rng.uniform(*FORCE_SCALE_RANGE)),observation_noise_std=sigma,
      independent_observation_noise=True)
    runtime.reset(mode='hanging' if task=='swing_up' else 'upright')
    if (task=='balancing' and episode%2==0) or (task=='swing_up' and episode%5==0):
      index=int(rng.integers(len(q)));runtime.data.qpos[:]=q[index];runtime.data.qvel[:]=v[index]
      mujoco.mj_forward(runtime.model,runtime.data)
    if hasattr(policy,'reset'):policy.reset()
    states=[];actions=[];previous=0.
    push_first=int(rng.integers(70,200));force=float(rng.uniform(-1.2*PUSH_FORCE_N,1.2*PUSH_FORCE_N)) if episode%2 else 0.
    for step in range(1000):
      clean=policy_observation(runtime.qpos,runtime.qvel,np,runtime.contract).astype(np.float32)
      states.append(clean);actions.append(previous)
      # Clean state is only a training label, never a policy input.
      observation=runtime.observation()
      if isinstance(policy,OnnxPolicy):
        action=policy(clean)  # bootstrap with competent clean source trajectories
      else:
        action=policy(observation)
      runtime.data.qfrc_applied[0]=force if push_first<=step<push_first+5 else 0.
      runtime.step(action);previous=float(np.clip(action,-1,1))
      if abs(runtime.qpos[0])>RAIL_FAILURE_M:break
      if task=='balancing':
        angles=np.array([runtime.qpos[1],runtime.qpos[1]+runtime.qpos[2]])-np.pi
        if np.max(np.abs(np.arctan2(np.sin(angles),np.cos(angles))))>FALL_ANGLE_RAD:break
    if len(states)>=32:
      trajectories.append((np.asarray(states),np.asarray(actions,dtype=np.float32)[:,None]))
  return trajectories


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--task',choices=('swing_up','balancing'),required=True)
  p.add_argument('--algorithm',choices=('sac','ppo'),required=True)
  p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source-onnx',type=Path,required=True)
  p.add_argument('--bank',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
  p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,required=True)
  p.add_argument('--rounds',type=int,default=3);p.add_argument('--updates-per-round',type=int,default=2000)
  p.add_argument('--collection-episodes',type=int,default=64)
  p.add_argument('--eval-episodes',type=int,default=32);p.add_argument('--eval-interval',type=int,default=500)
  p.add_argument('--learning-rate',type=float,default=3.e-4)
  p.add_argument('--observer-model',choices=('free_gru','model_gru'),default='free_gru')
  p.add_argument('--balanced-windows',action='store_true',help='Balance quiet, recovery and hanging training windows')
  p.add_argument('--window-frames',type=int,default=32)
  p.add_argument('--burn-in-frames',type=int,default=0)
  p.add_argument('--full-cross-gain',action='store_true')
  p.add_argument('--balancing-robust-case',choices=('balance_push','balance_randomized_no_push'),default='balance_push')
  args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
  if args.window_frames<32 or not 0<=args.burn_in_frames<args.window_frames:
    p.error('window must be at least32 and0 <= burn-in < window')
  if args.full_cross_gain and args.observer_model!='model_gru':
    p.error('full cross gain requires model_gru')
  run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
  config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
  config.update(method='causal_GRU_sensor_observer_frozen_expert_DAgger',hidden_size=64,window_frames=args.window_frames,
                inference_inputs=['noisy_observation','previous_applied_action','recurrent_memory'],
                clean_state_used_only_for_training_labels=True,reward_function_unchanged=True)
  (run/'train_config.json').write_text(json.dumps(config,indent=2)+'\n')
  load_bank(args.bank, CONTRACT)  # reject foreign banks even before baseline evaluation
  source=OnnxPolicy(args.source_onnx)
  if args.observer_model=='model_gru':
    from cart_double_pendulum.training.model_temporal_expert import ModelTemporalExpert
    model=ModelTemporalExpert(args.checkpoint,args.algorithm,full_cross_gain=args.full_cross_gain).to(args.device)
  else:
    model=TemporalExpert(args.checkpoint,args.algorithm).to(args.device)
  config['recurrent_state_size']=model.hidden_size
  (run/'train_config.json').write_text(json.dumps(config,indent=2)+'\n')
  trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
  optimizer=torch.optim.Adam(trainable,lr=args.learning_rate)
  sensor_profile=torch.tensor(noise_profile(CONTRACT),device=args.device,dtype=torch.float32)
  writer=SummaryWriter(str(run/'tensorboard'))
  robust_case=args.balancing_robust_case if args.task=='balancing' else None
  baseline=evaluate_policy(source,args.task,args.eval_episodes,811001,robust_case=robust_case)
  (run/'baseline_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
  trajectories=[];history=[];best=None;update=0;started=time.monotonic()
  rng=np.random.default_rng(args.seed)
  for round_index in range(args.rounds):
    behavior=source
    if (run/'policy.onnx').is_file():behavior=TemporalOnnxPolicy(run/'policy.onnx')
    fresh=collect(behavior,args.task,args.bank,args.collection_episodes,args.seed+round_index*10000)
    trajectories+=fresh
    trajectories=[item for item in trajectories if len(item[0])>=args.window_frames]
    if not trajectories:
      raise RuntimeError(f'No collected episode lasted {args.window_frames} control steps: the source policy '
                         'fails too early to adapt an observer. Train the expert further first.')
    # Tensors are indexed by trajectory, preventing windows crossing resets.
    clean=torch.from_numpy(np.concatenate([item[0] for item in trajectories])).to(args.device)
    actions=torch.from_numpy(np.concatenate([item[1] for item in trajectories])).to(args.device)
    offsets=np.cumsum([0]+[len(item[0]) for item in trajectories])
    lengths=np.diff(offsets)
    pools=[]
    if args.balanced_windows:
      all_states=clean.detach().cpu().numpy()
      all_actions=actions.detach().cpu().numpy().reshape(-1)
      angle1=np.arctan2(-all_states[:,2],-all_states[:,3])
      sin2=all_states[:,2]*all_states[:,5]+all_states[:,3]*all_states[:,4]
      cos2=all_states[:,3]*all_states[:,5]-all_states[:,2]*all_states[:,4]
      angle2=np.arctan2(-sin2,-cos2)
      error=np.maximum(np.abs(angle1),np.abs(angle2))
      angle=plant_scales().angle
      rate=.1268*plant_scales().rate/CONTRACT.joint_velocity_scale
      recovery=(error>.1897*angle)|(np.max(np.abs(all_states[:,6:8]),axis=1)>rate)|(np.abs(all_actions)>.015)
      starts=np.concatenate([np.arange(offsets[i],offsets[i+1]-args.window_frames+1) for i in range(len(trajectories))])
      middle=starts+args.window_frames//2
      phase=np.where(error[middle]>2.134*angle,2,np.where(recovery[middle],1,0))
      pools=[starts[phase==group] for group in range(3) if np.any(phase==group)]
    for _ in range(args.updates_per_round):
      update+=1
      if pools:
        starts=np.concatenate([rng.choice(pool,size=64//len(pools)+(index<64%len(pools))) for index,pool in enumerate(pools)])
        rng.shuffle(starts)
      else:
        ids=rng.integers(len(trajectories),size=64)
        starts=offsets[ids]+np.array([rng.integers(max(1,lengths[i]-args.window_frames+1)) for i in ids])
      indices=torch.as_tensor(starts[:,None]+np.arange(args.window_frames)[None,:],device=args.device)
      target=clean[indices];previous=actions[indices]
      sigma=torch.rand(64,1,1,device=args.device)*.02
      sigma[torch.rand(64,1,1,device=args.device)<.20]=0.
      measured=target+torch.randn_like(target)*sigma*sensor_profile
      predicted,estimate,_,confidence=model.sequence(measured,previous)
      with torch.no_grad():
        desired=model.clean_actions(target,previous) if hasattr(model,'clean_actions') else model.expert(target.reshape(-1,8)).reshape(64,args.window_frames,1)
      tail=slice(args.burn_in_frames,None)
      state_error=((estimate[:,tail]-target[:,tail])/model.input_std).square()
      state_loss=(confidence[:,tail]*state_error).mean() if args.observer_model=='model_gru' else state_error.mean()
      action_loss=(confidence[:,tail]*(predicted[:,tail]-desired[:,tail]).square()).mean()
      loss=state_loss+5.*action_loss
      optimizer.zero_grad(set_to_none=True);loss.backward()
      torch.nn.utils.clip_grad_norm_(trainable,1.)
      optimizer.step()
      if update%10==0:
        writer.add_scalar('loss/state_estimation',state_loss.item(),update)
        writer.add_scalar('loss/clean_action_target',action_loss.item(),update)
      if update%args.eval_interval==0:
        candidate=run/'candidate.onnx';export_temporal(model,candidate)
        metrics=evaluate_policy(TemporalOnnxPolicy(candidate),args.task,args.eval_episodes,811001,robust_case=robust_case)
        row={'update':update,'round':round_index,'metrics':metrics,
             'qualifies':qualifies(metrics,baseline,args.task,robust_case=robust_case),'elapsed_s':time.monotonic()-started}
        history.append(row)
        if row['qualifies'] and (best is None or rank(metrics,args.task,robust_case=robust_case)>rank(best['metrics'],args.task,robust_case=robust_case)):
          best=row;shutil.copy2(candidate,run/'policy.onnx')
          torch.save({'model':model.state_dict(),'config':config,'validation':metrics,'plant_metadata':plant_metadata()},run/'best.pt')
        (run/'validation_history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('TEMPORAL_EVAL '+json.dumps(row),flush=True)
  writer.close()
  result={'complete':True,'best':best,'baseline':baseline,'training_trajectories':len(trajectories),
          'training_states':len(clean),'elapsed_s':time.monotonic()-started,'config':config}
  (run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  print('TEMPORAL_COMPLETE '+str(run),flush=True)


if __name__=='__main__':main()
