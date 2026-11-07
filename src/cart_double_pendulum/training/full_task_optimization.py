"""Opt-in Full-task optimization; never installed for experts or Meta."""
import numpy as np
import torch

from cart_double_pendulum.rl.sac import SACAgent, TransitionBatch


def reflection_mask(dimension, *, device, dtype):
    frame = [-1., -1., -1., 1., -1., 1., -1., -1.]
    if dimension == 8:
        values = frame
    elif dimension >= 18 and dimension % 9 == 0:
        values = (frame + [-1.]) * (dimension // 9)
    else:
        raise ValueError('Full reflection supports raw8D or chronological 9D-frame history')
    return torch.tensor(values, device=device, dtype=dtype)


def reflect_batch(batch):
    def reflect(values):
        if values is None:
            return None
        mask = reflection_mask(values.shape[-1],device=values.device,dtype=values.dtype)
        return torch.cat((values, values * mask), dim=0)
    return TransitionBatch(reflect(batch.observations),
        torch.cat((batch.actions, -batch.actions),dim=0),
        torch.cat((batch.rewards,batch.rewards),dim=0),
        reflect(batch.next_observations),torch.cat((batch.dones,batch.dones),dim=0),
        reflect(batch.critic_observations),reflect(batch.next_critic_observations))


class FullSACAgent(SACAgent):
    """Same SAC losses, with exact plant-reflection replay augmentation."""
    def update(self, batch):
        return super().update(reflect_batch(batch))


def selection_key(metrics):
    """Joint success first; useful skill progress before force-energy ties."""
    cases = metrics['cases']
    values = list(cases.values())
    joint = [m['success_rate'] for m in values]
    capture = [m['full_process_success_rate'] for m in values]
    center = [m['precision_center_success_rate'] for m in values]
    return [float(min(joint)),float(np.mean(joint)),
        float(cases['nominal']['full_process_success_rate']),float(min(capture)),float(np.mean(capture)),
        float(np.mean(center)),
        float(np.mean([min(5.,m['mean_longest_center_hold_s'])/5. for m in values])),
        float(np.mean([min(10.,m['mean_longest_capture_hold_s'])/10. for m in values])),
        float(np.mean([m['center_occupancy'] for m in values])),
        float(np.mean([m['capture_occupancy'] for m in values])),
        float(np.mean([m['handoff_rate'] for m in values])),
        -float(np.mean([m['numerical_failure_rate'] for m in values])),
        -float(np.mean([m['rail_failure_rate'] for m in values])),
        -float(metrics['mean_force_energy'])]
