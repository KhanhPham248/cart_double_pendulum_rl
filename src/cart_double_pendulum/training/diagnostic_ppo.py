"""Opt-in metrics around upstream PPO.update; no extra optimizer steps or RNG draws."""
import torch
from rsl_rl.algorithms.ppo import PPO


def policy_update_metrics(log_prob, old_log_prob, kl, clip_param):
    with torch.no_grad():
        ratio = (log_prob.detach().reshape(-1) - old_log_prob.detach().reshape(-1)).exp()
        return {'diagnostic_kl': float(kl.detach().mean()),
                'diagnostic_clip_fraction': float(((ratio - 1.).abs() > clip_param).float().mean())}


class DiagnosticPPO(PPO):
    def update(self):
        # Explicitly scope the instrumentation; augmentation changes batch alignment.
        if self.symmetry or self.rnd or self.is_multi_gpu or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError('DiagnosticPPO currently supports nonrecurrent, single-device PPO without symmetry/RND')
        records = []
        current = [None]
        storage = self.storage
        original_generator = storage.mini_batch_generator
        original_log_prob = self.actor.get_output_log_prob
        had_generator = 'mini_batch_generator' in storage.__dict__
        had_log_prob = 'get_output_log_prob' in self.actor.__dict__

        def batches(*args, **kwargs):
            for batch in original_generator(*args, **kwargs):
                current[0] = batch
                yield batch
            current[0] = None

        def log_prob(*args, **kwargs):
            result = original_log_prob(*args, **kwargs)
            batch = current[0]
            if batch is not None:
                with torch.no_grad():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params,
                                                      self.actor.output_distribution_params)
                    records.append(policy_update_metrics(result, batch.old_actions_log_prob,
                                                         kl, self.clip_param))
            return result

        def critic_metrics(module, inputs, output):
            batch = current[0]
            if batch is not None and records:
                with torch.no_grad():
                    returns = batch.returns.detach().reshape(-1)
                    variance = returns.var(unbiased=False)
                    if variance > 1.e-12:
                        error = returns - output.detach().reshape(-1)
                        records[-1]['diagnostic_explained_variance'] = float(1. - error.var(unbiased=False) / variance)

        storage.mini_batch_generator = batches
        self.actor.get_output_log_prob = log_prob
        handle = self.critic.register_forward_hook(critic_metrics)
        try:
            result = super().update()
        finally:
            handle.remove()
            if had_generator:
                storage.mini_batch_generator = original_generator
            else:
                del storage.mini_batch_generator
            if had_log_prob:
                self.actor.get_output_log_prob = original_log_prob
            else:
                del self.actor.get_output_log_prob
        for key in {key for row in records for key in row}:
            values = [row[key] for row in records if key in row]
            result[key] = sum(values) / len(values)
        return result
