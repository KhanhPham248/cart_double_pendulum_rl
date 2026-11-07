"""Preserve pre-reset observations for SAC time-limit bootstrapping in mjlab 1.2."""
import torch
from mjlab.envs import ManagerBasedRlEnv


class SACEnvironment(ManagerBasedRlEnv):
    """Peek at each final frame before auto-reset, without advancing history twice.

    This project's SAC observation groups contain state/frame terms and no
    manager corruption/delay. Keep that contract explicit instead of silently
    constructing incorrect final observations for unsupported configurations.
    """
    def step(self, action):
        self.extras.pop('final_observation', None)
        self.extras.pop('final_observation_mask', None)
        self._capture_final_observation = True
        try:
            return super().step(action)
        finally:
            self._capture_final_observation = False

    def _reset_idx(self, env_ids=None):
        if (getattr(self, '_capture_final_observation', False)
                and (self.reset_time_outs & ~self.reset_terminated).any()):
            self.sim.forward()
            final = {}
            for group, cfg in self.observation_manager.cfg.items():
                if cfg.enable_corruption or not cfg.concatenate_terms or cfg.concatenate_dim != -1:
                    raise ValueError('SAC final observations require uncorrupted concatenated state terms')
                frames = []
                for name, term in cfg.terms.items():
                    if term.params.get('observer_checkpoint'):
                        raise ValueError('Recurrent observer peeking is not supported by SACEnvironment')
                    if term.delay_max_lag or term.noise is not None:
                        raise ValueError('SAC final observations do not support manager delay/noise')
                    frame = term.func(self, **term.params)
                    if term.clip is not None:
                        frame = frame.clamp(*term.clip)
                    if term.scale is not None:
                        frame = frame * term.scale
                    if term.history_length:
                        buffer = self.observation_manager._group_obs_term_history_buffer[group][name]
                        history = buffer.buffer
                        frame = torch.cat((history[:, 1:], frame[:, None]), dim=1)
                        if term.flatten_history_dim:
                            frame = frame.flatten(start_dim=1)
                    frames.append(frame)
                final[group] = torch.cat(frames, dim=-1).clone()
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids if env_ids is not None else slice(None)] = True
            self.extras['final_observation'] = final
            self.extras['final_observation_mask'] = mask
        super()._reset_idx(env_ids)


def replay_observations(observations, terminated, truncated, extras):
    """Use the actual transition successor; only true termination masks bootstrap."""
    timeouts = truncated.bool() & ~terminated.bool()
    if not timeouts.any():
        return observations
    final = extras.get('final_observation')
    mask = extras.get('final_observation_mask')
    if final is None or mask is None or not mask[timeouts].all():
        raise RuntimeError('SAC timeout is missing its pre-reset final observation')
    return {key: torch.where(timeouts[:, None], final[key], value) for key, value in observations.items()}
