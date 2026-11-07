"""Fixed two-link equilibria; EP bits encode absolute down(0)/up(1)."""
import math
from numbers import Integral


def validate_ep(target_ep):
  if isinstance(target_ep, bool) or not isinstance(target_ep, Integral) or not 0 <= target_ep <= 3:
    raise ValueError('target_ep must be an integer from 0 to 3')
  return int(target_ep)


def absolute_target(target_ep):
  ep = validate_ep(target_ep)
  return (math.pi if ep & 2 else 0., math.pi if ep & 1 else 0.)


def joint_target(target_ep):
  first, second = absolute_target(target_ep)
  return first, second - first


def target_errors(qpos, array_api, target_ep):
  if qpos.shape[-1] != 3:
    raise ValueError('EP tasks require two links and a cart')
  first, second = absolute_target(target_ep)
  return [array_api.arctan2(array_api.sin(error), array_api.cos(error))
          for error in (qpos[..., 1] - first, qpos[..., 1] + qpos[..., 2] - second)]


def target_mask(qpos, qvel, array_api, target_ep, *, cart_safe_m, angle_threshold_rad,
                cart_velocity_threshold_m_s, angular_velocity_threshold_rad_s):
  e1, e2 = target_errors(qpos, array_api, target_ep)
  return ((abs(qpos[..., 0]) < cart_safe_m) & (abs(e1) < angle_threshold_rad)
          & (abs(e2) < angle_threshold_rad)
          & (abs(qvel[..., 0]) < cart_velocity_threshold_m_s)
          & (abs(qvel[..., 1]) < angular_velocity_threshold_rad_s)
          & (abs(qvel[..., 1] + qvel[..., 2]) < angular_velocity_threshold_rad_s))
