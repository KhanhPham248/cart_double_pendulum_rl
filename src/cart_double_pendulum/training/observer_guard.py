"""Guard for the recurrent swing expert: reset the state estimate (hidden 64:70) when it
disagrees strongly with the measurement."""
import numpy as np

from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.onnx_policy import BatchedOnnxPolicy
from cart_double_pendulum.plant_scales import plant_scales

STATE_CHANNELS = slice(64, 70)
class GuardedObserver(BatchedOnnxPolicy):
  def __init__(self, path, contract=CONTRACT):
    super().__init__(path, contract=contract)
    S = plant_scales(contract)
    self.reset_thresholds = np.array([.5*S.rail, 3.793*S.angle, 3.793*S.angle,
                                      3.215*S.cart_speed, 8.456*S.rate, 8.456*S.rate])

  def __call__(self, observations, *, advance=True):
    if self.hidden is not None:
      obs = np.asarray(observations)
      measured = np.stack((obs[:, 0] * self.contract.rail_limit_m, np.arctan2(-obs[:, 2], -obs[:, 3]),
          np.arctan2(obs[:, 4], obs[:, 5]), self.contract.cart_velocity_scale * obs[:, 1],
          self.contract.joint_velocity_scale * obs[:, 6], self.contract.joint_velocity_scale * obs[:, 7]), axis=1)
      state = self.hidden[0, :, STATE_CHANNELS]
      delta = state - measured
      delta[:, 1:3] = np.arctan2(np.sin(delta[:, 1:3]), np.cos(delta[:, 1:3]))
      bad = (~np.isfinite(state).all(axis=1)) | (np.abs(delta) > self.reset_thresholds).any(axis=1)
      if np.any(bad):
        # A copy avoids mutating the caller's state for a feature peek.
        previous = self.hidden
        self.hidden = previous.copy()
        self.hidden[0, bad, STATE_CHANNELS] = measured[bad]
        result = super().__call__(observations, advance=advance)
        if not advance:
          self.hidden = previous
        return result
    return super().__call__(observations, advance=advance)
