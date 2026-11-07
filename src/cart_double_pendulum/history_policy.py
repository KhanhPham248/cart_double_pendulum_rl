"""History ONNX input: frames [observation (8), applied action (1)], oldest first.

As in the mjlab CircularBuffer, the first frame after a reset fills the window.
"""

from __future__ import annotations

import numpy as np


def history_length(inputs) -> int | None:
  """Recognize a single fixed-width 9*H input, H >= 2."""
  if len(inputs) != 1 or len(inputs[0].shape) != 2:
    return None
  width = inputs[0].shape[1]
  if isinstance(width, int) and width >= 18 and width % 9 == 0:
    return width // 9
  return None


class HistoryOnnxPolicy:
  """Batched history policy; ``advance=False`` peeks without changing memory."""

  def __init__(self, session):
    self.session = session
    inputs = session.get_inputs()
    self.history_length = history_length(inputs)
    if self.history_length is None:
      raise ValueError("history ONNX requires one [batch, 9*H] input, H >= 2")
    self.input_name = inputs[0].name
    self.output_name = session.get_outputs()[0].name
    self.reset()

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      self.history = None
      self.previous_action = None
      self.initialized = None
    elif self.history is not None:
      self.history[env_ids] = 0.0
      self.previous_action[env_ids] = 0.0
      self.initialized[env_ids] = False

  def record_applied_actions(self, actions) -> None:
    if self.previous_action is not None:
      self.previous_action[:] = np.clip(np.asarray(actions, dtype=np.float32).reshape(-1, 1),-1.,1.)

  def __call__(self, observations: np.ndarray, *, advance: bool = True) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8:
      raise ValueError("expected expert observations [batch, 8]")
    batch = len(values)
    if self.history is None or len(self.history) != batch:
      history = np.zeros((batch, self.history_length, 9), dtype=np.float32)
      previous = np.zeros((batch, 1), dtype=np.float32)
      initialized = np.zeros(batch, dtype=bool)
    else:
      history = self.history
      previous = self.previous_action
      initialized = self.initialized
    frame = np.concatenate((values, previous), axis=1)
    window = np.empty_like(history)
    window[:, :-1] = history[:, 1:]
    window[:, -1] = frame
    window[~initialized] = frame[~initialized, None, :]
    result = self.session.run(
      [self.output_name], {self.input_name: window.reshape(batch, -1)}
    )[0]
    actions = np.asarray(result, dtype=np.float32).reshape(batch, 1)
    if advance:
      self.history = window
      self.previous_action = np.clip(actions,-1.,1.).copy()
      self.initialized = np.ones(batch, dtype=bool)
    return actions
