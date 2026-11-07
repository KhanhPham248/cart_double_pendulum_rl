"""ONNX policy wrappers used by headless evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.history_policy import HistoryOnnxPolicy, history_length
from cart_double_pendulum.policy_metadata import policy_session


class OnnxPolicy:
  """Single-environment policy on raw 8-D observations; record the applied action after each call."""

  def __init__(self, path: str | Path, contract=CONTRACT):
    self.path = Path(path).resolve()
    self.session = policy_session(self.path, contract=contract)
    inputs = self.session.get_inputs()
    outputs = self.session.get_outputs()
    self.recurrent = {value.name for value in inputs} == {'observations','previous_action','hidden'}
    self.history_length = history_length(inputs)
    self.has_memory = self.recurrent or self.history_length is not None
    if self.recurrent:
      from cart_double_pendulum.training.temporal_expert import TemporalOnnxPolicy
      self.temporal = TemporalOnnxPolicy(self.path,session=self.session)
      return
    if len(inputs) != 1 or len(outputs) < 1:
      raise ValueError("cart policy ONNX contract must have one input and one output")
    self.input_name = inputs[0].name
    self.output_name = outputs[0].name
    if self.history_length is not None:
      self.history_policy = HistoryOnnxPolicy(self.session)

  def __call__(self, observation: np.ndarray) -> float:
    if self.recurrent:
      return self.temporal(observation)
    if self.history_length is not None:
      values = np.asarray(observation, dtype=np.float32).reshape(1, 8)
      return float(self.history_policy(values)[0, 0])
    values = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    action = self.session.run([self.output_name], {self.input_name: values})[0]
    return float(np.asarray(action).reshape(-1)[0])

  def reset(self):
    if self.recurrent:
      self.temporal.reset()
    elif self.history_length is not None:
      self.history_policy.reset()

  def record_applied_action(self,action):
    if self.recurrent:
      self.temporal.record_applied_action(action)
    elif self.history_length is not None:
      self.history_policy.record_applied_actions(action)
