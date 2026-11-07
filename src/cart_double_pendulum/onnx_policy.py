"""Batched ONNX actor inference for observation, history and recurrent policies."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.history_policy import HistoryOnnxPolicy, history_length
from cart_double_pendulum.policy_metadata import policy_session
OBSERVATION_DIM = 8


class BatchedOnnxPolicy:
  """Batched frozen expert (observation, history or recurrent ONNX) on raw 8-D observations."""

  def __init__(self, path: str | Path, contract=CONTRACT):
    self.contract = contract
    self.path = Path(path).resolve()
    self.session = policy_session(self.path, dynamic_batch=True, contract=contract)
    inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
    self.recurrent = {value.name for value in inputs} == {'observations','previous_action','hidden'}
    self.history_length = history_length(inputs)
    self.has_memory = self.recurrent or self.history_length is not None
    if self.recurrent:
      hidden = next(value for value in inputs if value.name == 'hidden')
      self.hidden_size = int(hidden.shape[2])
      self.hidden = None
      self.previous_action = None
    elif len(inputs) != 1 or len(outputs) != 1:
      raise ValueError("expert ONNX must have exactly one input and one output")
    self.input_name, self.output_name = inputs[0].name, outputs[0].name
    if self.history_length is not None:
      self.history_policy = HistoryOnnxPolicy(self.session)

  def reset(self, env_ids=None) -> None:
    if self.history_length is not None:
      self.history_policy.reset(env_ids)
      return
    if not self.recurrent:
      return
    if env_ids is None:
      self.hidden = None
      self.previous_action = None
    elif self.hidden is not None:
      self.hidden[:,env_ids,:] = 0.
      self.previous_action[env_ids] = 0.

  def record_applied_actions(self, action) -> None:
    if self.history_length is not None:
      self.history_policy.record_applied_actions(action)
      return
    if self.recurrent and self.previous_action is not None:
      self.previous_action[:] = np.asarray(action,dtype=np.float32).reshape(-1,1)

  def __call__(self, observations: np.ndarray, *, advance: bool = True) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != OBSERVATION_DIM:
      raise ValueError(f"expected expert observations [batch, {OBSERVATION_DIM}]")
    if self.history_length is not None:
      return self.history_policy(values, advance=advance)
    if self.recurrent:
      if self.hidden is None or self.hidden.shape[1] != len(values):
        self.hidden = np.zeros((1,len(values),self.hidden_size),dtype=np.float32)
        self.previous_action = np.zeros((len(values),1),dtype=np.float32)
      result,hidden = self.session.run(None,{'observations':values,'previous_action':self.previous_action,'hidden':self.hidden})
      if advance:
        self.hidden = hidden
        self.previous_action = np.asarray(result,dtype=np.float32).reshape(-1,1).copy()
      return np.asarray(result,dtype=np.float32).reshape(-1,1)
    result = self.session.run(
      [self.output_name], {self.input_name: values}
    )[0]
    return np.asarray(result, dtype=np.float32).reshape(-1, 1)
