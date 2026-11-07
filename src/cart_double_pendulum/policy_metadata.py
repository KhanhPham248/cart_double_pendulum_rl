"""Policy ONNX metadata: written by every exporter, checked by every loader (docs/HARDWARE.md)."""

from __future__ import annotations

from dataclasses import asdict
import json
import hashlib
from pathlib import Path

import onnx
import onnxruntime as ort

from cart_double_pendulum.common import (
  CONTRACT, GUARD_DAMPING, GUARD_FORCE, GUARD_MASS, RAIL_GUARD_FRACTION, CartDoublePendulumContract)
from cart_double_pendulum.history_policy import history_length


FORMAT = "cart_double_pendulum.policy.v1"
RECURRENT_INPUTS = {"observations", "previous_action", "hidden"}


def policy_interface(inputs) -> str:
  """Classify an ONNX input signature as one of the three actor interfaces."""

  names = {value.name for value in inputs}
  if names == RECURRENT_INPUTS:
    return "recurrent"
  if history_length(inputs) is not None:
    return "history"
  if len(inputs) == 1 and inputs[0].shape[-1] == CONTRACT.observation_dim:
    return "observation"
  raise ValueError(f"unrecognized policy ONNX inputs: {[(v.name, v.shape) for v in inputs]}")


def contract_metadata(contract: CartDoublePendulumContract = CONTRACT) -> dict[str, str]:
  """Values a policy depends on; exact string forms so equality is exact."""

  plant = {key: value for key, value in asdict(contract).items()
           if isinstance(value, float) and key not in {
             "control_timestep_s", "cart_velocity_scale", "joint_velocity_scale"}}
  return {
    "format": FORMAT,
    "model_sha256": hashlib.sha256(Path(contract.model_path).read_bytes()).hexdigest(),
    "control_timestep_s": repr(contract.control_timestep_s),
    "observation_names": ",".join(contract.observation_names),
    "rail_limit_m": repr(contract.rail_limit_m),
    "cart_velocity_scale_m_s": repr(contract.cart_velocity_scale),
    "joint_velocity_scale_rad_s": repr(contract.joint_velocity_scale),
    "action_names": ",".join(contract.action_names),
    "action_force_limit_n": repr(contract.actuator_force_limit_n),
    "plant": json.dumps(plant, sort_keys=True),
    "rail_guard": repr((RAIL_GUARD_FRACTION, GUARD_MASS, GUARD_DAMPING, GUARD_FORCE)),
  }


def write_policy_metadata(path: str | Path, contract: CartDoublePendulumContract = CONTRACT) -> None:
  """Stamp ``path`` in place, keeping metadata written by other tools."""

  path = Path(path)
  model = onnx.load(str(path))
  interface = policy_interface(ort.InferenceSession(
    model.SerializeToString(), providers=["CPUExecutionProvider"]).get_inputs())
  values = {**contract_metadata(contract), "interface": interface}
  kept = [entry for entry in model.metadata_props if entry.key not in values]
  del model.metadata_props[:]
  model.metadata_props.extend(kept)
  for key, value in values.items():
    model.metadata_props.add(key=key, value=value)
  onnx.save(model, str(path))


def check_policy_metadata(
  metadata: dict[str, str], contract: CartDoublePendulumContract = CONTRACT, *, where: str = "policy"
) -> None:
  if metadata.get("format") != FORMAT:
    raise ValueError(f"{where} has no {FORMAT} metadata; export it with this package")
  expected = contract_metadata(contract)
  wrong = [key for key, value in expected.items() if metadata.get(key) != value]
  if "plant" in wrong:
    trained, current = json.loads(metadata.get("plant", "{}")), json.loads(expected["plant"])
    wrong.remove("plant")
    wrong += [f"plant.{key}" for key in sorted(current) if trained.get(key) != current[key]]
  if wrong:
    raise ValueError(f"{where} was trained for a different plant or interface; mismatched: {wrong}")


def policy_session(
  path: str | Path,
  *,
  contract: CartDoublePendulumContract = CONTRACT,
  dynamic_batch: bool = False,
) -> ort.InferenceSession:
  """Checked single-threaded CPU session; ``dynamic_batch`` frees a fixed batch of 1."""

  path = Path(path).resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  model = onnx.load(str(path))
  check_policy_metadata({entry.key: entry.value for entry in model.metadata_props}, contract, where=str(path))
  if dynamic_batch:
    for value_info in (*model.graph.input, *model.graph.output):
      dimensions = value_info.type.tensor_type.shape.dim
      if dimensions and dimensions[0].dim_value == 1 and value_info.name not in ("hidden", "next_hidden"):
        dimensions[0].ClearField("dim_value")
        dimensions[0].dim_param = "batch"
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  options.inter_op_num_threads = 1
  return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
