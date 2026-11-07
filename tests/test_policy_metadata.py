"""Exported policies carry the contract they were trained for, and loaders check it."""
import re

import onnx
import pytest
import torch

from cart_double_pendulum.common import MODEL_PATH, load_contract
from cart_double_pendulum.policy_metadata import FORMAT, policy_session, write_policy_metadata
from cart_double_pendulum.runtime_policy import OnnxPolicy


def export_linear(path, width, input_name="observations"):
  torch.onnx.export(torch.nn.Linear(width, 1), torch.zeros(1, width), str(path),
                    input_names=[input_name], output_names=["actions"], opset_version=18, dynamo=False)
  return path


def metadata(path):
  return {entry.key: entry.value for entry in onnx.load(str(path)).metadata_props}


@pytest.mark.parametrize(("width", "interface"), [(8, "observation"), (72, "history")])
def test_stamped_policy_loads_and_names_its_interface(tmp_path, width, interface):
  path = export_linear(tmp_path / "policy.onnx", width)
  write_policy_metadata(path)
  values = metadata(path)
  assert values["format"] == FORMAT and values["interface"] == interface
  OnnxPolicy(path)(torch.zeros(8).numpy())


def test_unstamped_policy_is_refused(tmp_path):
  with pytest.raises(ValueError, match="no .* metadata"):
    policy_session(export_linear(tmp_path / "policy.onnx", 8))


def test_policy_for_another_plant_is_refused(tmp_path):
  path = export_linear(tmp_path / "policy.onnx", 8)
  write_policy_metadata(path)
  other = tmp_path / "model.xml"
  other.write_text(re.sub(r'mass="0.0112"', 'mass="0.0074"', MODEL_PATH.read_text()))
  with pytest.raises(ValueError, match="plant.link1_mass_kg"):
    policy_session(path, contract=load_contract(other))


def test_restamping_keeps_foreign_metadata(tmp_path):
  path = export_linear(tmp_path / "policy.onnx", 8)
  model = onnx.load(str(path))
  model.metadata_props.add(key="exporter", value="rsl_rl")
  onnx.save(model, str(path))
  write_policy_metadata(path)
  write_policy_metadata(path)
  keys = [entry.key for entry in onnx.load(str(path)).metadata_props]
  assert keys.count("format") == 1 and "exporter" in keys
