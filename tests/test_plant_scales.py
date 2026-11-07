"""Task parameters are plant-scale multiples, so an MJCF edit rescales them."""
import json
import re

import pytest

from conftest import ROOT
from cart_double_pendulum.common import CONTRACT, MODEL_PATH, load_contract
from cart_double_pendulum.plant_scales import plant_scales, resolve, scale_tag
from cart_double_pendulum.training.task_directory import read_json

# Keys whose values depend on the plant; hanging angles stay in plain radians.
SUFFIXES = ("_m", "_m_s", "_n", "_rad_s", "_rad")
SHARED_RATES = {"velocity", "velocity_threshold", "capture_velocity", "hanging_velocity", "upright_velocity"}
PLAIN = {"hanging_angle_rad"}

def physical(key):
  return key not in PLAIN and (key in SHARED_RATES or key.endswith(SUFFIXES))

def bare_physical_values(value, key=None):
  if scale_tag(value) is not None:
    return []
  if isinstance(value, dict):
    return [found for k, item in value.items() for found in bare_physical_values(item, k)]
  if isinstance(value, list):
    return [found for item in value for found in bare_physical_values(item, key)]
  numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
  return [key] if numeric and physical(key) else []

def test_scales_of_the_model():
  scales = plant_scales()
  assert scales.rail == CONTRACT.rail_limit_m and scales.force == CONTRACT.actuator_force_limit_n
  assert 0 < scales.angle < 1 and 0 < scales.rate < 20 and 0 < scales.cart_speed < 20
  assert scales.upright_pole > 0

@pytest.mark.parametrize("path", sorted((ROOT / "tasks/real").rglob("*.json")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_task_json_has_no_bare_physical_values(path):
  assert bare_physical_values(json.loads(path.read_text())) == []

def test_resolve_multiplies_by_the_scale():
  scales = plant_scales()
  assert resolve({"a": {"rail": 0.5}, "b": [{"angle": 2.0}], "c": 3, "rail": {"weight": 1}}) == {
    "a": 0.5 * scales.rail, "b": [2.0 * scales.angle], "c": 3, "rail": {"weight": 1}}
  with pytest.raises(ValueError, match="unknown plant scale"):
    scales.value("metre", 1.0)

def test_rail_edit_rescales_task_values(tmp_path):
  text = MODEL_PATH.read_text()
  edited = re.sub(r'(<joint name="cart_slide"[^>]*range=")-0\.2115 0\.2115', r"\g<1>-0.4 0.4", text, count=1)
  assert edited != text
  path = tmp_path / "model.xml"
  path.write_text(edited)
  contract = load_contract(path)
  scales = plant_scales(contract)
  assert scales.rail == 0.4
  terms = read_json(ROOT / "tasks/real/expert_swing_sac/reward.json", contract)["terms"]
  assert terms["rail_terminal"]["params"]["rail_limit_m"] == pytest.approx(0.95 * 0.4)
