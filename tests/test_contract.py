"""The MJCF file is the single source of plant constants."""
import re

import mujoco
import pytest

from cart_double_pendulum.common import CONTRACT, MODEL_PATH, PLANT_PARAMETERS, load_contract


def compiled():
  return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def body(model, name):
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_contract_matches_compiled_model():
  model = compiled()
  assert CONTRACT.cart_mass_kg == model.body_mass[body(model, "cart")]
  assert CONTRACT.link1_mass_kg == model.body_mass[body(model, "link1")]
  assert CONTRACT.link2_mass_kg == model.body_mass[body(model, "link2")]
  assert CONTRACT.link1_length_m == pytest.approx(abs(model.body_pos[body(model, "link2")][2]))
  assert CONTRACT.link1_com_m == pytest.approx(abs(model.body_ipos[body(model, "link1")][2]))
  assert CONTRACT.cart_armature_kg == model.dof_armature[0]
  assert CONTRACT.actuator_force_limit_n == model.actuator_forcerange[0][1]
  assert CONTRACT.rail_limit_m == model.jnt_range[0][1]
  assert CONTRACT.gravity_m_s2 == -model.opt.gravity[2]
  assert CONTRACT.physics_timestep_s == model.opt.timestep
  assert CONTRACT.decimation * CONTRACT.physics_timestep_s == pytest.approx(CONTRACT.control_timestep_s)
  assert set(CONTRACT.plant_parameters()) == set(PLANT_PARAMETERS)


def edited_model(tmp_path, pattern, replacement):
  text = MODEL_PATH.read_text()
  edited, count = re.subn(pattern, replacement, text, count=1)
  assert count == 1, pattern
  path = tmp_path / "model.xml"
  path.write_text(edited)
  return path


def test_contract_follows_xml_edits(tmp_path):
  path = edited_model(tmp_path, r'(<inertial pos="0 0 0" mass=")0.26(")', r"\g<1>1.25\g<2>")
  contract = load_contract(path)
  assert contract.cart_mass_kg == 1.25
  assert contract.link1_length_m == CONTRACT.link1_length_m


def test_force_and_rail_limits_follow_xml(tmp_path):
  text = MODEL_PATH.read_text().replace('ctrlrange="-54.625 54.625"', 'ctrlrange="-25 25"').replace(
    'forcerange="-54.625 54.625"', 'forcerange="-25 25"').replace('range="-0.2115 0.2115"', 'range="-0.15 0.15"', 1)
  path = tmp_path / "model.xml"
  path.write_text(text)
  contract = load_contract(path)
  assert contract.actuator_force_limit_n == 25.0
  assert contract.rail_limit_m == 0.15


def test_elbow_hardware_mass_moves_the_link1_centre_of_mass(tmp_path):
  path = edited_model(tmp_path, r'(<body name="link2")', '<geom type="sphere" size="0.008" pos="0 0 -0.150" '
                      r'mass="0.012"/>\n        \g<1>')
  contract = load_contract(path)
  mass = CONTRACT.link1_mass_kg + 0.012
  assert contract.link1_mass_kg == pytest.approx(mass)
  assert contract.link1_com_m == pytest.approx((CONTRACT.link1_mass_kg * CONTRACT.link1_com_m + 0.012 * 0.150) / mass)
  assert contract.link1_length_m == CONTRACT.link1_length_m


def test_centre_of_mass_off_the_link_line_is_rejected(tmp_path):
  path = edited_model(tmp_path, r'(<geom name="link1_geom" type="box" size="0.010 0.0015 0.075" pos=)"0 0 -0.075"',
                      r'\g<1>"0.02 0 -0.075"')
  with pytest.raises(ValueError, match="off the line"):
    load_contract(path)


def test_asymmetric_force_range_is_rejected(tmp_path):
  text = MODEL_PATH.read_text().replace('forcerange="-54.625 54.625"', 'forcerange="-30 54.625"')
  path = tmp_path / "model.xml"
  path.write_text(text)
  with pytest.raises(ValueError, match="symmetric"):
    load_contract(path)
