"""Plant sweep: on edited MJCF copies the plant stays feasible and every reward rule still holds."""
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT

RULES = "failure_costs or success_anywhere or reward_discounts or scales_of_the_model"


def scale_attribute(text, name, key, factor):
  """Multiply every number of attribute ``key`` on the element named ``name``."""
  def scaled(match):
    values = " ".join(f"{float(v) * factor:.6g}" for v in match.group(2).split())
    return f'{match.group(1)}{values}"'
  edited = re.sub(rf'(<\w+ name="{name}"[^>]*?\b{key}=")([^"]*)"', scaled, text, count=1, flags=re.S)
  assert edited != text, (name, key)
  return edited


def longer_links(text, factor):
  for name in ("link1_geom", "link2_geom"):
    text = re.sub(rf'(<geom name="{name}"[^>]*?size="\S+ \S+ )(\S+)"',
                  lambda m: f'{m.group(1)}{float(m.group(2)) * factor:.6g}"', text, count=1, flags=re.S)
  for name in ("link1_geom", "link2_geom", "elbow_hardware", "link2", "tip"):
    text = scale_attribute(text, name, "pos", factor)
  return text


VARIANTS = {
  "rail x0.5": lambda t: scale_attribute(t, "cart_slide", "range", 0.5),
  "force x0.5": lambda t: scale_attribute(scale_attribute(t, "cart_motor", "forcerange", 0.5), "cart_motor", "ctrlrange", 0.5),
  "damping x2": lambda t: scale_attribute(t, "cart_slide", "damping", 2.0),
  "links x1.5": lambda t: longer_links(t, 1.5),
  "link mass x3": lambda t: scale_attribute(scale_attribute(t, "link1_geom", "mass", 3.0), "link2_geom", "mass", 3.0),
}


@pytest.mark.parametrize("name", VARIANTS)
def test_plant_variant_keeps_the_rules(tmp_path, name):
  for folder in ("src", "tasks", "tests", "assets", "scripts"):
    shutil.copytree(ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__"))
  model = tmp_path / "assets/cart_double_pendulum_real.xml"
  model.write_text(VARIANTS[name](model.read_text()))
  env = {**os.environ, "PYTHONPATH": str(tmp_path / "src")}
  report = subprocess.run([sys.executable, "-c", "from cart_double_pendulum.plant_scales import plant_report; "
                           "r = plant_report(); assert not r['warnings'], r"], cwd=tmp_path, env=env,
                          capture_output=True, text=True)
  assert report.returncode == 0, report.stderr[-2000:]
  rules = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_environment.py",
                          "tests/test_plant_scales.py", "-k", RULES], cwd=tmp_path, env=env, capture_output=True, text=True)
  assert rules.returncode == 0, rules.stdout[-3000:]
