"""Every task directory is complete, consistent and free of plant constants."""
import inspect
import json

import pytest

from conftest import ROOT, TASKS
from cart_double_pendulum.common import CONTRACT, PLANT_PARAMETERS
from cart_double_pendulum.training import task_directory
import control_tasks

LEGACY_PLANT_KEYS = {"gravity", "link1_hinge_inertia_kg_m2", "link2_hinge_inertia_kg_m2"}


def walk(value):
  if isinstance(value, dict):
    for key, item in value.items():
      yield key
      yield from walk(item)
  elif isinstance(value, list):
    for item in value:
      yield from walk(item)


@pytest.mark.parametrize("name", TASKS)
def test_task_files_exist(name):
  folder = ROOT / "tasks/real" / name
  spec = json.loads((folder / "task.json").read_text())
  assert spec["name"] == name
  for key in ("environment_file", "evaluation_file", "curriculum_file", "observer_file", "optimization_file"):
    if key in spec:
      assert (folder / spec[key]).is_file(), key
  assert (folder / "run.py").is_file()


@pytest.mark.parametrize("name", TASKS)
def test_json_files_contain_no_plant_constants(name):
  for path in (ROOT / "tasks/real" / name).glob("*.json"):
    keys = set(walk(json.loads(path.read_text())))
    assert not keys & (set(PLANT_PARAMETERS) | LEGACY_PLANT_KEYS), path


@pytest.mark.parametrize("name", TASKS)
def test_rewards_resolve_and_receive_plant_constants(name):
  terms, _ = task_directory.reward_terms(ROOT / "tasks/real" / name)
  assert terms
  for term in terms.values():
    for parameter in getattr(term.func, "plant_parameters", ()):
      assert term.params[parameter] == getattr(CONTRACT, parameter)


@pytest.mark.parametrize("name", TASKS)
def test_reset_function_resolves(name):
  folder = ROOT / "tasks/real" / name
  reset = task_directory.read_json(folder / "environment.json")["reset_system"]
  func = task_directory.function(reset["function"])
  parameters = inspect.signature(func).parameters
  assert set(reset["params"]) <= set(parameters)


@pytest.mark.parametrize("name", TASKS)
def test_training_options_are_accepted_by_trainer(name):
  _, spec, values = control_tasks.load(name)
  values.pop("handoff_bank", None)
  script = ROOT / "scripts" / f"train_{spec['algorithm']}.py"
  flags = control_tasks.script_flags(script)
  assert all("--" + key.replace("_", "-") in flags for key in values)


def test_json_rejects_plant_constant(tmp_path):
  folder = tmp_path / "task"
  folder.mkdir()
  (folder / "reward.json").write_text(json.dumps({"scale_rewards_by_dt": False, "terms": {
    "energy": {"function": "swing_energy_target", "weight": 1.0, "params": {"gravity_m_s2": 9.81}}}}))
  with pytest.raises(ValueError, match="MJCF"):
    task_directory.reward_terms(folder)


def test_rail_thresholds_must_fit_the_mjcf_rail(tmp_path):
  folder = tmp_path / "task"
  folder.mkdir()
  (folder / "reward.json").write_text(json.dumps({"scale_rewards_by_dt": False, "terms": {
    "rail": {"function": "rail_terminal_cost", "weight": -1.0, "params": {"rail_limit_m": CONTRACT.rail_limit_m + .1}}}}))
  with pytest.raises(ValueError, match="rail range"):
    task_directory.reward_terms(folder)


def test_full_process_methods_share_one_evaluation():
  files = [(ROOT / "tasks/real" / name / "evaluation.json").read_text() for name in ("full_ppo", "full_sac")]
  assert files[0] == files[1]


def test_full_fall_penalty_uses_the_evaluator_thresholds():
  evaluation = json.loads((ROOT / "tasks/real/full_ppo/evaluation.json").read_text())
  for name in ("full_ppo", "full_sac"):
    params = json.loads((ROOT / "tasks/real" / name / "reward.json").read_text())["terms"]["full_fall"]["params"]
    assert params == {key: evaluation[key] for key in params}


def test_full_tasks_extend_the_swing_expert_reward_with_one_shared_reward():
  swing = json.loads((ROOT / "tasks/real/expert_swing_sac/reward.json").read_text())["terms"]
  for name in ("full_ppo", "full_sac"):
    base = json.loads((ROOT / "tasks/real" / name / "base_reward.json").read_text())["terms"]
    assert base == {key: swing[key] for key in base}
  rewards = [json.loads((ROOT / "tasks/real" / name / "reward.json").read_text()) for name in ("full_ppo", "full_sac")]
  assert rewards[0] == rewards[1]


def test_every_noisy_task_draws_one_noise_level_per_episode_from_the_shared_ranges():
  from cart_double_pendulum.tasks.config import DAMPING_SCALE_RANGE, FORCE_SCALE_RANGE, INERTIA_SCALE_RANGE
  for name in TASKS:
    training = json.loads((ROOT / "tasks/real" / name / "training.json").read_text())
    if training.get("actor_noise_max_std"):
      assert training["actor_noise_max_std"] == 0.02 and training.get("actor_noise_per_episode") is True, name
  ranges = json.loads((ROOT / "tasks/real/full_ppo/evaluation.json").read_text())["randomization"]
  assert (tuple(ranges["mass_scale"]), tuple(ranges["damping_scale"]), tuple(ranges["force_scale"]),
          tuple(ranges["observation_noise_std"])) == (INERTIA_SCALE_RANGE, DAMPING_SCALE_RANGE, FORCE_SCALE_RANGE, (0, 0.02))


def test_launcher_lists_every_task_directory():
  assert set(control_tasks.NAMES) == set(TASKS)
