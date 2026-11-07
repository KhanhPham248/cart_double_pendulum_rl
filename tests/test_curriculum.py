"""Evaluation-gated robustness curriculum: stages, ramps, pooled gates, no step back."""
import json

import pytest

from conftest import ROOT
from cart_double_pendulum import curriculum
from cart_double_pendulum.training import full_task_curriculum

BUDGET = 1000
RAMP = int(curriculum.RAMP_FRACTION * BUDGET)


@pytest.fixture(autouse=True)
def fresh():
  curriculum.start(BUDGET)
  yield
  curriculum._STATE = curriculum._State()


def evaluate(nominal, randomized, episodes=32):
  return curriculum.report(nominal, randomized, episodes)["stage"]


def test_inactive_curriculum_is_at_full_difficulty():
  curriculum._STATE = curriculum._State()
  assert curriculum.levels() == curriculum.Levels(1.0, 1.0, 1.0, 1.0)
  assert curriculum.evaluation_level() == 1.0


def test_nominal_gate_pools_two_evaluations():
  assert curriculum.levels() == curriculum.Levels(0.0, 0.0, 0.0, 0.0)
  assert evaluate(1.0, 0.0) == 1  # one evaluation is not enough
  assert evaluate(0.875, 0.0) == 1  # pooled 60/64 < 95%
  assert evaluate(1.0, 0.0) == 1  # pooled 60/64 again
  assert evaluate(1.0, 0.0) == 2  # pooled 64/64


def test_stages_ramp_then_gate_and_never_step_back():
  evaluate(1.0, 0.0)
  assert evaluate(1.0, 0.0) == 2
  curriculum.observe_step(RAMP // 2)
  assert curriculum.levels().domain == pytest.approx(curriculum.LIGHT_LEVEL / 2)
  assert curriculum.evaluation_level() == curriculum.LIGHT_LEVEL
  assert evaluate(1.0, 1.0) == 2 and evaluate(1.0, 1.0) == 2  # ramp not finished: not counted
  curriculum.observe_step(RAMP)
  evaluate(1.0, 0.95)
  assert evaluate(1.0, 0.9) == 3
  curriculum.observe_step(2 * RAMP)
  assert curriculum.levels().domain == 1.0 and curriculum.evaluation_level() == 1.0
  evaluate(1.0, 0.9)
  assert evaluate(1.0, 0.9) == 4
  assert curriculum.levels().push_amplitude == curriculum.PUSH_START
  curriculum.observe_step(3 * RAMP)
  assert curriculum.levels() == curriculum.Levels(1.0, 1.0, 1.0, 1.0)
  assert evaluate(0.0, 0.0) == 4


def test_benchmark_is_scored_at_full_level_while_the_gate_uses_the_stage_level():
  levels = []
  def run():
    levels.append(curriculum.evaluation_level())
    rate = 1.0 if levels[-1] < 1.0 else 0.25
    return {"nominal_success_rate": 1.0, "randomized_success_rate": rate, "cases": {"level": levels[-1]}}
  gate = lambda m: (m["nominal_success_rate"], m["randomized_success_rate"], None)
  metrics = curriculum.gate_and_benchmark(run, gate, 32)
  assert levels == [curriculum.LIGHT_LEVEL, 1.0] and curriculum.evaluation_level() == curriculum.LIGHT_LEVEL
  assert metrics["randomized_success_rate"] == 0.25 and metrics["benchmark_level"] == 1.0
  assert metrics["gate"] == {"level": curriculum.LIGHT_LEVEL, "cases": {"level": curriculum.LIGHT_LEVEL}}
  assert curriculum.state()["pooled"] == [(32.0, 32.0, 32)]  # the gate saw the stage-level result
  curriculum.start(BUDGET, first_stage=3)
  levels.clear()
  curriculum.gate_and_benchmark(run, gate, 32)
  assert levels == [1.0]  # already at full level: evaluated once


def test_full_schedule_rejects_robustness_columns(tmp_path):
  data = json.loads((ROOT / "tasks/real/full_ppo/curriculum.json").read_text())
  for point in data["schedule"]:
    point["domain_fraction"] = 0.0
  path = tmp_path / "curriculum.json"
  path.write_text(json.dumps(data))
  with pytest.raises(ValueError, match="cart_double_pendulum.curriculum"):
    full_task_curriculum.load_curriculum(str(path))


def test_centring_rate_is_also_required():
  curriculum.report(1.0, 0.0, 32, (0.4, 0.0))
  assert curriculum.report(1.0, 0.0, 32, (0.5, 0.0))["stage"] == 1  # pooled centring 0.45 < 0.5
  assert curriculum.report(1.0, 0.0, 32, (0.6, 0.0))["stage"] == 2


def test_final_stage_stops_before_pushes():
  curriculum.start(BUDGET, final_stage=3)
  evaluate(1.0, 0.0)
  evaluate(1.0, 0.0)
  curriculum.observe_step(RAMP)
  evaluate(1.0, 1.0)
  assert evaluate(1.0, 1.0) == 3
  curriculum.observe_step(2 * RAMP)
  evaluate(1.0, 1.0)
  assert evaluate(1.0, 1.0) == 3
  assert curriculum.levels() == curriculum.Levels(1.0, 1.0, 0.0, 0.0)


def test_expert_cannot_be_forced_into_randomization_after_failed_evaluations():
  curriculum.observe_step(BUDGET * 10)
  for _ in range(4):
    evaluate(0., 0.)
  assert curriculum.stage() == 1
  assert curriculum.state()['history'] == []
  assert curriculum.levels() == curriculum.Levels(0., 0., 0., 0.)


def test_expert_reset_ramp_cannot_grow_while_nominal_skill_is_zero():
  from cart_double_pendulum.tasks.mdp import _ramp_progress
  curriculum.observe_step(BUDGET)
  assert _ramp_progress(.15) == 0.
  evaluate(1., 0.)
  evaluate(1., 0.)
  assert _ramp_progress(.15) == pytest.approx(1/3)


def test_ramp_rounding_still_counts_the_evaluation():
  evaluate(1.0, 0.0)
  evaluate(1.0, 0.0)
  curriculum.observe_step(RAMP - 1)
  evaluate(1.0, 1.0)
  assert evaluate(1.0, 1.0) == 3




def test_collapse_rolls_back_and_pauses_every_difficulty_clock(tmp_path):
  first = curriculum.report(0.6, 0.0, 32)
  checkpoint = tmp_path / "step_1.pt"
  checkpoint.write_text("good")
  assert first["best_skill"] and curriculum.skill_checkpoint(first, checkpoint) is None
  assert not curriculum.report(0.2, 0.0, 32)["rollback"]  # one bad evaluation is noise
  collapsed = curriculum.report(0.2, 0.0, 32)
  assert collapsed["rollback"] and collapsed["paused"]
  assert curriculum.skill_checkpoint(collapsed, tmp_path / "step_2.pt") == tmp_path / "best_skill.pt"
  curriculum.observe_step(150)  # paused: the difficulty clock stands still
  assert curriculum.clock_fraction() == 0.0
  assert not curriculum.report(0.5, 0.0, 32)["paused"]  # 0.5 >= 0.75 * 0.6
  curriculum.observe_step(400)
  assert curriculum.clock_fraction() == .25  # exclude the 150 paused steps
  assert curriculum.stage() == 1
  evaluate(1., 0.)
  assert evaluate(1., 0.) == 2  # only a pooled skill gate advances the stage


def test_pause_is_capped():
  curriculum.report(0.6, 0.0, 32)
  curriculum.report(0.1, 0.0, 32)
  assert curriculum.report(0.1, 0.0, 32)["rollback"]
  curriculum.observe_step(BUDGET)
  assert not curriculum.state()["paused"]
  assert curriculum.state()["paused_steps"] == int(curriculum.MAX_PAUSE * BUDGET)


def full_schedule():
  data = full_task_curriculum.load_curriculum(str(ROOT / 'tasks/real/full_sac/curriculum.json'))
  curriculum.configure_full_schedule([p['fraction'] for p in data['schedule']], data['skill_gates'])
  return data


def test_full_difficulty_cannot_advance_on_elapsed_time():
  data = full_schedule()
  curriculum.observe_step(BUDGET * 10)
  for _ in range(4):
    curriculum.report(0., 0., 32, (0., 0.))
  assert curriculum.stage() == 1
  assert curriculum.full_progress() == 0.
  from types import SimpleNamespace
  env = SimpleNamespace(common_step_counter=BUDGET * 10)
  settings = full_task_curriculum.settings(env, ROOT / 'tasks/real/full_sac/curriculum.json')
  assert settings == data['schedule'][0]


def test_full_schedule_needs_one_evaluation_per_gate_and_centring_for_later_levels():
  data = full_schedule()
  for index, gate in enumerate(data['skill_gates']):
    assert gate['evaluations'] == 1
    if gate['centre_min']:
      curriculum.report(1., 1., 32, (0., 0.))
      assert curriculum.full_progress() == data['schedule'][index]['fraction']
    curriculum.report(1., 1., 32, (1., 1.))
    assert curriculum.full_progress() == data['schedule'][index + 1]['fraction']


def test_robustness_does_not_wait_for_the_full_schedule():
  full_schedule()
  curriculum.report(1., 1., 32, (1., 1.))
  assert curriculum.report(1., 1., 32, (1., 1.))['stage'] == 2
  assert curriculum.full_progress() < 1.


def test_resumed_environment_counter_keeps_the_clock_running():
  curriculum.start(BUDGET)
  curriculum.observe_step(300)
  curriculum.observe_step(300)
  curriculum.observe_step(100)  # new environment after a resume starts from zero
  assert curriculum.state()['control_step'] == 400


def test_full_skill_regression_pauses_schedule():
  full_schedule()
  curriculum.report(.8, 0., 32, (0., 0.))
  curriculum.report(0., 0., 32, (0., 0.))
  assert curriculum.report(0., 0., 32, (0., 0.))['paused']
  previous = curriculum.full_progress()
  curriculum.observe_step(BUDGET)
  assert curriculum.full_progress() == previous


def test_full_schedule_rejects_ungated_time_configuration(tmp_path):
  data = json.loads((ROOT / 'tasks/real/full_sac/curriculum.json').read_text())
  data['progress_units'] = 'fraction_of_requested_control_steps'
  path = tmp_path / 'curriculum.json'
  path.write_text(json.dumps(data))
  with pytest.raises(ValueError, match='skill_gated_schedule'):
    full_task_curriculum.load_curriculum(str(path))
