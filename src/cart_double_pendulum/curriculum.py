"""Evaluation-gated robustness curriculum (README, Robustness curriculum); inactive = full difficulty."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import shutil
import math

LIGHT_LEVEL = 0.5  # stage-2 randomization, as a fraction of the evaluator ranges
LINK_NOISE_STAGE3_MULTIPLIER = 1.5  # final angle/rate noise, as a multiple of stage 2
PUSH_START = 0.25  # first push amplitude and frequency, as a fraction of the target
RAMP_FRACTION = 0.10  # of the training budget, per stage ramp
RAMP_DONE = 0.99  # ramp counted as finished; absorbs step rounding at evaluations
POOLED_EVALUATIONS = 2
# Gate to leave each stage: (evaluation case, pooled success rate).
GATES = {1: ("nominal", 0.95), 2: ("randomized", 0.90), 3: ("randomized", 0.90)}
CENTRE_MINIMUM = 0.5  # pooled centring rate also required where a task reports one
FINAL_STAGE = 4
# Skill guard: nominal success below ROLLBACK_DROP x best (best >= ROLLBACK_FLOOR) for PATIENCE evaluations
# restores the best-skill checkpoint and pauses every difficulty clock until RECOVERED x best.
ROLLBACK_DROP, ROLLBACK_FLOOR, RECOVERED, PATIENCE, MAX_ROLLBACKS = 0.5, 0.25, 0.75, 2, 3
MAX_PAUSE = 0.20  # of the budget, in total


@dataclass(frozen=True)
class Levels:
  """Fractions of the target randomization, sensor noise and push amplitude and frequency."""
  domain: float
  noise: float
  push_amplitude: float
  push_frequency: float


@dataclass
class _State:
  active: bool = False
  stage: int = FINAL_STAGE
  final_stage: int = FINAL_STAGE
  budget_control_steps: int = 1
  step: int = 0
  counter: int = 0
  entered_at: int = 0
  paused: bool = False
  paused_steps: int = 0
  best_skill: float = 0.0
  low: int = 0
  rollbacks: int = 0
  pool: list = field(default_factory=list)
  history: list = field(default_factory=list)
  full_fractions: tuple = ()
  full_gates: tuple = ()
  full_index: int = 0
  full_pool: list = field(default_factory=list)
  full_last_scores: dict = field(default_factory=dict)


_STATE = _State()
_BENCHMARK = False


def start(budget_control_steps: int, final_stage: int = FINAL_STAGE, first_stage: int = 1) -> None:
  """Begin training at ``first_stage``; ramps last ``RAMP_FRACTION`` of this budget."""
  global _STATE
  if budget_control_steps <= 0 or not 1 <= first_stage <= final_stage <= FINAL_STAGE:
    raise ValueError("the curriculum needs a positive budget and 1 <= first stage <= final stage <= 4")
  _STATE = _State(active=True, stage=first_stage, final_stage=final_stage,
                  budget_control_steps=int(budget_control_steps))


def _clock() -> int:
  """Control steps of difficulty progress: training steps minus paused ones."""
  return _STATE.step - _STATE.paused_steps


def configure_full_schedule(fractions, gates) -> None:
  """Full reset/reward difficulty advances only after pooled nominal skill gates."""
  if not _STATE.active:
    raise RuntimeError('Start the training curriculum before configuring full skill gates')
  if len(gates) != len(fractions) - 1:
    raise ValueError('Each full schedule transition needs a skill gate')
  _STATE.full_fractions = tuple(fractions)
  _STATE.full_gates = tuple(dict(gate) for gate in gates)
  _STATE.full_index, _STATE.full_pool, _STATE.full_last_scores = 0, [], {}


def full_progress() -> float:
  """Unlocked full-task schedule position; elapsed time cannot advance it."""
  if not _STATE.active:
    return 1.0
  return _STATE.full_fractions[_STATE.full_index] if _STATE.full_fractions else 0.0


def _report_full(capture, centre, episodes):
  if not _STATE.full_fractions:
    return
  _STATE.full_last_scores = {'capture': capture, 'centre': centre}
  if _STATE.paused or _STATE.full_index == len(_STATE.full_gates):
    _STATE.full_pool = []
    return
  gate = _STATE.full_gates[_STATE.full_index]
  _STATE.full_pool = (_STATE.full_pool + [(capture * episodes, centre * episodes, episodes)])[-gate['evaluations']:]
  captured, centred, total = (sum(values) for values in zip(*_STATE.full_pool))
  if (len(_STATE.full_pool) == gate['evaluations']
      and captured >= gate['capture_min'] * total - 1e-9
      and centred >= gate['centre_min'] * total - 1e-9):
    _STATE.full_index += 1
    _STATE.full_pool = []
    _STATE.history.append({'full_schedule_index': _STATE.full_index,
                           'full_progress': full_progress(), 'control_step': _STATE.step})


def clock_fraction(inactive: float = 1.0) -> float:
  """Difficulty clock as a fraction of the budget; ``inactive`` when no curriculum runs."""
  return min(1.0, _clock() / _STATE.budget_control_steps) if _STATE.active else inactive


def reset_difficulty_cap() -> float:
  """Expert/meta reset ramps cannot outrun the robustness skills already earned."""
  if not _STATE.active:
    return 1.0
  return (_STATE.stage - 1) / max(1, _STATE.final_stage - 1)


def _advance() -> None:
  _STATE.stage += 1
  _STATE.entered_at = _clock()
  _STATE.pool = []
  _STATE.history.append({"stage": _STATE.stage, "control_step": _STATE.step})


def observe_step(counter: int) -> None:
  """Advance the difficulty clock by the environment's step counter; elapsed budget never unlocks a stage.

  A counter below the last one is a new environment (a resumed run) restarting from zero.
  """
  counter = int(counter)
  advance = counter - _STATE.counter if counter >= _STATE.counter else counter
  _STATE.counter = counter
  if _STATE.paused:
    cap = int(MAX_PAUSE * _STATE.budget_control_steps)
    _STATE.paused_steps = min(cap, _STATE.paused_steps + advance)
    _STATE.paused = _STATE.paused_steps < cap
  _STATE.step += advance


def _ramp() -> float:
  span = RAMP_FRACTION * _STATE.budget_control_steps
  return min(1.0, max(0.0, (_clock() - _STATE.entered_at) / span))


def levels() -> Levels:
  if not _STATE.active:
    return Levels(1.0, 1.0, 1.0, 1.0)
  ramp = _ramp()
  if _STATE.stage == 1:
    return Levels(0.0, 0.0, 0.0, 0.0)
  if _STATE.stage == 2:
    return Levels(LIGHT_LEVEL * ramp, LIGHT_LEVEL * ramp, 0.0, 0.0)
  if _STATE.stage == 3:
    level = LIGHT_LEVEL + (1.0 - LIGHT_LEVEL) * ramp
    return Levels(level, level, 0.0, 0.0)
  push = PUSH_START + (1.0 - PUSH_START) * ramp
  return Levels(1.0, 1.0, push, push)


def evaluation_level() -> float:
  """Randomization level of the evaluators' randomized cases; the benchmark always uses 1."""
  return LIGHT_LEVEL if _STATE.active and _STATE.stage <= 2 and not _BENCHMARK else 1.0


@contextmanager
def benchmark():
  """Evaluate at the full randomization level whatever the stage."""
  global _BENCHMARK
  previous, _BENCHMARK = _BENCHMARK, True
  try:
    yield
  finally:
    _BENCHMARK = previous


def gate_and_benchmark(evaluate, gate, episodes: int) -> dict:
  """Gate the curriculum at the stage's level; return the fixed full-level benchmark, which ranks checkpoints.

  ``evaluate()`` returns metrics; ``gate(metrics)`` returns (nominal, randomized, centre) for ``report``.
  """
  level = evaluation_level()
  staged = evaluate()
  if level == 1.0:
    metrics = staged
  else:
    with benchmark():
      metrics = evaluate()
  metrics = {**metrics, "benchmark_level": 1.0,
             "gate": {"level": level, "cases": staged["cases"]}}
  nominal, randomized, centre = gate(staged)
  metrics["curriculum"] = report(nominal, randomized, episodes, centre)
  return metrics


def link_noise_level(noise_level: float) -> float:
  """Keep stage 2 unchanged; ramp link sensors to 1.5x stage 2 at stage 3."""
  if not math.isfinite(noise_level) or not 0.0 <= noise_level <= 1.0:
    raise ValueError('sensor noise curriculum level must lie in [0, 1]')
  if noise_level <= LIGHT_LEVEL:
    return noise_level
  progress = (noise_level - LIGHT_LEVEL) / (1.0 - LIGHT_LEVEL)
  return LIGHT_LEVEL * (1.0 + (LINK_NOISE_STAGE3_MULTIPLIER - 1.0) * progress)


def scaled_range(scale_range: tuple[float, float], level: float) -> tuple[float, float]:
  return tuple(1.0 + level * (value - 1.0) for value in scale_range)


def report(nominal_success_rate: float, randomized_success_rate: float, episodes: int,
           centre: tuple[float, float] | None = None) -> dict:
  """Record one evaluation; the result's ``best_skill``/``rollback`` tell the trainer to keep/restore a checkpoint."""
  if _STATE.active and _STATE.stage < _STATE.final_stage and (_STATE.stage == 1 or _ramp() >= RAMP_DONE):
    case, threshold = GATES[_STATE.stage]
    rate = nominal_success_rate if case == "nominal" else randomized_success_rate
    centred = 1.0 if centre is None else centre[case == "randomized"]
    _STATE.pool = (_STATE.pool + [(rate * episodes, centred * episodes, episodes)])[-POOLED_EVALUATIONS:]
    successes, centred, total = (sum(values) for values in zip(*_STATE.pool))
    if (len(_STATE.pool) == POOLED_EVALUATIONS and successes >= threshold * total - 1e-9
        and centred >= CENTRE_MINIMUM * total - 1e-9):
      _advance()
  best, rollback = _STATE.active and nominal_success_rate > _STATE.best_skill, False
  if _STATE.active:
    _STATE.best_skill = max(_STATE.best_skill, nominal_success_rate)
    collapsed = _STATE.best_skill >= ROLLBACK_FLOOR and nominal_success_rate < ROLLBACK_DROP * _STATE.best_skill
    _STATE.low = _STATE.low + 1 if collapsed else 0
    if _STATE.low >= PATIENCE and _STATE.rollbacks < MAX_ROLLBACKS:
      rollback, _STATE.low, _STATE.paused = True, 0, True
      _STATE.rollbacks += 1
      _STATE.history.append({"rollback": _STATE.rollbacks, "control_step": _STATE.step})
    elif _STATE.paused and nominal_success_rate >= RECOVERED * _STATE.best_skill:
      _STATE.paused = False
    if centre is not None:
      _report_full(nominal_success_rate, centre[0], episodes)
  return {**state(), "best_skill": best, "rollback": rollback}


def state() -> dict:
  current = levels()
  return {"active": _STATE.active, "stage": _STATE.stage,
          "final_stage": _STATE.final_stage,
          "control_step": _STATE.step, "environment_counter": _STATE.counter,
          "paused": _STATE.paused, "paused_steps": _STATE.paused_steps,
          "best_nominal_skill": _STATE.best_skill, "rollbacks": _STATE.rollbacks,
          "entered_at": _STATE.entered_at, "pooled": list(_STATE.pool),
          "full_schedule": {'active': bool(_STATE.full_fractions), 'index': _STATE.full_index,
                            'progress': full_progress(), 'pooled': list(_STATE.full_pool),
                            'scores': dict(_STATE.full_last_scores)},
          "levels": asdict(current), "evaluation_level": evaluation_level(),
          "history": list(_STATE.history)}


def stage() -> int:
  return _STATE.stage


def skill_checkpoint(result: dict, checkpoint) -> Path | None:
  """Keep ``checkpoint`` as best_skill.pt when it is the best so far; return that file when a rollback is due."""
  kept = Path(checkpoint).parent / "best_skill.pt"
  if result.get("best_skill"):
    shutil.copy2(checkpoint, kept)
  return kept if result.get("rollback") and kept.is_file() else None
