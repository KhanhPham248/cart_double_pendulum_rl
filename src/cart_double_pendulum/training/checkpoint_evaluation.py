"""Checkpoint evaluator hook: installed by the launcher, called by the trainers."""
from pathlib import Path

_EVALUATOR = None


def install(evaluator) -> None:
  """Set ``evaluator(policy_path, task, episodes) -> metrics`` for this process."""
  global _EVALUATOR
  _EVALUATOR = evaluator


def evaluate_checkpoint(path: Path, task: str, episodes: int) -> dict:
  if _EVALUATOR is None:
    raise RuntimeError('No checkpoint evaluator installed; start training through tasks/real/<name>/run.py')
  return _EVALUATOR(path, task, episodes)
