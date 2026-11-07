"""Shared fixtures: import paths and small synthetic inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

TASKS = tuple(sorted(path.parent.name for path in (ROOT / "tasks/real").glob("*/task.json")))


@pytest.fixture
def synthetic_bank(tmp_path):
  """A small bank of near-upright states in the format the resets expect."""
  rng = np.random.default_rng(0)
  count = 64
  qpos = np.column_stack((rng.uniform(-.06, .06, count), np.pi + rng.uniform(-.2, .2, count), rng.uniform(-.2, .2, count)))
  qvel = rng.uniform(-.5, .5, (count, 3))
  path = tmp_path / "handoff_bank.npz"
  from cart_double_pendulum.plant_metadata import plant_metadata
  np.savez(path, plant_metadata=json.dumps(plant_metadata()), qpos=qpos, qvel=qvel, stage=np.ones(count, dtype=np.int8),
           episode_seed=np.arange(count), time_s=np.full(count, 5.0))
  return path
