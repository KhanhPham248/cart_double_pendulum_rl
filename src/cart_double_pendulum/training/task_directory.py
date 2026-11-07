"""Task directory loader: JSON files and reward terms; plant constants come from the MJCF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mjlab.managers import RewardTermCfg

from cart_double_pendulum.common import PROJECT_ROOT, CONTRACT, CartDoublePendulumContract
from cart_double_pendulum.plant_scales import plant_scales, resolve
from cart_double_pendulum.rl.sac import DEFAULT_DISCOUNT
from cart_double_pendulum.tasks import mdp, tip
from cart_double_pendulum.training import full_task_curriculum


_REWARD_MODULES = (full_task_curriculum, mdp, tip)


def read_json(path: str | Path, contract: CartDoublePendulumContract = CONTRACT) -> Any:
  """Load JSON, expanding ``${PROJECT_ROOT}`` and ``{"<scale>": multiple}`` plant-scale values."""

  text = Path(path).read_text().replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
  return resolve(json.loads(text), plant_scales(contract))


def function(name: str):
  """Resolve a reward or reset function name used in a task JSON file."""

  matches = [vars(module)[name] for module in _REWARD_MODULES
             if callable(vars(module).get(name)) and getattr(vars(module)[name], "__module__", None) == module.__name__]
  if len(matches) != 1:
    raise ValueError(f"task JSON function {name!r} matched {len(matches)} definitions")
  return matches[0]


def with_plant_parameters(func, params: dict, contract: CartDoublePendulumContract, where: str) -> dict:
  """Return ``params`` plus the plant constants that ``func`` declares."""

  names = getattr(func, "plant_parameters", ())
  clash = sorted(set(names) & set(params))
  if clash:
    raise ValueError(f"{where}: {clash} are read from the MJCF model; remove them from the JSON file")
  return {**params, **{name: getattr(contract, name) for name in names}}


# Task thresholds measured along the rail; each must lie inside the MJCF rail range.
RAIL_KEYS = ("rail_limit_m", "free_cart_m", "cart_safe_m", "capture_cart_m", "precision_cart_m")


def check_rail_thresholds(value: Any, contract: CartDoublePendulumContract, where: str) -> None:
  """Fail when a task threshold lies outside a (possibly shortened) MJCF rail."""

  if isinstance(value, dict):
    for key, item in value.items():
      if key in RAIL_KEYS and not 0.0 < float(item) < contract.rail_limit_m:
        raise ValueError(f"{where}: {key}={item} must lie inside the MJCF rail range +/-{contract.rail_limit_m} m")
      check_rail_thresholds(item, contract, where)
  elif isinstance(value, list):
    for item in value:
      check_rail_thresholds(item, contract, where)


def reward_terms(folder: Path, contract: CartDoublePendulumContract = CONTRACT) -> tuple[dict, bool]:
  """Build ``{name: RewardTermCfg}`` and ``scale_rewards_by_dt`` from reward.json."""

  folder = Path(folder)
  definition = read_json(folder / "reward.json", contract)
  for path in sorted(folder.glob("*.json")):
    check_rail_thresholds(read_json(path, contract), contract, f"{folder.name}/{path.name}")
  base_file = folder / "base_reward.json"
  if base_file.is_file():
    # Full tasks extend an expert reward; they must keep every base term as is.
    base = read_json(base_file, contract)["terms"]
    changed = sorted(key for key, value in base.items() if definition["terms"].get(key) != value)
    if changed:
      raise ValueError(f"{folder.name}: reward.json changes base reward terms {changed}")
  training = folder / "training.json"
  gamma = read_json(training, contract).get("gamma", DEFAULT_DISCOUNT) if training.is_file() else DEFAULT_DISCOUNT
  terms = {}
  for key, term in definition["terms"].items():
    where = f"{folder.name}/reward.json:{key}"
    func = function(term["function"])
    params = dict(term["params"])
    if "curriculum_file" in params:
      params["curriculum_file"] = str((folder / params["curriculum_file"]).resolve())
    params = with_plant_parameters(func, params, contract, where)
    if getattr(func, "uses_discount", False):
      if "discount" in params:
        raise ValueError(f"{where}: discount is the trainer's gamma; remove it from the JSON file")
      params["discount"] = gamma
    terms[key] = RewardTermCfg(func=func, weight=term["weight"], params=params)
  return terms, bool(definition["scale_rewards_by_dt"])
