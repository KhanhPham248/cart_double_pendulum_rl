"""python scripts/reward_terms.py <run>/train [--points N]: per-term reward means (steps: SAC transitions, PPO iterations)."""
import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _step(value: int) -> str:
  return f"{value / 1e6:.2f}M" if value >= 1e6 else f"{value / 1e3:.0f}k" if value >= 1e3 else str(value)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run", type=Path)
  parser.add_argument("--points", type=int, default=5, help="columns, evenly spaced over training")
  args = parser.parse_args()
  folders = sorted({path.parent for path in args.run.rglob("events.out.tfevents*")})
  series = {}
  for folder in folders:
    events = EventAccumulator(str(folder), size_guidance={"scalars": 0})
    events.Reload()
    for tag in events.Tags()["scalars"]:
      if tag.startswith("reward_terms/"):
        series[tag.split("/", 1)[1]] = events.Scalars(tag)
  if not series:
    raise SystemExit(f"no reward_terms/* in {args.run}; train with this version first")
  length = min(len(values) for values in series.values())
  columns = sorted({round(i * (length - 1) / max(1, args.points - 1)) for i in range(args.points)})
  first = next(iter(series.values()))
  print(f"{'term':34s}" + "".join(f"{_step(first[i].step):>11s}" for i in columns))
  for name, values in sorted(series.items(), key=lambda item: -abs(item[1][length - 1].value)):
    print(f"{name:34s}" + "".join(f"{values[i].value:>11.3f}" for i in columns))
  totals = [sum(values[i].value for values in series.values()) for i in columns]
  print(f"{'total':34s}" + "".join(f"{value:>11.3f}" for value in totals))


if __name__ == "__main__":
  main()
