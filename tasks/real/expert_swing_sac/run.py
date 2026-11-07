"""python run.py [inspect|train|train-observer|evaluate] ...; the task is this directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from control_tasks import task_main

if __name__ == '__main__':
    task_main(Path(__file__).resolve().parent.name)
