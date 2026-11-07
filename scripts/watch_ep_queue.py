#!/usr/bin/env python3
"""Persistent EP queue with a cap including existing processes on the physical GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def gpu_pids(gpu):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = dict(line.strip().split(', ', 1) for line in rows.splitlines())[str(gpu)]
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    return {int(pid) for line in rows.splitlines() for device, pid in [line.strip().split(', ', 1)] if device == uuid}


def save(path, state):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--max-tasks', type=int, default=2)
    parser.add_argument('--poll-seconds', type=float, default=20)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=['tip_swing_sac', *(f'sac_ep{i}' for i in range(4))],
                        help='ordered fresh-run task list; default EP0..EP3')
    args = parser.parse_args()
    names = args.tasks or [f'sac_ep{i}' for i in range(4)]
    if len(set(names)) != len(names):
        parser.error('task list must not contain duplicates')
    if args.max_tasks < 1 or args.poll_seconds <= 0:
        parser.error('positive max-tasks and poll-seconds required')
    root = Path(__file__).resolve().parents[1]
    directory = args.state_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / f'gpu{args.gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = directory / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else dict(gpu=args.gpu, max_tasks=args.max_tasks,
        tasks=[dict(name=name, status='pending') for name in names])
    if state['gpu'] != args.gpu or state['max_tasks'] != args.max_tasks:
        raise ValueError('queue state GPU/cap differs from requested configuration')
    if args.tasks is not None and [task['name'] for task in state['tasks']] != names:
        raise ValueError('task order differs from persisted queue')
    children = {}
    while True:
        for task in state['tasks']:
            if task['status'] != 'running':
                continue
            pid = task['pid']
            code = children[pid].poll() if pid in children else None
            if task['start_tick'] is not None and identity(pid) == task['start_tick'] and code is None:
                continue
            complete = 'TRAIN_COMPLETE run_dir=' in Path(task['log']).read_text(errors='replace')
            task.update(status='complete' if complete and code in (None, 0) else 'failed',
                        returncode=code, finished_at=time.time())
            print(f"FINISH {task['name']} {task['status']} code={code}", flush=True)
        save(path, state)
        if all(t['status'] in ('complete', 'failed') for t in state['tasks']):
            print('QUEUE_COMPLETE', flush=True)
            return
        try:
            occupied = gpu_pids(args.gpu)
        except (subprocess.SubprocessError, ValueError, KeyError) as exc:
            print(f'GPU_QUERY_FAILED {exc}; no launch', flush=True)
            time.sleep(args.poll_seconds)
            continue
        occupied.update(t['pid'] for t in state['tasks'] if t['status'] == 'running')
        state.update(observed_gpu_pids=sorted(occupied), checked_at=time.time())
        pending = next((t for t in state['tasks'] if t['status'] == 'pending'), None)
        if pending is not None and len(occupied) < args.max_tasks:
            log_path = directory / (pending['name'] + '.log')
            if log_path.exists():
                raise FileExistsError(f'Refusing to overwrite {log_path}')
            environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONPATH=str(root / 'src'))
            environment.pop('CUDA_VISIBLE_DEVICES', None)
            command = [sys.executable, '-u', str(root / 'tasks/real' / pending['name'] / 'run.py'),
                       'train', '--device', f'cuda:{args.gpu}']
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[child.pid] = child
            pending.update(status='running', pid=child.pid, start_tick=identity(child.pid),
                log=str(log_path), command=command, started_at=time.time())
            save(path, state)
            print(f"START {pending['name']} pid={child.pid} gpu={args.gpu} occupied_before={sorted(occupied)}", flush=True)
        save(path, state)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
