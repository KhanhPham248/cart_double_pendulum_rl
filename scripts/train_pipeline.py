#!/usr/bin/env python3
"""Snapshot and queue the complete training pipeline on one physical GPU."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
# Dependency pipeline only; independent TIP/EP tasks use their own launchers.
TASKS = ('expert_swing_sac', 'expert_balance_ppo', 'full_ppo', 'full_sac')


def now():
    return datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid):
    """Linux PID plus start tick, so a reused PID cannot hold the queue forever."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_tick': fields[19],
                'command': Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return None


def same_process(record):
    current = process_identity(record['pid'])
    return current is not None and current['start_tick'] == record['start_tick']


def gpu_processes(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,gpu_uuid',
                             '--format=csv,noheader'], check=True, capture_output=True, text=True, timeout=15)
    return [int(line.split(',')[0]) for line in result.stdout.splitlines() if line.strip()]


class CommandFailure(RuntimeError):
    def __init__(self, name, code, log):
        super().__init__(f'{name} exited {code}; see {log}')
        self.code, self.log = code, log


def transient_failure(error):
    """Only retry recognised startup/resource errors, never learning or code failures."""
    if not isinstance(error, CommandFailure):
        return False
    with error.log.open('rb') as f:
        f.seek(max(0, error.log.stat().st_size - 32768))
        tail = f.read().decode(errors='replace').lower()
    return any(message in tail for message in (
        'cuda error: initialization error', 'cuda driver initialization failed',
        'cuda-capable device(s) is/are busy or unavailable', 'resource temporarily unavailable'))


def plan():
    stages = []

    def add(name, kind, task=None, dependencies=(), source=None, extra=()):
        stages.append(dict(name=name, kind=kind, task=task, dependencies=list(dependencies),
                           source=source, extra=list(extra), status='pending'))

    add('swing_initial', 'train', 'expert_swing_sac')
    add('build_bank', 'bank', dependencies=('swing_initial',), source='swing_initial')
    add('swing_with_bank', 'train', 'expert_swing_sac', ('build_bank',))
    add('swing_observer', 'observer', 'expert_swing_sac', ('swing_with_bank',), 'swing_with_bank')
    add('validate_swing', 'validate', 'expert_swing_sac', ('swing_with_bank',), 'swing_with_bank')
    add('balance_stage1', 'train', 'expert_balance_ppo', ('build_bank',), extra=('--without-observer',))
    add('balance_observer', 'observer', 'expert_balance_ppo', ('balance_stage1',), 'balance_stage1')
    add('freeze_balance_observer', 'freeze_observer', dependencies=('balance_observer',), source='balance_stage1')
    add('balance_stage2', 'train', 'expert_balance_ppo', ('freeze_balance_observer',))
    add('validate_balance', 'validate', 'expert_balance_ppo', ('balance_stage2',), 'balance_stage2')
    for task in ('full_ppo', 'full_sac'):
        add(task, 'train', task, ('build_bank',))
        add('validate_' + task, 'validate', task, (task,), task)
    return stages


def prepare(directory, gpu, wait_pids, episodes):
    directory.mkdir(parents=True, exist_ok=False)
    project = directory / 'project'
    project.mkdir()
    # Copy only source/configuration. Old policies, banks and results cannot leak in.
    for name in ('src', 'scripts', 'tasks', 'assets', 'tests'):
        shutil.copytree(ROOT / name, project / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pt', '*.onnx', '*.npz'))
    for name in ('README.md', 'pyproject.toml', 'requirements.txt', 'LICENSE', 'NOTICE'):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, project / name)
    hashes = {str(p.relative_to(project)): digest(p) for p in sorted(project.rglob('*')) if p.is_file()}
    wait_for = [r for pid in wait_pids if (r := process_identity(pid)) is not None]
    state = dict(status='prepared', created_at=now(), original_project=str(ROOT), project=str(project),
                 physical_gpu=gpu, validation_episodes=episodes, validation_seed=281028001,
                 wait_for=wait_for, source_sha256=hashes, stages=plan(),
                 fresh_training=True, reward_and_budgets_unchanged=True)
    save(directory / 'status.json', state)
    return state


class Pipeline:
    def __init__(self, directory):
        self.directory = directory
        self.state = read(directory / 'status.json')
        self.project = Path(self.state['project'])
        self.stages = {s['name']: s for s in self.state['stages']}
        self.child = None
        self.environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.state['physical_gpu']),
                                PYTHONPATH=str(self.project / 'src'), PYTHONNOUSERSITE='1',
                                PYTHONUNBUFFERED='1', MUJOCO_GL='egl', OMP_NUM_THREADS='4',
                                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TZ='Asia/Ho_Chi_Minh')
        self.environment.pop('CONTROL_TASK_RESULTS', None)
        self.environment.pop('CONTROL_TASK_SUITE', None)

    def flush(self):
        save(self.directory / 'status.json', self.state)

    def command(self, name, command, results=None):
        path = self.directory / 'logs' / (name + '.log')
        path.parent.mkdir(exist_ok=True)
        attempt = 1
        while path.exists():
            attempt += 1
            path = self.directory / 'logs' / f'{name}.attempt{attempt}.log'
        environment = dict(self.environment)
        if results:
            environment['CONTROL_TASK_RESULTS'] = str(results)
        self.state['active_command'] = command
        self.state['active_log'] = str(path)
        self.flush()
        with path.open('x') as output:
            self.child = subprocess.Popen(command, cwd=self.project, env=environment,
                                          stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                self.state['child_pid'] = self.child.pid
                self.flush()
                code = self.child.wait()
            except BaseException:
                # The interrupted wait has now released Popen's internal waitpid
                # lock. Waiting recursively inside the signal handler deadlocks.
                self.stop_child()
                raise
            finally:
                self.child = None
                self.state.pop('child_pid', None)
                self.flush()
        if code:
            raise CommandFailure(name, code, path)

    def preflight(self):
        for name, expected in self.state['source_sha256'].items():
            if digest(self.project / name) != expected:
                raise ValueError(f'Snapshot changed since preparation: {name}')
        # CPU checks only: safe while the requested GPU is occupied.
        visible = self.environment['CUDA_VISIBLE_DEVICES']
        self.environment['CUDA_VISIBLE_DEVICES'] = ''
        try:
            self.command('preflight_tests', [sys.executable, '-m', 'pytest', '-q'])
            for task in TASKS:
                self.command('inspect_' + task,
                             [sys.executable, f'tasks/real/{task}/run.py', 'inspect'])
            for script in ('train_sac', 'train_ppo', 'train_temporal_expert', 'build_handoff_bank'):
                self.command('cli_' + script, [sys.executable, f'scripts/{script}.py', '--help'])
        finally:
            self.environment['CUDA_VISIBLE_DEVICES'] = visible
        self.state['preflight_passed_at'] = now()
        self.flush()

    def wait_gpu(self, poll=20., idle_seconds=60.):
        self.state['status'] = 'waiting_for_gpu'
        clear_since = None
        previous = None
        while True:
            guardians = [r for r in self.state['wait_for'] if same_process(r)]
            try:
                pids = gpu_processes(self.state['physical_gpu'])
            except (OSError, subprocess.SubprocessError) as error:
                # A temporary query failure is never evidence that the GPU is free.
                clear_since = None
                self.state.update(gpu_query_error=str(error), last_checked_at=now())
                self.flush()
                print(f'{now()} GPU_QUERY_RETRY {error}', flush=True)
                time.sleep(poll)
                continue
            self.state.pop('gpu_query_error', None)
            clear_since = time.monotonic() if not guardians and not pids and clear_since is None else clear_since
            if guardians or pids:
                clear_since = None
            blockers = dict(coordinators=[r['pid'] for r in guardians], gpu_compute_pids=pids)
            self.state.update(waiting_on=blockers, last_checked_at=now())
            self.flush()
            if blockers != previous:
                print(f'{now()} WAIT {json.dumps(blockers)}', flush=True)
                previous = blockers
            if clear_since is not None and time.monotonic() - clear_since >= idle_seconds:
                return
            time.sleep(poll)

    def source_run(self, stage):
        return Path(self.stages[stage['source']]['run_dir'])

    def action(self, stage):
        name, kind, task = stage['name'], stage['kind'], stage['task']
        results = self.directory / 'stages' / name / 'results'
        if stage.get('attempt', 1) > 1:
            results = self.directory / 'stages' / name / f"attempt_{stage['attempt']}" / 'results'
        if kind == 'train':
            self.command(name, [sys.executable, f'tasks/real/{task}/run.py', 'train', *stage['extra']], results)
            run = Path(read(results / task / 'latest_run.json')['run_dir'])
            evaluation_file = 'evaluation_best.json'
            for required in ('policy.onnx', 'checkpoints/best.pt', evaluation_file):
                if not (run / required).is_file():
                    raise FileNotFoundError(run / required)
            stage.update(run_dir=str(run), results=str(results), evaluation=read(run / evaluation_file))
        elif kind == 'bank':
            run = self.source_run(stage)
            for suffix in ('.npz', '.json'):
                previous = self.project / ('data/handoff_bank' + suffix)
                if previous.exists():
                    previous.rename(previous.with_name(f"handoff_bank.before_attempt_{stage.get('attempt', 1)}{suffix}"))
            self.command(name, [sys.executable, 'scripts/build_handoff_bank.py', '--policy', str(run / 'policy.onnx')])
            stage['metadata'] = read(self.project / 'data/handoff_bank.json')
            meta = stage['metadata']
            if meta['model_sha256'] != digest(self.project / 'assets/cart_double_pendulum_real.xml'):
                raise ValueError('Handoff bank belongs to another plant')
            # The builder refuses weak experts; recheck so an old or hand-made bank cannot slip through.
            if meta['handoff_states'] <= 0 or meta['episodes_reaching_handoff'] < 0.5 * meta['episodes']:
                raise ValueError(f"Handoff bank fails quality: {meta['episodes_reaching_handoff']}/{meta['episodes']} "
                                 f"episodes reached handoff, {meta['handoff_states']} handoff states")
        elif kind == 'observer':
            run = self.source_run(stage)
            output = run.parent / 'observer'
            if output.exists():
                output.rename(run.parent / f"observer.before_attempt_{stage.get('attempt', 1)}")
            self.command(name, [sys.executable, f'tasks/real/{task}/run.py', 'train-observer'], run.parents[1])
            stage['qualified'] = (run.parent / 'observer/best.pt').is_file()
            stage['result'] = read(run.parent / 'observer/result.json')
        elif kind == 'freeze_observer':
            source = self.source_run(stage).parent / 'observer/best.pt'
            if not source.is_file():
                raise RuntimeError('No balancing observer passed the task qualification; stage 2 cannot use it')
            target = self.project / 'tasks/real/expert_balance_ppo/frozen_observer.pt'
            shutil.copy2(source, target)
            stage.update(source_file=str(source), sha256=digest(source))
        elif kind == 'validate':
            run = self.source_run(stage)
            policy = run / 'policy.onnx'
            # Validate the observer-adapted swing export when available.
            if task == 'expert_swing_sac' and self.stages['swing_observer']['status'] == 'complete':
                observed = run.parent / 'observer/policy.onnx'
                if observed.is_file():
                    policy = observed
            seed = self.state['validation_seed']
            self.command(name, [sys.executable, f'tasks/real/{task}/run.py', 'evaluate',
                               '--policy', str(policy), '--episodes', str(self.state['validation_episodes']),
                               '--seed', str(seed)], run.parents[1])
            stage.update(metrics=read(run.parent / f'evaluation_{seed}.json'), policy=str(policy))
        else:
            raise ValueError(kind)

    def run(self):
        self.state.update(status='running')
        self.state.setdefault('training_started_at', now())
        for stage in self.state['stages']:
            if stage['status'] == 'complete':
                continue
            failed = [name for name in stage['dependencies'] if self.stages[name]['status'] != 'complete']
            if failed:
                stage.update(status='skipped', reason='Unfinished dependencies: ' + ', '.join(failed))
                self.flush()
                continue
            for retry in range(2):
                stage.update(status='running', started_at=now(), attempt=stage.get('attempt', 0) + 1)
                stage.pop('error', None)
                self.state['current_stage'] = stage['name']
                self.flush()
                print(f"{now()} START {stage['name']} attempt={stage['attempt']}", flush=True)
                retryable = False
                try:
                    self.action(stage)
                    stage['status'] = 'complete'
                except Exception as error:
                    stage.update(status='failed', error=traceback.format_exc())
                    retryable = retry == 0 and transient_failure(error)
                stage['finished_at'] = now()
                stage.setdefault('attempt_history', []).append({k: stage[k] for k in
                    ('attempt', 'status', 'started_at', 'finished_at', 'run_dir', 'error') if k in stage})
                self.flush()
                print(f"{now()} END {stage['name']} {stage['status']}", flush=True)
                if not retryable:
                    break
                print(f"{now()} RETRY {stage['name']} after GPU becomes idle", flush=True)
                self.wait_gpu()
                self.state['status'] = 'running'
        complete = all(stage['status'] == 'complete' for stage in self.state['stages'])
        self.state.update(status='complete' if complete else 'incomplete', finished_at=now())
        self.state.pop('current_stage', None)
        self.flush()

    def stop_child(self):
        if self.child and process_identity(self.child.pid) is not None:
            # An interrupted Popen.wait() can cache returncode=0 before the
            # child exits. Check /proc and clear that stale result before reaping.
            self.child.returncode = None
            for signum, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                try:
                    os.killpg(self.child.pid, signum)
                except ProcessLookupError:
                    pass
                try:
                    self.child.wait(timeout=timeout)
                    return
                except subprocess.TimeoutExpired:
                    if signum == signal.SIGKILL:
                        raise

    def interrupt(self, *_):
        # Only unwind here; command() stops/reaps the child outside the handler.
        raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, default=1)
    parser.add_argument('--wait-pid', type=int, action='append', default=[])
    parser.add_argument('--validation-episodes', type=int, default=64)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted pipeline; retain completed stages')
    args = parser.parse_args()
    if sum((args.prepare_only, args.run_prepared, args.resume)) > 1:
        parser.error('Choose prepare-only, run-prepared or resume')
    if args.physical_gpu < 0 or args.validation_episodes < 1:
        parser.error('GPU index must be nonnegative and episodes positive')
    directory = args.directory.resolve()
    if not (args.run_prepared or args.resume):
        prepare(directory, args.physical_gpu, args.wait_pid, args.validation_episodes)
    with (directory / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = Pipeline(directory)
        if pipeline.state['status'] != 'prepared' and not args.resume:
            parser.error('This pipeline has already started; choose a new output directory')
        pipeline.state.update(pid=os.getpid(), python=sys.executable)
        pipeline.flush()
        signal.signal(signal.SIGTERM, pipeline.interrupt)
        signal.signal(signal.SIGINT, pipeline.interrupt)
        try:
            if not (args.run_prepared or args.resume) or (args.resume and not pipeline.state.get('preflight_passed_at')):
                pipeline.preflight()
            if args.prepare_only:
                print(f'PREPARED {directory}', flush=True)
                return
            if not pipeline.state.get('preflight_passed_at'):
                raise RuntimeError('Prepared pipeline has not passed preflight')
            # Recheck hashes after a queue handoff; generated pycache files are ignored.
            for name, expected in pipeline.state['source_sha256'].items():
                if digest(pipeline.project / name) != expected:
                    raise ValueError(f'Snapshot changed: {name}')
            gpu = pipeline.state['physical_gpu']
            with Path(f'/tmp/cart_real_v2_gpu{gpu}.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pipeline.wait_gpu()
                pipeline.run()
        except BaseException:
            pipeline.state.update(status='interrupted' if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 'failed',
                                  error=traceback.format_exc(), finished_at=now())
            pipeline.flush()
            raise


if __name__ == '__main__':
    main()
