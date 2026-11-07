"""Exercise pipeline dependencies, source isolation and the GPU queue without training."""
import pytest
import subprocess
import os
import signal
import sys
import time

import train_pipeline as p


@pytest.fixture
def pipeline(tmp_path):
    directory = tmp_path / 'pipeline'
    p.prepare(directory, 1, [], 64)
    return p.Pipeline(directory)


def test_snapshot_has_exact_configuration_and_no_pretrained_inputs(pipeline):
    project = pipeline.project
    for task in p.TASKS:
        original = p.ROOT / 'tasks/real' / task / 'training.json'
        assert (project / 'tasks/real' / task / 'training.json').read_bytes() == original.read_bytes()
    assert not (project / 'results').exists()
    assert not (project / 'data').exists()
    assert not list(project.rglob('*.pt'))
    assert not list(project.rglob('*.onnx'))
    assert all(p.digest(project / name) == expected for name, expected in pipeline.state['source_sha256'].items())
    assert pipeline.environment['CUDA_VISIBLE_DEVICES'] == '1'
    assert pipeline.environment['PYTHONPATH'] == str(project / 'src')


def test_wait_honours_coordinator_and_resets_idle_period(pipeline, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(p.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(p.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    pipeline.state['wait_for'] = [{'pid': 123, 'start_tick': 'x'}]
    monkeypatch.setattr(p, 'same_process', lambda record: clock[0] < 40.)
    monkeypatch.setattr(p, 'gpu_processes', lambda gpu: [456] if clock[0] < 20. or 60. <= clock[0] < 80. else [])
    pipeline.wait_gpu(poll=20., idle_seconds=60.)
    assert clock[0] == 140.
    assert pipeline.state['waiting_on'] == {'coordinators': [], 'gpu_compute_pids': []}


def test_queue_fails_closed_when_gpu_query_fails(pipeline, monkeypatch):
    def unavailable(gpu):
        raise RuntimeError('nvidia-smi unavailable')
    monkeypatch.setattr(p, 'gpu_processes', unavailable)
    with pytest.raises(RuntimeError, match='unavailable'):
        pipeline.wait_gpu()
    assert pipeline.state['status'] == 'waiting_for_gpu'


def test_temporary_nvidia_query_failure_does_not_kill_the_queue(pipeline, monkeypatch):
    clock, calls = [0.], [0]
    monkeypatch.setattr(p.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(p.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    def query(gpu):
        calls[0] += 1
        if calls[0] == 1:
            raise subprocess.TimeoutExpired('nvidia-smi', 15)
        return []
    monkeypatch.setattr(p, 'gpu_processes', query)
    pipeline.wait_gpu(poll=20., idle_seconds=60.)
    assert clock[0] == 80.
    assert 'gpu_query_error' not in pipeline.state


def test_failed_balance_observer_skips_dependents_but_runs_full_tasks(pipeline, monkeypatch):
    executed = []
    def action(stage):
        executed.append(stage['name'])
        if stage['name'] == 'balance_observer':
            raise RuntimeError('observer training failed')
    monkeypatch.setattr(pipeline, 'action', action)
    pipeline.run()
    assert pipeline.stages['balance_observer']['status'] == 'failed'
    for name in ('balance_stage2',):
        assert pipeline.stages[name]['status'] == 'skipped'
        assert name not in executed
    assert pipeline.stages['full_ppo']['status'] == 'complete'
    assert pipeline.stages['full_sac']['status'] == 'complete'
    assert pipeline.state['status'] == 'incomplete'


def test_unqualified_observer_cannot_be_frozen(pipeline):
    stage = pipeline.stages['freeze_balance_observer']
    pipeline.stages['balance_stage1']['run_dir'] = str(pipeline.directory / 'stage1/results/expert_balance_ppo/train')
    with pytest.raises(RuntimeError, match='No balancing observer passed'):
        pipeline.action(stage)


def test_repeated_swing_runs_use_separate_result_namespaces(pipeline, monkeypatch):
    roots = []
    def command(name, command, results):
        roots.append(results)
        run = results / 'expert_swing_sac/train'
        (run / 'checkpoints').mkdir(parents=True)
        (run / 'policy.onnx').touch()
        (run / 'checkpoints/best.pt').touch()
        p.save(run / 'evaluation_best.json', {'success_rate': 0.})
        p.save(results / 'expert_swing_sac/latest_run.json', {'run_dir': str(run)})
    monkeypatch.setattr(pipeline, 'command', command)
    pipeline.action(pipeline.stages['swing_initial'])
    pipeline.action(pipeline.stages['swing_with_bank'])
    assert roots[0] != roots[1]
    assert pipeline.stages['swing_initial']['run_dir'] != pipeline.stages['swing_with_bank']['run_dir']


def test_validation_reads_report_from_the_source_task(pipeline, monkeypatch):
    run = pipeline.directory / 'stages/full_sac/results/full_sac/train'
    run.mkdir(parents=True)
    pipeline.stages['full_sac']['run_dir'] = str(run)
    def command(name, command, results):
        assert results == run.parents[1]
        p.save(run.parent / f"evaluation_{pipeline.state['validation_seed']}.json", {'success_rate': .75})
    monkeypatch.setattr(pipeline, 'command', command)
    stage = pipeline.stages['validate_full_sac']
    pipeline.action(stage)
    assert stage['metrics']['success_rate'] == .75




def test_resume_preserves_completed_training_stages(pipeline, monkeypatch):
    done = ('swing_initial', 'build_bank', 'swing_with_bank')
    for name in done:
        pipeline.stages[name]['status'] = 'complete'
    executed = []
    monkeypatch.setattr(pipeline, 'action', lambda stage: executed.append(stage['name']))
    pipeline.run()
    assert not set(done) & set(executed)
    assert pipeline.state['status'] == 'complete'


def test_only_known_transient_errors_get_one_automatic_retry(pipeline, monkeypatch):
    log = pipeline.directory / 'cuda.log'
    log.write_text('CUDA error: initialization error')
    calls = []
    def action(stage):
        calls.append((stage['name'], stage['attempt']))
        if stage['name'] == 'swing_initial' and stage['attempt'] == 1:
            raise p.CommandFailure(stage['name'], 1, log)
    monkeypatch.setattr(pipeline, 'action', action)
    monkeypatch.setattr(pipeline, 'wait_gpu', lambda: None)
    pipeline.run()
    assert ('swing_initial', 2) in calls
    assert pipeline.stages['swing_initial']['status'] == 'complete'
    assert len(pipeline.stages['swing_initial']['attempt_history']) == 2
    assert not p.transient_failure(RuntimeError('training failed'))


def test_retry_training_retains_the_first_attempt(pipeline, monkeypatch):
    roots = []
    def command(name, command, results):
        roots.append(results)
        run = results / 'expert_swing_sac/train'
        (run / 'checkpoints').mkdir(parents=True)
        (run / 'policy.onnx').touch()
        (run / 'checkpoints/best.pt').touch()
        p.save(run / 'evaluation_best.json', {})
        p.save(results / 'expert_swing_sac/latest_run.json', {'run_dir': str(run)})
    monkeypatch.setattr(pipeline, 'command', command)
    stage = pipeline.stages['swing_initial']
    for attempt in (1, 2):
        stage['attempt'] = attempt
        pipeline.action(stage)
    assert roots[0] != roots[1]
    assert all((root / 'expert_swing_sac/train/checkpoints/best.pt').is_file() for root in roots)


def test_signal_handler_does_not_wait_recursively(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, 'stop_child', lambda: pytest.fail('cleanup must not run inside a signal handler'))
    with pytest.raises(KeyboardInterrupt):
        pipeline.interrupt(signal.SIGTERM, None)


def test_sigterm_during_child_wait_reaps_child_and_exits(pipeline):
    script = '''
import signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import train_pipeline as p
pipeline = p.Pipeline(Path(sys.argv[2]))
signal.signal(signal.SIGTERM, pipeline.interrupt)
try:
    pipeline.command('interrupt_probe', [sys.executable, '-c', 'import time; time.sleep(60)'])
except KeyboardInterrupt:
    print('CLEAN_EXIT', flush=True)
'''
    runner = subprocess.Popen([sys.executable, '-c', script, str(p.ROOT / 'scripts'), str(pipeline.directory)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = p.read(pipeline.directory / 'status.json')
            child_pid = state.get('child_pid')
            if child_pid:
                break
            time.sleep(.01)
        assert child_pid and p.process_identity(child_pid)
        runner.send_signal(signal.SIGTERM)
        out, err = runner.communicate(timeout=5)
        assert runner.returncode == 0, err
        assert 'CLEAN_EXIT' in out
        assert p.process_identity(child_pid) is None
        assert 'child_pid' not in p.read(pipeline.directory / 'status.json')
    finally:
        if runner.poll() is None:
            os.killpg(runner.pid, signal.SIGKILL)
            runner.wait(timeout=5)
        if child_pid and p.process_identity(child_pid):
            os.killpg(child_pid, signal.SIGKILL)


def test_cleanup_does_not_trust_stale_popen_returncode(pipeline):
    """Popen may cache zero when wait is interrupted; a live child still needs reaping."""
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    pipeline.child = child
    try:
        child.returncode = 0
        pipeline.stop_child()
        assert p.process_identity(child.pid) is None
    finally:
        if p.process_identity(child.pid):
            os.killpg(child.pid, signal.SIGKILL)
            child.returncode = None
            child.wait()
