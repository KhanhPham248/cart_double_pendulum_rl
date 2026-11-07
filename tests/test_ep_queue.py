"""GPU capacity and persistent queue progression without launching training."""
import json
from types import SimpleNamespace
import pytest
import watch_ep_queue as watcher


@pytest.mark.parametrize('existing,expected', [({12, 13}, 0), ({12}, 1)])
def test_capacity_counts_external_gpu_processes(existing, expected, tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(watcher, 'gpu_pids', lambda _: existing.copy())
    monkeypatch.setattr(watcher, 'identity', lambda _: '123')
    monkeypatch.setattr(watcher.subprocess, 'Popen', lambda *a, **kw: launched.append(a[0]) or SimpleNamespace(pid=99))
    monkeypatch.setattr(watcher.sys, 'argv', ['watch', '--state-dir', str(tmp_path)])
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: (_ for _ in ()).throw(InterruptedError()))
    with pytest.raises(InterruptedError):
        watcher.main()
    assert len(launched) == expected
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['tasks'][0]['status'] == ('running' if expected else 'pending')
    if expected:
        assert launched[0][-2:] == ['--device', 'cuda:1']


def test_restart_counts_child_before_cuda_is_visible(tmp_path, monkeypatch):
    state = dict(gpu=1, max_tasks=2, tasks=[dict(name='sac_ep0', status='running',pid=99,start_tick='123'),
        dict(name='sac_ep1',status='pending')])
    (tmp_path / 'state.json').write_text(json.dumps(state))
    monkeypatch.setattr(watcher, 'gpu_pids', lambda _: {12})
    monkeypatch.setattr(watcher, 'identity', lambda _: '123')
    monkeypatch.setattr(watcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('cap exceeded'))
    monkeypatch.setattr(watcher.sys, 'argv', ['watch', '--state-dir', str(tmp_path)])
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: (_ for _ in ()).throw(InterruptedError()))
    with pytest.raises(InterruptedError): watcher.main()
    assert json.loads((tmp_path / 'state.json').read_text())['tasks'][1]['status'] == 'pending'


def test_centered_queue_can_include_tip_before_eps(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, 'gpu_pids', lambda _: {12, 13})
    names = ['tip_swing_sac', 'sac_ep0', 'sac_ep1', 'sac_ep2', 'sac_ep3']
    monkeypatch.setattr(watcher.sys, 'argv', ['watch', '--state-dir', str(tmp_path), '--tasks', *names])
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: (_ for _ in ()).throw(InterruptedError()))
    with pytest.raises(InterruptedError): watcher.main()
    state = json.loads((tmp_path / 'state.json').read_text())
    assert [task['name'] for task in state['tasks']] == names
    assert all(task['status'] == 'pending' for task in state['tasks'])
