"""No default-plant scales or foreign bank/observer may leak into a second plant."""
import json
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from conftest import ROOT
from cart_double_pendulum.common import CONTRACT, MODEL_PATH, load_contract
from cart_double_pendulum.plant_metadata import plant_metadata, load_bank
from cart_double_pendulum.plant_scales import plant_scales
from cart_double_pendulum.tasks import make_task, mdp
from cart_double_pendulum.training import full_task_curriculum
from cart_double_pendulum.training.model_temporal_expert import ModelTemporalExpert
from cart_double_pendulum.training.ppo_observer_adapter import load_observer
from cart_double_pendulum.rl.sac import SACAgent, SACConfig


@pytest.fixture
def other_plant(tmp_path):
    path = tmp_path / 'plant.xml'
    path.write_text(MODEL_PATH.read_text().replace('range="-0.2115 0.2115"', 'range="-0.12 0.12"'))
    return load_contract(path)


def bank(path, contract):
    np.savez(path, qpos=np.array([[0., math.pi, 0.]]), qvel=np.zeros((1, 3)),
             plant_metadata=json.dumps(plant_metadata(contract)))
    return path


def test_all_bank_consumers_check_plant(other_plant, tmp_path, monkeypatch):
    path = bank(tmp_path / 'bank.npz', other_plant)
    assert load_bank(path, other_plant)[0].shape == (1, 3)
    with pytest.raises(ValueError, match='different plant'):
        mdp.load_state_bank(path, 'cpu')
    import train_temporal_expert
    with pytest.raises(ValueError, match='different plant'):
        train_temporal_expert.collect(None, 'balancing', path, 1, 0)
    class Asset:
        def write_joint_state_to_sim(self, *args, **kw):
            pass
    env = SimpleNamespace(cfg=SimpleNamespace(plant_contract=CONTRACT), device='cpu',
                          num_envs=1, common_step_counter=0, scene={mdp.ASSET: Asset()})
    from cart_double_pendulum import curriculum
    previous = curriculum._STATE
    try:
        curriculum.start(100)
        with pytest.raises(ValueError, match='different plant'):
            # Force bank reset; full reset also uses the shared checked loader.
            with monkeypatch.context() as m:
                m.setattr(torch, 'rand', lambda n, **kw: torch.full((n,), .5))
                full_task_curriculum.reset_full_task(env, curriculum_file=ROOT / 'tasks/real/full_sac/curriculum.json',
                    bank_path=str(path), budget_control_steps=100)
    finally:
        curriculum._STATE = previous


def test_unstamped_bank_is_rejected_even_with_a_sidecar(tmp_path):
    path = tmp_path / 'legacy.npz'
    np.savez(path, qpos=np.zeros((1, 3)), qvel=np.zeros((1, 3)))
    path.with_suffix('.json').write_text(json.dumps({'model_sha256': plant_metadata()['model_sha256']}))
    with pytest.raises(ValueError, match='missing plant metadata'):
        load_bank(path)


def observer_checkpoint(tmp_path, contract):
    source = tmp_path / 'source.pt'
    agent = SACAgent(8, 1, 'cpu', SACConfig())
    torch.save({'agent': agent.state_dict(), 'plant_metadata': plant_metadata(contract)}, source)
    model = ModelTemporalExpert(source, 'sac', contract=contract)
    path = tmp_path / 'observer.pt'
    torch.save({'model': model.state_dict(), 'plant_metadata': plant_metadata(contract),
                'config': {'observer_model': 'model_gru', 'algorithm': 'sac'}}, path)
    return path


def test_observer_accepts_own_plant_and_rejects_foreign_and_unstamped(other_plant, tmp_path):
    path = observer_checkpoint(tmp_path, other_plant)
    model = load_observer(path, 'cpu', other_plant)
    assert model.contract == other_plant
    assert model.scales.rail == .12
    with pytest.raises(ValueError, match='different plant'):
        load_observer(path, 'cpu')
    payload = torch.load(path, weights_only=False)
    payload.pop('plant_metadata')
    torch.save(payload, path)
    with pytest.raises(ValueError, match='missing plant metadata'):
        load_observer(path, 'cpu', other_plant)


def test_two_plants_get_distinct_environment_and_reset_scales(other_plant):
    cfg = make_task('balancing', other_plant).make_env_cfg(4096, robust_training=True)
    scales = plant_scales(other_plant)
    assert cfg.plant_contract == other_plant
    assert cfg.terminations['rail_failure'].params['rail_limit_m'] == pytest.approx(.95 * .12)
    assert cfg.terminations['balancing_fall'].params['angle_limit_rad'] == pytest.approx(3.319 * scales.angle)
    assert cfg.events['robust_cart_impulse'].params['force_range'] == (-.25 * scales.force, .25 * scales.force)
    class Asset:
        def write_joint_state_to_sim(self, q, v, **kw):
            self.q, self.v = q, v
    asset = Asset()
    env = SimpleNamespace(cfg=cfg, scene={mdp.ASSET: asset}, device='cpu', num_envs=4096, common_step_counter=0)
    mdp.reset_expert_swing(env)
    assert float(asset.q[:, 0].abs().max()) <= .35 * .12 + 1e-7
    assert mdp.env_scales(env) == scales
    # Pure-state reward helpers must also resolve their supplied plant.
    q = torch.tensor([[0., math.pi, 0.]])
    v = torch.tensor([[.5 * scales.cart_speed_per_rate, 0., 0.]])
    assert mdp.stable_mask(q, v, cart_threshold_m=.1, angle_threshold_rad=.1,
                           velocity_threshold=1., contract=other_plant).item()


def test_full_schedule_and_history_resolve_custom_contract(other_plant, tmp_path):
    path = bank(tmp_path / 'own.npz', other_plant)
    cfg = make_task('swing_up', other_plant).make_env_cfg(1)
    full_task_curriculum.configure_environment(cfg, ROOT / 'tasks/real/full_sac',
                                               bank_path=str(path), budget_control_steps=100)
    assert cfg.observations['actor'].terms['state'].params['contract'] == other_plant
    data = full_task_curriculum.load_curriculum(str(ROOT / 'tasks/real/full_sac/curriculum.json'), other_plant)
    assert data['hanging_cart_m'] == pytest.approx(.18 * .12)


def test_simulation_options_match_xml_including_solver(other_plant):
    cfg = make_task('swing_up', other_plant).make_env_cfg(1)
    reference = mujoco.MjModel.from_xml_path(other_plant.model_path)
    model = mujoco.MjModel.from_xml_path(other_plant.model_path)
    model.opt.iterations = 100
    cfg.sim.mujoco.apply(model)
    for name in ('iterations', 'tolerance', 'ls_iterations', 'ls_tolerance', 'ccd_iterations',
                 'solver', 'integrator', 'jacobian', 'cone', 'impratio', 'timestep'):
        assert getattr(model.opt, name) == getattr(reference.opt, name), name
    assert model.opt.iterations == 80


def test_solver_iterations_follow_a_new_xml_value(tmp_path):
    path = tmp_path / 'solver.xml'
    path.write_text(MODEL_PATH.read_text().replace('iterations="80"', 'iterations="37"'))
    cfg = make_task('swing_up', load_contract(path)).make_env_cfg(1)
    assert cfg.sim.mujoco.iterations == 37


def test_custom_observer_export_keeps_its_plant_identity(other_plant, tmp_path):
    from cart_double_pendulum.training.temporal_expert import export_temporal
    from cart_double_pendulum.runtime_policy import OnnxPolicy
    model = load_observer(observer_checkpoint(tmp_path, other_plant), 'cpu', other_plant)
    path = tmp_path / 'policy.onnx'
    export_temporal(model, path)
    with pytest.raises(ValueError, match='different plant'):
        OnnxPolicy(path)
    observation = np.array([0., 0., 0., -1., 0., 1., 0., 0.], dtype=np.float32)
    assert np.isfinite(OnnxPolicy(path, contract=other_plant)(observation))
