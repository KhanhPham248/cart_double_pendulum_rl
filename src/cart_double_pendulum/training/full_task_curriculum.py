"""Full-task-only reset and reward schedules controlled by task-local JSON."""
from functools import lru_cache
import json
import math
from pathlib import Path

import torch

from cart_double_pendulum import curriculum
from cart_double_pendulum.tasks import mdp
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.plant_scales import resolve, plant_scales


@lru_cache(maxsize=16)
def load_curriculum(path, contract=CONTRACT):
    data = resolve(json.loads(Path(path).read_text()), plant_scales(contract))
    points = data['schedule']
    fractions = [p['fraction'] for p in points]
    if fractions[0] != 0. or fractions[-1] != 1. or any(
            a >= b for a, b in zip(fractions, fractions[1:])):
        raise ValueError('Full curriculum fractions must increase from zero to one')
    if data.get('progress_units') != 'skill_gated_schedule' or data.get('interpolation') != 'step':
        raise ValueError('Full schedule must use skill_gated_schedule with step interpolation')
    gates = data.get('skill_gates', [])
    if len(gates) != len(points) - 1:
        raise ValueError('Every full schedule transition requires a skill gate')
    for gate in gates:
        if (set(gate) != {'capture_min', 'centre_min', 'evaluations'}
            or not 0 < gate['capture_min'] <= 1 or not 0 <= gate['centre_min'] <= 1
            or isinstance(gate['evaluations'], bool) or not isinstance(gate['evaluations'], int)
            or gate['evaluations'] < 1):
            raise ValueError('Full skill gate requires rates in [0,1] and at least one evaluation')
    keys = set(points[0])
    for point in points:
        if set(point) != keys or not all(math.isfinite(float(v)) for v in point.values()):
            raise ValueError('Full curriculum points must have the same finite scalar fields')
        if not (0 <= point['bank_probability'] and .1 <= point['hanging_probability']
                and point['bank_probability'] + point['hanging_probability'] <= 1.):
            raise ValueError('Keep hanging resets in every stage, with valid mixture weights')
        if min(point['precision_sigma_m'], point['upright_cart_m'],
               point['upright_angle_rad'], point['upright_velocity']) <= 0:
            raise ValueError('Full reset/reward ranges must be positive')
        for key in ('settling_scale','precision_scale','center_weight_scale'):
            if key in point and not 0 <= point[key] <= 1:
                raise ValueError('Full reward curriculum scales must be in [0,1]')
    if any(key in points[0] for key in ('domain_fraction', 'noise_max_std', 'push_force_n')) or 'readiness' in data:
        raise ValueError('Randomization, noise and pushes follow cart_double_pendulum.curriculum, not the schedule')
    return data


def settings(env, curriculum_file):
    data = load_curriculum(str(curriculum_file), mdp.env_contract(env))
    progress = curriculum.full_progress()
    # Fractions label unlocked skill levels, not fractions of elapsed training time.
    return dict(next(point for point in reversed(data['schedule']) if point['fraction'] <= progress))


def reset_full_task(env, env_ids=None, *, curriculum_file, bank_path, budget_control_steps):
    env._full_budget_control_steps = budget_control_steps
    cfg = load_curriculum(str(curriculum_file), mdp.env_contract(env))
    stage = settings(env, curriculum_file)
    ids = mdp._reset_ids(env, env_ids)
    n = len(ids)
    q = torch.empty((n, 3), device=env.device)
    v = torch.empty_like(q).uniform_(-cfg['hanging_velocity'], cfg['hanging_velocity'])
    v[:, 0] *= mdp.env_scales(env).cart_speed_per_rate
    q[:, 0].uniform_(-cfg['hanging_cart_m'], cfg['hanging_cart_m'])
    q[:, 1:].uniform_(-cfg['hanging_angle_rad'], cfg['hanging_angle_rad'])
    draw = torch.rand(n, device=env.device)
    bank = (draw >= stage['hanging_probability']) & (
        draw < stage['hanging_probability'] + stage['bank_probability'])
    upright = draw >= stage['hanging_probability'] + stage['bank_probability']
    if bank.any():
        cache = getattr(env, '_full_training_bank', None)
        if cache is None or cache[0] != bank_path:
            from cart_double_pendulum.plant_metadata import load_bank
            bq, bv = load_bank(bank_path, mdp.env_contract(env))
            cache = (bank_path, torch.as_tensor(bq, device=env.device),
                     torch.as_tensor(bv, device=env.device))
            env._full_training_bank = cache
        sample = torch.randint(len(cache[1]), (int(bank.sum()),), device=env.device)
        q[bank], v[bank] = cache[1][sample], cache[2][sample]
    count = int(upright.sum())
    if count:
        angles = torch.empty((count, 2), device=env.device).uniform_(
            -stage['upright_angle_rad'], stage['upright_angle_rad'])
        omega = torch.empty((count, 2), device=env.device).uniform_(
            -stage['upright_velocity'], stage['upright_velocity'])
        q[upright, 0] = torch.empty(count, device=env.device).uniform_(
            -stage['upright_cart_m'], stage['upright_cart_m'])
        q[upright, 1] = math.pi + angles[:, 0]
        q[upright, 2] = angles[:, 1] - angles[:, 0]
        v[upright, 0] = torch.empty(count, device=env.device).uniform_(
            -stage['upright_velocity'], stage['upright_velocity']) * mdp.env_scales(env).cart_speed_per_rate
        v[upright, 1], v[upright, 2] = omega[:, 0], omega[:, 1] - omega[:, 0]
    # Mirror half the starts: the plant is symmetric under reflection.
    if cfg.get('mirror_resets', False):
        mirror = torch.rand(n, device=env.device) < .5
        q[mirror] *= -1.
        v[mirror] *= -1.
    env.scene['cart_double_pendulum'].write_joint_state_to_sim(
        q, v, joint_ids=slice(0, 3), env_ids=ids)


def full_return_settling(env, *, capture_sigma_rad, cart_velocity_reference_m_s,
                         angular_velocity_reference_rad_s, return_cart_reference_m_s,
                         return_distance_m, curriculum_file=None):
    """Permit centering motion away from the origin; retain angular braking."""
    if min(capture_sigma_rad, cart_velocity_reference_m_s, angular_velocity_reference_rad_s,
           return_cart_reference_m_s, return_distance_m) <= 0:
        raise ValueError('Full settling references must be positive')
    q, v = mdp.state(env)
    position_gate = (q[:, 0].abs() / return_distance_m).clamp(0., 1.)
    toward_center = q[:, 0] * v[:, 0] < 0.
    cart_reference = cart_velocity_reference_m_s + position_gate * toward_center * (
        return_cart_reference_m_s - cart_velocity_reference_m_s)
    speed = torch.stack((v[:, 0].abs() / cart_reference,
        v[:, 1].abs() / angular_velocity_reference_rad_s,
        (v[:, 1] + v[:, 2]).abs() / angular_velocity_reference_rad_s,
        v[:, 2].abs() / angular_velocity_reference_rad_s), dim=1).amax(dim=1)
    capture = mdp.double_upright_capture(env, angle_sigma_rad=capture_sigma_rad)
    scale = settings(env, curriculum_file).get('settling_scale', 1.) if curriculum_file else 1.
    return scale * capture * torch.exp(-.5 * speed.square())


def full_precision_center(env, *, curriculum_file, capture_sigma_rad=None, kernel='gaussian'):
    capture_sigma_rad = 1.66 * mdp.env_scales(env).angle if capture_sigma_rad is None else capture_sigma_rad
    q, _ = mdp.state(env)
    stage = settings(env, curriculum_file)
    ratio = (q[:, 0] / stage['precision_sigma_m']).square()
    if kernel == 'gaussian':
        centered = torch.exp(-.5 * ratio)
    elif kernel == 'cauchy':
        centered = 1. / (1. + ratio)
    else:
        raise ValueError('Unknown Full precision kernel')
    return stage.get('precision_scale', 1.) * mdp.double_upright_capture(
        env, angle_sigma_rad=capture_sigma_rad) * centered


def full_center_penalty(env, *, curriculum_file, reference_m=None, capture_sigma_rad=None):
    """Off-centre cost while upright, bounded by 1 so an off-centre capture never scores below no capture."""
    capture_sigma_rad = 1.66 * mdp.env_scales(env).angle if capture_sigma_rad is None else capture_sigma_rad
    reference_m = .25 * mdp.env_scales(env).rail if reference_m is None else reference_m
    scale = settings(env, curriculum_file).get('center_weight_scale', 1.)
    return scale * mdp.double_upright_capture(env, angle_sigma_rad=capture_sigma_rad) * (
        mdp.cart_center_cost(env, cart_sigma_m=reference_m))


def configure_environment(cfg, folder, *, bank_path, budget_control_steps):
    """Install the full task's history, reset and curriculum-scheduled events."""
    folder = Path(folder)
    task = json.loads((folder / 'task.json').read_text())
    if task['name'] not in ('full_ppo', 'full_sac') or task['phase'] != 'full':
        raise ValueError('Full configuration cannot be installed on an expert or Meta task')
    if not bank_path or not Path(bank_path).is_file():
        raise FileNotFoundError(f'{task["name"]} needs a handoff bank ({bank_path}); see docs/BANKS.md')
    curriculum_file = str((folder / task['curriculum_file']).resolve())
    contract = cfg.plant_contract
    load_curriculum(curriculum_file, contract)
    environment = resolve(json.loads((folder / task['environment_file']).read_text()), plant_scales(contract))
    cfg.episode_length_s = float(environment['episode_length_s'])
    frames = environment.get('actor_history_frames', 0)
    if isinstance(frames, bool) or not isinstance(frames, int) or frames == 1 or frames < 0:
        raise ValueError('Full sensor history must be zero or at least two frames')
    if frames:
        # SAC replay needs equal actor/critic widths: give the critic the clean history.
        for group in ('actor', 'critic'):
            term = cfg.observations[group].terms['state']
            term.func = mdp.actor_history_frame
            noise_cap = cfg.observations['actor'].terms['state'].params.get('max_noise_std', 0.)
            term.params = dict(contract=contract,
                max_noise_std=noise_cap if group == 'actor' else 0.,
                noise_per_episode=group == 'actor')
            term.history_length = frames
            term.flatten_history_dim = True
    reset = environment['reset_system']
    if reset['function'] != 'reset_full_task':
        raise ValueError('Full task environment must declare its full-process reset')
    reset_params = dict(reset['params'])
    if reset_params.get('curriculum_file') != task['curriculum_file']:
        raise ValueError('Full reset and task must reference the same local curriculum')
    reset_params.update(curriculum_file=curriculum_file,
                        bank_path=bank_path, budget_control_steps=budget_control_steps)
    cfg.events['reset_system'].func = reset_full_task
    cfg.events['reset_system'].params = reset_params
    return cfg
