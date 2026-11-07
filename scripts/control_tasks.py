"""Launcher behind every tasks/real/<name>/run.py: inspect | train | train-observer | evaluate."""
import argparse
import ast
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'tasks/real'
RESULTS = Path(os.environ.get('CONTROL_TASK_RESULTS', str(ROOT / 'results/real'))).resolve()
EP_TASKS = tuple(f'sac_ep{ep}' for ep in range(4))
NAMES = EP_TASKS + ('tip_swing_sac', 'expert_swing_sac', 'expert_balance_ppo', 'full_ppo', 'full_sac')
EXPERTS = ('tip_swing_sac', 'expert_swing_sac', 'expert_balance_ppo')
FULL = ('full_ppo', 'full_sac')
# Swing and fixed-EP tasks can train before a handoff bank exists.
BANK_OPTIONAL = EP_TASKS + ('expert_swing_sac', 'tip_swing_sac')


class RunLog:
    def __init__(self, terminal, log): self.terminal, self.log = terminal, log
    def write(self, value):
        self.log.write(value)
        return self.terminal.write(value)
    def flush(self): self.log.flush(); self.terminal.flush()
    def __getattr__(self, name): return getattr(self.terminal, name)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    from cart_double_pendulum.training.task_directory import read_json as read
    return read(path)


def load(name):
    folder = SUITE / name
    return folder, read_json(folder / 'task.json'), read_json(folder / 'training.json')


def latest_run(name):
    pointer = RESULTS / name / 'latest_run.json'
    if not pointer.is_file():
        raise FileNotFoundError(f'{name} has no training run under {RESULTS / name}; run "train" first')
    return Path(read_json(pointer)['run_dir'])


def check_inputs(name, values):
    """Fail before training when an input file is missing."""
    bank = values.get('handoff_bank')
    if bank and not Path(bank).is_file():
        if name not in BANK_OPTIONAL:
            raise FileNotFoundError(f'{name} needs the handoff bank {bank}; see docs/BANKS.md')
        print(f'NOTE: {bank} does not exist; training {name} without bank resets', flush=True)
        values['handoff_bank'] = None
    if bank and Path(bank).is_file():
        from cart_double_pendulum.plant_metadata import load_bank
        load_bank(bank)
    for key in ('resume_checkpoint', 'actor_observer_checkpoint'):
        path = values.get(key)
        if path and not Path(path).is_file():
            raise FileNotFoundError(f'{name}: {key} {path} does not exist; see docs/BANKS.md')


def script_flags(script):
    """The --flags a training script's argparse parser accepts."""
    tree = ast.parse(script.read_text())
    return {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
            for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            and arg.value.startswith('--')}


def train(name, extra):
    from cart_double_pendulum.training import task_directory
    folder, spec, values = load(name)
    if '--without-observer' in extra:
        if name != 'expert_balance_ppo':
            raise ValueError('--without-observer applies only to expert_balance_ppo')
        extra = [item for item in extra if item != '--without-observer']
        values['actor_observer_checkpoint'] = None
    check_inputs(name, values)
    # The bank feeds the reset installed below; trainers never read it.
    bank = values.pop('handoff_bank', None)
    environment = read_json(folder / spec['environment_file'])
    script = ROOT / 'scripts' / f"train_{spec['algorithm']}.py"
    flags = script_flags(script)
    unsupported = sorted(key for key in values if '--' + key.replace('_', '-') not in flags)
    if unsupported:
        raise ValueError(f'{name}/training.json has options {script.name} does not accept: {unsupported}')
    if '--seed' in extra:
        values['seed'] = int(extra[extra.index('--seed') + 1])
    protected = ('--run-dir', '--task', '--handoff-bank', '--actor-observer-checkpoint',
                 '--resume-checkpoint')
    if any(item.split('=')[0] in protected for item in extra):
        raise ValueError('Command-line overrides may not change the task, its inputs or its output; '
                         'edit training.json instead')
    output = RESULTS / name / 'train'
    if 'CONTROL_TASK_RESULTS' not in os.environ:
        output = RESULTS / name / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / 'train'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing training run: {output}')
    values['run_dir'] = str(output)

    def effective_option(key, default):
        flag = '--' + key.replace('_', '-')
        result = values.get(key, default)
        for i, item in enumerate(extra):
            if item == flag: result = extra[i + 1]
            elif item.startswith(flag + '='): result = item.split('=', 1)[1]
        return result

    # Training budget in control steps per world.
    if spec['algorithm'] == 'ppo':
        training_budget = int(effective_option('max_iterations', 2000)) * int(effective_option('rollout_steps', 24))
    else:
        total = int(effective_option('total_transitions', 2000000))
        envs = int(effective_option('num_envs', 512))
        training_budget = (total + envs - 1) // envs
    budget_control_steps = training_budget if name in FULL else None
    # Plant feasibility against this run's time scales; printed and saved with the run.
    from cart_double_pendulum.common import CONTRACT
    from cart_double_pendulum.plant_scales import plant_report
    from cart_double_pendulum.rl.sac import DEFAULT_DISCOUNT
    report = dict(plant_report())
    report['discount_horizon_s'] = CONTRACT.control_timestep_s / (1 - float(effective_option('gamma', DEFAULT_DISCOUNT)))
    report['episode_length_s'] = float(environment['episode_length_s'])
    pump = report['pump_to_upright_energy_s'] if name != 'expert_balance_ppo' else None
    report['warnings'] = report['warnings'] + [message for failed, message in (
        (pump is not None and pump > report['discount_horizon_s'], 'swing-up pump time exceeds the discount horizon'),
        (pump is not None and pump > 0.5 * report['episode_length_s'], 'swing-up pump time exceeds half an episode'))
        if failed]
    for message in report['warnings']:
        print(f'PLANT_WARNING {message}', flush=True)
    save(output / 'plant_report.json', report)
    command = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        command.extend(['--' + key.replace('_', '-')] if value is True else ['--' + key.replace('_', '-'), str(value)])

    from cart_double_pendulum.tasks import config
    from cart_double_pendulum.training import checkpoint_evaluation, full_task_curriculum
    original = config.Task.make_env_cfg

    def configured(self, *args, **kwargs):
        cfg = original(self, *args, **kwargs)
        cfg.episode_length_s = float(environment['episode_length_s'])
        if name not in FULL:
            reset = environment['reset_system']
            func = task_directory.function(reset['function'])
            params = dict(reset['params'])
            if 'bank_path' in inspect.signature(func).parameters:
                params['bank_path'] = bank
            cfg.events['reset_system'].func = func
            cfg.events['reset_system'].params = params
        cfg.rewards, cfg.scale_rewards_by_dt = task_directory.reward_terms(folder, self.contract)
        if name in FULL:
            full_task_curriculum.configure_environment(cfg, folder, bank_path=bank,
                                                       budget_control_steps=budget_control_steps)
        save(output / 'effective_reward.json', {
            'terms': {key: {'function': term.func.__name__, 'weight': term.weight, 'params': term.params}
                      for key, term in cfg.rewards.items()},
            'scale_rewards_by_dt': cfg.scale_rewards_by_dt})
        state = cfg.observations['actor'].terms['state']
        save(output / 'effective_environment.json', {
            'gravity': list(cfg.sim.mujoco.gravity), 'episode_length_s': cfg.episode_length_s,
            'physics_timestep_s': cfg.sim.mujoco.timestep, 'decimation': cfg.decimation,
            'reset_function': cfg.events['reset_system'].func.__name__,
            'actor_function': state.func.__name__, 'actor_history_frames': state.history_length,
            'actor_noise_max_std': state.params.get('max_noise_std', 0.),
            'critic_term_names': list(cfg.observations['critic'].terms),
            'event_names': list(cfg.events), 'scale_rewards_by_dt': cfg.scale_rewards_by_dt,
            'curriculum_budget_control_steps': budget_control_steps})
        return cfg
    config.Task.make_env_cfg = configured

    from cart_double_pendulum import curriculum
    # (first, final) stage: swing hands over before the push stage.
    first_stage, final_stage = {'tip_swing_sac': (1, 3), 'expert_swing_sac': (1, 3)}.get(name, (1, 4))
    curriculum.start(training_budget, final_stage=final_stage, first_stage=first_stage)
    if name in FULL:
        schedule = full_task_curriculum.load_curriculum(str(folder / spec['curriculum_file']))
        curriculum.configure_full_schedule([point['fraction'] for point in schedule['schedule']],
                                           schedule['skill_gates'])

    def staged(evaluate, gate=lambda m: (m['nominal_success_rate'], m['randomized_success_rate'], None)):
        """Gate the curriculum at the stage's level; rank checkpoints on the full-level benchmark."""
        def run(path, task, episodes):
            metrics = curriculum.gate_and_benchmark(lambda: evaluate(path, task, episodes), gate, episodes)
            save(output / 'curriculum_state.json', metrics['curriculum'])
            return metrics
        return run

    evaluation_file = folder / spec['evaluation_file']
    if name in EP_TASKS:
        from cart_double_pendulum.training.ep_evaluation import evaluate_ep_task
        checkpoint_evaluation.install(staged(
            lambda path, task, episodes: evaluate_ep_task(name, path, episodes, evaluation_file)))
    elif name in EXPERTS:
        from cart_double_pendulum.training.task_local_evaluation import evaluate_expert_task
        checkpoint_evaluation.install(staged(
            lambda path, task, episodes: evaluate_expert_task(name, path, episodes, evaluation_file)))
    elif name in FULL:
        from cart_double_pendulum.training.full_task_evaluation import evaluate_full_task, full_gate
        from cart_double_pendulum.training.full_task_optimization import selection_key
        optimization = read_json(folder / spec['optimization_file'])
        if name == 'full_sac' and optimization.get('mirror_replay', False):
            from cart_double_pendulum.rl import sac
            from cart_double_pendulum.training.full_task_optimization import FullSACAgent
            sac.SACAgent = FullSACAgent

        def evaluate_full(path, task, episodes):
            metrics = evaluate_full_task(name, path, episodes, evaluation_file)
            metrics['full_task_selection_key'] = selection_key(metrics)
            metrics['full_task_selection_protocol'] = 'joint_success_then_capture_center_progress'
            return metrics
        checkpoint_evaluation.install(staged(evaluate_full, full_gate))

    snapshot = [ROOT / 'scripts/control_tasks.py', script, ROOT / 'assets/cart_double_pendulum_real.xml',
                *sorted((ROOT / 'src/cart_double_pendulum').rglob('*.py')),
                *sorted(path for path in folder.iterdir() if path.is_file())]
    if values.get('resume_checkpoint'):
        from cart_double_pendulum.plant_metadata import check_plant_metadata
        import torch
        payload = torch.load(values['resume_checkpoint'], map_location='cpu', weights_only=False)
        check_plant_metadata(payload.get('plant_metadata'), CONTRACT, where=values['resume_checkpoint'])
        origin = Path(values['resume_checkpoint']).resolve().parent.parent / 'launch_contract.json'
        if not origin.is_file() or read_json(origin).get('task') != name:
            raise ValueError('resume checkpoint must belong to the same named task')
        snapshot.append(Path(values['resume_checkpoint']))
        snapshot.append(origin)
    if values.get('actor_observer_checkpoint'):
        snapshot.append(Path(values['actor_observer_checkpoint']))
    if bank:
        snapshot.append(Path(bank))
    provenance = []
    for source in snapshot:
        record = {'source': str(source), 'sha256': digest(source)}
        if source.is_relative_to(ROOT) and source.suffix in ('.py', '.json', '.xml'):
            target = output / 'source_snapshot' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        provenance.append(record)
    save(output / 'launch_contract.json', {
        'task': name, 'source': provenance, 'command': [str(script), *command, *extra],
        'without_observer': name == 'expert_balance_ppo' and values.get('actor_observer_checkpoint') is None,
        'seed': values['seed']})
    print('COMMAND', str(script), *command, *extra, flush=True)
    sys.argv = [str(script), *command, *extra]
    save(RESULTS / name / 'latest_run.json', {'task': name, 'run_dir': str(output), 'configuration': str(folder)})
    stdout, stderr = sys.stdout, sys.stderr
    with (output / 'console.log').open('x', buffering=1) as log:
        sys.stdout, sys.stderr = RunLog(stdout, log), RunLog(stderr, log)
        try: runpy.run_path(str(script), run_name='__main__')
        finally: sys.stdout, sys.stderr = stdout, stderr


def evaluate(name, episodes, seed, policy):
    folder, spec, values = load(name)
    policy = Path(policy) if policy else latest_run(name) / 'policy.onnx'
    if not policy.is_file():
        raise FileNotFoundError(policy)
    evaluation_file = folder / spec['evaluation_file']
    if name in FULL:
        from cart_double_pendulum.training.full_task_evaluation import evaluate_full_task
        metrics = evaluate_full_task(name, policy, episodes, evaluation_file, seed=seed)
    elif name in EP_TASKS:
        from cart_double_pendulum.training.ep_evaluation import evaluate_ep_task
        metrics = evaluate_ep_task(name, policy, episodes, evaluation_file, seed=seed)
    elif name in EXPERTS:
        from cart_double_pendulum.training.task_local_evaluation import evaluate_expert_task
        metrics = evaluate_expert_task(name, policy, episodes, evaluation_file, seed=seed)
    else:
        raise ValueError(f'unsupported task: {name}')
    metrics['policy'] = str(policy)
    metrics['policy_sha256'] = digest(policy)
    save(RESULTS / name / f'evaluation_{seed if seed is not None else "default"}.json', metrics)
    print(json.dumps(metrics, indent=2))


def train_observer(name, extra):
    """Adapt a recurrent observer to the latest run of an expert task (docs/BANKS.md)."""
    if name not in EXPERTS:
        raise ValueError('Observer adaptation is defined for the two expert tasks only')
    folder, spec, values = load(name)
    actor_run = latest_run(name)
    if not actor_run.resolve().is_relative_to((RESULTS / name).resolve()):
        raise ValueError('Observer source must belong to this task')
    checkpoint, policy = actor_run / 'checkpoints/best.pt', actor_run / 'policy.onnx'
    if not checkpoint.is_file() or not policy.is_file():
        raise FileNotFoundError(f'{actor_run} has no best.pt/policy.onnx; train and validate the expert first')
    if name == 'expert_balance_ppo' and not read_json(actor_run / 'launch_contract.json').get('without_observer'):
        raise ValueError('Train expert_balance_ppo with --without-observer before adapting its observer')
    if not values.get('handoff_bank') or not Path(values['handoff_bank']).is_file():
        raise FileNotFoundError(f"Observer adaptation needs the handoff bank {values.get('handoff_bank')}; see docs/BANKS.md")
    observer = read_json(folder / spec['observer_file'])
    output = actor_run.parent / 'observer'
    if output.exists():
        raise FileExistsError(output)
    command = ['--task', values['task'], '--algorithm', spec['algorithm'], '--checkpoint', str(checkpoint),
               '--source-onnx', str(policy), '--bank', values['handoff_bank'], '--run-dir', str(output)]
    for key, value in observer.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value: command.append(flag)
        else: command.extend([flag, str(value)])
    if any(item.split('=')[0] in ('--checkpoint', '--source-onnx', '--run-dir', '--task', '--algorithm', '--bank')
           for item in extra):
        raise ValueError('Observer overrides cannot change task identity')
    script = ROOT / 'scripts/train_temporal_expert.py'
    namespace = runpy.run_path(str(script), run_name='_task_local_observer')
    module_globals = namespace['main'].__globals__
    from cart_double_pendulum.training.task_local_evaluation import evaluate_expert_task
    module_globals['evaluate_policy'] = lambda p, t, episodes, seed, **kw: evaluate_expert_task(
        name, p.path, episodes, folder / spec['evaluation_file'])['cases']
    output.mkdir(parents=True)
    for filename in (spec['observer_file'], spec['evaluation_file'], 'task.json'):
        shutil.copy2(folder / filename, output / filename)
    save(output / 'launch_contract.json', {'task': name, 'actor_run': str(actor_run),
        'checkpoint_sha256': digest(checkpoint), 'policy_sha256': digest(policy),
        'command': [str(script), *command, *extra]})
    sys.argv = [str(script), *command, *extra]
    stdout, stderr = sys.stdout, sys.stderr
    with (output / 'console.log').open('x', buffering=1) as log:
        sys.stdout, sys.stderr = RunLog(stdout, log), RunLog(stderr, log)
        try: module_globals['main']()
        finally: sys.stdout, sys.stderr = stdout, stderr


def inspect_task(name):
    folder, spec, training = load(name)
    files = {key: read_json(folder / spec[key]) for key in
             ('environment_file', 'evaluation_file', 'curriculum_file', 'observer_file', 'optimization_file')
             if key in spec}
    from dataclasses import asdict
    from cart_double_pendulum.common import CONTRACT
    from cart_double_pendulum.plant_scales import plant_report, plant_scales
    print(json.dumps({'task': spec, 'training': training, 'reward': read_json(folder / 'reward.json'),
                      'files': files, 'plant': CONTRACT.plant_parameters(),
                      'plant_scales': asdict(plant_scales()), 'plant_report': plant_report(),
                      'results_root': str(RESULTS / name)}, indent=2))


def task_main(name):
    if name not in NAMES:
        raise ValueError(name)
    if any(arg == '--task' or arg.startswith('--task=') for arg in sys.argv[1:]):
        raise ValueError('The per-task entrypoint cannot launch a different task')
    sys.argv = [sys.argv[0], *(sys.argv[1:] or ['inspect']), '--task', name]
    main()


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('inspect', 'train', 'train-observer', 'evaluate'))
    parser.add_argument('--task', choices=NAMES, required=True)
    parser.add_argument('--episodes', type=int, default=3, help='evaluate: episodes per condition')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--policy', help='evaluate: ONNX policy (default: latest run policy.onnx)')
    args, extra = parser.parse_known_args()
    if args.action == 'inspect':
        inspect_task(args.task)
    elif args.action == 'evaluate':
        evaluate(args.task, args.episodes, args.seed, args.policy)
    else:
        if args.seed is not None:
            extra.extend(['--seed', str(args.seed)])
        (train_observer if args.action == 'train-observer' else train)(args.task, extra)


if __name__ == '__main__':
    main()
