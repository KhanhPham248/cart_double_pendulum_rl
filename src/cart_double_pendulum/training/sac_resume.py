"""Resume SAC state into a new objective/run; replay is deliberately recollected."""
import hashlib
import json
from pathlib import Path
import torch
from cart_double_pendulum.plant_metadata import check_plant_metadata


def restore(agent, checkpoint, task, contract):
    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if payload.get('format') != 'cart_double_pendulum_sac_v1' or payload.get('task') != task:
        raise ValueError('resume checkpoint format/task mismatch')
    check_plant_metadata(payload.get('plant_metadata'), contract, where=str(path))
    previous = payload['agent']['config']
    for key in ('gamma', 'tau', 'hidden_dims'):
        old, new = previous[key], getattr(agent.config, key)
        differs = tuple(old) != tuple(new) if key == 'hidden_dims' else old != new
        if differs:
            raise ValueError(f'resume SAC config mismatch: {key}')
    start = int(payload['transitions'])
    if start < 0:
        raise ValueError('resume transition counter must be nonnegative')
    agent.load_state_dict(payload['agent'])
    origin = path.parent.parent / 'launch_contract.json'
    return dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_transitions=start, restored_update_count=agent.update_count,
                restored=['actor', 'critics', 'targets', 'optimizers', 'alpha', 'normalizers'],
                replay_restored=False, environment_restored=False,
                phase_clock='reset for new objective and additional transition budget',
                source_launch=json.loads(origin.read_text()) if origin.is_file() else None)
