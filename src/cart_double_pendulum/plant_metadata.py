"""Plant identity for non-ONNX artifacts; checked at every bank/observer load."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from cart_double_pendulum.common import CONTRACT


def plant_metadata(contract=CONTRACT):
    values = asdict(contract)
    values.pop('model_path')
    # JSON-normalise tuples so a saved and reloaded record compares identically.
    return {'format': 'cart_double_pendulum.plant.v1',
            'model_sha256': hashlib.sha256(Path(contract.model_path).read_bytes()).hexdigest(),
            'contract': json.loads(json.dumps(values))}


def check_plant_metadata(metadata, contract=CONTRACT, *, where='artifact'):
    expected = plant_metadata(contract)
    if not isinstance(metadata, dict) or metadata.get('format') != expected['format']:
        raise ValueError(f'{where}: missing plant metadata; rebuild the artifact for the current plant')
    wrong = [key for key in expected if metadata.get(key) != expected[key]]
    if wrong:
        raise ValueError(f'{where}: different plant ({", ".join(wrong)}); rebuild for the current MJCF')


def load_bank(path, contract=CONTRACT):
    """Return validated states; metadata lives inside the NPZ, not an optional sidecar."""
    with np.load(path, allow_pickle=False) as bank:
        q = np.asarray(bank['qpos'], dtype=np.float32)
        v = np.asarray(bank['qvel'], dtype=np.float32)
        if q.shape != v.shape or q.ndim != 2 or q.shape[1] != 3 or not len(q):
            raise ValueError(f'{path}: bank must contain matching nonempty N x 3 qpos/qvel arrays')
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError(f'{path}: bank contains non-finite states')
        metadata = json.loads(str(bank['plant_metadata'].item())) if 'plant_metadata' in bank else None
        check_plant_metadata(metadata, contract, where=str(path))
    return q, v
