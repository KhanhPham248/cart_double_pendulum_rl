# Fixed-equilibrium and TIP tasks

## Target convention

Targets refer to the two **absolute link angles**. The second joint in the MJCF
is relative: absolute angles are `q1` and `q1 + q2`, with hanging equal to zero.

| Task | Link 1 | Link 2 | Absolute target (rad) |
|---|---|---|---|
| `sac_ep0` | down | down | (0, 0) |
| `sac_ep1` | down | up | (0, π) |
| `sac_ep2` | up | down | (π, 0) |
| `sac_ep3` | up | up | (π, π) |

## Configuration

Every `tasks/real/<task>/` directory contains its own task, training, environment,
reward and evaluation JSON. These are the source of truth for new runs.

EP tasks share the TIP product reward with velocity multiplier `rate_scale=5`,
a discounted rail/numerical-failure penalty and a target-aware precision-center
bonus of weight 3. The center bonus combines broad and fine Cauchy kernels,
a target-angle gate and a settling factor. Its fine tolerance and width are
both 5 mm. This is a soft reward, not a hard position constraint.

`tip_swing_sac` uses rate scale 1 and additional near-upright braking/capture terms.
TIP and fixed-EP rewards and evaluation protocols are therefore distinct.

The default independent SAC profile uses 128 environments, 20M transitions,
16,384 learning-start transitions, batch 512, replay capacity 1M, learning rate
0.0003, UTD ratio 0.03125, a 256×256 backbone and gamma 0.995. Task seeds differ.
Training snapshots preserve the exact settings used for each run.

## Training and evaluation

```bash
python tasks/real/sac_ep3/run.py inspect
python tasks/real/sac_ep3/run.py train
python tasks/real/sac_ep3/run.py evaluate --episodes 32
```

Replace `sac_ep3` with another task. EP resets mix near-target starts, starts near
other equilibria and moving transitions. The reset curriculum widens over time.
Evaluation compares the state against the selected absolute target and requires
one continuous second inside angle/speed thresholds and cart **±5 mm**. Later
rail/numerical failures invalidate the episode. Nominal, randomized and moving
randomized cases have separate seeds and conditions in `evaluation.json`.

Do not compare success rates from different protocol versions as if they were
one benchmark. Video scenarios are documented separately in the video gallery.

## Queue

```bash
python scripts/watch_ep_queue.py --gpu 1 --max-tasks 2 \
  --tasks sac_ep1 sac_ep2 sac_ep3 --state-dir watcher_gpu1
```

The cap counts all observed compute processes on the selected physical GPU and
managed children still starting. State and logs live in the selected state directory.
Restart with the same state directory to continue pending jobs. A file lock prevents
duplicate watchers. Completed/failed jobs are not retried automatically.

## Continue from a checkpoint

Set `resume_checkpoint` in the task's `training.json`. The checkpoint must belong
to the same named task and compatible plant/agent contract, with its original
`launch_contract.json` alongside the source run. With resume configured,
`total_transitions` means **additional transitions**.

Restoration includes actor, critics, target networks, optimizer states, entropy
state and observation normalization. Old checkpoints do not contain replay,
physical state or RNG state: replay is recollected with the restored actor;
updates wait for the configured refill budget. The phase/curriculum clock restarts.
The initial restored policy is rescored under the current objective before selecting
new best checkpoints. Provenance is recorded in `resume.json` and `launch_contract.json`.

## Supplied demonstration policies

EP1–EP3 are frozen best exports from a centered continuation run, each trained
for an additional 10M transitions from its earlier best checkpoint. EP0 is an
older best export and did not receive that continuation. Policy hashes and source
run paths are recorded in `policies/ep*/provenance.json`. Source paths describe
provenance and are not dependencies needed to run the supplied policies.
