# Cart Double Pendulum RL

MuJoCo reinforcement learning for swing-up, stabilization and cart centering of a
measured two-link cart pendulum. The repository contains independent SAC/PPO tasks,
training launchers, evaluation, checkpoint continuation and reproducible demonstration videos.

## Tasks

| Task | Algorithm | Target |
|---|---|---|
| `expert_swing_sac` | SAC | Hanging to calm upright |
| `expert_balance_ppo` | PPO | Upright balance with history and observer |
| `full_sac`, `full_ppo` | SAC / PPO | Swing-up, capture and centering |
| `tip_swing_sac` | SAC | Upright TIP reward with braking and capture |
| `sac_ep0` | SAC | Both links down |
| `sac_ep1` | SAC | Link 1 down, link 2 up |
| `sac_ep2` | SAC | Link 1 up, link 2 down |
| `sac_ep3` | SAC | Both links up |

EP targets use **absolute link angles**, not the relative second-joint angle.
TIP/EP task definitions include a target-aware centering reward and a ±5 mm evaluation
criterion. Policies shipped for EP1–EP3 come from centered continuation runs; EP0 is
an earlier best policy and was not retrained for that criterion.

## Videos

See the [video gallery](videos/README.md) for EP0–EP3 under nominal conditions,
sensor/force noise and signed cart pushes. Each video is a deterministic single-rollout
demonstration with its protocol and policy hashes recorded; it is not a success-rate benchmark.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,video]'
```

Python and dependency versions are declared in [pyproject.toml](pyproject.toml).
For offscreen rendering, install FFmpeg and EGL/OpenGL drivers. No GPU is required
for ONNX inference; training can use CUDA.

## Quick start

Run commands from this repository's root:

```bash
python tasks/real/sac_ep3/run.py inspect
python tasks/real/sac_ep3/run.py train
python tasks/real/sac_ep3/run.py evaluate --episodes 32
```

Select any of the nine task folders. Each run writes a separate directory under
`results/real/<task>/runs/`; generated training artifacts are ignored by Git.
See [EP task guide](docs/EP_TASKS.md) for rewards, profiles, evaluation and resume semantics.

### Queue selected EP tasks

```bash
python scripts/watch_ep_queue.py --gpu 1 --max-tasks 2 \
  --tasks sac_ep1 sac_ep2 sac_ep3 --state-dir watcher_gpu1
```

The cap counts other compute processes on the selected physical GPU. The watcher
queues training; it does not render videos. No training starts during installation.

### Expert/full training pipeline

```bash
python scripts/train_pipeline.py --directory ~/cart_runs/experiment --physical-gpu 0
python scripts/train_pipeline.py --directory ~/cart_runs/experiment --resume
```

The dependency pipeline snapshots source, trains swing, builds a handoff bank,
trains observer/balance stages and runs the two full tasks. Banks and frozen
observers are generated inputs, not supplied pretrained artifacts. TIP/EP tasks
run independently and do not need that bank.

### Checkpoint continuation

Set `resume_checkpoint` in the selected SAC task's `training.json`, keeping the
original checkpoint's `launch_contract.json` in its expected parent directory.
`total_transitions` then means **additional** transitions. Actor, critics, targets,
optimizers and normalization state are restored. Replay, environment and RNG state
are recollected/reset; this is not bit-exact rollout recovery.

## Reproduce videos

```bash
MUJOCO_GL=egl python scripts/render_ep_videos.py
```

The renderer uses only this repository's model, runtime and four supplied EP policy
bundles. The script records scenario parameters, hashes, push events and trajectory
summaries in `videos/manifest.json`. Existing filenames are overwritten on rerender.

## Validate

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The suite exercises plant contracts, rewards, task launchers, optimizer/ONNX smoke,
checkpoint restoration and GPU queue behavior. Passing tests does not establish
physical hardware acceptance.

## Layout

```text
assets/       Measured MJCF plant
src/          Environment, rewards, SAC, inference and evaluators
scripts/      Launchers, training pipeline, queue and video renderer
tasks/real/   Nine independent task configurations
tests/        Automated verification
policies/     Four frozen EP ONNX bundles and provenance
videos/       Scenario gallery, MP4 clips and reproducibility manifest
docs/         Task configuration and training guide
```

## License

Apache-2.0. Original copyright and attribution are preserved in
[LICENSE](LICENSE) and [NOTICE](NOTICE).
