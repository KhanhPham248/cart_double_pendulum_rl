# Handoff bank and observer inputs

The balancing and full-process tasks need reset samples near the upright handoff
region. These samples are generated from a qualified swing policy, not bundled
as a pretrained training input.

```bash
python tasks/real/expert_swing_sac/run.py train
python scripts/build_handoff_bank.py --policy <swing-run>/train/policy.onnx
```

The builder records states and plant identity in `data/handoff_bank.npz` and a
JSON sidecar. Inspect the report before using the bank; artifacts from another
MJCF plant are rejected. Consult `--help` for output and sampling options.

A typical balance sequence is:

```bash
python tasks/real/expert_balance_ppo/run.py train --without-observer
python tasks/real/expert_balance_ppo/run.py train-observer
cp <balance-run>/observer/best.pt tasks/real/expert_balance_ppo/frozen_observer.pt
python tasks/real/expert_balance_ppo/run.py train
```

Observer qualification must succeed before copying `best.pt`. The automated
pipeline handles dependencies and checks input availability. See the root README
for its commands. Independent TIP/EP tasks can train without the handoff bank.
