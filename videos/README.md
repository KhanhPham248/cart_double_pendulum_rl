# EP0–EP3 video gallery

Four fixed-target policies, each shown in three scenarios. Click a thumbnail to
open its MP4 file; use **Raw / Download** if your GitHub client does not play it inline.
All clips are 20 seconds at 25 fps, 800×600, H.264. Thumbnails use the same 8.04 s
timestamp, not a selected best moment.

| Target | Nominal | Noise | Push |
|---|---|---|---|
| EP0 | [![EP0 nominal](ep0_nominal.jpg)](ep0_nominal.mp4) | [![EP0 noise](ep0_noise.jpg)](ep0_noise.mp4) | [![EP0 push](ep0_push.jpg)](ep0_push.mp4) |
| EP1 | [![EP1 nominal](ep1_nominal.jpg)](ep1_nominal.mp4) | [![EP1 noise](ep1_noise.jpg)](ep1_noise.mp4) | [![EP1 push](ep1_push.jpg)](ep1_push.mp4) |
| EP2 | [![EP2 nominal](ep2_nominal.jpg)](ep2_nominal.mp4) | [![EP2 noise](ep2_noise.jpg)](ep2_noise.mp4) | [![EP2 push](ep2_push.jpg)](ep2_push.mp4) |
| EP3 | [![EP3 nominal](ep3_nominal.jpg)](ep3_nominal.mp4) | [![EP3 noise](ep3_noise.jpg)](ep3_noise.mp4) | [![EP3 push](ep3_push.jpg)](ep3_push.mp4) |

## Protocol

- **Nominal:** measured plant, mass/damping/actuator scales all 1, sensor noise off,
  no force noise and no external push.
- **Noise:** same plant/reset, sensor noise scalar std **0.010**, cart force OU noise
  std **1% of actuator force limit**, correlation 0.25 s, clipped at ±3 std.
  The sensor scalar multiplies the runtime's channel-specific noise profile; it
  is not 0.010 m or 0.010 rad in every channel. This matches GUI medium noise
  after its amplitude was doubled.
- **Push:** nominal sensors/plant, +1 N along world X on the cart at 8 s and −1 N
  at 14 s, each for 0.1 s. Pushes use external body forces; the learned action
  still passes through the runtime rail guard. Noise is off in this scenario.
- **Reset:** near the opposite equilibrium, cart sampled within ±10 mm and joint
  perturbations within ±0.03 rad, zero initial velocities. Seed 42 is paired
  across the three scenarios for each target. EP0 therefore starts near upright
  instead of being initialized at its already-satisfied down/down target.
- **Indicator:** “CENTERED + STILL” combines task target-angle/speed thresholds
  with cart ±5 mm. “Longest hold” is a demonstration statistic, not the official
  task benchmark. Any detected rail-limit event remains visibly flagged.

## Policy provenance

EP0 uses an earlier best checkpoint, without the later centered continuation.
EP1–EP3 use best policies from the additional-10M centered continuation. All four
frozen ONNX files and hashes are in `../policies/ep*/provenance.json`.

Each clip is one complete fixed-seed rollout; no failed segment is removed and no
seed search is performed. These videos illustrate behavior and do not establish
success rates, robustness across all randomizations or physical hardware readiness.

## Reproduce

From repository root, install the video extra and FFmpeg/EGL support:

```bash
python -m pip install -e '.[video]'
MUJOCO_GL=egl python scripts/render_ep_videos.py
```

The [manifest](manifest.json) records model/policy hashes, initial states, parameters,
push events and per-clip summaries. Matching JSON sidecars and sampled traces
are provided beside each MP4. Output filenames are overwritten on rerender.
