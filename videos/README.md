# EP0–EP3 video gallery

Four fixed-target policies, each shown in nominal, noisy and pushed conditions. The
animated GIFs below play inline and loop; clicking an animation opens that GIF directly.
Each GIF preserves 800×600 resolution at 20 fps with a 256-color palette and high-quality
dithering. The `init_` files identify EP0, which uses the earlier policy; `update_` identifies
EP1–EP3 centered updates.

| Target | Nominal | Noise | Push |
|---|---|---|---|
| EP0 | [![EP0 nominal](init_ep0_nominal.gif)](init_ep0_nominal.gif) | [![EP0 noise](init_ep0_noise.gif)](init_ep0_noise.gif) | [![EP0 push](init_ep0_push.gif)](init_ep0_push.gif) |
| EP1 | [![EP1 nominal](update_ep1_nominal.gif)](update_ep1_nominal.gif) | [![EP1 noise](update_ep1_noise.gif)](update_ep1_noise.gif) | [![EP1 push](update_ep1_push.gif)](update_ep1_push.gif) |
| EP2 | [![EP2 nominal](update_ep2_nominal.gif)](update_ep2_nominal.gif) | [![EP2 noise](update_ep2_noise.gif)](update_ep2_noise.gif) | [![EP2 push](update_ep2_push.gif)](update_ep2_push.gif) |
| EP3 | [![EP3 nominal](update_ep3_nominal.gif)](update_ep3_nominal.gif) | [![EP3 noise](update_ep3_noise.gif)](update_ep3_noise.gif) | [![EP3 push](update_ep3_push.gif)](update_ep3_push.gif) |

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
python scripts/create_ep_gifs.py --remove-source-mp4
```

The renderer creates temporary MP4s for palette generation. The final command
updates the manifest and sidecars, verifies the GIF files, then deletes those MP4s.

The [manifest](manifest.json) records model/policy hashes, initial states, parameters,
push events, GIF settings and per-clip summaries. Matching JSON sidecars and sampled
traces are provided beside each GIF. Output filenames are overwritten on rerender.
