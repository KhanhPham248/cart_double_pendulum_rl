# Repository validation — 2026-11-07

## Automated checks

- Full CPU suite: **255 passed**, 174 ONNX-export deprecation warnings, 168.74 s.
- Follow-up task-directory and pipeline checks: **66 passed**, 11.94 s.
- `pyflakes scripts src tests`: passed after final launcher cleanup.
- Python source compilation: passed.

The generic ONNX actor runner is a standalone module used by expert evaluation
and observer inference. Nine task configurations are retained; all use the same
measured plant and their own reward/evaluation contracts.

## Video artifacts

- 12 complete clips: EP0–EP3 × nominal/noise/push.
- Every clip verified with FFprobe: H.264, 800×600, 25 fps, 20 s, 500 frames.
- Four frozen ONNX hashes checked before inference.
- Finite state checked every simulation step; no rail-limit event in these clips.
- Each push applies +0.25 N along world +X at the link-2 geometric center for 0.10 s at t=8.00 s. The GIF marks the contact point in red, and states force direction, magnitude, timing and duration. The script records the peak angle error and time to recover and hold for 1 s.
- Source policy hashes, reset states, protocol and traces are included with the clips.

The light push causes at least 45° target-angle error for each EP; peak error is 85.5–178.2°. All four policies return and hold within 5 mm in 1.80–2.94 s in these fixed-seed clips. These are demonstrations, not a statistical evaluation or
physical hardware qualification. EP0 predates the centered continuation used for
EP1–EP3; this difference is documented in the gallery and policy provenance.

## Repository scope

The folder is independent of the original checkout. Training outputs, local
credentials, caches, banks and generated observers are excluded by `.gitignore`.
Only the four small demonstration ONNX exports are included as pretrained policies.
Original license and attribution are preserved.

## Inline GIFs — 2026-11-07

- 12 looping GIFs generated from the full-resolution rendered clips.
- 800×600, 20 fps, 400 frames per clip, 256-color palette and Sierra dithering.
- Pillow metadata, frame count, infinite-loop flag and dimensions checked for every GIF;
  policy artifact hashes are recorded in the policy provenance files.
- GitHub gallery embeds each GIF and links directly to the same GIF.
