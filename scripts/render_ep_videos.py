#!/usr/bin/env python3
"""Render paired, reproducible EP0–EP3 demonstrations; no training is launched."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cart_double_pendulum.common import CONTRACT
from cart_double_pendulum.equilibria import joint_target, target_errors, target_mask
from cart_double_pendulum.onnx_policy import BatchedOnnxPolicy
from cart_double_pendulum.plant_scales import plant_scales
from cart_double_pendulum.runtime import CartDoublePendulumRuntime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_point(point, camera, width, height, fovy_deg):
    """Project a world point into the viewer image for a temporary hit marker."""
    forward = np.asarray(camera.forward, dtype=float)
    up = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    delta = np.asarray(point) - np.asarray(camera.pos)
    depth = float(np.dot(delta, forward))
    if depth <= 0:
        return None
    tangent = math.tan(math.radians(fovy_deg) / 2)
    ndc_x = float(np.dot(delta, right)) / (depth * tangent * width / height)
    ndc_y = float(np.dot(delta, up)) / (depth * tangent)
    return (int(width * (0.5 + ndc_x / 2)), int(height * (0.5 - ndc_y / 2)))


def render(ep, scenario, args):
    seed = args.seed
    noisy = scenario == 'noise'
    runtime = CartDoublePendulumRuntime(CONTRACT.model_path, seed=seed,
        independent_observation_noise=True, observation_noise_std=args.noise_std if noisy else 0.)
    runtime.reset(mode='hanging', cart_position_m=0., angle_rad=0., velocity=0.)
    rng = np.random.default_rng(seed)
    # Start near the opposite EP, so EP0 also demonstrates a transition.
    q = np.array([rng.uniform(-.01, .01), *joint_target(3 - ep)])
    q[1:] += rng.uniform(-.03, .03, 2)
    runtime.data.qpos[list(runtime.qpos_indices)] = q
    mujoco.mj_forward(runtime.model, runtime.data)
    policy_path = ROOT / f'policies/ep{ep}/policy.onnx'
    source = json.loads(policy_path.with_name('provenance.json').read_text())
    if digest(policy_path) != source['files']['policy']['sha256']:
        raise ValueError('policy hash mismatch')
    policy = BatchedOnnxPolicy(policy_path)
    observation = runtime.observation()
    scales = plant_scales(CONTRACT)
    thresholds = dict(cart_safe_m=.65 * scales.rail, angle_threshold_rad=1.138 * scales.angle,
        cart_velocity_threshold_m_s=.418 * scales.cart_speed,
        angular_velocity_threshold_rad_s=.5285 * scales.rate)
    dt = CONTRACT.control_timestep_s
    every = round(1 / (args.fps * dt))
    if every < 1 or not math.isclose(every * dt, 1 / args.fps):
        raise ValueError('fps must divide the control frequency exactly')
    prefix = 'init' if ep == 0 else 'update'
    thumbnail_time = {'nominal': 3.0, 'noise': 10.0, 'push': 8.04}[scenario]
    output = args.output / f'ep{ep}_{scenario}.mp4'
    thumbnail = args.output / f'{prefix}_ep{ep}_{scenario}.jpg'
    encoder = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', f'{args.width}x{args.height}',
        '-framerate', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)],
        stdin=subprocess.PIPE)
    renderer = mujoco.Renderer(runtime.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 90.; camera.elevation = -8.; camera.distance = 1.65
    camera.lookat[:] = [0., -.04, .10]
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    small = ImageFont.truetype('DejaVuSans.ttf', 14)
    cart = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, 'cart')
    link2 = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, 'link2')
    link2_geom = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, 'link2_geom')
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 17001]))
    force_noise = 0.; hold = longest = 0.; rail = False
    fell = False; peak_error_deg = 0.; recovery_time_s = None
    push_start = 8.; push_end = push_start + args.push_duration
    center_steps = 0; max_cart = 0.; events = []; trace = []
    total_steps = round(args.seconds / dt)
    try:
        for step in range(total_steps):
            t = step * dt
            push_active = scenario == 'push' and push_start <= t < push_end
            if scenario == 'push' and step == round(push_start / dt):
                events.append(dict(time_s=t, force_n=args.push_force,
                                   duration_s=args.push_duration, target='link2',
                                   point='link2_geom center', direction='world +X'))
            runtime.data.xfrc_applied[:] = 0.
            if noisy:
                decay = math.exp(-dt / .25)
                std = args.force_fraction * CONTRACT.actuator_force_limit_n
                force_noise = float(np.clip(decay * force_noise + std * math.sqrt(1-decay**2)
                    * noise_rng.normal(), -3*std, 3*std))
            runtime.data.xfrc_applied[cart, 0] = force_noise
            if push_active:
                point = runtime.data.geom_xpos[link2_geom].copy()
                force = np.array([args.push_force, 0., 0.])
                runtime.data.xfrc_applied[link2, :3] = force
                runtime.data.xfrc_applied[link2, 3:] = np.cross(
                    point - runtime.data.xipos[link2], force)
            action = float(policy(observation[None, :])[0, 0])
            policy.record_applied_actions([[action]])
            observation = runtime.step(action)
            if not np.isfinite(runtime.qpos).all() or not np.isfinite(runtime.qvel).all():
                raise FloatingPointError('non-finite simulation state')
            inside = bool(target_mask(runtime.qpos, runtime.qvel, np, ep, **thresholds))
            centered = inside and abs(runtime.qpos[0]) <= .005
            hold = hold + dt if centered else 0.; longest = max(longest, hold)
            angle_error_deg = float(np.max(np.abs(np.degrees(target_errors(runtime.qpos, np, ep)))))
            if scenario == 'push' and t >= push_start:
                peak_error_deg = max(peak_error_deg, angle_error_deg)
                fell = fell or angle_error_deg >= args.fall_angle_deg
                if recovery_time_s is None and t >= push_end and hold >= args.recovery_hold_s:
                    recovery_time_s = max(0., t + dt - push_end)
            center_steps += int(centered)
            max_cart = max(max_cart, abs(runtime.qpos[0]))
            rail = rail or abs(runtime.qpos[0]) >= .95 * scales.rail
            if step % every == every - 1:
                frame_time = t + dt
                renderer.update_scene(runtime.data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                if scenario == 'push' and push_start <= frame_time <= push_end + args.push_indicator_s:
                    marker = project_point(runtime.data.geom_xpos[link2_geom], renderer.scene.camera[0],
                                           args.width, args.height, runtime.model.vis.global_.fovy)
                    if marker is not None:
                        x, y = marker
                        draw.ellipse((x-18, y-18, x+18, y+18), outline='#ff304f', width=5)
                        draw.ellipse((x-5, y-5, x+5, y+5), fill='#ff304f')
                draw.rectangle((0, 0, args.width, 78), fill='#101827')
                label = ['DOWN / DOWN', 'DOWN / UP', 'UP / DOWN', 'UP / UP'][ep]
                draw.text((16, 8), f'EP{ep}  {label}  |  {scenario.upper()}', font=font, fill='#f8fafc')
                detail = (f'Sensor std {args.noise_std:g} + cart force noise {args.force_fraction*100:g}%' if noisy else
                          f'PUSH LINK 2 (+X): {args.push_force:g} N at {push_start:.2f} s for {args.push_duration:.2f} s' if scenario == 'push' else
                          'Nominal plant | noise OFF | no external push')
                draw.text((16, 40), detail, font=small, fill='#b9c8df')
                draw.rectangle((0, args.height-65, args.width, args.height), fill='#101827')
                errors = np.degrees(np.asarray(target_errors(runtime.qpos, np, ep)))
                draw.text((16, args.height-55),
                    f't={runtime.data.time:5.2f}s   cart={runtime.qpos[0]*1000:+7.2f} mm   error={np.max(np.abs(errors)):.1f} deg',
                    font=small, fill='#f8fafc')
                if rail:
                    status = 'RAIL LIMIT REACHED'
                elif push_active:
                    status = 'PUSH ON LINK 2'
                elif scenario == 'push' and recovery_time_s is not None:
                    status = f'RECOVERED + STABLE in {recovery_time_s:.2f}s'
                elif scenario == 'push' and fell:
                    status = 'FALLEN / RECOVERING'
                else:
                    status = 'CENTERED + STILL' if centered else 'TRANSITION / RECOVERY'
                draw.text((16, args.height-30), f'{status}   |   target dev {angle_error_deg:.1f} deg   |   +/-5 mm hold {longest:.2f}s',
                    font=small, fill='#fb7185' if push_active or rail or (scenario == 'push' and fell and recovery_time_s is None) else '#7dd3fc')
                if abs(frame_time - thumbnail_time) <= .5 / args.fps:
                    frame.save(thumbnail, quality=92)
                encoder.stdin.write(np.asarray(frame).tobytes())
                trace.append(dict(time_s=float(runtime.data.time), cart_m=float(runtime.qpos[0]),
                                  max_angle_error_deg=float(np.max(np.abs(errors))),
                                  push_link2_n=args.push_force if push_active else 0.))
    finally:
        renderer.close()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited {code}')
    result = dict(target_ep=ep, scenario=scenario, file=output.name, sha256=digest(output),
        policy_sha256=digest(policy_path), thumbnail=thumbnail.name, thumbnail_time_s=thumbnail_time,
        camera=dict(azimuth=90., elevation=-8., distance=1.65, lookat_m=[0., -.04, .10]), source=source['source'], centered_training=source['centered_training'],
        seed=seed, duration_s=args.seconds, fps=args.fps, initial_qpos=q.tolist(),
        observation_noise_std=args.noise_std if noisy else 0.,
        force_noise_fraction=args.force_fraction if noisy else 0., noise_correlation_s=.25 if noisy else None,
        push_events=events, push_caused_fall=fell, peak_target_angle_error_deg=peak_error_deg,
        recovery_hold_s=args.recovery_hold_s if scenario == 'push' else None,
        recovery_time_after_push_s=recovery_time_s, recovery_success=recovery_time_s is not None if scenario == 'push' else None,
        longest_center_hold_s=longest, center_occupancy=center_steps/total_steps,
        rail_limit_reached=bool(rail), max_cart_excursion_m=float(max_cart), final_qpos=runtime.qpos.tolist())
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    output.with_suffix('.trace.json').write_text(json.dumps(trace)+'\n')
    print(json.dumps({k: result[k] for k in ('file', 'longest_center_hold_s', 'rail_limit_reached')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'videos')
    parser.add_argument('--eps', nargs='+', type=int, choices=range(4), default=list(range(4)))
    parser.add_argument('--scenarios', nargs='+', choices=['nominal', 'noise', 'push'], default=['nominal', 'noise', 'push'])
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-std', type=float, default=.01)
    parser.add_argument('--force-fraction', type=float, default=.01)
    parser.add_argument('--push-force', type=float, default=.25)
    parser.add_argument('--push-duration', type=float, default=.1)
    parser.add_argument('--push-indicator-s', type=float, default=.75)
    parser.add_argument('--fall-angle-deg', type=float, default=45.)
    parser.add_argument('--recovery-hold-s', type=float, default=1.)
    args = parser.parse_args()
    if shutil.which('ffmpeg') is None:
        parser.error('ffmpeg is required')
    args.output.mkdir(parents=True, exist_ok=True)
    records = [render(ep, scenario, args) for ep in args.eps for scenario in args.scenarios]
    (args.output/'manifest.json').write_text(json.dumps(dict(model_sha256=digest(CONTRACT.model_path),
        clips=records, note='Single fixed-seed demonstrations, not an evaluation benchmark.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
