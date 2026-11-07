#!/usr/bin/env python3
"""Create full-resolution GitHub GIFs and optionally remove temporary MP4s."""
from __future__ import annotations
import hashlib
import json
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / 'videos'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remove-source-mp4', action='store_true',
        help='remove generated MP4s after GIFs and manifest entries are verified')
    parser.add_argument('--remove-jpg-json', action='store_true',
        help='remove gallery thumbnails, traces and JSON sidecars after conversion')
    args = parser.parse_args()
    if args.remove_jpg_json and not args.remove_source_mp4:
        parser.error('--remove-jpg-json requires --remove-source-mp4')
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg is required')
    sources = sorted(VIDEO_DIR.glob('ep[0-3]_[a-z]*.mp4'))
    if not sources:
        raise SystemExit('no temporary EP MP4 files found to convert')
    for source in sources:
        record = json.loads(source.with_suffix('.json').read_text())
        prefix = 'init' if record['target_ep'] == 0 else 'update'
        output = VIDEO_DIR / f"{prefix}_ep{record['target_ep']}_{record['scenario']}.gif"
        palette_filter = ('[0:v]fps=20,split[a][b];'
            '[a]palettegen=max_colors=256:stats_mode=full[p];'
            '[b][p]paletteuse=dither=sierra2_4a')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source),
            '-filter_complex', palette_filter, '-loop', '0', str(output)], check=True)
        record.update(gif=output.name, gif_sha256=sha256(output), gif_width=800,
                      gif_height=600, gif_fps=20, gif_loop='infinite', gif_colors=256)
        source.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
        print(f'{output.name}: {output.stat().st_size/1024:.0f} KiB sha256={sha256(output)}', flush=True)
    manifest_path = VIDEO_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for clip in manifest['clips']:
        record = json.loads((VIDEO_DIR / clip['file'].replace('.mp4','.json')).read_text())
        for key in ('gif','gif_sha256','gif_width','gif_height','gif_fps','gif_loop','gif_colors'):
            clip[key] = record[key]
        if args.remove_source_mp4:
            source = clip.pop('file')
            source_hash = clip.pop('sha256')
            clip['render_source_mp4'] = source
            clip['render_source_mp4_sha256'] = source_hash
            clip['file'] = record['gif']
            clip['sha256'] = record['gif_sha256']
            record['render_source_mp4'] = record.pop('file')
            record['render_source_mp4_sha256'] = record.pop('sha256')
            record['file'] = record['gif']
            record['sha256'] = record['gif_sha256']
            (VIDEO_DIR / source).unlink()
            (VIDEO_DIR / source.replace('.mp4','.json')).write_text(json.dumps(record, indent=2)+'\n')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    if args.remove_jpg_json:
        for pattern in ('*.jpg', '*.json'):
            for path in VIDEO_DIR.glob(pattern):
                path.unlink()


if __name__ == '__main__':
    main()
