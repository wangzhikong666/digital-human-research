"""Render saved GAGAvatar Track parameters without accumulating all video frames.

Run in the GAGAvatar-5060 environment with --tracker-root, --run-dir and --output.
Uses the official FLAME model, renderer and visualization layout; preserves fits.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import torch
import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker-root', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fps', required=True, type=float)
    args = parser.parse_args()
    assert not args.output.exists(), 'Refusing to overwrite a video'
    report_path = args.output.with_suffix('.render.json')
    assert not report_path.exists(), 'Refusing to overwrite a render record'
    sys.path.insert(0, str(args.tracker_root))
    from engines import FLAMEModel, LMDBEngine, RenderMesh

    start = time.perf_counter()
    params_path = args.run_dir / 'smoothed.pkl'
    before = hashlib.sha256(params_path.read_bytes()).hexdigest()
    with params_path.open('rb') as stream:
        results = pickle.load(stream)
    frames = sorted(results, key=lambda name: int(name.split('_')[-1]))
    model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0).to('cuda')
    renderer = RenderMesh(512, faces=model.get_faces().cpu().numpy(), device='cuda')
    lmdb = LMDBEngine(str(args.run_dir / 'img_lmdb'), write=False)
    argv = [shutil.which('ffmpeg'), '-nostdin', '-n', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1536x512', '-r', str(args.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(args.output)]
    writer = subprocess.Popen(argv, stdin=subprocess.PIPE)
    try:
        with torch.inference_mode():
            for idx, key in enumerate(frames):
                params = results[key]
                vertices, _ = model(
                    shape_params=torch.as_tensor(params['shapecode'], device='cuda')[None],
                    expression_params=torch.as_tensor(params['expcode'], device='cuda')[None],
                    pose_params=torch.as_tensor(params['posecode'], device='cuda')[None].float(),
                    eye_pose_params=torch.as_tensor(params['eyecode'], device='cuda')[None].float(),
                )
                images, alpha = renderer(vertices, focal_length=12.0, transform_matrix=torch.as_tensor(params['transform_matrix'], device='cuda')[None])
                alpha = alpha[0].expand(3, -1, -1)
                original = lmdb[key].to('cuda').float()
                overlay = original.clone()
                overlay[alpha > 0.5] *= 0.3
                overlay[alpha > 0.5] += images[0, alpha > 0.5] * 0.7
                grid = torchvision.utils.make_grid([original.cpu(), overlay.cpu().to(torch.uint8), images[0].cpu()], nrow=3, padding=0)
                rgb = grid.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
                writer.stdin.write(rgb.tobytes())
                if idx % 50 == 0:
                    print(f'Rendered {idx + 1}/{len(frames)} frames', flush=True)
    finally:
        writer.stdin.close()
        code = writer.wait()
        lmdb.close()
    assert code == 0, code
    after = hashlib.sha256(params_path.read_bytes()).hexdigest()
    assert before == after
    report = {'passed': True, 'rendered_frames': len(frames), 'fps': args.fps, 'parameter_file': str(params_path), 'parameter_sha256_before': before, 'parameter_sha256_after': after, 'render_wall_time_seconds': time.perf_counter() - start, 'layout': 'official original/overlay/mesh, 1536x512', 'focal_length': 12.0, 'verts_scale': 5.0, 'encoder_argv': argv, 'reason': 'Render saved official fits one frame at a time to avoid accumulating the full video in memory.'}
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
