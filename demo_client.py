"""
Demo client for server.py.

Shows how to call POST /infer with multipart/form-data:
  - `metadata` form field: JSON string with camera calibration + query points
  - `cam_<i>_frames` form fields: 16 PNG bytes per camera, in temporal order

Usage:
    pixi run python demo_client.py [--url http://localhost:8000] \\
                                   [--n-cams 2] [--n-points 3] \\
                                   [--height 480] [--width 640]
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np
import requests


N_FRAMES = 16


def png_bytes(rgb_image: np.ndarray) -> bytes:
    """Encode an RGB uint8 image as PNG bytes."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return buf.tobytes()


def make_synthetic_camera(idx: int, height: int, width: int) -> dict:
    """Build a plausible pinhole camera with identity-ish extrinsics shifted along x."""
    return {
        'name':   f'cam_{idx}',
        # World->camera SE3. Cameras translated along x by 0.5*idx, looking at Z=0 plane.
        'ext':    [[1.0, 0.0, 0.0, float(idx) * 0.5],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 2.0],
                   [0.0, 0.0, 0.0, 1.0]],
        # K with focal ~400 px and principal point at image center.
        'mat':    [[400.0,   0.0, width  / 2.0],
                   [  0.0, 400.0, height / 2.0],
                   [  0.0,   0.0,           1.0]],
        # Zero distortion (undistorted images).
        'dist':   [0.0, 0.0, 0.0, 0.0, 0.0],
        # No crop offset (full-frame images).
        'offset': [0.0, 0.0],
    }


def make_synthetic_frames(n_cams: int, height: int, width: int) -> list[list[bytes]]:
    """Return [n_cams][N_FRAMES] of PNG bytes filled with random noise."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_cams):
        per_cam = []
        for _ in range(N_FRAMES):
            img = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            per_cam.append(png_bytes(img))
        out.append(per_cam)
    return out


def make_synthetic_queries(n_points: int) -> list[list[float]]:
    """N query points scattered near the world origin (in front of the cameras)."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-0.2, 0.2, size=(n_points, 2))
    return [[float(x), float(y), 1.0 + 0.3 * i] for i, (x, y) in enumerate(xy)]


def call_infer(url: str, cameras: list[dict], frames_per_cam: list[list[bytes]],
               query_coords: list[list[float]], query_times: list[int] | None = None):
    """POST to /infer and return the parsed response dict."""

    metadata = {
        'cameras':      cameras,
        'query_coords': query_coords,
    }
    if query_times is not None:
        metadata['query_times'] = query_times

    # Build multipart files list. requests accepts a list of (field_name, (filename, bytes, mime))
    # tuples; repeated field_names are sent as multiple files for the same form field, which is
    # what the server expects for cam_<i>_frames.
    files = []
    for cam_idx, frames in enumerate(frames_per_cam):
        field = f'cam_{cam_idx}_frames'
        for t, blob in enumerate(frames):
            files.append((field, (f'cam{cam_idx}_t{t:02d}.png', blob, 'image/png')))

    data = {'metadata': json.dumps(metadata)}

    t0 = time.perf_counter()
    resp = requests.post(f'{url}/infer', data=data, files=files, timeout=600)
    dt = time.perf_counter() - t0

    if resp.status_code != 200:
        print(f'request failed: {resp.status_code}', file=sys.stderr)
        print(resp.text[:2000], file=sys.stderr)
        resp.raise_for_status()

    print(f'POST /infer  status=200  elapsed={dt:.2f}s')
    return resp.json()


def summarize(response: dict) -> None:
    print(f'  n_cameras={response["n_cameras"]}  '
          f'n_points={response["n_points"]}  n_frames={response["n_frames"]}')
    print('  output shapes:')
    for k, sh in response['shapes'].items():
        print(f'    {k:24s} {sh}')

    coords = np.asarray(response['outputs']['coords_pred'])  # [B, T, N, 3]
    print(f'  coords_pred[0, 0, :, :] (first frame, all points):')
    for n, p in enumerate(coords[0, 0]):
        print(f'    point {n}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--url',      type=str, default='http://localhost:8000',
                   help='base URL of the server (default: %(default)s)')
    p.add_argument('--n-cams',   type=int, default=2)
    p.add_argument('--n-points', type=int, default=3)
    p.add_argument('--height',   type=int, default=480, help='source image height (px)')
    p.add_argument('--width',    type=int, default=640, help='source image width (px)')
    return p.parse_args()


def main():
    args = parse_args()

    # /health first to fail fast if server is down.
    health = requests.get(f'{args.url}/health', timeout=10).json()
    print(f'GET  /health  -> {health}')

    cameras        = [make_synthetic_camera(i, args.height, args.width)
                      for i in range(args.n_cams)]
    frames_per_cam = make_synthetic_frames(args.n_cams, args.height, args.width)
    query_coords   = make_synthetic_queries(args.n_points)

    response = call_infer(args.url, cameras, frames_per_cam, query_coords,
                          query_times=[0] * args.n_points)
    summarize(response)


if __name__ == '__main__':
    main()
