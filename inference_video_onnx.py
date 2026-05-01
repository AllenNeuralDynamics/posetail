"""
ONNX-based video inference. Mirrors inference_video.py but runs the exported
tracker_encoder.onnx via onnxruntime instead of the PyTorch TrackerEncoder.

Example invocation:

    python inference_video_onnx.py \
        --onnx tracker_encoder.onnx \
        --trial-path /path/to/session/trial/ \
        --start-frame 0 \
        --n-frames 256 \
        --n-overlap 2 \
        --providers CPUExecutionProvider \
        --outpath /path/to/output.npz

The ONNX model has fixed T=16, H=256, W=256 and dynamic cams/B/N. See
ONNX_INTERFACE.md for the input/output contract.
"""
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

import onnxruntime as ort

from inference_video import (
    build_video_readers,
    load_multiview_clip,
    crop_camera_group_to_queries,
    resize_camera_group,
    load_camera_group_from_metadata,
    load_trial,
    resolve_video_paths,
    camera_group_to_device,
)


IMAGE_SIZE = 256
N_FRAMES   = 16


ONNX_OUTPUT_NAMES = [
    "coords_pred", "3d_pred_cams_direct", "3d_pred_cams_rays",
    "conf_3d", "3d_pred_direct", "3d_pred_rays", "3d_pred_triangulate",
    "2d_pred", "vis_pred", "conf_pred",
    "vis_pred_2d", "conf_pred_2d", "depth_pred",
]


def stack_camera_group(camera_group):
    """Convert list-of-dicts camera_group into the 4 stacked tensors the ONNX
    model expects (cam_ext, cam_mat, cam_dist, cam_offset)."""
    cam_ext    = torch.stack([c['ext']    for c in camera_group]).to(torch.float32)
    cam_mat    = torch.stack([c['mat']    for c in camera_group]).to(torch.float32)
    cam_dist   = torch.stack([c['dist']   for c in camera_group]).to(torch.float32)
    cam_offset = torch.stack([c['offset'] for c in camera_group]).to(torch.float32)
    return cam_ext, cam_mat, cam_dist, cam_offset


def build_session(onnx_path, providers=None):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    active = session.get_providers()
    print(f'onnxruntime providers active: {active}')
    if 'CUDAExecutionProvider' in providers and 'CUDAExecutionProvider' not in active:
        print('  warning: CUDAExecutionProvider requested but not active — '
              'falling back to CPU. Install onnxruntime-gpu or check CUDA setup.')
    return session


def run_onnx_tracker_on_videos(
    session,
    video_paths,
    camera_group,
    query_points_3d,
    start_frame=0,
    n_frames=128,
    n_overlap=2,
):
    """ONNX equivalent of run_tracker_encoder_on_videos. Returns the same dict
    keys: coords_pred, vis_pred, conf_pred, frame_numbers, crop_history."""
    camera_group = camera_group_to_device(camera_group, 'cpu')

    readers, reader_lengths = build_video_readers(video_paths)

    max_available = min(reader_lengths)
    if start_frame < 0:
        raise ValueError('start_frame must be >= 0')
    if n_frames <= 0:
        raise ValueError('n_frames must be > 0')
    if n_overlap < 1:
        raise ValueError('n_overlap must be >= 1')
    if n_overlap >= N_FRAMES:
        raise ValueError(f'n_overlap ({n_overlap}) must be less than N_FRAMES ({N_FRAMES})')
    if start_frame >= max_available:
        raise ValueError('start_frame is beyond the available frames in at least one video')

    current_frame = start_frame
    end_frame = min(start_frame + n_frames, max_available)

    current_queries = query_points_3d.to(dtype=torch.float32)
    if current_queries.ndim == 2:
        current_queries = current_queries.unsqueeze(0)

    coords_pred_all = []
    vis_pred_all = []
    conf_pred_all = []
    frame_numbers_all = []
    crop_history = []
    query_times = None  # torch.int64 tensor once set

    pbar = tqdm(total=end_frame - start_frame, desc='Tracking (onnx)', unit='frames')
    while current_frame < end_frame:
        remaining = end_frame - current_frame
        current_clip_len = N_FRAMES

        camera_group_chunk, crop_boxes = crop_camera_group_to_queries(
            camera_group=camera_group,
            query_coords_3d=current_queries,
            min_crop_dim=IMAGE_SIZE,
            padding=20,
        )
        camera_group_chunk = resize_camera_group(camera_group_chunk, IMAGE_SIZE)

        target_sizes = [tuple(cam['size'].tolist()) for cam in camera_group_chunk]

        views, actual_clip_len = load_multiview_clip(
            readers,
            current_frame,
            current_clip_len,
            crop_boxes=crop_boxes,
            target_sizes=target_sizes,
        )

        if actual_clip_len == 0:
            break

        # Pad to exactly N_FRAMES by repeating the last frame
        if actual_clip_len < N_FRAMES:
            pad_len = N_FRAMES - actual_clip_len
            for i in range(len(views)):
                last_frame = views[i][-1:]
                padding = last_frame.expand(pad_len, -1, -1, -1)
                views[i] = torch.cat([views[i], padding], dim=0)

        keep_len = min(actual_clip_len, remaining)

        for i in range(len(views)):
            if views[i].shape[0] != actual_clip_len:
                raise ValueError('All videos must provide the same number of frames')

        crop_history.append({
            'start_frame': current_frame,
            'n_frames': actual_clip_len,
            'crop_boxes': crop_boxes,
        })

        # [cams, B=1, T, H, W, C] in [0, 1]
        views_stacked = torch.stack([v.to(torch.float32) / 255.0 for v in views]).unsqueeze(1)

        cam_ext, cam_mat, cam_dist, cam_offset = stack_camera_group(camera_group_chunk)

        if query_times is None:
            qtimes = np.zeros((current_queries.shape[0], current_queries.shape[1]),
                              dtype=np.int64)
        else:
            qtimes = query_times.cpu().to(torch.int64).numpy()

        feed = {
            "views":       views_stacked.numpy(),
            "coords":      current_queries.cpu().numpy().astype(np.float32),
            "query_times": qtimes,
            "cam_ext":     cam_ext.numpy(),
            "cam_mat":     cam_mat.numpy(),
            "cam_dist":    cam_dist.numpy(),
            "cam_offset":  cam_offset.numpy(),
        }

        ort_out = session.run(None, feed)
        coords_pred = torch.from_numpy(ort_out[ONNX_OUTPUT_NAMES.index("coords_pred")])
        vis_pred    = torch.from_numpy(ort_out[ONNX_OUTPUT_NAMES.index("vis_pred")])
        conf_pred   = torch.from_numpy(ort_out[ONNX_OUTPUT_NAMES.index("conf_pred")])

        coords_pred = coords_pred[:, :keep_len]
        vis_pred    = vis_pred[:, :keep_len]
        conf_pred   = conf_pred[:, :keep_len]

        if coords_pred.shape[1] == 0:
            break

        if query_times is not None:
            discard = min(n_overlap, keep_len)
        else:
            discard = 0

        coords_pred_all.append(coords_pred[:, discard:])
        vis_pred_all.append(vis_pred[:, discard:])
        conf_pred_all.append(conf_pred[:, discard:])
        frame_numbers_all.append(
            torch.arange(current_frame + discard, current_frame + keep_len, dtype=torch.int64))

        current_queries = coords_pred[:, -1]

        query_times = torch.full(
            (current_queries.shape[0], current_queries.shape[1]),
            n_overlap - 1,
            dtype=torch.int64,
        )

        current_frame += keep_len - n_overlap
        pbar.update(keep_len - discard)

        if current_frame + n_overlap >= end_frame:
            break

    pbar.close()
    del readers

    if len(coords_pred_all) == 0:
        return {
            'coords_pred': torch.empty((0, 0, 0, 3)),
            'vis_pred': torch.empty((0, 0, 0, 1)),
            'conf_pred': torch.empty((0, 0, 0, 1)),
            'frame_numbers': torch.empty((0,), dtype=torch.int64),
            'crop_history': crop_history,
        }

    return {
        'coords_pred': torch.cat(coords_pred_all, dim=1),
        'vis_pred': torch.cat(vis_pred_all, dim=1),
        'conf_pred': torch.cat(conf_pred_all, dim=1),
        'frame_numbers': torch.cat(frame_numbers_all, dim=0),
        'crop_history': crop_history,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--onnx', type=str, default='tracker_encoder.onnx',
                        help='Path to exported ONNX model')
    parser.add_argument('--trial-path', type=str, required=True,
                        help='Path to a trial directory containing metadata.yaml, '
                             'pose3d.npz, and an img/ or vid/ folder')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--n-frames', type=int, default=128)
    parser.add_argument('--n-overlap', type=int, default=2)
    parser.add_argument('--per-subject', action='store_true', default=False,
                        help='Track each subject independently instead of concatenating all keypoints')
    parser.add_argument('--providers', type=str, nargs='+',
                        default=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                        help='onnxruntime execution providers, in priority order. '
                             'Default tries CUDA first, falls back to CPU.')
    parser.add_argument('--outpath', type=str, default=None,
                        help='Optional output .npz path')

    return parser.parse_args()


def main():
    args = parse_args()

    session = build_session(args.onnx, providers=args.providers)

    metadata_path, video_paths, query_points_3d, per_subject_queries = load_trial(
        args.trial_path, start_frame=args.start_frame)

    video_paths = resolve_video_paths(video_paths)

    camera_group = load_camera_group_from_metadata(metadata_path, device='cpu')

    if len(video_paths) != len(camera_group):
        raise ValueError(
            f'Number of video paths ({len(video_paths)}) does not match '
            f'number of cameras ({len(camera_group)}) in metadata'
        )

    if args.per_subject:
        all_subject_outputs = []
        for subj_idx, subj_queries in enumerate(per_subject_queries):
            if subj_queries.shape[0] == 0:
                print(f'Skipping subject {subj_idx}: no valid query points')
                continue
            print(f'Tracking subject {subj_idx} ({subj_queries.shape[0]} keypoints)')
            subj_out = run_onnx_tracker_on_videos(
                session=session,
                video_paths=video_paths,
                camera_group=camera_group,
                query_points_3d=subj_queries,
                start_frame=args.start_frame,
                n_frames=args.n_frames,
                n_overlap=args.n_overlap,
            )
            subj_out['subject_idx'] = subj_idx
            all_subject_outputs.append(subj_out)

        if len(all_subject_outputs) == 0:
            outputs = {
                'coords_pred': torch.empty((0, 0, 0, 3)),
                'vis_pred': torch.empty((0, 0, 0, 1)),
                'conf_pred': torch.empty((0, 0, 0, 1)),
                'frame_numbers': torch.empty((0,), dtype=torch.int64),
                'crop_history': [],
            }
        else:
            outputs = {
                'frame_numbers': all_subject_outputs[0]['frame_numbers'],
                'crop_history': [],
            }
            coords_list, vis_list, conf_list = [], [], []
            for so in all_subject_outputs:
                coords_list.append(so['coords_pred'])
                vis_list.append(so['vis_pred'])
                conf_list.append(so['conf_pred'])
                outputs['crop_history'].extend(so['crop_history'])

            outputs['coords_pred'] = torch.cat(coords_list, dim=2)
            outputs['vis_pred']    = torch.cat(vis_list,    dim=2)
            outputs['conf_pred']   = torch.cat(conf_list,   dim=2)

            outputs['subject_kpt_counts'] = np.array(
                [so['coords_pred'].shape[2] for so in all_subject_outputs], dtype=np.int32)
            outputs['subject_indices'] = np.array(
                [so['subject_idx'] for so in all_subject_outputs], dtype=np.int32)
    else:
        outputs = run_onnx_tracker_on_videos(
            session=session,
            video_paths=video_paths,
            camera_group=camera_group,
            query_points_3d=query_points_3d,
            start_frame=args.start_frame,
            n_frames=args.n_frames,
            n_overlap=args.n_overlap,
        )

    outputs['video_paths'] = np.array(video_paths, dtype=str)
    outputs['metadata_path'] = metadata_path
    outputs['trial_path'] = args.trial_path
    outputs['onnx_path'] = os.path.abspath(args.onnx)
    outputs['start_frame'] = args.start_frame
    outputs['n_frames_requested'] = args.n_frames
    outputs['n_overlap'] = args.n_overlap
    outputs['per_subject'] = args.per_subject
    if isinstance(outputs['coords_pred'], torch.Tensor) and outputs['coords_pred'].ndim >= 2:
        outputs['n_frames_returned'] = outputs['coords_pred'].shape[1]
    else:
        outputs['n_frames_returned'] = 0

    if args.outpath is not None:
        save_dict = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                save_dict[k] = v.cpu().numpy()
            else:
                save_dict[k] = v

        crop_history_serializable = [
            {
                'start_frame': int(ch['start_frame']),
                'n_frames': int(ch['n_frames']),
                'crop_boxes': [cb.cpu().numpy().tolist() for cb in ch['crop_boxes']],
            }
            for ch in outputs['crop_history']
        ]
        save_dict['crop_history'] = json.dumps(crop_history_serializable)

        outdir = os.path.dirname(args.outpath)
        if outdir != '':
            os.makedirs(outdir, exist_ok=True)
        np.savez(args.outpath, **save_dict)
        print(f'Saved outputs to {args.outpath}')
    else:
        print('Inference completed.')
        print(f'onnx_path: {outputs["onnx_path"]}')
        n_returned = outputs['coords_pred'].shape[1] if outputs['coords_pred'].ndim >= 2 else 0
        print(f'n_frames_returned: {n_returned}')


if __name__ == '__main__':
    main()
