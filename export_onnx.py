"""
Export TrackerEncoder to ONNX.

Usage:
    python export_onnx.py [--output tracker_encoder.onnx] [--opset 18]

Loads the checkpoint and config from the hard-coded run directory, wraps the
model in TrackerEncoderONNX, and calls torch.onnx.export with dynamic axes for
cams, B, and N.  Fixed: T=16, H=256, W=256 (from model config).
"""

import argparse
import os
import sys
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
RUN_DIR = (
    "/groups/karashchuk/home/karashchukl/results/posetail-test-vjepa/"
    "wandb/run-20260410_131633-lwauuvci/files"
)
CONFIG_PATH     = os.path.join(RUN_DIR, "config.toml")
CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoints", "checkpoint_00552960.pth")

sys.path.insert(0, os.path.dirname(__file__))

from train_utils import load_config
from posetail.posetail.tracker_encoder import TrackerEncoder
from posetail.posetail.export_wrapper import TrackerEncoderONNX


def build_dummy_inputs(cams, B, N, T=16, H=256, W=256, device="cpu"):
    """Create random inputs matching the expected dtypes and shapes."""
    views      = torch.randint(0, 200, (cams, B, T, H, W, 3), dtype=torch.float32, device=device)
    coords     = torch.randn(B, N, 3, device=device)
    qtimes     = torch.zeros(B, N, dtype=torch.long, device=device)
    cam_ext    = torch.eye(4, device=device).unsqueeze(0).expand(cams, -1, -1).contiguous()
    cam_mat    = torch.eye(3, device=device).unsqueeze(0).expand(cams, -1, -1).contiguous()
    # Give non-zero focal lengths so projection_sensitivity doesn't produce NaNs
    cam_mat = cam_mat.clone()
    cam_mat[:, 0, 0] = 500.0
    cam_mat[:, 1, 1] = 500.0
    cam_mat[:, 0, 2] = W / 2.0
    cam_mat[:, 1, 2] = H / 2.0
    cam_dist   = torch.zeros(cams, 5, device=device)
    cam_offset = torch.zeros(cams, 2, device=device)
    return (views, coords, qtimes,
            cam_ext, cam_mat, cam_dist, cam_offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tracker_encoder.onnx")
    parser.add_argument("--opset",  type=int, default=18)
    parser.add_argument("--cams",   type=int, default=2,
                        help="cameras in the dummy trace input (cams/B/N are dynamic axes — "
                             "the exported model accepts any value at runtime)")
    parser.add_argument("--batch",  type=int, default=1,
                        help="batch size in the dummy trace input (dynamic at runtime)")
    parser.add_argument("--npts",   type=int, default=15,
                        help="query points in the dummy trace input (dynamic at runtime)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading config from {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)

    print(f"Building TrackerEncoder ({cfg.model.video_encoder_version})…")
    model = TrackerEncoder(**cfg.model)
    model.eval()

    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.scene_encoder.encoder.use_activation_checkpointing = False
    model.to(device)

    wrapper = TrackerEncoderONNX(model).eval().to(device)

    # ── Dummy inputs ──────────────────────────────────────────────────────────
    dummy = build_dummy_inputs(args.cams, args.batch, args.npts, device=device)

    print("Running one forward pass to verify the wrapper…")
    with torch.inference_mode():
        out = wrapper(*dummy)
    print(f"  Output tuple length: {len(out)}")
    for i, t in enumerate(out):
        print(f"  [{i}] shape={t.shape}  dtype={t.dtype}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    input_names = [
        "views", "coords", "query_times",
        "cam_ext", "cam_mat", "cam_dist", "cam_offset",
    ]
    output_names = [
        "coords_pred", "3d_pred_cams_direct", "3d_pred_cams_rays",
        "conf_3d", "3d_pred_direct", "3d_pred_rays", "3d_pred_triangulate",
        "2d_pred", "vis_pred", "conf_pred",
        "vis_pred_2d", "conf_pred_2d", "depth_pred",
    ]
    dynamic_axes = {
        "views":        {0: "cams", 1: "batch"},
        "coords":       {0: "batch", 1: "npts"},
        "query_times":  {0: "batch", 1: "npts"},
        "cam_ext":      {0: "cams"},
        "cam_mat":      {0: "cams"}, "cam_dist":    {0: "cams"},
        "cam_offset":   {0: "cams"},
        # outputs
        "coords_pred":         {0: "batch", 2: "npts"},
        "3d_pred_cams_direct": {0: "cams",  1: "batch", 3: "npts"},
        "3d_pred_cams_rays":   {0: "cams",  1: "batch", 3: "npts"},
        "conf_3d":             {0: "cams",  1: "batch", 3: "npts"},
        "3d_pred_direct":      {0: "batch", 2: "npts"},
        "3d_pred_rays":        {0: "batch", 2: "npts"},
        "3d_pred_triangulate": {0: "batch", 2: "npts"},
        "2d_pred":             {0: "cams",  1: "batch", 3: "npts"},
        "vis_pred":            {0: "batch", 2: "npts"},
        "conf_pred":           {0: "batch", 2: "npts"},
        "vis_pred_2d":         {0: "cams",  1: "batch", 3: "npts"},
        "conf_pred_2d":        {0: "cams",  1: "batch", 3: "npts"},
        "depth_pred":          {0: "cams",  1: "batch", 3: "npts"},
    }

    print(f"\nExporting to {args.output} (opset {args.opset})…")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
        )

    print(f"Saved → {args.output}  ({os.path.getsize(args.output)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
