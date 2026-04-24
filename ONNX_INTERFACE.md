# TrackerEncoder ONNX Interface

Exported from `TrackerEncoderONNX` via `export_onnx.py`.  
Files: `tracker_encoder.onnx` (graph) + `tracker_encoder.onnx.data` (weights).

## Fixed dimensions

| Symbol | Value | Meaning |
|--------|-------|---------|
| `T` | 16 | frames per clip |
| `H` | 256 | frame height (pixels) |
| `W` | 256 | frame width (pixels) |

## Dynamic dimensions

| Symbol | Meaning |
|--------|---------|
| `cams` | number of cameras |
| `B` | batch size |
| `N` | number of query points |

---

## Inputs

| Name | Shape | dtype | Description |
|------|-------|-------|-------------|
| `views` | `[cams, B, 16, 256, 256, 3]` | float32 | RGB frames in `[0, 1]` (e.g. `uint8 / 255`). ImageNet normalization is applied internally. Channel order is `[R, G, B]`. |
| `coords` | `[B, N, 3]` | float32 | Query point 3D coordinates in world space. These are the initial locations of the points to track. |
| `query_times` | `[B, N]` | int64 | Frame index at which each query point was observed. Values in `[0, T-1]`. |
| `cam_ext` | `[cams, 4, 4]` | float32 | World-to-camera extrinsic matrices. Each is a 4×4 homogeneous transform `[R | t; 0 1]` mapping world points into camera space. The inverse and camera centers are computed internally from this. |
| `cam_mat` | `[cams, 3, 3]` | float32 | Camera intrinsic matrix K. Layout: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`. |
| `cam_dist` | `[cams, 5]` | float32 | Distortion coefficients `[k1, k2, p1, p2, k3]` (OpenCV convention). Pass zeros for undistorted cameras. |
| `cam_offset` | `[cams, 2]` | float32 | Pixel offset subtracted after projection, as `[offset_x, offset_y]`. Pass zeros if images are not cropped. |

---

## Outputs

All outputs are float32. The model produces 13 tensors in a fixed order.

### Primary 3D predictions

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 0 | `coords_pred` | `[B, T, N, 3]` | **Main output.** Predicted 3D world-space coordinates at every frame. Computed as the confidence-weighted sum of per-camera direct predictions (`3d_pred_cams_direct`). Same values as `3d_pred_direct` (index 4). |
| 4 | `3d_pred_direct` | `[B, T, N, 3]` | Predicted 3D positions via camera-space offset unprojection, aggregated across cameras using `conf_3d` weights. |
| 5 | `3d_pred_rays` | `[B, T, N, 3]` | Predicted 3D positions via ray-casting: each camera predicts a 2D location and depth, then a ray is cast and the results are aggregated using `conf_3d` weights. |
| 6 | `3d_pred_triangulate` | `[B, T, N, 3]` | Predicted 3D positions via weighted midpoint triangulation across all cameras. Meaningful only when `cams >= 2`. |

### Per-camera 3D predictions

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 1 | `3d_pred_cams_direct` | `[cams, B, T, N, 3]` | Per-camera direct 3D predictions (before confidence-weighted aggregation). |
| 2 | `3d_pred_cams_rays` | `[cams, B, T, N, 3]` | Per-camera ray-based 3D predictions (before confidence-weighted aggregation). |

### 2D predictions

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 7 | `2d_pred` | `[cams, B, T, N, 2]` | Predicted 2D pixel coordinates in each camera view, in the same pixel space as the input images (after subtracting `cam_offset`). |

### Confidence and visibility

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 3 | `conf_3d` | `[cams, B, T, N]` | Per-camera confidence weights for 3D aggregation. Softmaxed across the camera dimension (sums to 1 over `cams`). |
| 8 | `vis_pred` | `[B, T, N, 1]` | Predicted visibility of each query point at each frame, aggregated across cameras as the max. Values in `[0, 1]`. |
| 9 | `conf_pred` | `[B, T, N, 1]` | Predicted confidence of the 2D localization, aggregated across cameras as the max. Values in `[0, 1]`. |
| 10 | `vis_pred_2d` | `[cams, B, T, N]` | Per-camera visibility logits (pre-sigmoid). Apply `sigmoid` to get probabilities. |
| 11 | `conf_pred_2d` | `[cams, B, T, N]` | Per-camera 2D confidence logits (pre-sigmoid). Apply `sigmoid` to get probabilities. |

### Depth

| Index | Name | Shape | Description |
|-------|------|-------|-------------|
| 12 | `depth_pred` | `[cams, B, T, N]` | Predicted depth of each query point from each camera center, in world units. |

---

## Coordinate conventions

- **World space**: right-handed, units determined by your calibration. `coords`, `cam_center`, and all `*_3d_*` outputs share this space.
- **Camera space**: standard OpenCV convention — X right, Y down, Z into the scene.
- **2D pixel space**: origin at top-left; `[0, 0]` is the top-left corner of the (possibly cropped) image. `cam_offset` shifts the coordinate origin if the image is a crop of a larger sensor.
- **Distortion**: OpenCV radial-tangential model with coefficients `[k1, k2, p1, p2, k3]`.

## Typical usage

For most tracking applications, `coords_pred` (index 0) is the only output you need. Use `vis_pred` (index 8) to mask out frames where the point is not visible, and `conf_pred` (index 9) as a quality filter on the 2D localization. When you need camera-specific information (e.g. for re-projection error), use `2d_pred` (index 7) and `depth_pred` (index 12).
