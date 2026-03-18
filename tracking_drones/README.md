# Bearing-only two-camera drone demo

This folder contains a simple, demo-robust **bearing-only** pipeline:

- Use **Y (luma)** only
- Motion-first candidate generation (frame differencing)
- Track blobs over time (lightweight Kalman + greedy association)
- Per-track drone-vs-bird heuristic score (motion smoothness)
- Convert track center → **undistorted bearing vector** using `calib_auto.npz`
- Two-camera fusion by **angular agreement** (no triangulation)

It also computes an optional **rough 3D point** from the two rays (closest approach). This is reported in the **left camera frame** as `P_L=(x,y,z)` meters.

## Run (live cameras)

From repo root:

- `python -m tracking_drones.run_bearing_demo`

Common setup (matches the rest of this repo’s default devices/rotations):

- `python -m tracking_drones.run_bearing_demo --cam-left /dev/video2 --cam-right /dev/video0 --rot-left ccw --rot-right cw --fourcc YUYV --w 1280 --h 720`

Useful knobs:

- `--diff-thresh 15` (more sensitive motion)
- `--min-area 6 --max-area 1200` (blob size gate)
- `--max-match-dist 35` (association gate)
- `--min-score 0.55` (how “drone-like” a track must be)
- `--max-angle 2.0` (two-camera bearing agreement tolerance)

If cross-camera angles are way off (e.g. ~20°), you may be scaling intrinsics from the wrong calibration resolution. Try:

- `--calib-w 480 --calib-h 640` (default; common for rotated 640x480 captures)
- or `--calib-w 640 --calib-h 480` (if you calibrated without rotation)

Match zoom / FOV:

- `--zoom-left 1.0` and `--zoom-right 1.0` are default (no zoom)
- Example: make both wide cams look equally zoomed: `--zoom-left 1.5 --zoom-right 1.5`
- Example: if left is more zoomed than right, bump right up: `--zoom-right 1.8`

Quit with `q`.

## “Coordinates” / Geo coordinates

Right now the demo can output a 3D point in the **left camera coordinate frame** (meters).

To turn that into **latitude/longitude/altitude**, you still need the rig pose in the world:

- Left camera position: GPS (lat/lon/alt) or surveyed point
- Left camera orientation: yaw/heading + pitch + roll (IMU), or a calibrated transform into ENU/NED

Once you have that, converting `P_L` to ENU/NED and then to WGS84 is straightforward.

## Calibration

The demo reads `calib_auto.npz` (keys: `mtxL`, `distL`, `mtxR`, `distR`, `R`, `T`).

If you need to regenerate calibration, the repo already has scripts and the images under:

- `data/data_cam1/` and `data/data_cam2/` (intrinsics)
- `data/extrinsicLeftCam1/` and `data/extrinsicRightCam2/` (stereo/extrinsics)
