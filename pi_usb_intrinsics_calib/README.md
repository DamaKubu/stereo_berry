# Pi USB Camera Intrinsics Calibration (8x5)

Small, clean project for calibrating a **USB camera on Raspberry Pi**.

Goal: capture **100 good chessboard views** and calibrate intrinsics with **< 1 px** reprojection error.

## What this saves (no JPEG)

- Lossless preview images: `output/captures/*.png`
- “YUY2” frames: `output/captures/*.yuy2` (packed YUYV422)
- Per-frame metadata: `output/captures/*.json`

Note: OpenCV typically delivers decoded BGR frames even if the camera is negotiated as YUYV.
This project writes a packed `.yuy2` frame derived from the captured image. If you need the
**exact raw bytes from the camera**, use the optional `--raw-ffmpeg` mode in the capture script.

## Install (Pi)

- Recommended (fastest on Pi):
  - `sudo apt-get update`
  - `sudo apt-get install -y python3-opencv v4l-utils ffmpeg python3-numpy`

If you prefer pip OpenCV (slower builds):
- `pip install opencv-python numpy`

## Check camera modes

- List formats + sizes:
  - `v4l2-ctl -d /dev/video0 --list-formats-ext`

You want **YUYV** / **YUY2** (aka `YUYV`, `YUY2`, `YUYV422` depending on tool).

## 1) Capture 100 frames

Interactive capture (recommended):

- `python3 scripts/capture_chessboard.py --dev /dev/video0 --count 100 --board 8x5 --square-mm 25`

Controls:
- `s` save a frame (only saves if chessboard is detected unless `--allow-miss`)
- `q` quit

More automation:
- `--auto` will auto-save when the board is stable.

## 2) Calibrate intrinsics

- `python3 scripts/calibrate_intrinsics.py --input output/captures --board 8x5 --square-mm 25`

Outputs:
- `output/intrinsics.yaml`
- `output/intrinsics.npz`

## Tips to hit <1px

- Fill the frame with the chessboard (different positions + tilts)
- Include strong perspective angles (not only fronto-parallel)
- Avoid motion blur; use good lighting
- Use a stiff printed board; flat surface
- Ensure correct `--square-mm` (physical square size)

