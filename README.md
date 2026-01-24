# stereo_berry
two cameras one cup (raspberry pi 5)

## Chessboard Calibration with YUYV Format

This repository contains tools for stereo camera calibration on Raspberry Pi 5 using uncompressed YUYV format instead of MJPEG.

### Requirements

```bash
pip install -r requirements.txt
```

### Usage

#### Simple Chessboard Calibration

The `chessboard_calibration.py` script captures images from a camera using uncompressed YUYV format and performs calibration using a chessboard pattern.

Basic usage:
```bash
python3 chessboard_calibration.py
```

With custom parameters:
```bash
python3 chessboard_calibration.py --camera 0 --width 1280 --height 720 --rows 6 --cols 9 --images 20 --output calibration_cam0.npz
```

Arguments:
- `--camera`: Camera index (0 or 1 for stereo setup)
- `--width`: Image width (default: 640)
- `--height`: Image height (default: 480)
- `--rows`: Number of internal corners in chessboard rows (default: 6)
- `--cols`: Number of internal corners in chessboard columns (default: 9)
- `--images`: Number of calibration images to capture (default: 20)
- `--output`: Output filename for calibration data (default: calibration.npz)

#### Calibration Process

1. Run the script with desired parameters
2. Position the chessboard in the camera's field of view
3. When the chessboard is detected (green text), press 'c' to capture
4. Move the chessboard to different positions and angles
5. Capture at least 20 images from various angles
6. Press 'q' to finish early or wait until all images are captured
7. Calibration parameters will be saved to the specified output file

### YUYV Format

This implementation uses YUYV (YUV 4:2:2) uncompressed format instead of MJPEG because:
- Lower CPU overhead for encoding/decoding
- More consistent frame timing
- Better for real-time processing on Raspberry Pi
- Higher quality (no compression artifacts) 
