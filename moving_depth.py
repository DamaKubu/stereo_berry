"""moving_depth.py

Workflow
- Capture synchronized frames from 2 cameras (V4L2).
- Rotate both frames to upright.
- Stereo-rectify using calibration from CALIB_NPZ.
- Compute disparity (StereoSGBM) on rectified grayscale.
- Convert disparity -> depth in meters using rectified focal length and baseline.
- Detect moving objects on rectified LEFT image using background subtraction in a ROI
  (useful for your scene: sky on top, buildings bottom).
- For each moving blob, estimate distance as robust median depth inside the blob.
- Render debug windows: rectified left/right, disparity, depth colormap, motion mask,
  and overlay with boxes+distance.

Why these parameters (high level)
- Uses rectified baseline derived from P2 to avoid drift if T scale differs.
- Uses ROI cropping to ignore sky and stabilize motion detection.
- Uses median depth to reduce speckle/outliers from SGBM.

Keep this file <1000 lines; tuning is done by auto_dev_loop.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


# =========================
# ====== PARAMETERS =======
# =========================

# Cameras
CAM_LEFT = "/dev/video2"
CAM_RIGHT = "/dev/video0"
ROT_LEFT = cv2.ROTATE_90_COUNTERCLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_CLOCKWISE

# Capture format (best-effort; device may ignore)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FOURCC_PREFERENCE = ["NV12", "YUYV"]  # best-effort; driver may ignore

# Calibration
CALIB_NPZ = "calib_auto.npz"
RECTIFY_ALPHA = 0.8  # 0=crop/zoom, 1=keep full FOV with black borders

# Distortion strength (1.0 = full correction from calibration, 0.0 = no distortion correction).
# If rectification feels "fisheye"/over-warped, try 0.6..0.9.
DISTORTION_SCALE = 1.0

# Extrinsics experiment mode:
# - "calib": use R,T from calib_auto.npz
# - "identity": force R = I and T = [-baseline,0,0]^T
EXTRINSICS_MODE = "calib"  # "calib" | "identity"

# If you know the physical baseline, set it here (meters).
# In EXTRINSICS_MODE="calib": rescales calibration T magnitude.
# In EXTRINSICS_MODE="identity": uses this baseline directly.
FORCE_BASELINE_M: Optional[float] = 0.98

# Motion ROI (fractions of height). Example: ignore sky by starting lower.
ROI_Y0_FRAC = 0.40
ROI_Y1_FRAC = 1.00

# Background subtractor
MOG2_HISTORY = 250
MOG2_VAR_THRESHOLD = 16
MOG2_DETECT_SHADOWS = False

# Motion blob filtering
MIN_BLOB_AREA = 800
MAX_BLOB_AREA = 200000

# Stereo SGBM parameters (the auto-loop tunes these)
SGBM_MIN_DISP = 0
SGBM_NUM_DISP = 128  # must be multiple of 16
SGBM_BLOCK_SIZE = 5
SGBM_UNIQUENESS = 7
SGBM_SPECKLE_WINDOW = 80
SGBM_SPECKLE_RANGE = 2
SGBM_DISP12_MAXDIFF = 1

# Depth post-processing
VALID_DISP_MIN = 1.0
DEPTH_MAX_VIS_M = 60.0

# Rendering
WINDOW_SCALE = 1.0

# Auto-tuning hook: auto_dev_loop.py writes this file.
TUNED_PARAMS_JSON = "tuned_params.json"


# =========================
# ====== UTILITIES ========
# =========================

def _on_trackbar(_val: int) -> None:
    return


def _try_set_capture(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    for fourcc in FOURCC_PREFERENCE:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            break
        except Exception:
            continue


def _apply_overrides_from_json(path: str) -> None:
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    except Exception:
        return

    # Only allow a known-safe set of parameters to be overridden.
    allowed = {
        "ROI_Y0_FRAC",
        "ROI_Y1_FRAC",
        "MOG2_HISTORY",
        "MOG2_VAR_THRESHOLD",
        "MIN_BLOB_AREA",
        "MAX_BLOB_AREA",
        "SGBM_MIN_DISP",
        "SGBM_NUM_DISP",
        "SGBM_BLOCK_SIZE",
        "SGBM_UNIQUENESS",
        "SGBM_SPECKLE_WINDOW",
        "SGBM_SPECKLE_RANGE",
        "SGBM_DISP12_MAXDIFF",
        "VALID_DISP_MIN",
        "RECTIFY_ALPHA",
        "DISTORTION_SCALE",
        "EXTRINSICS_MODE",
    }

    for k, v in data.items():
        if k not in allowed:
            continue
        globals()[k] = v


def _colorize_depth(depth_m: NDArray[Any], max_vis_m: float) -> NDArray[Any]:
    d = np.nan_to_num(depth_m, nan=max_vis_m, posinf=max_vis_m, neginf=0.0)
    d = np.clip(d, 0.0, max_vis_m)
    d8 = (d / max_vis_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - d8, cv2.COLORMAP_TURBO)


@dataclass
class Rectify:
    mapLx: NDArray[Any]
    mapLy: NDArray[Any]
    mapRx: NDArray[Any]
    mapRy: NDArray[Any]
    f_px: float
    baseline_m: float
    Q: NDArray[Any]


def load_calibration(npz_path: str):
    calib = np.load(npz_path)
    mtxL, distL = calib["mtxL"], calib["distL"]
    mtxR, distR = calib["mtxR"], calib["distR"]
    rot, trans = calib["R"], calib["T"]

    # Soften distortion if desired.
    try:
        scale = float(DISTORTION_SCALE)
    except Exception:
        scale = 1.0
    scale = max(0.0, min(1.0, scale))
    distL = (distL * scale).astype(np.float64)
    distR = (distR * scale).astype(np.float64)

    # Override extrinsics for quick experiments.
    if str(EXTRINSICS_MODE) == "identity":
        rot = np.eye(3, dtype=np.float64)
        baseline = float(FORCE_BASELINE_M or 0.98)
        # OpenCV convention typically expects camera2 shifted along -x.
        trans = np.array([[-baseline, 0.0, 0.0]], dtype=np.float64).T

    return mtxL, distL, mtxR, distR, rot, trans


def build_rectify(image_size_wh: Tuple[int, int], mtxL, distL, mtxR, distR, R, T, alpha: float) -> Rectify:
    w, h = image_size_wh
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL,
        distL,
        mtxR,
        distR,
        (w, h),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_16SC2)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_16SC2)

    f_px = float(P1[0, 0])
    baseline_m = float(-P2[0, 3] / P2[0, 0])

    return Rectify(mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy, f_px=f_px, baseline_m=baseline_m, Q=Q)


def make_stereo_matcher() -> cv2.StereoSGBM:
    num_disp = int(SGBM_NUM_DISP)
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16

    block_size = int(SGBM_BLOCK_SIZE)
    if block_size % 2 == 0:
        block_size += 1

    stereo = cv2.StereoSGBM_create(
        minDisparity=int(SGBM_MIN_DISP),
        numDisparities=int(num_disp),
        blockSize=int(block_size),
        P1=8 * 1 * int(block_size) ** 2,
        P2=32 * 1 * int(block_size) ** 2,
        disp12MaxDiff=int(SGBM_DISP12_MAXDIFF),
        uniquenessRatio=int(SGBM_UNIQUENESS),
        speckleWindowSize=int(SGBM_SPECKLE_WINDOW),
        speckleRange=int(SGBM_SPECKLE_RANGE),
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return stereo


def _roi_slices(h: int) -> Tuple[slice, slice]:
    y0 = int(max(0.0, min(1.0, ROI_Y0_FRAC)) * h)
    y1 = int(max(0.0, min(1.0, ROI_Y1_FRAC)) * h)
    if y1 <= y0:
        y0, y1 = 0, h
    return slice(y0, y1), slice(0, None)


def _median_depth_in_mask(depth_m: NDArray[Any], mask_u8: NDArray[Any]) -> Optional[float]:
    pts = depth_m[mask_u8 > 0]
    if pts.size < 50:
        return None
    pts = pts[np.isfinite(pts)]
    if pts.size < 50:
        return None
    # Robust median
    return float(np.median(pts))


def run() -> None:
    _apply_overrides_from_json(TUNED_PARAMS_JSON)
    mtxL, distL, mtxR, distR, R, T = load_calibration(CALIB_NPZ)

    if EXTRINSICS_MODE == "calib" and FORCE_BASELINE_M is not None:
        tnorm = float(np.linalg.norm(T))
        if tnorm > 1e-9:
            T = (T * (float(FORCE_BASELINE_M) / tnorm)).astype(np.float64)

    capL = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Cannot open cameras")

    _try_set_capture(capL)
    _try_set_capture(capR)

    stereo = make_stereo_matcher()

    rectify: Optional[Rectify] = None
    bg = cv2.createBackgroundSubtractorMOG2(
        history=int(MOG2_HISTORY),
        varThreshold=float(MOG2_VAR_THRESHOLD),
        detectShadows=bool(MOG2_DETECT_SHADOWS),
    )

    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("MotionMask", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("near(cm)", "Overlay", 50, 10000, _on_trackbar)
    cv2.createTrackbar("far(cm)", "Overlay", 6000, 10000, _on_trackbar)

    last_t = time.time()
    fps_smooth = 0.0

    while True:
        capL.grab()
        capR.grab()
        retL, frameL = capL.retrieve()
        retR, frameR = capR.retrieve()
        if not (retL and retR):
            continue

        frameL = cv2.rotate(frameL, ROT_LEFT)
        frameR = cv2.rotate(frameR, ROT_RIGHT)

        h, w = frameL.shape[:2]
        if rectify is None:
            rectify = build_rectify((w, h), mtxL, distL, mtxR, distR, R, T, RECTIFY_ALPHA)

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        grayLr = cv2.remap(grayL, rectify.mapLx, rectify.mapLy, cv2.INTER_LINEAR)
        grayRr = cv2.remap(grayR, rectify.mapRx, rectify.mapRy, cv2.INTER_LINEAR)

        disp = stereo.compute(grayLr, grayRr).astype(np.float32) / 16.0
        valid = disp >= float(VALID_DISP_MIN)

        depth = np.full_like(disp, np.nan, dtype=np.float32)
        depth[valid] = (rectify.f_px * rectify.baseline_m) / disp[valid]

        # Depth range mask (optional)
        near_cm = cv2.getTrackbarPos("near(cm)", "Overlay")
        far_cm = cv2.getTrackbarPos("far(cm)", "Overlay")
        if far_cm <= near_cm:
            far_cm = near_cm + 1
            cv2.setTrackbarPos("far(cm)", "Overlay", far_cm)
        near_m = near_cm / 100.0
        far_m = far_cm / 100.0

        roi_y, roi_x = _roi_slices(h)
        roi = grayLr[roi_y, roi_x]

        fg = bg.apply(roi)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = cv2.cvtColor(grayLr, cv2.COLOR_GRAY2BGR)
        depth_vis = _colorize_depth(depth, max_vis_m=float(min(DEPTH_MAX_VIS_M, max(5.0, far_m))))

        # Draw ROI line
        y0 = roi_y.start if roi_y.start is not None else 0
        cv2.line(overlay, (0, y0), (w - 1, y0), (0, 255, 255), 2)

        # Estimate distance for each blob
        for c in contours:
            area = cv2.contourArea(c)
            if area < float(MIN_BLOB_AREA) or area > float(MAX_BLOB_AREA):
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            y_full = y + y0

            # Blob mask in full image coords
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(blob_mask, (x, y_full), (x + bw, y_full + bh), 255, -1)

            # Depth restriction + blob restriction
            depth_mask = np.zeros((h, w), dtype=np.uint8)
            valid_depth = np.isfinite(depth) & (depth >= near_m) & (depth <= far_m)
            depth_mask[valid_depth] = 255
            blob_depth_mask = cv2.bitwise_and(depth_mask, blob_mask)

            dist = _median_depth_in_mask(depth, blob_depth_mask)

            cv2.rectangle(overlay, (x, y_full), (x + bw, y_full + bh), (0, 255, 0), 2)
            if dist is not None:
                cv2.putText(
                    overlay,
                    f"{dist:0.1f} m",
                    (x, max(20, y_full - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # FPS
        now = time.time()
        dt = now - last_t
        last_t = now
        fps = 1.0 / max(1e-6, dt)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps
        cv2.putText(
            overlay,
            f"FPS {fps_smooth:0.1f}  f={rectify.f_px:0.0f}px  B={rectify.baseline_m:0.3f}m",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Show
        if WINDOW_SCALE != 1.0:
            overlay_show = cv2.resize(overlay, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
            depth_show = cv2.resize(depth_vis, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
            fg_show = cv2.resize(fg, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE)
        else:
            overlay_show = overlay
            depth_show = depth_vis
            fg_show = fg

        cv2.imshow("Overlay", overlay_show)
        cv2.imshow("Depth", depth_show)
        cv2.imshow("MotionMask", fg_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
