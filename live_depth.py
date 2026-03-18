import cv2
import numpy as np
from typing import Any, Optional, Tuple
from numpy.typing import NDArray

# ---------------- CONFIG ----------------
CAM_LEFT = "/dev/video2"
CAM_RIGHT = "/dev/video0"

CALIB_FILE = "calib_auto.npz"

# Rectification scaling:
# - 0.0 = crop to valid pixels (often looks zoomed)
# - 1.0 = keep full FOV (adds black borders)
RECTIFY_ALPHA = 1.0

# Depth mask range (meters). You can also tweak live via trackbars.
DEFAULT_NEAR_M = 0.20
DEFAULT_FAR_M = 3.00

ROT_LEFT = cv2.ROTATE_90_COUNTERCLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_CLOCKWISE

EXPORT_YAML_ON_START = False
EXPORT_YAML_PATH = "calib_auto.yml"


def load_calibration(npz_path: str):
    calib = np.load(npz_path)
    mtxL, distL = calib["mtxL"], calib["distL"]
    mtxR, distR = calib["mtxR"], calib["distR"]
    R, T = calib["R"], calib["T"]
    E = calib["E"] if "E" in calib else None
    F = calib["F"] if "F" in calib else None
    return mtxL, distL, mtxR, distR, R, T, E, F


def save_readable_calib(
    filename: str,
    mtxL: NDArray[Any],
    distL: NDArray[Any],
    mtxR: NDArray[Any],
    distR: NDArray[Any],
    R: NDArray[Any],
    T: NDArray[Any],
    E: Optional[NDArray[Any]],
    F: Optional[NDArray[Any]],
) -> None:
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("mtxL", mtxL)
    fs.write("distL", distL)
    fs.write("mtxR", mtxR)
    fs.write("distR", distR)
    fs.write("R", R)
    fs.write("T", T)
    if E is not None:
        fs.write("E", E)
    if F is not None:
        fs.write("F", F)
    fs.release()
    print(f"Calibration saved to {filename} (human-readable YAML)")


def build_rectify_maps(
    image_size_wh: Tuple[int, int],
    mtxL: NDArray[Any],
    distL: NDArray[Any],
    mtxR: NDArray[Any],
    distR: NDArray[Any],
    R: NDArray[Any],
    T: NDArray[Any],
    alpha: float,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], float, float, NDArray[Any]]:
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

    # Baseline consistent with rectified projection matrices.
    # With P2[0,3] = -f*B, so B = -P2[0,3]/f.
    f = float(P1[0, 0])
    baseline = float(-P2[0, 3] / P2[0, 0])
    return (mapLx, mapLy, mapRx, mapRy, f, baseline, Q)


# Stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # must be multiple of 16
    blockSize=5,
    P1=8*3*5**2,
    P2=32*3*5**2,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=50,
    speckleRange=2
)


def _colorize_depth(depth_m: NDArray[Any], max_vis_m: float = 5.0) -> NDArray[Any]:
    d = np.clip(depth_m, 0.0, max_vis_m)
    d8 = (d / max_vis_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - d8, cv2.COLORMAP_TURBO)


def _on_trackbar(_val: int) -> None:
    return

# ---------------- DEPTH VIEWER ----------------
def depth_viewer() -> None:
    mtxL, distL, mtxR, distR, R, T, E, F = load_calibration(CALIB_FILE)
    if EXPORT_YAML_ON_START:
        save_readable_calib(EXPORT_YAML_PATH, mtxL, distL, mtxR, distR, R, T, E, F)

    capL = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)

    # Force resolution & YUYV
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Depth Masked", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("near(cm)", "Depth Masked", int(DEFAULT_NEAR_M * 100), 1000, _on_trackbar)
    cv2.createTrackbar("far(cm)", "Depth Masked", int(DEFAULT_FAR_M * 100), 1000, _on_trackbar)

    rectify = None

    while True:
        capL.grab(); capR.grab()
        retL, frameL = capL.retrieve()
        retR, frameR = capR.retrieve()
        if not (retL and retR):
            continue

        frameL = cv2.rotate(frameL, ROT_LEFT)
        frameR = cv2.rotate(frameR, ROT_RIGHT)

        if rectify is None:
            h, w = frameL.shape[:2]
            rectify = build_rectify_maps((w, h), mtxL, distL, mtxR, distR, R, T, RECTIFY_ALPHA)
            mapLx, mapLy, mapRx, mapRy, f, baseline, q_matrix = rectify
        else:
            mapLx, mapLy, mapRx, mapRy, f, baseline, q_matrix = rectify

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # Rectify (undistort + align epipolar lines)
        grayL = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
        grayR = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)

        # Compute disparity
        disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        valid_disp = disp > 0.5

        # Depth (meters): Z = f*B / disparity
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid_disp] = (f * baseline) / disp[valid_disp]

        near_cm = cv2.getTrackbarPos("near(cm)", "Depth Masked")
        far_cm = cv2.getTrackbarPos("far(cm)", "Depth Masked")
        if far_cm <= near_cm:
            far_cm = near_cm + 1
            cv2.setTrackbarPos("far(cm)", "Depth Masked", far_cm)

        near_m = near_cm / 100.0
        far_m = far_cm / 100.0

        mask = valid_disp & (depth >= near_m) & (depth <= far_m)
        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_u8 = cv2.medianBlur(mask_u8, 5)

        depth_color = _colorize_depth(depth, max_vis_m=max(0.5, far_m))
        depth_masked = cv2.bitwise_and(depth_color, depth_color, mask=mask_u8)

        # Overlay masked depth on rectified left image for intuition
        left_vis = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(left_vis, 0.6, depth_masked, 0.8, 0.0)

        # Show
        disp_vis = disp.copy()
        disp_vis[~valid_disp] = 0
        disp_vis_norm = np.zeros_like(disp_vis, dtype=np.float32)
        cv2.normalize(disp_vis, disp_vis_norm, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis_norm.astype(np.uint8)

        cv2.imshow("Left (raw)", frameL)
        cv2.imshow("Right (raw)", frameR)
        cv2.imshow("Left (rectified gray)", grayL)
        cv2.imshow("Disparity", disp_vis)
        cv2.imshow("Depth Masked", depth_masked)
        cv2.imshow("Overlay", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    depth_viewer()
