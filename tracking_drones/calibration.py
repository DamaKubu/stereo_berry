from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np


CamSide = Literal["left", "right"]


@dataclass(frozen=True)
class StereoCalibration:
    mtxL: Any
    distL: Any
    mtxR: Any
    distR: Any
    # OpenCV stereo convention: X_R = R * X_L + T
    R: Any
    T: Any


def load_stereo_calibration_npz(path: str | Path) -> StereoCalibration:
    npz = np.load(str(path))
    return StereoCalibration(
        mtxL=npz["mtxL"],
        distL=npz["distL"],
        mtxR=npz["mtxR"],
        distR=npz["distR"],
        R=npz["R"],
        T=npz["T"],
    )


def undistort_normalized_point(
    uv_px: tuple[float, float],
    mtx: Any,
    dist: Any,
) -> tuple[float, float]:
    """Undistort a single pixel coordinate to normalized camera coordinates.

    Returns (x, y) such that the bearing in camera frame is proportional to [x, y, 1].
    """

    u, v = uv_px
    pts = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, mtx, dist)
    x, y = float(und[0, 0, 0]), float(und[0, 0, 1])
    return x, y


def bearing_from_uv(
    uv_px: tuple[float, float],
    mtx: Any,
    dist: Any,
) -> Any:
    x, y = undistort_normalized_point(uv_px, mtx=mtx, dist=dist)
    b = np.array([x, y, 1.0], dtype=np.float64)
    n = np.linalg.norm(b)
    if n <= 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return b / n


def bearing_to_az_el_deg(bearing: Any) -> tuple[float, float]:
    """Convert a unit bearing (camera frame) to (azimuth, elevation) degrees.

    Convention:
    - camera frame is x-right, y-down, z-forward
    - azimuth is positive to the right
    - elevation is positive up
    """

    x, y, z = float(bearing[0]), float(bearing[1]), float(bearing[2])
    az = np.degrees(np.arctan2(x, z))
    el = np.degrees(np.arctan2(-y, np.sqrt(x * x + z * z)))
    return az, el


def right_bearing_in_left_frame(calib: StereoCalibration, b_right: Any) -> Any:
    """Rotate a right-camera bearing into the left-camera frame."""

    b = calib.R.T @ b_right
    n = float(np.linalg.norm(b))
    if n <= 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return b / n


def scale_camera_matrix_for_image(mtx: Any, image_hw: tuple[int, int]) -> Any:
    """Best-effort scale of intrinsics to match a new image size.

    This repo often calibrates at one resolution then captures at another.
    Distortion coefficients remain valid in normalized coordinates; scaling K
    is a practical way to keep bearings reasonable for demo work.

    The calibration image size is inferred from K assuming principal point ~ center.
    """

    h, w = int(image_hw[0]), int(image_hw[1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])
    w0 = max(1.0, 2.0 * cx)
    h0 = max(1.0, 2.0 * cy)

    sx = float(w) / w0
    sy = float(h) / h0

    K = np.array(mtx, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def scale_camera_matrix_explicit(mtx: Any, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> Any:
    """Scale intrinsics from a known source image size to a destination size."""

    src_w, src_h = float(src_wh[0]), float(src_wh[1])
    dst_w, dst_h = float(dst_wh[0]), float(dst_wh[1])
    sx = dst_w / max(1.0, src_w)
    sy = dst_h / max(1.0, src_h)

    K = np.array(mtx, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def adjust_camera_matrix_for_crop_and_resize(
    mtx: Any,
    crop_xywh: tuple[int, int, int, int],
    out_wh: tuple[int, int],
) -> Any:
    """Adjust intrinsics for a crop (x,y,w,h) followed by resize to out_wh.

    If you center-crop then resize back to the original output size, this is a
    practical way to implement digital zoom while keeping bearings consistent.
    """

    x0, y0, cw, ch = crop_xywh
    out_w, out_h = int(out_wh[0]), int(out_wh[1])

    sx = float(out_w) / float(max(1, cw))
    sy = float(out_h) / float(max(1, ch))

    K = np.array(mtx, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - float(x0)) * sx
    K[1, 2] = (K[1, 2] - float(y0)) * sy
    return K


def rotate_camera_matrix_90(
    mtx: Any,
    rotation: Literal["none", "cw", "ccw", "180"],
    image_wh: tuple[int, int],
) -> Any:
    """Return intrinsics for an image rotated in pixel space.

    Parameters
    - mtx: original intrinsics for an image of size image_wh
    - rotation: the same rotation you apply to the image
    - image_wh: (width, height) BEFORE rotation

    Notes
    - Distortion coefficients stay the same; only K changes.
    - For 90-degree rotations, fx/fy swap.
    """

    w, h = int(image_wh[0]), int(image_wh[1])
    K = np.array(mtx, dtype=np.float64).copy()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    if rotation == "none":
        return K

    if rotation == "180":
        K[0, 2] = (w - 1.0) - cx
        K[1, 2] = (h - 1.0) - cy
        return K

    # 90 degree rotations: output size is (h, w)
    if rotation == "cw":
        # u' = h-1 - v ; v' = u
        K[0, 0] = fy
        K[1, 1] = fx
        K[0, 2] = (h - 1.0) - cy
        K[1, 2] = cx
        return K

    if rotation == "ccw":
        # u' = v ; v' = w-1 - u
        K[0, 0] = fy
        K[1, 1] = fx
        K[0, 2] = cy
        K[1, 2] = (w - 1.0) - cx
        return K

    raise ValueError(f"unknown rotation: {rotation}")
