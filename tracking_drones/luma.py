from __future__ import annotations

import cv2
import numpy as np


def extract_luma(frame: np.ndarray) -> np.ndarray:
    """Return a uint8 grayscale (luma) image from various OpenCV capture formats.

    Supports:
    - BGR frames (H,W,3)
    - YUYV-ish frames often exposed as (H,W,2) where channel 0 is Y
    - already-grayscale frames (H,W)
    """

    if frame is None:
        raise ValueError("frame is None")

    if frame.ndim == 2:
        if frame.dtype != np.uint8:
            return frame.astype(np.uint8)
        return frame

    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Some V4L2 modes (e.g., YUYV) come through as 2 channels per pixel.
    if frame.ndim == 3 and frame.shape[2] == 2:
        y = frame[:, :, 0]
        if y.dtype != np.uint8:
            y = y.astype(np.uint8)
        return y

    # Fallback: do a best-effort conversion
    try:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        raise ValueError(f"Unsupported frame shape for luma extraction: {frame.shape}") from e
