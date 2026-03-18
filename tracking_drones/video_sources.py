from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal

import glob

import cv2


Rotation = Literal["none", "cw", "ccw", "180"]


def _rot_code(rot: Rotation) -> int | None:
    if rot == "none":
        return None
    if rot == "cw":
        return cv2.ROTATE_90_CLOCKWISE
    if rot == "ccw":
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if rot == "180":
        return cv2.ROTATE_180
    raise ValueError(f"unknown rotation: {rot}")


def apply_rotation(frame: Any, rot: Rotation) -> Any:
    code = _rot_code(rot)
    if code is None:
        return frame
    return cv2.rotate(frame, code)


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "YUYV"  # common uncompressed
    rotate: Rotation = "none"


def configure_capture(cap: cv2.VideoCapture, cfg: CameraConfig) -> None:
    if cfg.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.height))
    cap.set(cv2.CAP_PROP_FPS, int(cfg.fps))


class SingleCameraReader:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {cfg.device}")

        configure_capture(self.cap, cfg)

        # Best-effort: keep raw-ish output when possible.
        try:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        except Exception:
            pass

    def read(self) -> Any | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return apply_rotation(frame, self.cfg.rotate)

    def release(self) -> None:
        self.cap.release()


class DualCameraReader:
    def __init__(self, left: CameraConfig, right: CameraConfig) -> None:
        self.left_cfg = left
        self.right_cfg = right

        self.capL = cv2.VideoCapture(left.device, cv2.CAP_V4L2)
        self.capR = cv2.VideoCapture(right.device, cv2.CAP_V4L2)

        if not self.capL.isOpened() or not self.capR.isOpened():
            raise RuntimeError("Failed to open one or both cameras")

        configure_capture(self.capL, left)
        configure_capture(self.capR, right)

        # Best-effort: keep raw-ish output when possible.
        try:
            self.capL.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            self.capR.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        except Exception:
            pass

    def read(self) -> tuple[Any, Any] | None:
        # Grab both then retrieve to keep them as close as OpenCV allows.
        self.capL.grab()
        self.capR.grab()

        okL, frameL = self.capL.retrieve()
        okR, frameR = self.capR.retrieve()
        if not (okL and okR):
            return None

        frameL = apply_rotation(frameL, self.left_cfg.rotate)
        frameR = apply_rotation(frameR, self.right_cfg.rotate)
        return frameL, frameR

    def release(self) -> None:
        self.capL.release()
        self.capR.release()


def iter_images(glob_path: str) -> Iterator[Any]:
    paths = sorted(glob.glob(glob_path))
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        yield im
