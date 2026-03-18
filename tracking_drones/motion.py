from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2


@dataclass(frozen=True)
class MotionBlob:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area: float


@dataclass(frozen=True)
class MotionConfig:
    diff_thresh: int = 20
    blur_sigma: float = 1.0
    min_area: int = 6
    max_area: int = 2000
    aspect_min: float = 0.2
    aspect_max: float = 5.0
    morph_open: int = 1
    morph_close: int = 3


class FrameDiffMotion:
    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self._prev: Any | None = None

    def reset(self) -> None:
        self._prev = None

    def step(self, gray: Any) -> tuple[list[MotionBlob], Any | None]:
        """Return blobs + optional debug mask."""

        if gray.ndim != 2:
            raise ValueError("expected gray (H,W)")

        if self._prev is None:
            self._prev = gray.copy()
            return [], None

        diff = cv2.absdiff(gray, self._prev)
        self._prev = gray.copy()

        if self.cfg.blur_sigma > 0:
            # Convert sigma to kernel size (odd). Keep small.
            k = int(max(3, round(self.cfg.blur_sigma * 6 + 1)))
            if k % 2 == 0:
                k += 1
            diff = cv2.GaussianBlur(diff, (k, k), self.cfg.blur_sigma)

        _, mask = cv2.threshold(diff, int(self.cfg.diff_thresh), 255, cv2.THRESH_BINARY)

        if self.cfg.morph_open > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_open, self.cfg.morph_open))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        if self.cfg.morph_close > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_close, self.cfg.morph_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs: list[MotionBlob] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.cfg.min_area or area > self.cfg.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            ar = float(w) / float(h)
            if ar < self.cfg.aspect_min or ar > self.cfg.aspect_max:
                continue

            cx = float(x + w * 0.5)
            cy = float(y + h * 0.5)
            blobs.append(MotionBlob(bbox=(x, y, w, h), centroid=(cx, cy), area=area))

        return blobs, mask
