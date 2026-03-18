from __future__ import annotations

import cv2
import numpy as np

from .calibration import bearing_to_az_el_deg
from .tracker import Track


def draw_tracks(frame_bgr: np.ndarray, tracks: list[Track], label: str) -> np.ndarray:
    out = frame_bgr
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 240, 240), 2)

    for trk in tracks:
        if trk.last_det is None:
            continue

        x, y, w, h = trk.last_det.bbox
        color = (0, 255, 0) if trk.class_score > 0.6 else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        cx, cy = trk.last_det.centroid
        cv2.circle(out, (int(cx), int(cy)), 2, (255, 255, 255), -1)

        txt = f"id={trk.id} s={trk.class_score:.2f}"
        if trk.bearing is not None:
            az, el = bearing_to_az_el_deg(trk.bearing)
            txt += f" az={az:+.1f} el={el:+.1f}"

        cv2.putText(out, txt, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return out


def stack_debug(left: np.ndarray, right: np.ndarray, max_w: int = 1600) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (int(left.shape[1] * (h / left.shape[0])), h))
        right = cv2.resize(right, (int(right.shape[1] * (h / right.shape[0])), h))

    stacked = np.hstack([left, right])
    if stacked.shape[1] > max_w:
        scale = max_w / stacked.shape[1]
        stacked = cv2.resize(stacked, (int(stacked.shape[1] * scale), int(stacked.shape[0] * scale)))
    return stacked
