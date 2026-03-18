from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import StereoCalibration, right_bearing_in_left_frame
from .tracker import Track


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = float(np.linalg.norm(a))
    bb = float(np.linalg.norm(b))
    if aa <= 0 or bb <= 0:
        return 180.0
    c = float(np.clip(np.dot(a, b) / (aa * bb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


@dataclass(frozen=True)
class FusionMatch:
    left_id: int
    right_id: int
    ang_deg: float
    confidence: float


@dataclass(frozen=True)
class FusionConfig:
    max_angle_deg: float = 2.0


def fuse_two_cameras(
    calib: StereoCalibration,
    left_tracks: list[Track],
    right_tracks: list[Track],
    cfg: FusionConfig,
) -> list[FusionMatch]:
    # Only consider tracks that have a bearing.
    L = [t for t in left_tracks if t.bearing is not None]
    R = [t for t in right_tracks if t.bearing is not None]

    matches: list[tuple[float, int, int, float]] = []
    for tl in L:
        for tr in R:
            br_in_L = right_bearing_in_left_frame(calib, tr.bearing)  # type: ignore[arg-type]
            ang = angle_between_deg(tl.bearing, br_in_L)  # type: ignore[arg-type]
            if ang <= cfg.max_angle_deg:
                conf = float(min(tl.class_score, tr.class_score))
                matches.append((ang, tl.id, tr.id, conf))

    matches.sort(key=lambda x: x[0])
    used_L: set[int] = set()
    used_R: set[int] = set()

    out: list[FusionMatch] = []
    for ang, lid, rid, conf in matches:
        if lid in used_L or rid in used_R:
            continue
        used_L.add(lid)
        used_R.add(rid)
        out.append(FusionMatch(left_id=lid, right_id=rid, ang_deg=float(ang), confidence=float(conf)))

    return out
