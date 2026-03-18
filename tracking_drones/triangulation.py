from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TriangulationResult:
    point_L: Any  # 3-vector in left camera frame
    separation_m: float
    t_left: float
    t_right: float


def _unit(v: Any) -> Any:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n


def triangulate_two_rays(
    origin1: Any,
    dir1: Any,
    origin2: Any,
    dir2: Any,
    min_forward_m: float = 0.0,
) -> TriangulationResult | None:
    """Closest-point triangulation between two rays.

    Returns midpoint of closest approach in the coordinate frame of the inputs.

    If the rays are nearly parallel or the solution is behind either origin
    (t < min_forward_m), returns None.
    """

    o1 = np.asarray(origin1, dtype=np.float64).reshape(3)
    o2 = np.asarray(origin2, dtype=np.float64).reshape(3)
    d1 = _unit(dir1)
    d2 = _unit(dir2)

    w0 = o1 - o2

    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))

    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return None

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom

    if t1 < min_forward_m or t2 < min_forward_m:
        return None

    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2

    pm = 0.5 * (p1 + p2)
    sep = float(np.linalg.norm(p1 - p2))

    return TriangulationResult(point_L=pm, separation_m=sep, t_left=float(t1), t_right=float(t2))
