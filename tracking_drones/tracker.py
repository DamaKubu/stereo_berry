from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _ema(prev: float, value: float, alpha: float) -> float:
    return (1.0 - alpha) * prev + alpha * value


def _ema_vec(prev: Any, value: Any, alpha: float) -> Any:
    v = (1.0 - alpha) * prev + alpha * value
    n = float(np.linalg.norm(v))
    if n <= 0:
        return value
    return v / n


@dataclass
class TrackConfig:
    # Association
    max_match_dist_px: float = 35.0

    # Lifecycle
    max_misses: int = 8
    min_confirmed_hits: int = 3

    # History window
    history: int = 30

    # Drone/bird heuristic thresholds
    vel_var_thresh: float = 2.5
    acc_var_thresh: float = 3.5
    area_var_thresh: float = 0.35

    # Scoring
    class_ema_alpha: float = 0.25


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: float


@dataclass
class Track:
    id: int

    x: Any  # state [x,y,vx,vy]
    P: Any

    age: int = 0
    hits: int = 0
    misses: int = 0

    last_det: Detection | None = None

    history_pos: list[tuple[float, float]] = field(default_factory=list)
    history_area: list[float] = field(default_factory=list)
    history_vel: list[tuple[float, float]] = field(default_factory=list)

    bearing: Any | None = None  # unit vector in camera frame
    class_score: float = 0.0

    def is_confirmed(self, cfg: TrackConfig) -> bool:
        return self.hits >= cfg.min_confirmed_hits


class TrackManager:
    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

        # Fixed model (dt=1 frame).
        self.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )
        self.Q = np.diag([1.0, 1.0, 3.0, 3.0]).astype(np.float64)
        self.R = np.diag([8.0, 8.0]).astype(np.float64)

    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def step(self, detections: list[Detection]) -> list[Track]:
        # Predict
        for trk in self._tracks.values():
            trk.x = self.F @ trk.x
            trk.P = self.F @ trk.P @ self.F.T + self.Q
            trk.age += 1

        # Associate via greedy nearest-neighbor
        matches, unmatched_track_ids, unmatched_det_ids = self._associate(detections)

        # Update matched
        for tid, did in matches:
            trk = self._tracks[tid]
            det = detections[did]

            z = np.array([[det.centroid[0]], [det.centroid[1]]], dtype=np.float64)
            y = z - (self.H @ trk.x)
            S = self.H @ trk.P @ self.H.T + self.R
            K = trk.P @ self.H.T @ np.linalg.inv(S)

            trk.x = trk.x + (K @ y)
            trk.P = (np.eye(4) - (K @ self.H)) @ trk.P

            trk.hits += 1
            trk.misses = 0
            trk.last_det = det

            self._append_history(trk, det)
            self._update_class_score(trk)

        # Misses
        for tid in unmatched_track_ids:
            trk = self._tracks[tid]
            trk.misses += 1

        # New tracks for unmatched detections
        for did in unmatched_det_ids:
            det = detections[did]
            self._tracks[self._next_id] = self._init_track(self._next_id, det)
            self._next_id += 1

        # Prune
        dead = [tid for tid, t in self._tracks.items() if t.misses >= self.cfg.max_misses]
        for tid in dead:
            del self._tracks[tid]

        return self.tracks()

    def _init_track(self, track_id: int, det: Detection) -> Track:
        x = np.array([[det.centroid[0]], [det.centroid[1]], [0.0], [0.0]], dtype=np.float64)
        P = np.diag([50.0, 50.0, 200.0, 200.0]).astype(np.float64)
        trk = Track(id=track_id, x=x, P=P)
        trk.last_det = det
        trk.hits = 1
        trk.age = 1
        self._append_history(trk, det)
        trk.class_score = 0.0
        return trk

    def _append_history(self, trk: Track, det: Detection) -> None:
        trk.history_pos.append(det.centroid)
        trk.history_area.append(float(det.area))
        vx, vy = float(trk.x[2, 0]), float(trk.x[3, 0])
        trk.history_vel.append((vx, vy))

        if len(trk.history_pos) > self.cfg.history:
            trk.history_pos = trk.history_pos[-self.cfg.history :]
            trk.history_area = trk.history_area[-self.cfg.history :]
            trk.history_vel = trk.history_vel[-self.cfg.history :]

    def _associate(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_ids = list(self._tracks.keys())
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))

        # Compute all candidate distances
        candidates: list[tuple[float, int, int]] = []
        for tid in track_ids:
            trk = self._tracks[tid]
            px, py = float(trk.x[0, 0]), float(trk.x[1, 0])
            for did, det in enumerate(detections):
                dx = det.centroid[0] - px
                dy = det.centroid[1] - py
                d = float(np.hypot(dx, dy))
                if d <= self.cfg.max_match_dist_px:
                    candidates.append((d, tid, did))

        candidates.sort(key=lambda t: t[0])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _d, tid, did in candidates:
            if tid in used_tracks or did in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(did)
            matches.append((tid, did))

        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
        unmatched_dets = [did for did in range(len(detections)) if did not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _update_class_score(self, trk: Track) -> None:
        # Need some history
        if len(trk.history_vel) < 6 or len(trk.history_area) < 6:
            return

        v = np.array(trk.history_vel[-self.cfg.history :], dtype=np.float64)
        speed = np.linalg.norm(v, axis=1)
        vel_var = float(np.var(speed))

        a = np.diff(v, axis=0)
        acc = np.linalg.norm(a, axis=1)
        acc_var = float(np.var(acc))

        area = np.array(trk.history_area[-self.cfg.history :], dtype=np.float64)
        area_var = float(np.std(area) / (float(np.mean(area)) + 1e-6))

        score = 0.0
        if vel_var < self.cfg.vel_var_thresh:
            score += 0.4
        if acc_var < self.cfg.acc_var_thresh:
            score += 0.4
        if area_var < self.cfg.area_var_thresh:
            score += 0.2

        trk.class_score = _ema(trk.class_score, score, self.cfg.class_ema_alpha)


def smooth_bearing(prev: Any | None, new: Any, alpha: float = 0.2) -> Any:
    if prev is None:
        n = float(np.linalg.norm(new))
        return new / n if n > 0 else new
    return _ema_vec(prev, new, alpha)
