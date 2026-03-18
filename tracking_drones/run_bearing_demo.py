from __future__ import annotations

import argparse
import time
from typing import Any

# Allow running this file directly from inside the tracking_drones folder.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from tracking_drones.calibration import (
    StereoCalibration,
    adjust_camera_matrix_for_crop_and_resize,
    bearing_from_uv,
    bearing_to_az_el_deg,
    load_stereo_calibration_npz,
    scale_camera_matrix_explicit,
    scale_camera_matrix_for_image,
)
from tracking_drones.fusion import FusionConfig, fuse_two_cameras
from tracking_drones.luma import extract_luma
from tracking_drones.motion import FrameDiffMotion, MotionConfig
from tracking_drones.tracker import Detection, TrackConfig, TrackManager, smooth_bearing
from tracking_drones.triangulation import triangulate_two_rays
from tracking_drones.video_sources import CameraConfig, DualCameraReader
from tracking_drones.viz import draw_tracks, stack_debug
from tracking_drones.calibration import right_bearing_in_left_frame


def _ensure_bgr(frame: Any) -> Any:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 2:
        # best-effort: show luma
        y = frame[:, :, 0]
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    return frame


def _tracks_for_output(tracks: list[Any], min_score: float) -> list[Any]:
    return [t for t in tracks if t.bearing is not None and t.class_score >= min_score]


def _digital_zoom(frame: Any, zoom: float) -> tuple[Any, tuple[int, int, int, int]]:
    """Center-crop and resize back to original size. Returns (frame, crop_xywh)."""

    if zoom <= 1.0:
        h, w = frame.shape[:2]
        return frame, (0, 0, w, h)

    h, w = frame.shape[:2]
    cw = max(1, int(round(w / zoom)))
    ch = max(1, int(round(h / zoom)))
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)

    cropped = frame[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized, (x0, y0, cw, ch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-camera bearing-only motion+track demo (Y channel only).")
    ap.add_argument("--calib", default="calib_auto.npz")

    ap.add_argument("--cam-left", default="/dev/video2")
    ap.add_argument("--cam-right", default="/dev/video0")

    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="YUYV")

    ap.add_argument("--rot-left", default="ccw", choices=["none", "cw", "ccw", "180"])
    ap.add_argument("--rot-right", default="cw", choices=["none", "cw", "ccw", "180"])

    ap.add_argument(
        "--calib-w",
        type=int,
        default=480,
        help="Calibration image width that K was estimated on (default 480; common for rotated 640x480 capture).",
    )
    ap.add_argument(
        "--calib-h",
        type=int,
        default=640,
        help="Calibration image height that K was estimated on (default 640; common for rotated 640x480 capture).",
    )

    ap.add_argument(
        "--zoom-left",
        type=float,
        default=1.0,
        help="Digital zoom factor (>=1). 1.0 = no zoom, 2.0 = 2x zoom.",
    )
    ap.add_argument(
        "--zoom-right",
        type=float,
        default=1.0,
        help="Digital zoom factor (>=1). 1.0 = no zoom, 2.0 = 2x zoom.",
    )

    ap.add_argument("--diff-thresh", type=int, default=20)
    ap.add_argument("--min-area", type=int, default=6)
    ap.add_argument("--max-area", type=int, default=2000)

    ap.add_argument("--max-match-dist", type=float, default=35.0)
    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--max-angle", type=float, default=2.0)

    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()

    calib: StereoCalibration = load_stereo_calibration_npz(args.calib)

    motion_cfg = MotionConfig(
        diff_thresh=int(args.diff_thresh),
        min_area=int(args.min_area),
        max_area=int(args.max_area),
    )

    track_cfg = TrackConfig(max_match_dist_px=float(args.max_match_dist))

    motionL = FrameDiffMotion(motion_cfg)
    motionR = FrameDiffMotion(motion_cfg)

    tmL = TrackManager(track_cfg)
    tmR = TrackManager(track_cfg)

    reader = DualCameraReader(
        left=CameraConfig(
            device=args.cam_left,
            width=args.w,
            height=args.h,
            fps=args.fps,
            fourcc=args.fourcc,
            rotate=args.rot_left,
        ),
        right=CameraConfig(
            device=args.cam_right,
            width=args.w,
            height=args.h,
            fps=args.fps,
            fourcc=args.fourcc,
            rotate=args.rot_right,
        ),
    )

    fuse_cfg = FusionConfig(max_angle_deg=float(args.max_angle))

    scaled_K_L = None
    scaled_K_R = None

    # Right camera center in LEFT camera frame.
    # OpenCV stereo: X_R = R * X_L + T.
    # Right origin (X_R=0) in left frame: X_L = -R^T * T.
    C_R_in_L = -(calib.R.T @ calib.T).reshape(3)

    last_t = time.time()
    fps_ema = 0.0

    try:
        while True:
            pair = reader.read()
            if pair is None:
                continue

            frameL, frameR = pair

            frameL, cropL = _digital_zoom(frameL, float(args.zoom_left))
            frameR, cropR = _digital_zoom(frameR, float(args.zoom_right))

            yL = extract_luma(frameL)
            yR = extract_luma(frameR)

            if scaled_K_L is None:
                # calib_auto.npz in this repo is commonly computed on already-rotated images.
                # Since we rotate frames before processing, just scale K to the current processed frame size.
                dst_wh = (yL.shape[1], yL.shape[0])
                src_wh = (int(args.calib_w), int(args.calib_h))
                scaled_K_L = scale_camera_matrix_explicit(calib.mtxL, src_wh=src_wh, dst_wh=dst_wh)
            if scaled_K_R is None:
                dst_wh = (yR.shape[1], yR.shape[0])
                src_wh = (int(args.calib_w), int(args.calib_h))
                scaled_K_R = scale_camera_matrix_explicit(calib.mtxR, src_wh=src_wh, dst_wh=dst_wh)

            K_L = adjust_camera_matrix_for_crop_and_resize(
                scaled_K_L, cropL, (yL.shape[1], yL.shape[0])
            )
            K_R = adjust_camera_matrix_for_crop_and_resize(
                scaled_K_R, cropR, (yR.shape[1], yR.shape[0])
            )

            blobsL, maskL = motionL.step(yL)
            blobsR, maskR = motionR.step(yR)

            detsL = [Detection(b.bbox, b.centroid, b.area) for b in blobsL]
            detsR = [Detection(b.bbox, b.centroid, b.area) for b in blobsR]

            tracksL = tmL.step(detsL)
            tracksR = tmR.step(detsR)

            # Bearing update (camera frame). Undistort only the track center.
            for trk in tracksL:
                if not trk.is_confirmed(track_cfg):
                    continue
                u = float(trk.x[0, 0])
                v = float(trk.x[1, 0])
                b = bearing_from_uv((u, v), K_L, calib.distL)
                trk.bearing = smooth_bearing(trk.bearing, b, alpha=0.2)

            for trk in tracksR:
                if not trk.is_confirmed(track_cfg):
                    continue
                u = float(trk.x[0, 0])
                v = float(trk.x[1, 0])
                b = bearing_from_uv((u, v), K_R, calib.distR)
                trk.bearing = smooth_bearing(trk.bearing, b, alpha=0.2)

            outL = draw_tracks(_ensure_bgr(frameL.copy()), tracksL, "LEFT")
            outR = draw_tracks(_ensure_bgr(frameR.copy()), tracksR, "RIGHT")

            # Fusion: only output-ish tracks
            out_tracksL = _tracks_for_output(tracksL, min_score=float(args.min_score))
            out_tracksR = _tracks_for_output(tracksR, min_score=float(args.min_score))
            fused = fuse_two_cameras(calib, out_tracksL, out_tracksR, fuse_cfg)

            # Debug: best angular agreement (helps diagnose K/rotation/zoom mismatch)
            best_ang = None
            best_pair = None
            for tl in out_tracksL:
                for tr in out_tracksR:
                    br_in_L = right_bearing_in_left_frame(calib, tr.bearing)
                    c = float(np.clip(np.dot(tl.bearing, br_in_L), -1.0, 1.0))
                    ang = float(np.degrees(np.arccos(c)))
                    if best_ang is None or ang < best_ang:
                        best_ang = ang
                        best_pair = (tl.id, tr.id)

            # HUD
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

            cv2.putText(
                outL,
                f"fps~{fps_ema:.1f} tracks={len(out_tracksL)}",
                (10, outL.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if best_ang is not None and best_pair is not None:
                cv2.putText(
                    outL,
                    f"best_pair L{best_pair[0]} R{best_pair[1]} ang={best_ang:.2f}deg (gate={fuse_cfg.max_angle_deg:.1f})",
                    (10, outL.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
            cv2.putText(
                outR,
                f"tracks={len(out_tracksR)}",
                (10, outR.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Show fused matches on left pane
            y0 = 55
            for i, m in enumerate(fused[:6]):
                tl = next((t for t in out_tracksL if t.id == m.left_id), None)
                tr = next((t for t in out_tracksR if t.id == m.right_id), None)
                if tl is None or tr is None or tl.bearing is None or tr.bearing is None:
                    continue
                azL, elL = bearing_to_az_el_deg(tl.bearing)

                # Triangulate: left ray from origin, right ray from C_R_in_L (both in left frame)
                bR_in_L = (calib.R.T @ tr.bearing).reshape(3)
                tri = triangulate_two_rays(
                    origin1=np.array([0.0, 0.0, 0.0]),
                    dir1=tl.bearing,
                    origin2=C_R_in_L,
                    dir2=bR_in_L,
                    min_forward_m=0.05,
                )

                if tri is not None:
                    x, y, z = float(tri.point_L[0]), float(tri.point_L[1]), float(tri.point_L[2])
                    rng = float(np.linalg.norm(tri.point_L))
                    txt = (
                        f"MATCH L{m.left_id} R{m.right_id} ang={m.ang_deg:.2f} conf={m.confidence:.2f} "
                        f"az={azL:+.1f} el={elL:+.1f} "
                        f"P_L=({x:+.2f},{y:+.2f},{z:+.2f})m r={rng:.1f}m sep={tri.separation_m:.2f}m"
                    )
                else:
                    txt = f"MATCH L{m.left_id} R{m.right_id} ang={m.ang_deg:.2f} conf={m.confidence:.2f} az={azL:+.1f} el={elL:+.1f}"
                cv2.putText(outL, txt, (10, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if not args.no_gui:
                stacked = stack_debug(outL, outR)
                cv2.imshow("bearing_demo", stacked)

                if maskL is not None and maskR is not None:
                    mask_stack = stack_debug(
                        cv2.cvtColor(maskL, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(maskR, cv2.COLOR_GRAY2BGR),
                        max_w=1600,
                    )
                    cv2.imshow("motion_mask", mask_stack)

                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
            else:
                # headless: print the best match occasionally
                if fused and (int(now) % 2 == 0):
                    m = fused[0]
                    print(f"match L{m.left_id} R{m.right_id} ang={m.ang_deg:.2f} conf={m.confidence:.2f}")

    finally:
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
