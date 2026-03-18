from __future__ import annotations

import argparse
import time
from typing import Any

# Allow running this file directly from inside the tracking_drones folder.
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob

import cv2

from tracking_drones.video_sources import CameraConfig, DualCameraReader, SingleCameraReader
from tracking_drones.viz import stack_debug


def _ensure_bgr(frame: Any) -> Any:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 2:
        # best-effort: show luma
        y = frame[:, :, 0]
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    return frame


def _list_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Live camera preview (single or stereo). Press 'q' to quit.")

    ap.add_argument("--list", action="store_true", help="List /dev/video* devices and exit.")

    # Single-camera mode
    ap.add_argument("--cam", default=None, help="Single camera device path (e.g. /dev/video0).")

    # Stereo mode (defaults match run_bearing_demo)
    ap.add_argument("--cam-left", default="/dev/video2")
    ap.add_argument("--cam-right", default="/dev/video0")

    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="YUYV")

    ap.add_argument("--rot", default="none", choices=["none", "cw", "ccw", "180"], help="Rotation for --cam.")
    ap.add_argument("--rot-left", default="ccw", choices=["none", "cw", "ccw", "180"])
    ap.add_argument("--rot-right", default="cw", choices=["none", "cw", "ccw", "180"])

    ap.add_argument("--max-w", type=int, default=1600, help="Max stacked preview width.")
    args = ap.parse_args()

    if args.list:
        devs = _list_devices()
        if not devs:
            print("No /dev/video* devices found.")
        else:
            print("Found video devices:")
            for d in devs:
                print(f"  {d}")
        return

    last_t = time.time()
    fps_ema = 0.0

    if args.cam:
        reader: Any = SingleCameraReader(
            CameraConfig(
                device=str(args.cam),
                width=int(args.w),
                height=int(args.h),
                fps=int(args.fps),
                fourcc=str(args.fourcc),
                rotate=str(args.rot),
            )
        )
        win = "preview"

        try:
            while True:
                frame = reader.read()
                if frame is None:
                    continue

                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

                out = _ensure_bgr(frame)
                cv2.putText(
                    out,
                    f"fps~{fps_ema:.1f} {args.cam}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(win, out)

                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
        finally:
            reader.release()
            cv2.destroyAllWindows()
        return

    # Stereo mode
    reader2 = DualCameraReader(
        left=CameraConfig(
            device=str(args.cam_left),
            width=int(args.w),
            height=int(args.h),
            fps=int(args.fps),
            fourcc=str(args.fourcc),
            rotate=str(args.rot_left),
        ),
        right=CameraConfig(
            device=str(args.cam_right),
            width=int(args.w),
            height=int(args.h),
            fps=int(args.fps),
            fourcc=str(args.fourcc),
            rotate=str(args.rot_right),
        ),
    )

    try:
        while True:
            pair = reader2.read()
            if pair is None:
                continue
            frameL, frameR = pair

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

            left = _ensure_bgr(frameL)
            right = _ensure_bgr(frameR)

            cv2.putText(
                left,
                f"LEFT fps~{fps_ema:.1f} {args.cam_left}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                right,
                f"RIGHT {args.cam_right}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            stacked = stack_debug(left, right, max_w=int(args.max_w))
            cv2.imshow("preview_stereo", stacked)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
    finally:
        reader2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
