from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from utils_v4l2 import pick_max_mode, v4l2_list_modes


def _parse_board(s: str) -> tuple[int, int]:
    if "x" in s:
        a, b = s.lower().split("x", 1)
        return int(a), int(b)
    if "," in s:
        a, b = s.split(",", 1)
        return int(a), int(b)
    raise ValueError("Board must be like 8x5")


def _fourcc(fourcc: str) -> int:
    fn = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(fn):
        return cast(int, fn(*fourcc))

    fn2 = getattr(getattr(cv2, "VideoWriter", object), "fourcc", None)
    if callable(fn2):
        return cast(int, fn2(*fourcc))

    raise RuntimeError("OpenCV does not expose VideoWriter_fourcc")


def negotiated_props(cap: cv2.VideoCapture) -> dict[str, Any]:
    def _fourcc_str(v: float) -> str:
        try:
            i = int(v)
        except Exception:
            return ""
        return "".join([chr((i >> (8 * k)) & 0xFF) for k in range(4)])

    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "fourcc": _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC) or 0.0),
    }


def pack_bgr_to_yuy2(bgr: NDArray[np.uint8]) -> bytes:
    """Pack BGR image into YUYV422 (a.k.a. YUY2/YUYV).

    This is derived from the captured BGR frame (not necessarily identical to the camera's raw).
    Output layout per two pixels: Y0 U Y1 V.
    """

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected BGR image")

    h, w, _ = bgr.shape
    if w % 2 != 0:
        # Drop last column to keep 4:2:2 packing simple.
        bgr = bgr[:, : w - 1, :]
        w -= 1

    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.uint8)
    u = yuv[:, :, 1].astype(np.uint8)
    v = yuv[:, :, 2].astype(np.uint8)

    # Subsample U/V horizontally (average pairs)
    u2 = ((u[:, 0::2].astype(np.uint16) + u[:, 1::2].astype(np.uint16)) // 2).astype(np.uint8)
    v2 = ((v[:, 0::2].astype(np.uint16) + v[:, 1::2].astype(np.uint16)) // 2).astype(np.uint8)

    y0 = y[:, 0::2]
    y1 = y[:, 1::2]

    out = np.empty((h, w // 2, 4), dtype=np.uint8)
    out[:, :, 0] = y0
    out[:, :, 1] = u2
    out[:, :, 2] = y1
    out[:, :, 3] = v2
    return out.tobytes()


def find_corners(gray: Any, board_wh: tuple[int, int]) -> tuple[bool, Any]:
    # SB is more robust than the classic detector.
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cast(Any, cv2).findChessboardCornersSB(gray, board_wh, flags)
        if ok:
            return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cast(Any, cv2).findChessboardCorners(gray, board_wh, flags)
    if not ok:
        return False, None

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cast(Any, cv2).cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return True, corners2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/video0")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--board", type=str, default="8x5")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--prefer-fourcc", type=str, default="YUYV", help="V4L2 fourcc to request")
    ap.add_argument("--max-mode", action="store_true", default=True)
    ap.add_argument("--no-max-mode", dest="max_mode", action="store_false")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--rotate", choices=["none", "cw", "ccw", "180"], default="none")
    ap.add_argument("--out", type=str, default="output/captures")
    ap.add_argument("--auto", action="store_true", help="Auto-save when stable chessboard is detected")
    ap.add_argument("--stable-frames", type=int, default=8)
    ap.add_argument("--min-sep", type=float, default=0.6, help="Minimum seconds between auto-saves")
    ap.add_argument("--allow-miss", action="store_true", help="Allow saving even if no chessboard")
    ap.add_argument(
        "--raw-ffmpeg",
        action="store_true",
        help="(Optional) also dump the exact camera raw frame using ffmpeg. Requires ffmpeg and may stutter preview.",
    )
    args = ap.parse_args()

    board = _parse_board(args.board)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {args.dev}")

    # Request max res for the chosen format.
    mode = None
    if args.max_mode:
        try:
            modes = v4l2_list_modes(args.dev)
            mode = pick_max_mode(modes, prefer_fourcc=args.prefer_fourcc)
        except Exception:
            mode = None

    if mode is not None:
        cap.set(cv2.CAP_PROP_FOURCC, _fourcc(mode.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(mode.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(mode.height))
        if args.fps > 0 and mode.max_fps:
            cap.set(cv2.CAP_PROP_FPS, float(min(args.fps, mode.max_fps)))
        elif args.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(args.fps))
    else:
        cap.set(cv2.CAP_PROP_FOURCC, _fourcc(args.prefer_fourcc))
        if args.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    rot_code = {
        "none": None,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }[args.rotate]

    stable = 0
    last_save_t = 0.0
    saved = 0

    print("Controls: s=save, q=quit")
    print("Negotiated:", negotiated_props(cap))

    try:
        while True:
            cap.grab()
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            gray = cast(Any, cv2).cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, board)
            if found:
                stable += 1
                cast(Any, cv2).drawChessboardCorners(frame, board, corners, found)
            else:
                stable = 0

            hud = frame.copy()
            cv2.putText(
                hud,
                f"saved {saved}/{args.count}  stable {stable}/{args.stable_frames}  found={found}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow("capture", hud)

            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            do_save = False
            if key == ord("q"):
                break
            if key == ord("s"):
                do_save = True
            if args.auto and stable >= int(args.stable_frames) and (now - last_save_t) >= float(args.min_sep):
                do_save = True

            if not do_save:
                continue

            if (not found) and (not args.allow_miss):
                print("Skip: no chessboard detected (use --allow-miss to force)")
                continue

            stem = f"frame_{saved:04d}"
            png_path = out_dir / f"{stem}.png"
            yuy2_path = out_dir / f"{stem}.yuy2"
            meta_path = out_dir / f"{stem}.json"

            cv2.imwrite(str(png_path), frame)
            yuy2_bytes = pack_bgr_to_yuy2(frame)
            yuy2_path.write_bytes(yuy2_bytes)

            meta = {
                "dev": args.dev,
                "board": {"w": board[0], "h": board[1]},
                "square_mm": float(args.square_mm),
                "timestamp": now,
                "found": bool(found),
                "negotiated": negotiated_props(cap),
                "png": str(png_path),
                "yuy2": str(yuy2_path),
            }
            meta_path.write_text(json.dumps(meta, indent=2))

            if args.raw_ffmpeg:
                # Best-effort: dump one exact frame via ffmpeg (reopens the device).
                # This is optional and can stutter; the derived yuy2 is always written.
                raw_path = out_dir / f"{stem}.raw_from_cam.yuy2"
                w = meta["negotiated"]["width"]
                h = meta["negotiated"]["height"]
                cmd = (
                    f"ffmpeg -hide_banner -loglevel error -f v4l2 -input_format yuyv422 "
                    f"-video_size {w}x{h} -i {args.dev} -frames:v 1 -f rawvideo -pix_fmt yuyv422 {raw_path}"
                )
                import subprocess

                subprocess.run(cmd, shell=True, check=False)

            last_save_t = now
            saved += 1
            stable = 0
            print(f"Saved {saved}/{args.count}: {png_path.name}")

            if saved >= int(args.count):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
