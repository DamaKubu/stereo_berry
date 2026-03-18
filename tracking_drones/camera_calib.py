from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Allow running this file directly from inside the tracking_drones folder.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


@dataclass(frozen=True)
class V4L2Mode:
    fourcc: str
    width: int
    height: int
    max_fps: float | None


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout


def list_video_devices() -> list[str]:
    devs = sorted(Path("/dev").glob("video*"))
    return [str(p) for p in devs]


def udev_info(dev: str) -> dict[str, str]:
    # Stable identity hints: ID_PATH, ID_SERIAL, DEVPATH, etc.
    out = _run(["udevadm", "info", "--query=property", "--name", dev])
    info: dict[str, str] = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        info[k.strip()] = v.strip()
    return info


_SIZE_RE = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
_FMT_RE = re.compile(r"\[(\d+)\]:\s+'([A-Z0-9]{4})'\s+\(")
_INT_RE = re.compile(r"Interval:\s+Discrete\s+([0-9.]+)s\s+\(([0-9.]+)\s+fps\)")


def v4l2_list_modes(dev: str) -> list[V4L2Mode]:
    # Parses `v4l2-ctl --list-formats-ext` output.
    out = _run(["v4l2-ctl", "-d", dev, "--list-formats-ext"])
    modes: list[V4L2Mode] = []

    current_fourcc: str | None = None
    current_size: tuple[int, int] | None = None
    current_max_fps: float | None = None

    def flush() -> None:
        nonlocal current_fourcc, current_size, current_max_fps
        if current_fourcc and current_size:
            modes.append(
                V4L2Mode(
                    fourcc=current_fourcc,
                    width=current_size[0],
                    height=current_size[1],
                    max_fps=current_max_fps,
                )
            )
        current_size = None
        current_max_fps = None

    for line in out.splitlines():
        mfmt = _FMT_RE.search(line)
        if mfmt:
            flush()
            current_fourcc = mfmt.group(2)
            continue

        msz = _SIZE_RE.search(line)
        if msz:
            flush()
            current_size = (int(msz.group(1)), int(msz.group(2)))
            continue

        mint = _INT_RE.search(line)
        if mint and current_size is not None:
            fps = float(mint.group(2))
            if current_max_fps is None or fps > current_max_fps:
                current_max_fps = fps

    flush()
    return modes


def pick_max_mode(modes: Iterable[V4L2Mode], prefer_fourcc: str) -> V4L2Mode | None:
    prefer = prefer_fourcc.upper()
    cand = [m for m in modes if m.fourcc.upper() == prefer]
    if not cand:
        return None
    # Max by area, then width.
    return max(cand, key=lambda m: (m.width * m.height, m.width))


def _parse_index(dev: str) -> int | None:
    if dev.isdigit():
        return int(dev)
    if dev.startswith("/dev/video"):
        tail = dev[len("/dev/video") :]
        if tail.isdigit():
            return int(tail)
    return None


def open_capture(dev: str) -> cv2.VideoCapture:
    # Some OpenCV builds struggle with V4L2 "by name"; try several options.
    attempts: list[tuple[Any, int | None]] = [(dev, cv2.CAP_V4L2), (dev, cv2.CAP_ANY)]
    idx = _parse_index(dev)
    if idx is not None:
        attempts += [(idx, cv2.CAP_V4L2), (idx, cv2.CAP_ANY)]

    for src, api in attempts:
        try:
            cap = cv2.VideoCapture(src, api) if api is not None else cv2.VideoCapture(src)
        except Exception:
            continue
        if cap.isOpened():
            return cap

    # Last resort.
    cap = cv2.VideoCapture(dev)
    return cap


def set_fourcc(cap: cv2.VideoCapture, fourcc: str) -> None:
    # Avoid stub issues by using getattr.
    fn = getattr(cv2, "VideoWriter_fourcc", None)
    if fn is None:
        fn = getattr(getattr(cv2, "VideoWriter", object), "fourcc", None)
    if callable(fn):
        val = fn(*fourcc)
        if isinstance(val, int):
            cap.set(cv2.CAP_PROP_FOURCC, val)


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


def ensure_bgr(frame: Any) -> Any:
    if frame is None:
        return frame
    if getattr(frame, "ndim", 0) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 2:
        # Likely YUYV-ish layout; show luma.
        y = frame[:, :, 0]
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    return frame


def _stack_two(left: Any, right: Any, max_w: int = 1600) -> Any:
    # Minimal side-by-side stack (avoid importing other modules).
    if left is None or right is None:
        return left if right is None else right
    if left.shape[0] != right.shape[0]:
        h = min(int(left.shape[0]), int(right.shape[0]))
        left = cv2.resize(left, (int(left.shape[1] * (h / left.shape[0])), h))
        right = cv2.resize(right, (int(right.shape[1] * (h / right.shape[0])), h))
    stacked = np.hstack([left, right])
    if stacked.shape[1] > int(max_w):
        scale = float(max_w) / float(stacked.shape[1])
        stacked = cv2.resize(stacked, (int(stacked.shape[1] * scale), int(stacked.shape[0] * scale)))
    return stacked


def find_chessboard(gray: np.ndarray, board_wh: tuple[int, int]) -> tuple[bool, Any]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, board_wh, flags)
    if not ok:
        return False, None

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return True, corners2


def preview(dev: str, *, prefer_fourcc: str, max_mode: bool, fps: float, rotate: str) -> None:
    modes = v4l2_list_modes(dev)
    mode = pick_max_mode(modes, prefer_fourcc=prefer_fourcc) if max_mode else None

    cap = open_capture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {dev}")

    try:
        if mode is not None:
            set_fourcc(cap, mode.fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(mode.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(mode.height))
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(min(fps, mode.max_fps or fps)))
        else:
            set_fourcc(cap, prefer_fourcc)
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(fps))

        rot_code = {
            "none": None,
            "cw": cv2.ROTATE_90_CLOCKWISE,
            "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "180": cv2.ROTATE_180,
        }[rotate]

        last_t = time.time()
        fps_ema = 0.0

        while True:
            cap.grab()
            ok, frame = cap.retrieve()
            if not ok:
                continue

            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)

            out = ensure_bgr(frame)
            p = negotiated_props(cap)
            cv2.putText(
                out,
                f"{dev} fps~{fps_ema:.1f} negotiated {p['width']}x{p['height']} {p['fourcc']}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("preview", out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def capture_intrinsics(
    dev: str,
    out_dir: Path,
    *,
    board_wh: tuple[int, int],
    prefer_fourcc: str,
    max_mode: bool,
    fps: float,
    rotate: str,
    every_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    modes = v4l2_list_modes(dev)
    mode = pick_max_mode(modes, prefer_fourcc=prefer_fourcc) if max_mode else None

    cap = open_capture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {dev}")

    rot_code = {
        "none": None,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }[rotate]

    last_save_t = 0.0
    saved = 0

    try:
        if mode is not None:
            set_fourcc(cap, mode.fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(mode.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(mode.height))
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(min(fps, mode.max_fps or fps)))
        else:
            set_fourcc(cap, prefer_fourcc)
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(fps))

        while True:
            # Timestamp bounds around the acquisition.
            t0_ns = time.time_ns()
            m0_ns = time.monotonic_ns()

            cap.grab()
            ok, frame = cap.retrieve()

            m1_ns = time.monotonic_ns()
            t1_ns = time.time_ns()

            if not ok:
                continue

            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            bgr = ensure_bgr(frame)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            found, corners = find_chessboard(gray, board_wh)
            vis = bgr.copy()
            if found:
                cv2.drawChessboardCorners(vis, board_wh, corners, found)

            p = negotiated_props(cap)
            cv2.putText(
                vis,
                f"{dev} negotiated {p['width']}x{p['height']} {p['fourcc']} | board {board_wh[0]}x{board_wh[1]} found={found} saved={saved}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                "Keys: s=save (when found), a=autosave toggle, q=quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                2,
            )

            cv2.imshow("intrinsics_capture", vis)
            k = cv2.waitKey(1) & 0xFF

            now = time.time()
            autosave = (every_s > 0)
            do_save = False
            if k == ord("s"):
                do_save = True
            elif k == ord("a"):
                every_s = 0.0 if autosave else max(0.5, every_s)
            elif k in (ord("q"), 27):
                break

            if every_s > 0 and (now - last_save_t) >= every_s:
                do_save = True

            if do_save and found:
                ts = time.time()
                fn = f"{saved:04d}_{int(ts * 1e6):d}.png"
                img_path = images_dir / fn
                cv2.imwrite(str(img_path), bgr)

                meta = {
                    "device": dev,
                    "filename": fn,
                    "t0_ns": t0_ns,
                    "t1_ns": t1_ns,
                    "mono0_ns": m0_ns,
                    "mono1_ns": m1_ns,
                    "negotiated": p,
                    "shape": [int(bgr.shape[0]), int(bgr.shape[1]), int(bgr.shape[2])],
                    "found": True,
                }
                with meta_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(meta) + "\n")

                saved += 1
                last_save_t = now

    finally:
        cap.release()
        cv2.destroyAllWindows()


def capture_frames(
    dev: str,
    out_dir: Path,
    *,
    prefer_fourcc: str,
    max_mode: bool,
    fps: float,
    rotate: str,
    every_s: float,
) -> None:
    """Capture generic frames (no chessboard requirement).

    Intended for fast dataset collection (e.g., moon shots) with good timestamps.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    modes = v4l2_list_modes(dev)
    mode = pick_max_mode(modes, prefer_fourcc=prefer_fourcc) if max_mode else None

    cap = open_capture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {dev}")

    rot_code = {
        "none": None,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }[rotate]

    last_save_t = 0.0
    saved = 0

    try:
        if mode is not None:
            set_fourcc(cap, mode.fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(mode.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(mode.height))
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(min(fps, mode.max_fps or fps)))
        else:
            set_fourcc(cap, prefer_fourcc)
            if fps > 0:
                cap.set(cv2.CAP_PROP_FPS, float(fps))

        while True:
            # Timestamp bounds around the acquisition.
            t0_ns = time.time_ns()
            m0_ns = time.monotonic_ns()

            cap.grab()
            ok, frame = cap.retrieve()

            m1_ns = time.monotonic_ns()
            t1_ns = time.time_ns()

            if not ok:
                continue

            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            bgr = ensure_bgr(frame)
            vis = bgr.copy()

            p = negotiated_props(cap)
            cv2.putText(
                vis,
                f"{dev} negotiated {p['width']}x{p['height']} {p['fourcc']} | saved={saved}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                "Keys: s=save, a=autosave toggle, q=quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 200),
                2,
            )

            cv2.imshow("capture", vis)
            k = cv2.waitKey(1) & 0xFF

            now = time.time()
            autosave = (every_s > 0)
            do_save = False
            if k == ord("s"):
                do_save = True
            elif k == ord("a"):
                every_s = 0.0 if autosave else max(0.5, every_s)
            elif k in (ord("q"), 27):
                break

            if every_s > 0 and (now - last_save_t) >= every_s:
                do_save = True

            if do_save:
                ts = time.time()
                fn = f"{saved:04d}_{int(ts * 1e6):d}.png"
                img_path = images_dir / fn
                cv2.imwrite(str(img_path), bgr)

                meta = {
                    "device": dev,
                    "filename": fn,
                    "t0_ns": t0_ns,
                    "t1_ns": t1_ns,
                    "mono0_ns": m0_ns,
                    "mono1_ns": m1_ns,
                    "negotiated": p,
                    "shape": [int(bgr.shape[0]), int(bgr.shape[1]), int(bgr.shape[2])],
                }
                with meta_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(meta) + "\n")

                saved += 1
                last_save_t = now

    finally:
        cap.release()
        cv2.destroyAllWindows()


def capture_stereo_frames(
    left_dev: str,
    right_dev: str,
    out_dir: Path,
    *,
    prefer_fourcc: str,
    max_mode: bool,
    fps: float,
    rot_left: str,
    rot_right: str,
    every_s: float,
    max_w: int = 1600,
) -> None:
    """Capture synchronized stereo pairs every N seconds with timestamp bounds.

    Saves:
    - out/left/*.png
    - out/right/*.png
    - out/meta.jsonl (one line per saved pair)
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    left_dir = out_dir / "left"
    right_dir = out_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    modesL = v4l2_list_modes(left_dev)
    modesR = v4l2_list_modes(right_dev)
    modeL = pick_max_mode(modesL, prefer_fourcc=prefer_fourcc) if max_mode else None
    modeR = pick_max_mode(modesR, prefer_fourcc=prefer_fourcc) if max_mode else None

    capL = open_capture(left_dev)
    capR = open_capture(right_dev)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Failed to open one or both cameras")

    rot_code = {
        "none": None,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    rotL = rot_code[rot_left]
    rotR = rot_code[rot_right]

    last_save_t = 0.0
    saved = 0

    try:
        # Configure left
        if modeL is not None:
            set_fourcc(capL, modeL.fourcc)
            capL.set(cv2.CAP_PROP_FRAME_WIDTH, int(modeL.width))
            capL.set(cv2.CAP_PROP_FRAME_HEIGHT, int(modeL.height))
            if fps > 0:
                capL.set(cv2.CAP_PROP_FPS, float(min(fps, modeL.max_fps or fps)))
        else:
            set_fourcc(capL, prefer_fourcc)
            if fps > 0:
                capL.set(cv2.CAP_PROP_FPS, float(fps))

        # Configure right
        if modeR is not None:
            set_fourcc(capR, modeR.fourcc)
            capR.set(cv2.CAP_PROP_FRAME_WIDTH, int(modeR.width))
            capR.set(cv2.CAP_PROP_FRAME_HEIGHT, int(modeR.height))
            if fps > 0:
                capR.set(cv2.CAP_PROP_FPS, float(min(fps, modeR.max_fps or fps)))
        else:
            set_fourcc(capR, prefer_fourcc)
            if fps > 0:
                capR.set(cv2.CAP_PROP_FPS, float(fps))

        while True:
            # Timestamp bounds around acquisition of both frames.
            t0_ns = time.time_ns()
            m0_ns = time.monotonic_ns()

            capL.grab()
            capR.grab()
            okL, frameL = capL.retrieve()
            okR, frameR = capR.retrieve()

            m1_ns = time.monotonic_ns()
            t1_ns = time.time_ns()

            if not (okL and okR):
                continue

            if rotL is not None:
                frameL = cv2.rotate(frameL, rotL)
            if rotR is not None:
                frameR = cv2.rotate(frameR, rotR)

            bgrL = ensure_bgr(frameL)
            bgrR = ensure_bgr(frameR)

            pL = negotiated_props(capL)
            pR = negotiated_props(capR)

            visL = bgrL.copy()
            visR = bgrR.copy()
            cv2.putText(
                visL,
                f"LEFT {left_dev} {pL['width']}x{pL['height']} {pL['fourcc']} saved={saved}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                visR,
                f"RIGHT {right_dev} {pR['width']}x{pR['height']} {pR['fourcc']}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                visL,
                "Keys: s=save now, q=quit (autosave uses --autosave)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 255, 200),
                2,
            )

            stacked = _stack_two(visL, visR, max_w=int(max_w))
            cv2.imshow("capture_stereo", stacked)
            k = cv2.waitKey(1) & 0xFF

            now = time.time()
            do_save = False
            if k == ord("s"):
                do_save = True
            elif k in (ord("q"), 27):
                break

            if every_s > 0 and (now - last_save_t) >= every_s:
                do_save = True

            if do_save:
                ts = time.time()
                fn = f"{saved:04d}_{int(ts * 1e6):d}.png"
                cv2.imwrite(str(left_dir / fn), bgrL)
                cv2.imwrite(str(right_dir / fn), bgrR)

                meta = {
                    "left_device": left_dev,
                    "right_device": right_dev,
                    "filename": fn,
                    "t0_ns": t0_ns,
                    "t1_ns": t1_ns,
                    "mono0_ns": m0_ns,
                    "mono1_ns": m1_ns,
                    "negotiated_left": pL,
                    "negotiated_right": pR,
                    "shape_left": [int(bgrL.shape[0]), int(bgrL.shape[1]), int(bgrL.shape[2])],
                    "shape_right": [int(bgrR.shape[0]), int(bgrR.shape[1]), int(bgrR.shape[2])],
                }
                with meta_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(meta) + "\n")

                saved += 1
                last_save_t = now

    finally:
        capL.release()
        capR.release()
        cv2.destroyAllWindows()


def calibrate_intrinsics(
    dataset_dir: Path,
    *,
    board_wh: tuple[int, int],
    square_size: float,
    model: str = "k1k2",
) -> Path:
    images_dir = dataset_dir / "images"
    paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No images found in {images_dir}")

    objp = np.zeros((board_wh[0] * board_wh[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_wh[0], 0 : board_wh[1]].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints: list[Any] = []
    imgpoints: list[Any] = []

    image_size: tuple[int, int] | None = None

    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = find_chessboard(gray, board_wh)
        if not found:
            continue
        objpoints.append(objp)
        imgpoints.append(corners)

    if len(objpoints) < 8:
        raise RuntimeError(f"Not enough valid chessboard detections ({len(objpoints)}). Capture more views.")

    assert image_size is not None

    # Pinhole intrinsics with a minimal radial distortion model.
    # - model=k1k2: k1,k2 free; tangential (p1,p2)=0; k3 fixed to 0.
    # OpenCV uses distCoeffs = [k1,k2,p1,p2,k3] (for the common 5-coeff model).

    w, h = int(image_size[0]), int(image_size[1])
    K_init = np.array(
        [[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_init = np.zeros((5, 1), dtype=np.float64)

    flags = 0
    if model.lower() == "k1k2":
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_FIX_K3
    else:
        # Default OpenCV behavior (more degrees of freedom).
        K_init = None
        dist_init = None

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        K_init,  # type: ignore[arg-type]
        dist_init,  # type: ignore[arg-type]
        flags=flags,
    )

    out_npz = dataset_dir / "intrinsics.npz"
    np.savez(
        str(out_npz),
        K=K,
        dist=dist,
        image_size=np.array(image_size, dtype=np.int32),
        rms=float(rms),
        board_wh=np.array(board_wh, dtype=np.int32),
        square_size=float(square_size),
        n_images=int(len(objpoints)),
    )

    # Also write a small JSON summary.
    out_json = dataset_dir / "intrinsics_summary.json"
    summary = {
        "rms": float(rms),
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "board": {"inner_corners": [int(board_wh[0]), int(board_wh[1])], "square_size": float(square_size)},
        "n_images_used": int(len(objpoints)),
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return out_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-camera capture + intrinsics calibration (8x5 chessboard).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List /dev/video* with udev identity + max modes.")
    sp.add_argument("--fourcc", default="YUYV", help="FOURCC to consider for max-mode reporting.")

    sp = sub.add_parser("preview", help="Live preview of one camera.")
    sp.add_argument("--cam", required=True)
    sp.add_argument("--fourcc", default="YUYV")
    sp.add_argument("--max", action="store_true", help="Use max resolution for selected fourcc.")
    sp.add_argument("--fps", type=float, default=5.0)
    sp.add_argument("--rot", default="none", choices=["none", "cw", "ccw", "180"])

    sp = sub.add_parser("capture", help="Capture generic photos with timestamps (no chessboard required).")
    sp.add_argument("--cam", required=True)
    sp.add_argument("--out", required=True, help="Output dataset folder.")
    sp.add_argument("--fourcc", default="YUYV")
    sp.add_argument("--max", action="store_true", help="Use max resolution for selected fourcc.")
    sp.add_argument("--fps", type=float, default=5.0, help="Capture FPS request (low recommended).")
    sp.add_argument("--rot", default="none", choices=["none", "cw", "ccw", "180"])
    sp.add_argument("--autosave", type=float, default=0.0, help="Autosave period seconds (0=off).")

    sp = sub.add_parser("capture-stereo", help="Capture synchronized stereo pairs every N seconds (e.g., moon).")
    sp.add_argument("--cam-left", required=True)
    sp.add_argument("--cam-right", required=True)
    sp.add_argument("--out", required=True, help="Output folder (creates left/right + meta.jsonl).")
    sp.add_argument("--fourcc", default="YUYV")
    sp.add_argument("--max", action="store_true", help="Use max resolution for selected fourcc.")
    sp.add_argument("--fps", type=float, default=5.0)
    sp.add_argument("--rot-left", default="none", choices=["none", "cw", "ccw", "180"])
    sp.add_argument("--rot-right", default="none", choices=["none", "cw", "ccw", "180"])
    sp.add_argument("--autosave", type=float, default=10.0, help="Autosave period seconds.")
    sp.add_argument("--max-w", type=int, default=1600)

    sp = sub.add_parser("capture-intrinsics", help="Capture chessboard images with live preview + timestamps.")
    sp.add_argument("--cam", required=True)
    sp.add_argument("--out", required=True, help="Output dataset folder.")
    sp.add_argument("--board", default="8x5", help="Inner corners like 8x5.")
    sp.add_argument("--square", type=float, default=1.0, help="Square size (units arbitrary for intrinsics).")
    sp.add_argument("--fourcc", default="YUYV")
    sp.add_argument("--max", action="store_true", help="Use max resolution for selected fourcc.")
    sp.add_argument("--fps", type=float, default=5.0, help="Capture FPS request (low recommended).")
    sp.add_argument("--rot", default="none", choices=["none", "cw", "ccw", "180"])
    sp.add_argument("--autosave", type=float, default=0.0, help="Autosave period seconds (0=off).")

    sp = sub.add_parser("calibrate-intrinsics", help="Calibrate intrinsics from a captured dataset folder.")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--board", default="8x5")
    sp.add_argument("--square", type=float, default=1.0)
    sp.add_argument(
        "--model",
        default="k1k2",
        choices=["k1k2", "opencv"],
        help="Distortion model: k1k2 fixes p1,p2,k3; opencv lets OpenCV pick defaults.",
    )

    args = ap.parse_args()

    if args.cmd == "list":
        devs = list_video_devices()
        if not devs:
            print("No /dev/video* devices found")
            return
        for d in devs:
            info = udev_info(d)
            ident = info.get("ID_PATH") or info.get("ID_SERIAL") or info.get("DEVPATH") or ""
            print(f"{d}  {ident}")
            try:
                modes = v4l2_list_modes(d)
                m = pick_max_mode(modes, prefer_fourcc=str(args.fourcc))
                if m:
                    print(f"  max {m.fourcc} {m.width}x{m.height} fps<={m.max_fps or 0:.1f}")
            except Exception as e:
                print(f"  (v4l2-ctl parse failed: {e})")
        return

    if args.cmd == "preview":
        preview(
            str(args.cam),
            prefer_fourcc=str(args.fourcc),
            max_mode=bool(args.max),
            fps=float(args.fps),
            rotate=str(args.rot),
        )
        return

    if args.cmd == "capture":
        capture_frames(
            str(args.cam),
            Path(str(args.out)),
            prefer_fourcc=str(args.fourcc),
            max_mode=bool(args.max),
            fps=float(args.fps),
            rotate=str(args.rot),
            every_s=float(args.autosave),
        )
        return

    if args.cmd == "capture-stereo":
        capture_stereo_frames(
            str(args.cam_left),
            str(args.cam_right),
            Path(str(args.out)),
            prefer_fourcc=str(args.fourcc),
            max_mode=bool(args.max),
            fps=float(args.fps),
            rot_left=str(args.rot_left),
            rot_right=str(args.rot_right),
            every_s=float(args.autosave),
            max_w=int(args.max_w),
        )
        return

    if args.cmd == "capture-intrinsics":
        bw, bh = (int(x) for x in str(args.board).lower().split("x"))
        capture_intrinsics(
            str(args.cam),
            Path(str(args.out)),
            board_wh=(bw, bh),
            prefer_fourcc=str(args.fourcc),
            max_mode=bool(args.max),
            fps=float(args.fps),
            rotate=str(args.rot),
            every_s=float(args.autosave),
        )
        return

    if args.cmd == "calibrate-intrinsics":
        bw, bh = (int(x) for x in str(args.board).lower().split("x"))
        out = calibrate_intrinsics(
            Path(str(args.dataset)),
            board_wh=(bw, bh),
            square_size=float(args.square),
            model=str(args.model),
        )
        print(f"Wrote {out}")
        return


if __name__ == "__main__":
    main()
