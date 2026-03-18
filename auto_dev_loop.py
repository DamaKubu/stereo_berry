"""auto_dev_loop.py

Autonomous 2-hour tuning loop (parameter-only) for stereo depth + motion.

What it does
- Captures short frame bursts from both cameras.
- Rectifies using calib_auto.npz.
- Computes disparity+depth using StereoSGBM.
- Computes objective metrics (valid disparity %, speckle, temporal stability, FPS).
- Saves a debug composite image per iteration.
- Optionally uses a GPT-5.2 vision-capable model to *suggest parameter updates*.
- Writes overrides to tuned_params.json (read by moving_depth.py) and repeats.

Why parameter-only (instead of rewriting code)
- Lets you leave this running safely for hours.
- Avoids “LLM broke the program” failure mode.
- Still converges fast because the biggest wins are in SGBM + ROI + motion thresholds.

Model
- Uses GPT-5.2 if you enable vision. (Change VISION_MODEL as needed.)

How to run
- Basic (no API calls):
    python auto_dev_loop.py
- With vision suggestions:
    export OPENAI_API_KEY=...  # preferred
    python auto_dev_loop.py --vision

If OPENAI_API_KEY is not set and you pass --vision, it will prompt once using getpass
(so it won’t echo). It never writes the key to disk.

Outputs
- auto_runs/<timestamp>/iter_000/ ... composite.png, metrics.json, params.json
- tuned_params.json at repo root (used by moving_depth.py)

"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import random
from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


# =========================
# ====== PARAMETERS =======
# =========================

# Cameras / orientation
CAM_LEFT = "/dev/video2"
CAM_RIGHT = "/dev/video0"
ROT_LEFT = cv2.ROTATE_90_COUNTERCLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_CLOCKWISE

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FOURCC_PREFERENCE = ["NV12", "YUYV"]

# Calibration
CALIB_NPZ = "calib_auto.npz"
RECTIFY_ALPHA = 0.8

# Distortion strength (1.0 = full correction). If rectification looks over-warped, try 0.7..0.9.
DISTORTION_SCALE = 0.9

# Extrinsics experiment mode for tuning:
# - "calib": use R,T from calib_auto.npz (but rescale baseline if FORCE_BASELINE_M is set)
# - "identity": force R = I and T = [-baseline,0,0]^T
EXTRINSICS_MODE = "calib"  # "calib" | "identity"
FORCE_BASELINE_M = 0.98

# Scene knowledge: ignore sky (your top ~half is sky)
ROI_Y0_FRAC = 0.55
ROI_Y1_FRAC = 1.00

# Motion / blobs (also tunable)
MOG2_HISTORY = 250
MOG2_VAR_THRESHOLD = 16
MIN_BLOB_AREA = 800

# Matching pre-processing (helps outdoor exposure differences)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = 8

# Penalize warp inconsistency (high values usually correlate with “fishy” rectification/mismatch)
WARP_ERR_WEIGHT = 60.0

_RNG = random.Random()
_RNG.seed(int(time.time() * 1e6) ^ (os.getpid() << 16))

# Depth mask for evaluation (meters): your buildings ~30-60m.
EVAL_NEAR_M = 5.0
EVAL_FAR_M = 80.0

# SGBM initial guess (tunable)
INIT_PARAMS: Dict[str, Any] = {
    "SGBM_MIN_DISP": 0,
    "SGBM_NUM_DISP": 128,
    "SGBM_BLOCK_SIZE": 5,
    "SGBM_UNIQUENESS": 7,
    "SGBM_SPECKLE_WINDOW": 80,
    "SGBM_SPECKLE_RANGE": 2,
    "SGBM_DISP12_MAXDIFF": 1,
    "VALID_DISP_MIN": 1.0,
    "RECTIFY_ALPHA": RECTIFY_ALPHA,
    "DISTORTION_SCALE": DISTORTION_SCALE,
    "EXTRINSICS_MODE": EXTRINSICS_MODE,
    "FORCE_BASELINE_M": FORCE_BASELINE_M,
    "ROI_Y0_FRAC": ROI_Y0_FRAC,
    "ROI_Y1_FRAC": ROI_Y1_FRAC,
    "MOG2_HISTORY": MOG2_HISTORY,
    "MOG2_VAR_THRESHOLD": MOG2_VAR_THRESHOLD,
    "MIN_BLOB_AREA": MIN_BLOB_AREA,
    "USE_CLAHE": USE_CLAHE,
    "CLAHE_CLIP_LIMIT": CLAHE_CLIP_LIMIT,
    "CLAHE_TILE_GRID": CLAHE_TILE_GRID,
}

# Time budget (defaults tuned for quick iteration).
# Use --seconds to override.
TOTAL_SECONDS = 60
BURST_SECONDS = 4.0
SLEEP_BETWEEN_ITERS = 0.2

# Vision model
VISION_MODEL = "gpt-5.2"  # configurable
VISION_MAX_SUGGESTIONS = 1

# If you can't paste into terminal, put your key in this file (first line) and run with:
#   python auto_dev_loop.py --vision --api-key-file .openai_api_key
DEFAULT_API_KEY_FILE = ".openai_api_key"

# Where to write overrides for moving_depth.py
TUNED_PARAMS_JSON = "tuned_params.json"


# =========================
# ====== RECTIFY ===========
# =========================


@dataclass
class Rectify:
    mapLx: NDArray[Any]
    mapLy: NDArray[Any]
    mapRx: NDArray[Any]
    mapRy: NDArray[Any]
    f_px: float
    baseline_m: float


@dataclass
class Calib:
    mtxL: NDArray[Any]
    distL: NDArray[Any]
    mtxR: NDArray[Any]
    distR: NDArray[Any]
    rot: NDArray[Any]
    trans: NDArray[Any]


def load_calibration(npz_path: str):
    calib = np.load(npz_path)
    mtxL, distL = calib["mtxL"], calib["distL"]
    mtxR, distR = calib["mtxR"], calib["distR"]
    R, T = calib["R"], calib["T"]
    return mtxL, distL, mtxR, distR, R, T


def _apply_distortion_scale(dist: NDArray[Any], scale: float) -> NDArray[Any]:
    s = float(scale)
    s = max(0.0, min(1.0, s))
    return (dist.astype(np.float64) * s).astype(np.float64)


def _apply_extrinsics_mode(R: NDArray[Any], T: NDArray[Any], mode: str, baseline_m: float) -> Tuple[NDArray[Any], NDArray[Any]]:
    m = str(mode)
    if m == "identity":
        rot = np.eye(3, dtype=np.float64)
        trans = np.array([[-float(baseline_m), 0.0, 0.0]], dtype=np.float64).T
        return rot, trans

    # default: use calibrated direction but force baseline magnitude
    trans = T.astype(np.float64)
    tnorm = float(np.linalg.norm(trans))
    if tnorm > 1e-9:
        trans = trans * (float(baseline_m) / tnorm)
    return R.astype(np.float64), trans


def _rectify_for_params(calib: Calib, image_size_wh: Tuple[int, int], params: Dict[str, Any]) -> Rectify:
    dist_scale = float(params.get("DISTORTION_SCALE", DISTORTION_SCALE))
    distL = _apply_distortion_scale(calib.distL, dist_scale)
    distR = _apply_distortion_scale(calib.distR, dist_scale)

    rot, trans = _apply_extrinsics_mode(
        calib.rot,
        calib.trans,
        str(params.get("EXTRINSICS_MODE", EXTRINSICS_MODE)),
        float(params.get("FORCE_BASELINE_M", FORCE_BASELINE_M)),
    )

    alpha = float(params.get("RECTIFY_ALPHA", RECTIFY_ALPHA))
    return build_rectify(image_size_wh, calib.mtxL, distL, calib.mtxR, distR, rot, trans, alpha=alpha)


def build_rectify(image_size_wh: Tuple[int, int], mtxL, distL, mtxR, distR, R, T, alpha: float) -> Rectify:
    w, h = image_size_wh
    R1, R2, P1, P2, _Q, *_ = cv2.stereoRectify(
        mtxL,
        distL,
        mtxR,
        distR,
        (w, h),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_16SC2)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_16SC2)

    f_px = float(P1[0, 0])
    baseline_m = float(-P2[0, 3] / P2[0, 0])

    return Rectify(mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy, f_px=f_px, baseline_m=baseline_m)


# =========================
# ====== CAPTURE ===========
# =========================


def _try_set_capture(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    for fourcc in FOURCC_PREFERENCE:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            break
        except Exception:
            continue


def _grab_pair(capL: cv2.VideoCapture, capR: cv2.VideoCapture) -> Optional[Tuple[NDArray[Any], NDArray[Any]]]:
    capL.grab()
    capR.grab()
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if not (retL and retR):
        return None
    frameL = cv2.rotate(frameL, ROT_LEFT)
    frameR = cv2.rotate(frameR, ROT_RIGHT)
    return frameL, frameR


# =========================
# ====== STEREO ============
# =========================


def _make_stereo_matcher(params: Dict[str, Any]) -> cv2.StereoSGBM:
    num_disp = int(params.get("SGBM_NUM_DISP", 128))
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16

    block_size = int(params.get("SGBM_BLOCK_SIZE", 5))
    if block_size % 2 == 0:
        block_size += 1

    stereo = cv2.StereoSGBM_create(
        minDisparity=int(params.get("SGBM_MIN_DISP", 0)),
        numDisparities=int(num_disp),
        blockSize=int(block_size),
        P1=8 * 1 * int(block_size) ** 2,
        P2=32 * 1 * int(block_size) ** 2,
        disp12MaxDiff=int(params.get("SGBM_DISP12_MAXDIFF", 1)),
        uniquenessRatio=int(params.get("SGBM_UNIQUENESS", 7)),
        speckleWindowSize=int(params.get("SGBM_SPECKLE_WINDOW", 80)),
        speckleRange=int(params.get("SGBM_SPECKLE_RANGE", 2)),
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return stereo


def _roi_slices(h: int, params: Dict[str, Any]) -> Tuple[slice, slice]:
    y0 = int(max(0.0, min(1.0, float(params.get("ROI_Y0_FRAC", ROI_Y0_FRAC)))) * h)
    y1 = int(max(0.0, min(1.0, float(params.get("ROI_Y1_FRAC", ROI_Y1_FRAC)))) * h)
    if y1 <= y0:
        y0, y1 = 0, h
    return slice(y0, y1), slice(0, None)


def _colorize_disparity(disp: NDArray[Any]) -> NDArray[Any]:
    d = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0).copy()
    d[d < 0] = 0
    if np.max(d) <= 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    denom = float(np.percentile(d[d > 0], 99.0))
    if not np.isfinite(denom) or denom <= 1e-6:
        denom = 1.0
    d8 = (np.clip(d / denom, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def _colorize_depth(depth_m: NDArray[Any], max_vis_m: float = 80.0) -> NDArray[Any]:
    max_vis = float(max_vis_m)
    if not np.isfinite(max_vis) or max_vis <= 1e-6:
        max_vis = 1.0
    d = np.nan_to_num(depth_m, nan=max_vis, posinf=max_vis, neginf=0.0)
    d = np.clip(d, 0.0, max_vis)
    d8 = (d / max_vis * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - d8, cv2.COLORMAP_TURBO)


# =========================
# ====== EVAL ==============
# =========================


@dataclass
class Metrics:
    fps: float
    valid_disp_frac: float
    depth_med_m: float
    depth_std_m: float
    speckle_frac: float
    motion_frac: float
    warp_err: float
    score: float


def _evaluate_burst(
    capL: cv2.VideoCapture,
    capR: cv2.VideoCapture,
    calib: Calib,
    image_size_wh: Tuple[int, int],
    params: Dict[str, Any],
    burst_seconds: float,
) -> Tuple[Metrics, NDArray[Any]]:
    rectify = _rectify_for_params(calib, image_size_wh, params)
    stereo = _make_stereo_matcher(params)
    clahe = None
    if bool(params.get("USE_CLAHE", USE_CLAHE)):
        clip = float(params.get("CLAHE_CLIP_LIMIT", CLAHE_CLIP_LIMIT))
        tile = int(params.get("CLAHE_TILE_GRID", CLAHE_TILE_GRID))
        tile = int(np.clip(tile, 4, 16))
        clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip, 1.0, 6.0)), tileGridSize=(tile, tile))
    bg = cv2.createBackgroundSubtractorMOG2(
        history=int(params.get("MOG2_HISTORY", MOG2_HISTORY)),
        varThreshold=float(params.get("MOG2_VAR_THRESHOLD", MOG2_VAR_THRESHOLD)),
        detectShadows=False,
    )

    t0 = time.time()
    frames = 0
    last_overlay: Optional[NDArray[Any]] = None

    valid_fracs = []
    depth_meds = []
    depth_stds = []
    speckle_fracs = []
    motion_fracs = []
    warp_errs = []

    while time.time() - t0 < burst_seconds:
        pair = _grab_pair(capL, capR)
        if pair is None:
            continue
        frameL, frameR = pair
        h, w = frameL.shape[:2]

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        grayLr = cv2.remap(grayL, rectify.mapLx, rectify.mapLy, cv2.INTER_LINEAR)
        grayRr = cv2.remap(grayR, rectify.mapRx, rectify.mapRy, cv2.INTER_LINEAR)

        if clahe is not None:
            grayLr = clahe.apply(grayLr)
            grayRr = clahe.apply(grayRr)

        disp = stereo.compute(grayLr, grayRr).astype(np.float32) / 16.0
        valid = disp >= float(params.get("VALID_DISP_MIN", 1.0))

        depth = np.full_like(disp, np.nan, dtype=np.float32)
        depth[valid] = (rectify.f_px * rectify.baseline_m) / disp[valid]

        roi_y, roi_x = _roi_slices(h, params)
        disp_roi = disp[roi_y, roi_x]
        valid_roi = valid[roi_y, roi_x]
        depth_roi = depth[roi_y, roi_x]

        # Warp right->left using disparity and measure residual.
        # Good rectification+matching yields low residual; “fishy” warps tend to spike it.
        try:
            left_roi = grayLr[roi_y, roi_x]
            right_roi = grayRr[roi_y, roi_x]
            hroi, wroi = left_roi.shape[:2]
            xs = np.arange(wroi, dtype=np.float32)[None, :].repeat(hroi, axis=0)
            ys = np.arange(hroi, dtype=np.float32)[:, None].repeat(wroi, axis=1)
            map_x = xs - disp_roi.astype(np.float32)
            map_y = ys
            warped_right = cv2.remap(right_roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid_w = valid_roi & np.isfinite(disp_roi) & (map_x >= 0.0) & (map_x < float(wroi - 1))
            if np.count_nonzero(valid_w) > 200:
                err = float(np.mean(np.abs(left_roi[valid_w].astype(np.float32) - warped_right[valid_w].astype(np.float32))) / 255.0)
                warp_errs.append(err)
        except Exception:
            pass

        valid_frac = float(np.count_nonzero(valid_roi) / max(1, valid_roi.size))
        valid_fracs.append(valid_frac)

        depth_clip = depth_roi[np.isfinite(depth_roi)]
        depth_clip = depth_clip[(depth_clip >= EVAL_NEAR_M) & (depth_clip <= EVAL_FAR_M)]
        if depth_clip.size > 500:
            depth_meds.append(float(np.median(depth_clip)))
            depth_stds.append(float(np.std(depth_clip)))

        # Speckle proxy: fraction of tiny connected components in valid disparity mask
        valid_u8 = (valid_roi.astype(np.uint8) * 255)
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_u8, connectivity=8)
        if nlabels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            tiny = np.sum(areas < 30)
            speckle_fracs.append(float(tiny / max(1, areas.size)))

        # Motion in ROI
        fg = bg.apply(grayLr[roi_y, roi_x])
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
        motion_fracs.append(float(np.count_nonzero(fg) / max(1, fg.size)) / 255.0)

        # Composite for vision/debug
        disp_color = _colorize_disparity(disp_roi)
        depth_color = _colorize_depth(depth_roi, max_vis_m=EVAL_FAR_M)
        fg_color = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
        left_color = cv2.cvtColor(grayLr[roi_y, roi_x], cv2.COLOR_GRAY2BGR)

        right_color = cv2.cvtColor(grayRr[roi_y, roi_x], cv2.COLOR_GRAY2BGR)
        lr_diff = cv2.absdiff(grayLr[roi_y, roi_x], grayRr[roi_y, roi_x])
        lr_diff = cv2.cvtColor(lr_diff, cv2.COLOR_GRAY2BGR)

        # stack 2x3
        top = np.hstack([left_color, right_color, disp_color])
        bot = np.hstack([depth_color, fg_color, lr_diff])
        last_overlay = np.vstack([top, bot])

        frames += 1

    dt = max(1e-6, time.time() - t0)
    fps = frames / dt

    v = float(np.mean(valid_fracs)) if valid_fracs else 0.0
    s = float(np.mean(speckle_fracs)) if speckle_fracs else 1.0
    m = float(np.mean(motion_fracs)) if motion_fracs else 1.0
    werr = float(np.median(warp_errs)) if warp_errs else 1.0

    dmed = float(np.mean(depth_meds)) if depth_meds else float("nan")
    dstd = float(np.mean(depth_stds)) if depth_stds else float("nan")

    # Score: reward valid disparity & FPS, penalize speckle and warp inconsistency.
    score = 110.0 * v + 2.0 * fps - 35.0 * s - WARP_ERR_WEIGHT * werr

    # Outdoor prior: if median depth is implausibly close, it's usually mismatches.
    if not np.isfinite(dmed):
        score -= 30.0
    elif dmed < 15.0:
        score -= 25.0
    elif dmed > 120.0:
        score -= 10.0

    overlay = last_overlay if last_overlay is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    return Metrics(
        fps=fps,
        valid_disp_frac=v,
        depth_med_m=dmed,
        depth_std_m=dstd,
        speckle_frac=s,
        motion_frac=m,
        warp_err=werr,
        score=score,
    ), overlay


# =========================
# ====== VISION ============
# =========================


def _maybe_get_openai_key(enable_vision: bool) -> Optional[str]:
    if not enable_vision:
        return None
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    return None


def _read_api_key_file(path: str) -> Optional[str]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            return None
        # allow either raw key or KEY=... lines
        if "=" in text and text.split("=", 1)[0].strip().upper() in {"OPENAI_API_KEY", "API_KEY"}:
            return text.split("=", 1)[1].strip()
        return text.splitlines()[0].strip()
    except Exception:
        return None


def _openai_ping(api_key: str, model: str) -> bool:
    try:
        from openai import OpenAI
    except Exception as e:
        print("OpenAI SDK import failed:", e)
        return False

    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
        )
        out = (resp.output_text or "").strip()
        print("OpenAI ping OK. Output:", out[:120])
        return True
    except Exception as e:
        print("OpenAI ping FAILED:", repr(e))
        return False


def _vision_suggest_params(
    api_key: str,
    model: str,
    overlay_bgr: NDArray[Any],
    current_params: Dict[str, Any],
    metrics: Metrics,
) -> Optional[Dict[str, Any]]:
    # Lazy import so script runs without openai installed.
    try:
        from openai import OpenAI
    except Exception:
        return None

    # Encode image
    ok, buf = cv2.imencode(".png", overlay_bgr)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    prompt = (
        "You are tuning a stereo depth + motion pipeline. "
        "The debug image is a 2x2 grid: "
        "top-left=rectified left ROI gray, top-right=disparity colormap, "
        "bottom-left=depth colormap, bottom-right=motion mask. "
        "Suggest parameter-only changes to improve disparity density and reduce speckle, "
        "without destroying edges or making rectification look overly warped/fisheye. "
        "Return ONLY JSON.\n\n"
        f"Current params: {json.dumps(current_params)}\n"
        f"Metrics: {json.dumps(metrics.__dict__)}\n\n"
        "Allowed keys: SGBM_NUM_DISP, SGBM_BLOCK_SIZE, SGBM_UNIQUENESS, "
        "SGBM_SPECKLE_WINDOW, SGBM_SPECKLE_RANGE, VALID_DISP_MIN, RECTIFY_ALPHA, "
        "DISTORTION_SCALE, ROI_Y0_FRAC, ROI_Y1_FRAC, MOG2_HISTORY, MOG2_VAR_THRESHOLD, MIN_BLOB_AREA.\n"
        "Rules: keep SGBM_NUM_DISP multiple of 16 (64..256), block size odd (3..11). "
        "DISTORTION_SCALE in [0.2, 1.0] is reasonable; lower values reduce over-warp/fisheye. "
        "Do not include any text outside JSON."
    )

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    text = (resp.output_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


# =========================
# ====== TUNING ============
# =========================


def _clamp_params(p: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(p)

    def clamp_int(key: str, lo: int, hi: int) -> None:
        if key in out:
            out[key] = int(np.clip(int(out[key]), lo, hi))

    def clamp_float(key: str, lo: float, hi: float) -> None:
        if key in out:
            out[key] = float(np.clip(float(out[key]), lo, hi))

    clamp_int("SGBM_NUM_DISP", 64, 256)
    if out.get("SGBM_NUM_DISP", 128) % 16 != 0:
        out["SGBM_NUM_DISP"] = int(out["SGBM_NUM_DISP"] // 16 * 16)
    clamp_int("SGBM_BLOCK_SIZE", 3, 11)
    if out.get("SGBM_BLOCK_SIZE", 5) % 2 == 0:
        out["SGBM_BLOCK_SIZE"] = int(out["SGBM_BLOCK_SIZE"] + 1)

    clamp_int("SGBM_UNIQUENESS", 0, 25)
    clamp_int("SGBM_SPECKLE_WINDOW", 0, 300)
    clamp_int("SGBM_SPECKLE_RANGE", 0, 10)
    clamp_int("SGBM_DISP12_MAXDIFF", -1, 25)

    clamp_float("VALID_DISP_MIN", 0.5, 4.0)
    clamp_float("RECTIFY_ALPHA", 0.0, 1.0)
    clamp_float("DISTORTION_SCALE", 0.0, 1.0)
    clamp_float("ROI_Y0_FRAC", 0.0, 0.9)
    clamp_float("ROI_Y1_FRAC", 0.1, 1.0)

    if "EXTRINSICS_MODE" in out:
        out["EXTRINSICS_MODE"] = str(out["EXTRINSICS_MODE"]) if str(out["EXTRINSICS_MODE"]) in {"calib", "identity"} else "calib"
    if "FORCE_BASELINE_M" in out:
        out["FORCE_BASELINE_M"] = float(np.clip(float(out["FORCE_BASELINE_M"]), 0.05, 5.0))

    if "USE_CLAHE" in out:
        out["USE_CLAHE"] = bool(out["USE_CLAHE"])
    if "CLAHE_CLIP_LIMIT" in out:
        out["CLAHE_CLIP_LIMIT"] = float(np.clip(float(out["CLAHE_CLIP_LIMIT"]), 1.0, 6.0))
    if "CLAHE_TILE_GRID" in out:
        out["CLAHE_TILE_GRID"] = int(np.clip(int(out["CLAHE_TILE_GRID"]), 4, 16))

    clamp_int("MOG2_HISTORY", 50, 2000)
    clamp_float("MOG2_VAR_THRESHOLD", 4.0, 64.0)
    clamp_int("MIN_BLOB_AREA", 100, 20000)

    # Ensure ROI ordering
    if out.get("ROI_Y1_FRAC", 1.0) <= out.get("ROI_Y0_FRAC", 0.4):
        out["ROI_Y0_FRAC"] = 0.4
        out["ROI_Y1_FRAC"] = 1.0

    return out


def _mutate_params(p: Dict[str, Any]) -> Dict[str, Any]:
    # Small random-ish mutations (no RNG dependency on vision)
    q = dict(p)

    def pick_delta(options):
        return _RNG.choice(options)

    # Explore typical knobs
    q["SGBM_BLOCK_SIZE"] = int(np.clip(int(q["SGBM_BLOCK_SIZE"]) + pick_delta([-2, 0, 2]), 3, 11))
    if q["SGBM_BLOCK_SIZE"] % 2 == 0:
        q["SGBM_BLOCK_SIZE"] += 1

    q["SGBM_UNIQUENESS"] = int(np.clip(int(q["SGBM_UNIQUENESS"]) + pick_delta([-3, 0, 3]), 0, 25))
    q["SGBM_SPECKLE_WINDOW"] = int(np.clip(int(q["SGBM_SPECKLE_WINDOW"]) + pick_delta([-30, 0, 30]), 0, 300))
    q["SGBM_SPECKLE_RANGE"] = int(np.clip(int(q["SGBM_SPECKLE_RANGE"]) + pick_delta([-1, 0, 1]), 0, 10))

    # Occasionally widen disparities
    if _RNG.random() < 0.15:
        q["SGBM_NUM_DISP"] = int(np.clip(int(q["SGBM_NUM_DISP"]) + pick_delta([-32, 0, 32]), 64, 256))
        q["SGBM_NUM_DISP"] = int(q["SGBM_NUM_DISP"] // 16 * 16)

    # ROI tweak to suppress sky
    if _RNG.random() < 0.10:
        q["ROI_Y0_FRAC"] = float(np.clip(float(q["ROI_Y0_FRAC"]) + pick_delta([-0.05, 0.0, 0.05]), 0.0, 0.9))

    # Explore rectification / preprocessing knobs (these matter for “fishy” warps)
    if _RNG.random() < 0.20:
        q["DISTORTION_SCALE"] = float(
            np.clip(
                float(q.get("DISTORTION_SCALE", DISTORTION_SCALE)) + pick_delta([-0.15, -0.05, 0.0, 0.05, 0.15]),
                0.0,
                1.0,
            )
        )
    if _RNG.random() < 0.10:
        q["RECTIFY_ALPHA"] = float(
            np.clip(
                float(q.get("RECTIFY_ALPHA", RECTIFY_ALPHA)) + pick_delta([-0.3, -0.1, 0.0, 0.1, 0.3]),
                0.0,
                1.0,
            )
        )
    if _RNG.random() < 0.08:
        q["EXTRINSICS_MODE"] = "identity" if str(q.get("EXTRINSICS_MODE", EXTRINSICS_MODE)) == "calib" else "calib"
    if _RNG.random() < 0.10:
        q["USE_CLAHE"] = not bool(q.get("USE_CLAHE", USE_CLAHE))
    if _RNG.random() < 0.10:
        q["CLAHE_CLIP_LIMIT"] = float(
            np.clip(float(q.get("CLAHE_CLIP_LIMIT", CLAHE_CLIP_LIMIT)) + pick_delta([-1.0, -0.5, 0.0, 0.5, 1.0]), 1.0, 6.0)
        )
    if _RNG.random() < 0.10:
        q["CLAHE_TILE_GRID"] = int(np.clip(int(q.get("CLAHE_TILE_GRID", CLAHE_TILE_GRID)) + pick_delta([-2, 0, 2]), 4, 16))

    # Occasionally jump to a “no-fishy” preset (less distortion correction, minimal warp)
    if _RNG.random() < 0.03:
        q["EXTRINSICS_MODE"] = "identity"
        q["DISTORTION_SCALE"] = 0.35
        q["RECTIFY_ALPHA"] = 1.0
        q["USE_CLAHE"] = True
        q["CLAHE_CLIP_LIMIT"] = 3.0
        q["CLAHE_TILE_GRID"] = 8

    return _clamp_params(q)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true", help="Enable GPT-5.2 vision suggestions")
    ap.add_argument("--ping", action="store_true", help="Only test OpenAI API and exit")
    ap.add_argument(
        "--api-key-file",
        default=DEFAULT_API_KEY_FILE,
        help="Path to a file containing your OpenAI API key (first line)",
    )
    ap.add_argument(
        "--prompt-key",
        action="store_true",
        help="Prompt for API key via hidden input (getpass)",
    )
    ap.add_argument("--seconds", type=int, default=TOTAL_SECONDS, help="Total runtime budget")
    args = ap.parse_args()

    enable_vision = bool(args.vision)
    api_key = _maybe_get_openai_key(enable_vision)
    if enable_vision and not api_key:
        api_key = _read_api_key_file(str(args.api_key_file))
    if enable_vision and not api_key and args.prompt_key:
        api_key = getpass("OpenAI API key (hidden): ").strip() or None

    if args.ping:
        if not api_key:
            print(
                "No API key provided. Options:\n"
                "- export OPENAI_API_KEY=...\n"
                f"- create {DEFAULT_API_KEY_FILE} and pass --api-key-file {DEFAULT_API_KEY_FILE}\n"
                "- or run with --prompt-key"
            )
            return
        ok = _openai_ping(api_key, VISION_MODEL)
        raise SystemExit(0 if ok else 2)

    out_root = Path("auto_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    print("Auto tuning run dir:", str(out_root), flush=True)
    print(
        "Vision:",
        ("ON" if enable_vision and bool(api_key) else "OFF"),
        "Model:",
        VISION_MODEL,
        flush=True,
    )

    # Open cameras
    capL = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Cannot open cameras")
    _try_set_capture(capL)
    _try_set_capture(capR)

    # First frame for sizing
    pair = None
    for _ in range(30):
        pair = _grab_pair(capL, capR)
        if pair is not None:
            break
        time.sleep(0.05)
    if pair is None:
        raise RuntimeError("Could not grab initial frames")

    frameL0, _frameR0 = pair
    h, w = frameL0.shape[:2]

    mtxL, distL, mtxR, distR, rot, trans = load_calibration(CALIB_NPZ)
    calib = Calib(mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, rot=rot, trans=trans)

    best_params = _clamp_params(dict(INIT_PARAMS))
    best_score = -1e9

    start = time.time()
    it = 0

    while time.time() - start < float(args.seconds):
        # Evaluate candidate
        cand = best_params if it == 0 else _mutate_params(best_params)

        metrics, overlay = _evaluate_burst(capL, capR, calib, (w, h), cand, burst_seconds=BURST_SECONDS)

        iter_dir = out_root / f"iter_{it:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(iter_dir / "composite.png"), overlay)
        _write_json(iter_dir / "metrics.json", metrics.__dict__)
        _write_json(iter_dir / "params.json", cand)

        # Optional vision suggestion
        if enable_vision and api_key:
            try:
                suggestion = _vision_suggest_params(api_key, VISION_MODEL, overlay, cand, metrics)
            except Exception as e:
                suggestion = None
                _write_json(iter_dir / "vision_error.json", {"error": repr(e)})

            if isinstance(suggestion, dict) and suggestion:
                cand2 = dict(cand)
                cand2.update({k: v for k, v in suggestion.items() if isinstance(k, str)})
                cand2 = _clamp_params(cand2)
                metrics2, overlay2 = _evaluate_burst(
                    capL,
                    capR,
                    calib,
                    (w, h),
                    cand2,
                    burst_seconds=max(3.0, BURST_SECONDS * 0.5),
                )
                cv2.imwrite(str(iter_dir / "composite_after_vision.png"), overlay2)
                _write_json(iter_dir / "metrics_after_vision.json", metrics2.__dict__)
                _write_json(iter_dir / "params_after_vision.json", cand2)
                if metrics2.score > metrics.score:
                    cand, metrics = cand2, metrics2

        # Progress line (every iter)
        print(
            f"iter={it:04d} score={metrics.score:7.2f} best={best_score:7.2f} "
            f"valid={metrics.valid_disp_frac*100:5.1f}% fps={metrics.fps:4.1f} "
            f"speckle={metrics.speckle_frac:4.2f} warp={metrics.warp_err:4.2f} motion={metrics.motion_frac:4.2f}",
            flush=True,
        )

        # Keep best
        if metrics.score > best_score:
            best_score = metrics.score
            best_params = cand
            _write_json(Path(TUNED_PARAMS_JSON), best_params)
            _write_json(out_root / "best.json", {"score": best_score, "params": best_params, "metrics": metrics.__dict__})
            print("  new best -> wrote", TUNED_PARAMS_JSON, flush=True)

        it += 1
        time.sleep(max(0.0, SLEEP_BETWEEN_ITERS))

    capL.release()
    capR.release()

    _write_json(out_root / "final_best.json", {"score": best_score, "params": best_params})
    print("Done. Best score:", best_score)
    print("Best params written to:", TUNED_PARAMS_JSON)
    print("Run moving viewer:")
    print("  python moving_depth.py")


if __name__ == "__main__":
    main()
