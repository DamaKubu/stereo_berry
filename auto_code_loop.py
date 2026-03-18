"""auto_code_loop.py

Self-editing development loop.

Unlike auto_dev_loop.py (parameter tuning), this script can ask a GPT-5.2 model to
propose full code edits (complete file rewrites) and then apply them locally.

Safety model
- Only edits files on an allowlist.
- Applies changes by writing full file content with backups.
- Runs python syntax checks (py_compile) after edits.
- Optionally runs an OFFLINE stereo evaluation on saved image pairs in data/left + data/right.
- If evaluation gets worse, automatically reverts.

This is intentionally conservative: it’s “LLM-assisted coding with guardrails”,
not an uncontrolled self-modifying agent.

Run
- Dry run (generate suggestions, don’t apply):
    python auto_code_loop.py --vision --dry-run

- Apply edits with offline eval:
    python auto_code_loop.py --vision --apply --offline

API key
- export OPENAI_API_KEY=...   (preferred)
- or put it in .openai_api_key (first line) and pass --api-key-file

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray


VISION_MODEL_DEFAULT = "gpt-5.2"
DEFAULT_API_KEY_FILE = ".openai_api_key"

# Default timeout for a single LLM request (seconds). Also enforced via SIGALRM on Linux.
DEFAULT_LLM_TIMEOUT_S = 90

# Default number of retries for the LLM request.
DEFAULT_LLM_RETRIES = 2

DEFAULT_ALLOWLIST = [
    "moving_depth.py",
    "auto_dev_loop.py",
    "live_depth.py",
    "evaluate.py",
]

AUTO_CODE_RUNS_DIR = Path("auto_code_runs")


_ROT_CHOICES = {"none", "cw", "ccw", "180"}

_T = TypeVar("_T")


def _parse_rot(s: str) -> Optional[int]:
    s = str(s).lower().strip()
    if s not in _ROT_CHOICES:
        return None
    if s == "cw":
        return cv2.ROTATE_90_CLOCKWISE
    if s == "ccw":
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if s == "180":
        return cv2.ROTATE_180
    return None


# -------------------------
# ----- Key handling ------
# -------------------------

def _read_api_key_file(path: str) -> Optional[str]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            return None
        if "=" in text and text.split("=", 1)[0].strip().upper() in {"OPENAI_API_KEY", "API_KEY"}:
            return text.split("=", 1)[1].strip()
        return text.splitlines()[0].strip()
    except Exception:
        return None


def _get_api_key(api_key_file: str) -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip() or None
    return _read_api_key_file(api_key_file)


def _api_key_source(api_key_file: str) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return "env:OPENAI_API_KEY"
    if _read_api_key_file(api_key_file):
        return f"file:{api_key_file}"
    return "missing"


# -------------------------
# ----- Offline eval ------
# -------------------------

@dataclass
class OfflineMetrics:
    pairs: int
    valid_disp_frac: float
    speckle_frac: float
    warp_err: float


def _load_calib(npz_path: str) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    calib = np.load(npz_path)
    return calib["mtxL"], calib["distL"], calib["mtxR"], calib["distR"], calib["R"], calib["T"]


def _apply_distortion_scale(dist: NDArray[Any], scale: float) -> NDArray[Any]:
    s = float(max(0.0, min(1.0, scale)))
    return (dist.astype(np.float64) * s).astype(np.float64)


def _apply_extrinsics_mode(R: NDArray[Any], T: NDArray[Any], mode: str, baseline_m: float) -> Tuple[NDArray[Any], NDArray[Any]]:
    mode = str(mode)
    if mode == "identity":
        rot2 = np.eye(3, dtype=np.float64)
        trans2 = np.array([[-abs(float(baseline_m))], [0.0], [0.0]], dtype=np.float64)
        return rot2, trans2

    # default: calib, but optionally rescale baseline magnitude
    rot2 = R.astype(np.float64)
    trans2 = T.astype(np.float64)
    b = float(np.linalg.norm(trans2))
    if b > 1e-9 and float(baseline_m) > 0:
        trans2 = trans2 * (float(baseline_m) / b)
    return rot2, trans2


def _build_rectify(
    w: int,
    h: int,
    mtxL: NDArray[Any],
    distL: NDArray[Any],
    mtxR: NDArray[Any],
    distR: NDArray[Any],
    R: NDArray[Any],
    T: NDArray[Any],
    alpha: float,
):
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
    return mapLx, mapLy, mapRx, mapRy, f_px, baseline_m


def _make_sgbm(params: Dict[str, Any]) -> cv2.StereoSGBM:
    num_disp = int(params.get("SGBM_NUM_DISP", 128))
    if num_disp % 16 != 0:
        num_disp = (num_disp // 16 + 1) * 16

    block_size = int(params.get("SGBM_BLOCK_SIZE", 5))
    if block_size % 2 == 0:
        block_size += 1

    return cv2.StereoSGBM_create(
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


def _offline_eval(
    calib_npz: str,
    left_dir: str,
    right_dir: str,
    params: Dict[str, Any],
    max_pairs: int = 20,
    rot_left: Optional[int] = None,
    rot_right: Optional[int] = None,
) -> OfflineMetrics:
    exts = ["*.png", "*.jpg", "*.jpeg"]
    left_paths = sorted([p for ext in exts for p in Path(left_dir).glob(ext)])
    right_paths = sorted([p for ext in exts for p in Path(right_dir).glob(ext)])
    n = min(len(left_paths), len(right_paths), int(max_pairs))
    if n <= 0:
        return OfflineMetrics(pairs=0, valid_disp_frac=0.0, speckle_frac=1.0, warp_err=1.0)

    mtxL, distL0, mtxR, distR0, R0, T0 = _load_calib(calib_npz)

    dist_scale = float(params.get("DISTORTION_SCALE", 1.0))
    distL = _apply_distortion_scale(distL0, dist_scale)
    distR = _apply_distortion_scale(distR0, dist_scale)

    R, T = _apply_extrinsics_mode(
        R0,
        T0,
        str(params.get("EXTRINSICS_MODE", "calib")),
        float(params.get("FORCE_BASELINE_M", 0.98)),
    )

    alpha = float(params.get("RECTIFY_ALPHA", 0.8))

    valid_fracs: List[float] = []
    speckle_fracs: List[float] = []
    warp_errs: List[float] = []

    clahe = None
    if bool(params.get("USE_CLAHE", True)):
        clip = float(params.get("CLAHE_CLIP_LIMIT", 2.0))
        tile = int(params.get("CLAHE_TILE_GRID", 8))
        tile = int(np.clip(tile, 4, 16))
        clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip, 1.0, 6.0)), tileGridSize=(tile, tile))

    stereo = _make_sgbm(params)

    for i in range(n):
        imL = cv2.imread(str(left_paths[i]), cv2.IMREAD_COLOR)
        imR = cv2.imread(str(right_paths[i]), cv2.IMREAD_COLOR)
        if imL is None or imR is None:
            continue

        if rot_left is not None:
            imL = cv2.rotate(imL, rot_left)
        if rot_right is not None:
            imR = cv2.rotate(imR, rot_right)

        # Assume files are already rotated consistently (same as your capture pipeline).
        h, w = imL.shape[:2]
        mapLx, mapLy, mapRx, mapRy, _f, _b = _build_rectify(w, h, mtxL, distL, mtxR, distR, R, T, alpha)

        grayL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
        grayLr = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
        grayRr = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)
        if clahe is not None:
            grayLr = clahe.apply(grayLr)
            grayRr = clahe.apply(grayRr)

        disp = stereo.compute(grayLr, grayRr).astype(np.float32) / 16.0
        valid = disp >= float(params.get("VALID_DISP_MIN", 1.0))
        valid_fracs.append(float(np.count_nonzero(valid) / max(1, valid.size)))

        valid_u8 = (valid.astype(np.uint8) * 255)
        nlabels, _labels, stats, _ = cv2.connectedComponentsWithStats(valid_u8, connectivity=8)
        if nlabels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            tiny = np.sum(areas < 30)
            speckle_fracs.append(float(tiny / max(1, areas.size)))

        # Warp consistency error
        try:
            xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
            ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
            map_x = xs - disp.astype(np.float32)
            map_y = ys
            warped_right = cv2.remap(grayRr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid_w = valid & np.isfinite(disp) & (map_x >= 0.0) & (map_x < float(w - 1))
            if np.count_nonzero(valid_w) > 500:
                err = float(np.mean(np.abs(grayLr[valid_w].astype(np.float32) - warped_right[valid_w].astype(np.float32))) / 255.0)
                warp_errs.append(err)
        except Exception:
            pass

    if not valid_fracs:
        return OfflineMetrics(pairs=0, valid_disp_frac=0.0, speckle_frac=1.0, warp_err=1.0)

    return OfflineMetrics(
        pairs=len(valid_fracs),
        valid_disp_frac=float(np.mean(valid_fracs)),
        speckle_frac=float(np.mean(speckle_fracs)) if speckle_fracs else 1.0,
        warp_err=float(np.median(warp_errs)) if warp_errs else 1.0,
    )


def _offline_score(m: OfflineMetrics) -> float:
    # Higher better.
    return 120.0 * m.valid_disp_frac - 35.0 * m.speckle_frac - 80.0 * m.warp_err


def _colorize_disp(disp: NDArray[Any]) -> NDArray[Any]:
    d = np.nan_to_num(disp.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    d[d < 0] = 0
    if float(np.max(d)) <= 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    denom = float(np.percentile(d[d > 0], 99.0)) if np.any(d > 0) else 1.0
    if not np.isfinite(denom) or denom <= 1e-6:
        denom = 1.0
    d8 = (np.clip(d / denom, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)


def _debug_composite_one_pair(
    calib_npz: str,
    left_path: Path,
    right_path: Path,
    params: Dict[str, Any],
    rot_left: Optional[int] = None,
    rot_right: Optional[int] = None,
) -> Optional[NDArray[Any]]:
    imL = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    imR = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if imL is None or imR is None:
        return None
    if rot_left is not None:
        imL = cv2.rotate(imL, rot_left)
    if rot_right is not None:
        imR = cv2.rotate(imR, rot_right)

    h, w = imL.shape[:2]
    mtxL, distL0, mtxR, distR0, R0, T0 = _load_calib(calib_npz)

    dist_scale = float(params.get("DISTORTION_SCALE", 1.0))
    distL = _apply_distortion_scale(distL0, dist_scale)
    distR = _apply_distortion_scale(distR0, dist_scale)
    R, T = _apply_extrinsics_mode(
        R0,
        T0,
        str(params.get("EXTRINSICS_MODE", "calib")),
        float(params.get("FORCE_BASELINE_M", 0.98)),
    )
    alpha = float(params.get("RECTIFY_ALPHA", 0.8))
    mapLx, mapLy, mapRx, mapRy, _f, _b = _build_rectify(w, h, mtxL, distL, mtxR, distR, R, T, alpha)

    grayL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
    grayLr = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
    grayRr = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)

    clahe = None
    if bool(params.get("USE_CLAHE", True)):
        clip = float(params.get("CLAHE_CLIP_LIMIT", 2.0))
        tile = int(params.get("CLAHE_TILE_GRID", 8))
        tile = int(np.clip(tile, 4, 16))
        clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip, 1.0, 6.0)), tileGridSize=(tile, tile))
    if clahe is not None:
        grayLr = clahe.apply(grayLr)
        grayRr = clahe.apply(grayRr)

    stereo = _make_sgbm(params)
    disp = stereo.compute(grayLr, grayRr).astype(np.float32) / 16.0
    valid = disp >= float(params.get("VALID_DISP_MIN", 1.0))

    left_bgr = cv2.cvtColor(grayLr, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(grayRr, cv2.COLOR_GRAY2BGR)
    disp_color = _colorize_disp(disp)
    diff = cv2.absdiff(grayLr, grayRr)
    diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

    valid_vis = (valid.astype(np.uint8) * 255)
    valid_vis = cv2.cvtColor(valid_vis, cv2.COLOR_GRAY2BGR)

    top = np.hstack([left_bgr, right_bgr])
    bot = np.hstack([disp_color, diff])
    out = np.vstack([top, bot])
    cv2.putText(out, f"disp valid={100.0*float(np.mean(valid)):.1f}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(out, f"DIST_SCALE={params.get('DISTORTION_SCALE', '')} alpha={params.get('RECTIFY_ALPHA', '')} ext={params.get('EXTRINSICS_MODE', '')}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


# -------------------------
# ----- Patch logic -------
# -------------------------

_FORBIDDEN_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\.(Popen|call|run)\b",
    r"\bshutil\.rmtree\b",
    r"\brm\s+-rf\b",
]


def _is_safe_content(text: str) -> bool:
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, text):
            return False
    return True


def _py_compile(paths: List[str]) -> Tuple[bool, str]:
    try:
        cmd = ["python3", "-m", "py_compile", *paths]
        p = subprocess.run(cmd, capture_output=True, text=True)
        ok = p.returncode == 0
        out = (p.stdout or "") + (p.stderr or "")
        return ok, out.strip()
    except Exception as e:
        return False, repr(e)


def _read_files(paths: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        out[p] = Path(p).read_text(encoding="utf-8")
    return out


def _write_with_backup(path: str, new_text: str, backup_dir: Path) -> None:
    src = Path(path)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / (src.name + ".bak")
    if src.exists():
        shutil.copy2(src, backup_path)
    tmp = src.with_suffix(src.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(src)


def _restore_backups(backup_dir: Path, paths: List[str]) -> None:
    for p in paths:
        src = Path(p)
        bak = backup_dir / (src.name + ".bak")
        if bak.exists():
            tmp = src.with_suffix(src.suffix + ".tmp")
            tmp.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
            tmp.replace(src)


# -------------------------
# ----- LLM call ----------
# -------------------------

SYSTEM_PROMPT = """You are a senior engineer working on a stereo depth pipeline.
You will propose code edits as FULL FILE REPLACEMENTS (not diffs).

Constraints:
- Only edit files listed in allowlist.
- Output MUST be valid JSON matching the schema.
- Do not add external dependencies.
- Prefer small, focused changes, but you MAY refactor substantially if needed.
- Do not use os.system or subprocess.

Goal:
- Reduce \"fishy\" rectification artifacts and improve disparity consistency.
- Improve offline score (provided).
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _write_stage(iter_dir: Path, stage: str) -> None:
    _write_text(iter_dir / "llm_stage.txt", stage + "\n")


def _with_alarm_timeout(seconds: int, fn: Callable[[], _T]) -> _T:
    """Run fn() with a hard timeout on Unix via SIGALRM.

    This prevents "hung forever" network calls from stalling the loop silently.
    """

    if seconds <= 0:
        return fn()

    def _handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"LLM call exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _call_openai(
    api_key: str,
    model: str,
    payload: Dict[str, Any],
    *,
    timeout_s: int,
    iter_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Call OpenAI Responses API.

    Writes stage markers to iter_dir (if provided) so we can see where it hangs.
    """

    if iter_dir is not None:
        _write_stage(iter_dir, "import_openai")

    from openai import OpenAI

    # Build the request payload once (so if we hang, we at least wrote llm_stage).
    if iter_dir is not None:
        _write_stage(iter_dir, "serialize_payload")
    user_text = json.dumps(payload)

    def _do_request():
        if iter_dir is not None:
            _write_stage(iter_dir, "create_client")
        # Pass per-request timeout as well, and keep client creation simple.
        client = OpenAI(api_key=api_key)

        if iter_dir is not None:
            _write_stage(iter_dir, "responses_create")
        req = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
        }
        # Some SDK versions accept timeout here; if not, fall back.
        try:
            resp = client.responses.create(**req, timeout=float(timeout_s))
        except TypeError:
            resp = client.responses.create(**req)

        if iter_dir is not None:
            _write_stage(iter_dir, "parse_response")
        text = (resp.output_text or "").strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}\s*$", text)
            if m:
                return json.loads(m.group(0))
            raise

    return _with_alarm_timeout(int(timeout_s), _do_request)


def _openai_ping(api_key: str, model: str, timeout_s: int) -> bool:
    try:
        from openai import OpenAI
    except Exception as e:
        print("OpenAI SDK import failed:", repr(e))
        return False

    try:
        client = OpenAI(api_key=api_key)

        def _do():
            req = {
                "model": model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
            }
            try:
                _ = client.responses.create(**req, timeout=float(timeout_s))
            except TypeError:
                _ = client.responses.create(**req)

        _with_alarm_timeout(int(timeout_s), _do)
        return True
    except Exception as e:
        print("OpenAI ping failed:", repr(e))
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true", help="Enable GPT-5.2 code-edit suggestions")
    ap.add_argument("--ping", action="store_true", help="Test OpenAI API call and exit")
    ap.add_argument("--apply", action="store_true", help="Apply edits (otherwise just save proposals)")
    ap.add_argument("--dry-run", action="store_true", help="Alias for not applying edits")
    ap.add_argument("--offline", action="store_true", help="Run offline eval on data/left + data/right")
    ap.add_argument("--show", action="store_true", help="Show a live OpenCV preview window each iteration")
    ap.add_argument("--interactive", action="store_true", help="Prompt for feedback each iteration (type 'shit' etc)")
    ap.add_argument("--calib", default="calib_auto.npz")
    ap.add_argument("--left-dir", default="data/left")
    ap.add_argument("--right-dir", default="data/right")
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--rot-left", default="none", choices=sorted(_ROT_CHOICES))
    ap.add_argument("--rot-right", default="none", choices=sorted(_ROT_CHOICES))
    ap.add_argument("--model", default=VISION_MODEL_DEFAULT)
    ap.add_argument("--api-key-file", default=DEFAULT_API_KEY_FILE)
    ap.add_argument("--llm-timeout", type=int, default=DEFAULT_LLM_TIMEOUT_S, help="Timeout per LLM request (seconds)")
    ap.add_argument("--llm-retries", type=int, default=DEFAULT_LLM_RETRIES, help="Retries for LLM request")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--allow", nargs="*", default=DEFAULT_ALLOWLIST)
    ap.add_argument(
        "--goal",
        default="Make rectification less fishy and improve disparity stability for outdoor far scene.",
    )
    args = ap.parse_args()

    enable = bool(args.vision)
    do_apply = bool(args.apply) and not bool(args.dry_run)

    allow = [str(x) for x in args.allow]
    allow = [a for a in allow if Path(a).exists()]
    if not allow:
        print("No allowlisted files found. Pass --allow with existing files.")
        raise SystemExit(2)

    api_key = _get_api_key(str(args.api_key_file)) if enable else None
    if enable and not api_key:
        print("No API key found. Set OPENAI_API_KEY or create .openai_api_key.")
        raise SystemExit(2)

    if enable:
        assert api_key is not None

    if bool(args.ping):
        if not api_key:
            print("No API key found. Set OPENAI_API_KEY or create .openai_api_key.")
            raise SystemExit(2)
        model = str(args.model)
        timeout_s = int(args.llm_timeout)
        src = _api_key_source(str(args.api_key_file))
        print(f"OpenAI ping: model={model} timeout={timeout_s}s key_source={src}")
        ok = _openai_ping(api_key=api_key, model=model, timeout_s=timeout_s)
        print("OpenAI ping:", "OK" if ok else "FAILED")
        raise SystemExit(0 if ok else 2)

    run_dir = AUTO_CODE_RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Auto code run dir:", run_dir)

    # Baseline offline score
    base_params = {}
    tuned = Path("tuned_params.json")
    if tuned.exists():
        try:
            base_params = json.loads(tuned.read_text(encoding="utf-8"))
        except Exception:
            base_params = {}

    baseline_score = None
    baseline_metrics = None
    left_dir_used = str(args.left_dir)
    right_dir_used = str(args.right_dir)
    rot_left = _parse_rot(args.rot_left)
    rot_right = _parse_rot(args.rot_right)

    if bool(args.offline):
        left_dir = left_dir_used
        right_dir = right_dir_used
        # If data/left/right is empty, fall back to the existing capture folders.
        if Path(left_dir).exists() and Path(right_dir).exists():
            exts = ["*.png", "*.jpg", "*.jpeg"]
            has_left = any(Path(left_dir).glob(ext) for ext in exts)
            has_right = any(Path(right_dir).glob(ext) for ext in exts)
        else:
            has_left = has_right = False
        if not (has_left and has_right):
            if Path("data/data_cam1/images").exists() and Path("data/data_cam2/images").exists():
                left_dir = "data/data_cam1/images"
                right_dir = "data/data_cam2/images"

        left_dir_used = left_dir
        right_dir_used = right_dir

        m0 = _offline_eval(
            args.calib,
            left_dir,
            right_dir,
            base_params,
            max_pairs=int(args.max_pairs),
            rot_left=rot_left,
            rot_right=rot_right,
        )
        baseline_metrics = m0
        baseline_score = _offline_score(m0)
        print("Baseline offline:", m0, "score=", baseline_score)

        if bool(args.show):
            # Show baseline composite (first pair)
            exts = ["*.png", "*.jpg", "*.jpeg"]
            left_paths = sorted([p for ext in exts for p in Path(left_dir_used).glob(ext)])
            right_paths = sorted([p for ext in exts for p in Path(right_dir_used).glob(ext)])
            if left_paths and right_paths:
                comp0 = _debug_composite_one_pair(args.calib, left_paths[0], right_paths[0], base_params, rot_left=rot_left, rot_right=rot_right)
                if comp0 is not None:
                    cv2.imshow("auto_code_loop (baseline)", comp0)
                    cv2.waitKey(1)

    original_texts = _read_files(allow)

    user_feedback: str = ""

    for it in range(int(args.iters)):
        iter_dir = run_dir / f"iter_{it:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== iter {it}/{int(args.iters)-1} ===")
        print("Allowlist:", ", ".join(allow))
        print("Offline data:", left_dir_used, "|", right_dir_used)

        payload = {
            "goal": str(args.goal),
            "user_feedback": user_feedback.strip() or None,
            "allowlist": allow,
            "baseline_offline_metrics": baseline_metrics.__dict__ if baseline_metrics else None,
            "baseline_offline_score": baseline_score,
            "tuned_params_json": base_params,
            "files": original_texts,
        }
        _write_json(iter_dir / "payload.json", payload)

        if not enable:
            print("--vision not enabled; stopping.")
            break

        print(
            "Calling model:",
            args.model,
            f"(timeout={int(args.llm_timeout)}s, retries={int(args.llm_retries)}) ...",
        )
        sys.stdout.flush()

        (iter_dir / "llm_started_at.txt").write_text(datetime.now().isoformat() + "\n", encoding="utf-8")

        proposal: Optional[Dict[str, Any]] = None

        for attempt in range(int(args.llm_retries) + 1):
            try:
                _write_stage(iter_dir, f"attempt_{attempt}")
                proposal = _call_openai(
                    api_key=api_key,
                    model=str(args.model),
                    payload=payload,
                    timeout_s=int(args.llm_timeout),
                    iter_dir=iter_dir,
                )
                break
            except KeyboardInterrupt:
                print("Interrupted; stopping.")
                return
            except BaseException as e:
                print("LLM call failed:", repr(e))
                _write_text(iter_dir / "llm_error.txt", repr(e) + "\n")
                _write_text(iter_dir / "llm_traceback.txt", traceback.format_exc() + "\n")
                _write_json(
                    iter_dir / "llm_error.json",
                    {
                        "error_type": type(e).__name__,
                        "error": repr(e),
                        "attempt": attempt,
                        "model": str(args.model),
                        "timeout_s": int(args.llm_timeout),
                    },
                )
                if attempt < int(args.llm_retries):
                    time.sleep(1.0 + 0.5 * attempt)

        (iter_dir / "llm_finished_at.txt").write_text(datetime.now().isoformat() + "\n", encoding="utf-8")

        if proposal is None:
            # Failed all attempts.
            _write_stage(iter_dir, "failed_all_attempts")
            continue

        _write_json(iter_dir / "proposal.json", proposal)

        files = proposal.get("files")
        if not isinstance(files, dict) or not files:
            print("No files in proposal; stopping.")
            break

        # Validate proposal
        edited_paths = []
        for path, content in files.items():
            if not isinstance(path, str) or not isinstance(content, str):
                continue
            if path not in allow:
                continue
            if not _is_safe_content(content):
                print("Rejected proposal: forbidden pattern in", path)
                continue
            edited_paths.append(path)

        if not edited_paths:
            print("Proposal produced no allowed edits; stopping.")
            break

        print("Proposed edits:")
        for p in edited_paths:
            print(" -", p, f"({len(str(files[p]))} chars)")

        if not do_apply:
            # Save proposed new files without touching workspace
            for p in edited_paths:
                (iter_dir / (Path(p).name + ".proposed.py")).write_text(str(files[p]), encoding="utf-8")
            print("Dry-run: saved proposals to", iter_dir)

            if bool(args.show) and bool(args.offline):
                exts = ["*.png", "*.jpg", "*.jpeg"]
                left_paths = sorted([p for ext in exts for p in Path(left_dir_used).glob(ext)])
                right_paths = sorted([p for ext in exts for p in Path(right_dir_used).glob(ext)])
                if left_paths and right_paths:
                    comp = _debug_composite_one_pair(args.calib, left_paths[0], right_paths[0], base_params, rot_left=rot_left, rot_right=rot_right)
                    if comp is not None:
                        cv2.imshow("auto_code_loop (current)", comp)
                        cv2.waitKey(1)

            if bool(args.interactive):
                try:
                    fb = input("Feedback for next iter (enter to skip, 'q' to stop): ").strip()
                except KeyboardInterrupt:
                    print("Interrupted; stopping.")
                    break
                if fb.lower() == "q":
                    break
                user_feedback = fb
            continue

        # Apply with backups
        backup_dir = iter_dir / "backups"
        for p in edited_paths:
            _write_with_backup(p, str(files[p]), backup_dir)

        ok, compile_out = _py_compile(edited_paths)
        (iter_dir / "py_compile.txt").write_text(compile_out + "\n", encoding="utf-8")
        if not ok:
            print("Syntax check failed; reverting.")
            _restore_backups(backup_dir, edited_paths)
            continue

        if bool(args.offline):
            m1 = _offline_eval(
                args.calib,
                left_dir_used,
                right_dir_used,
                base_params,
                max_pairs=int(args.max_pairs),
                rot_left=rot_left,
                rot_right=rot_right,
            )
            s1 = _offline_score(m1)
            (iter_dir / "offline_metrics_after.json").write_text(json.dumps(m1.__dict__, indent=2), encoding="utf-8")
            (iter_dir / "offline_score_after.txt").write_text(str(s1) + "\n", encoding="utf-8")

            print("Offline after:", m1, "score=", s1)

            if bool(args.show):
                exts = ["*.png", "*.jpg", "*.jpeg"]
                left_paths = sorted([p for ext in exts for p in Path(left_dir_used).glob(ext)])
                right_paths = sorted([p for ext in exts for p in Path(right_dir_used).glob(ext)])
                if left_paths and right_paths:
                    comp = _debug_composite_one_pair(args.calib, left_paths[0], right_paths[0], base_params, rot_left=rot_left, rot_right=rot_right)
                    if comp is not None:
                        cv2.imshow("auto_code_loop (current)", comp)
                        cv2.waitKey(1)

            if baseline_score is not None and s1 < float(baseline_score) - 1e-6:
                print("Offline score got worse; reverting.")
                _restore_backups(backup_dir, edited_paths)
                continue

            baseline_score = s1
            baseline_metrics = m1
            original_texts = _read_files(allow)
            print("Kept edits. New offline score:", s1)
        else:
            # No offline eval => keep edits if syntax OK
            original_texts = _read_files(allow)
            print("Kept edits (syntax OK).")

        if bool(args.interactive):
            try:
                fb = input("Feedback for next iter (enter to skip, 'q' to stop): ").strip()
            except KeyboardInterrupt:
                print("Interrupted; stopping.")
                break
            if fb.lower() == "q":
                break
            user_feedback = fb

        time.sleep(0.2)

    if bool(args.show):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
