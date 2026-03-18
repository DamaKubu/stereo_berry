# pyright: reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false

import cv2
import numpy as np
import os
import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

# ---------------- OPENAI ----------------

def _load_dotenv_if_present() -> None:
    """Load KEY=VALUE pairs from .env into os.environ (best-effort).

    Python does not read .env automatically; this keeps the script self-contained
    without requiring python-dotenv.
    """

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    def _parse_value(raw: str) -> str:
        v = raw.strip()
        if not v:
            return v

        # Quoted value: keep content inside the first/last quote.
        if v[0] in ('"', "'"):
            q = v[0]
            end = v.find(q, 1)
            if end != -1:
                v = v[1:end]
            else:
                v = v.strip(q)
        else:
            # Unquoted: strip inline comments and trailing junk.
            if "#" in v:
                v = v.split("#", 1)[0].strip()

        # If someone pasted extra text after the key, keep only first token.
        # (OpenAI keys never contain whitespace.)
        v = v.split()[0] if v.split() else ""
        return v

    # Keys we allow .env to override even if already exported in the shell.
    # This avoids the very common "I updated .env but it still uses old key" issue.
    override_keys = {"OPENAI_API_KEY", "OPENAI_MODEL"}

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                if "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                key = key.strip()
                value = _parse_value(raw_value)
                if not key or not value:
                    continue

                if key in override_keys:
                    os.environ[key] = value
                elif key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return

    # Final cleanup: if the env already had a broken key with whitespace, fix it.
    k = os.environ.get("OPENAI_API_KEY")
    if k and any(ch.isspace() for ch in k):
        os.environ["OPENAI_API_KEY"] = k.split()[0]


_load_dotenv_if_present()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


def _get_openai_client() -> OpenAI | None:
    # In case the working directory differs, retry loading from the script dir.
    _load_dotenv_if_present()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _safe_openai_env_debug() -> str:
    """Non-secret debug string for understanding what the script is using."""
    k = os.getenv("OPENAI_API_KEY", "")
    prefix = k[:10] if k else "(missing)"
    length = len(k)
    model = os.getenv("OPENAI_MODEL") or "(default)"
    return f"OPENAI key={prefix}… len={length} model={model}"


def _extract_first_json_object(text: str) -> str:
    """Extract the first {...} JSON object from text (handles code fences, prose)."""
    s = text.strip()
    if s.startswith("```"):
        # Drop the first fence line and the trailing fence if present.
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s

# ---------------- CONFIG ----------------
CAM_LEFT = "/dev/video2"
CAM_RIGHT = "/dev/video0"

CAPTURE_FRAME_WIDTH = 640
CAPTURE_FRAME_HEIGHT = 480

ROT_LEFT = cv2.ROTATE_90_COUNTERCLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_CLOCKWISE

# Load calibration
calib = np.load("calib_auto.npz")
mtxL, distL = calib["mtxL"], calib["distL"]
mtxR, distR = calib["mtxR"], calib["distR"]
R, T = calib["R"], calib["T"]

BASELINE = 0.98  # meters, actual distance between cameras

# Initial SGBM parameters
params = {
    "numDisparities": 64,  # multiple of 16
    "blockSize": 5,
    "uniquenessRatio": 5,
    "speckleWindowSize": 50
}


@dataclass(frozen=True)
class RectifyMaps:
    image_size: tuple[int, int]  # (w, h)
    # Keep these as Any to avoid strict numpy typing noise in scripts.
    map_lx: Any
    map_ly: Any
    map_rx: Any
    map_ry: Any
    Q: Any


def _scale_translation_to_baseline(T_vec: Any, baseline_m: float) -> Any:
    norm = float(np.linalg.norm(T_vec))
    if norm <= 1e-9:
        return T_vec
    return (T_vec / norm) * baseline_m


def build_rectification_maps(
    image_size: tuple[int, int],
    mtxL_: Any,
    distL_: Any,
    mtxR_: Any,
    distR_: Any,
    R_: Any,
    T_: Any,
    baseline_m: float,
) -> RectifyMaps:
    # Use the measured baseline for correct metric depth scaling.
    T_scaled = _scale_translation_to_baseline(T_, baseline_m)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL_,
        distL_,
        mtxR_,
        distR_,
        image_size,
        R_,
        T_scaled,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    map_lx, map_ly = cv2.initUndistortRectifyMap(
        mtxL_, distL_, R1, P1, image_size, cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        mtxR_, distR_, R2, P2, image_size, cv2.CV_32FC1
    )

    return RectifyMaps(
        image_size=image_size,
        map_lx=map_lx,
        map_ly=map_ly,
        map_rx=map_rx,
        map_ry=map_ry,
        Q=Q,
    )


def sanitize_sgbm_params(candidate: dict[str, Any], image_width: int) -> dict[str, int]:
    """Clamp and normalize StereoSGBM params to avoid OpenCV runtime errors."""

    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    normalized = dict(params)
    normalized.update(candidate)

    block_size = _to_int(normalized.get("blockSize"), 5)
    block_size = max(3, block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = min(51, block_size)

    num_disparities = _to_int(normalized.get("numDisparities"), 64)
    num_disparities = max(16, num_disparities)
    num_disparities = (num_disparities // 16) * 16

    # OpenCV constraint (see error message):
    # width - (minDisparity + numDisparities) > blockSize/2
    max_num = ((image_width - (block_size // 2) - 1) // 16) * 16
    max_num = max(16, max_num)
    num_disparities = min(num_disparities, max_num)

    uniqueness = _to_int(normalized.get("uniquenessRatio"), 5)
    uniqueness = max(0, min(100, uniqueness))

    speckle = _to_int(normalized.get("speckleWindowSize"), 50)
    speckle = max(0, min(500, speckle))

    return {
        "numDisparities": num_disparities,
        "blockSize": block_size,
        "uniquenessRatio": uniqueness,
        "speckleWindowSize": speckle,
    }


def _open_cameras() -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    capL = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)

    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not capL.isOpened() or not capR.isOpened():
        capL.release()
        capR.release()
        raise RuntimeError(
            f"Cannot open cameras. L={CAM_LEFT} opened={capL.isOpened()} R={CAM_RIGHT} opened={capR.isOpened()}"
        )

    return capL, capR


def _grab_pair(capL: cv2.VideoCapture, capR: cv2.VideoCapture) -> tuple[Any, Any]:
    capL.grab()
    capR.grab()
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if not (retL and retR):
        raise RuntimeError("Failed to retrieve frames")

    frameL = cv2.rotate(frameL, ROT_LEFT)
    frameR = cv2.rotate(frameR, ROT_RIGHT)
    return frameL, frameR


def _make_sgbm(p: dict[str, int]) -> cv2.StereoSGBM:
    block_size = int(p["blockSize"])
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=int(p["numDisparities"]),
        blockSize=block_size,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=int(p["uniquenessRatio"]),
        speckleWindowSize=int(p["speckleWindowSize"]),
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def run_capture(
    p: dict[str, int],
    maps: RectifyMaps,
    show_debug: bool = False,
) -> tuple[Any, Any, Any]:
    """Capture a frame pair, rectify, compute disparity, and return (disp, rectL, rectR)."""
    capL, capR = _open_cameras()
    try:
        frameL, frameR = _grab_pair(capL, capR)
    finally:
        capL.release()
        capR.release()

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    rectL = cv2.remap(grayL, maps.map_lx, maps.map_ly, cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, maps.map_rx, maps.map_ry, cv2.INTER_LINEAR)

    stereo = _make_sgbm(p)
    disp = stereo.compute(rectL, rectR).astype(np.float32) / 16.0

    if show_debug:
        # Visual check: rectified images should align horizontally.
        visL = cv2.cvtColor(rectL, cv2.COLOR_GRAY2BGR)
        visR = cv2.cvtColor(rectR, cv2.COLOR_GRAY2BGR)
        for y in range(0, visL.shape[0], 40):
            cv2.line(visL, (0, y), (visL.shape[1] - 1, y), (0, 255, 0), 1)
            cv2.line(visR, (0, y), (visR.shape[1] - 1, y), (0, 255, 0), 1)
        cv2.imshow("rectL", visL)
        cv2.imshow("rectR", visR)
        disp_u8 = np.empty_like(disp, dtype=np.uint8)
        cv2.normalize(disp, disp_u8, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("disp", disp_u8)
        cv2.waitKey(1)

    return disp, rectL, rectR

def compute_depth(disparity: Any, f: float, baseline: float) -> Any:
    """Convert disparity to depth in meters."""
    return f * baseline / disparity

def evaluate_depth(depth: Any) -> float:
    """Return fraction of pixels with reasonable depth (0.5m–50m)."""
    valid = (depth > 0.5) & (depth < 50)
    valid_frac = np.count_nonzero(valid) / depth.size * 100
    print(f"Valid depth fraction: {valid_frac:.1f}%")
    return valid_frac

def ask_llm_for_params(params: dict[str, int], valid_frac: float) -> dict[str, int]:
    """Ask OpenAI to suggest new stereo parameters."""
    # After rotation, a 640x480 capture becomes 480 pixels wide.
    image_width = CAPTURE_FRAME_HEIGHT
    prompt = f"""
Current stereo SGBM parameters: {params}
Valid depth fraction: {valid_frac:.1f}%
Suggest new parameters (numDisparities, blockSize, uniquenessRatio, speckleWindowSize)
to improve depth quality.

Constraints:
- numDisparities must be a multiple of 16 and <= {image_width - 1}
- blockSize must be an odd integer >= 3

Return ONLY JSON.
"""
    client = _get_openai_client()
    if client is None:
        print("OPENAI_API_KEY not set; skipping LLM tuning.")
        return params

    print("LLM:", _safe_openai_env_debug())

    # Ask for strict JSON (no eval) to keep it safe.
    prompt = (
        prompt
        + "\nOutput ONLY JSON with keys: numDisparities, blockSize, uniquenessRatio, speckleWindowSize."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        suggested = (resp.choices[0].message.content or "").strip()
        if not suggested:
            raise ValueError("LLM returned empty content")

        json_text = _extract_first_json_object(suggested)
        new_params_raw: Any = json.loads(json_text)
        if not isinstance(new_params_raw, dict):
            raise ValueError("LLM did not return a JSON object")
        new_params = sanitize_sgbm_params(new_params_raw, image_width=image_width)
        print("LLM suggested new params (sanitized):", new_params)
        return new_params
    except Exception as e:
        print("LLM failed, using previous params:", e)
        return params

# ---------------- ITERATIVE TUNING ----------------
def auto_tune(max_iters: int = 10, threshold: float = 50) -> None:
    global params

    # After rotation: input 640x480 -> 480x640 (w=480).
    image_size = (CAPTURE_FRAME_HEIGHT, CAPTURE_FRAME_WIDTH)  # (w, h)
    image_width = image_size[0]
    maps = build_rectification_maps(
        image_size=image_size,
        mtxL_=mtxL,
        distL_=distL,
        mtxR_=mtxR,
        distR_=distR,
        R_=R,
        T_=T,
        baseline_m=BASELINE,
    )

    # Depth via Q is the most stable way; Z is meters after scaling T above.
    def disparity_to_depth_z(disp: Any) -> Any:
        disp_safe = disp.copy()
        disp_safe[disp_safe <= 0] = np.nan
        points = cv2.reprojectImageTo3D(disp_safe, maps.Q)
        return points[:, :, 2]

    for i in range(max_iters):
        print(f"\n--- Iteration {i} ---")
        params = sanitize_sgbm_params(params, image_width=image_width)
        disparity, _, _ = run_capture(params, maps=maps, show_debug=True)
        depth_z = disparity_to_depth_z(disparity)
        valid_frac = evaluate_depth(depth_z)

        if valid_frac >= threshold:
            print("Depth map looks good! Stopping iterations.")
            break
        else:
            print("Depth poor -> asking LLM for new parameters...")
            params = ask_llm_for_params(params, valid_frac)

    # Save final calibration + params
    np.savez(
        "calib_viewer.npz",
        params_json=json.dumps(params),
        numDisparities=np.int32(params["numDisparities"]),
        blockSize=np.int32(params["blockSize"]),
        uniquenessRatio=np.int32(params["uniquenessRatio"]),
        speckleWindowSize=np.int32(params["speckleWindowSize"]),
    )
    print("Final stereo params:", params)

# ---------------- RUN ----------------
if __name__ == "__main__":
    auto_tune()
