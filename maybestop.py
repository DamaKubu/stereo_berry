import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _load_dotenv_if_present() -> None:
    """Best-effort .env loader (no deps).

    NOTE: Overrides OPENAI_API_KEY/OPENAI_MODEL on purpose in other scripts,
    but this file doesn't use OpenAI. Still useful if you want to keep
    camera/config values in .env later.
    """

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    def _parse_value(raw: str) -> str:
        v = raw.strip()
        if not v:
            return v

        if v[0] in ('"', "'"):
            q = v[0]
            end = v.find(q, 1)
            if end != -1:
                v = v[1:end]
            else:
                v = v.strip(q)
        else:
            if "#" in v:
                v = v.split("#", 1)[0].strip()

        v = v.split()[0] if v.split() else ""
        return v

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
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return


_load_dotenv_if_present()


@dataclass(frozen=True)
class RectifyMaps:
    image_size: tuple[int, int]  # (w, h)
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
    mtxL: Any,
    distL: Any,
    mtxR: Any,
    distR: Any,
    R: Any,
    T: Any,
    baseline_m: float,
) -> RectifyMaps:
    T_scaled = _scale_translation_to_baseline(T, baseline_m)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL,
        distL,
        mtxR,
        distR,
        image_size,
        R,
        T_scaled,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    map_lx, map_ly = cv2.initUndistortRectifyMap(
        mtxL, distL, R1, P1, image_size, cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        mtxR, distR, R2, P2, image_size, cv2.CV_32FC1
    )

    return RectifyMaps(
        image_size=image_size,
        map_lx=map_lx,
        map_ly=map_ly,
        map_rx=map_rx,
        map_ry=map_ry,
        Q=Q,
    )


def _open_cameras(cam_left: str, cam_right: str, w: int, h: int) -> tuple[Any, Any]:
    capL = cv2.VideoCapture(cam_left, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(cam_right, cv2.CAP_V4L2)

    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not capL.isOpened() or not capR.isOpened():
        capL.release()
        capR.release()
        raise RuntimeError(
            f"Cannot open cameras. L={cam_left} opened={capL.isOpened()} R={cam_right} opened={capR.isOpened()}"
        )

    return capL, capR


def _grab_pair(capL: Any, capR: Any, rot_left: int, rot_right: int) -> tuple[Any, Any]:
    capL.grab()
    capR.grab()
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if not (retL and retR):
        raise RuntimeError("Failed to retrieve frames")

    if rot_left != -1:
        frameL = cv2.rotate(frameL, rot_left)
    if rot_right != -1:
        frameR = cv2.rotate(frameR, rot_right)
    return frameL, frameR


def _capture_rectified_pairs(
    *,
    cam_left: str,
    cam_right: str,
    capture_w: int,
    capture_h: int,
    rot_left: int,
    rot_right: int,
    maps: RectifyMaps,
    pairs: int,
    warmup: int,
) -> list[tuple[Any, Any]]:
    capL, capR = _open_cameras(cam_left, cam_right, capture_w, capture_h)
    try:
        # Warm up (let auto-exposure settle a little)
        for _ in range(max(0, warmup)):
            _ = _grab_pair(capL, capR, rot_left, rot_right)

        result: list[tuple[Any, Any]] = []
        for _ in range(pairs):
            frameL, frameR = _grab_pair(capL, capR, rot_left, rot_right)
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            rectL = cv2.remap(grayL, maps.map_lx, maps.map_ly, cv2.INTER_LINEAR)
            rectR = cv2.remap(grayR, maps.map_rx, maps.map_ry, cv2.INTER_LINEAR)
            result.append((rectL, rectR))
        return result
    finally:
        capL.release()
        capR.release()


def _sanitize_sgbm_params(p: dict[str, int], image_width: int) -> dict[str, int]:
    block_size = int(p["blockSize"])
    block_size = max(3, min(51, block_size))
    if block_size % 2 == 0:
        block_size += 1

    num_disp = int(p["numDisparities"])
    num_disp = max(16, (num_disp // 16) * 16)
    max_num = ((image_width - (block_size // 2) - 1) // 16) * 16
    max_num = max(16, max_num)
    num_disp = min(num_disp, max_num)

    uniq = int(p["uniquenessRatio"])
    uniq = max(0, min(100, uniq))

    speckle = int(p["speckleWindowSize"])
    speckle = max(0, min(500, speckle))

    return {
        "numDisparities": num_disp,
        "blockSize": block_size,
        "uniquenessRatio": uniq,
        "speckleWindowSize": speckle,
    }


def _make_sgbm(p: dict[str, int]) -> Any:
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


def _valid_depth_fraction(depth_z: Any, z_min: float, z_max: float) -> float:
    valid = np.isfinite(depth_z) & (depth_z > z_min) & (depth_z < z_max)
    return float(np.count_nonzero(valid)) / float(depth_z.size) * 100.0


def _disparity_to_depth_z(disp: Any, Q: Any) -> Any:
    disp_safe = disp.copy()
    disp_safe[disp_safe <= 0] = np.nan
    points = cv2.reprojectImageTo3D(disp_safe, Q)
    return points[:, :, 2]


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuner for StereoSGBM on rectified frames.")
    parser.add_argument("--calib", default="calib_auto.npz")
    parser.add_argument("--baseline", type=float, default=0.98)

    parser.add_argument("--cam-left", default="/dev/video2")
    parser.add_argument("--cam-right", default="/dev/video0")

    parser.add_argument("--capture-w", type=int, default=640)
    parser.add_argument("--capture-h", type=int, default=480)

    # Rotations: match your current wiring setup
    parser.add_argument("--rot-left", default="ccw", choices=["none", "cw", "ccw", "180"])
    parser.add_argument("--rot-right", default="cw", choices=["none", "cw", "ccw", "180"])

    parser.add_argument("--pairs", type=int, default=3, help="How many frame pairs to average per trial")
    parser.add_argument("--warmup", type=int, default=5)

    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--study", default="stereo_sgbm")
    parser.add_argument("--storage", default="sqlite:///optuna_study.db")

    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=50.0)

    parser.add_argument("--history", default="tuning_history.jsonl")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    try:
        import optuna  # type: ignore
    except Exception:
        raise SystemExit(
            "Optuna is not installed. Install it with: pip install optuna\n"
            "(In this repo venv: .venv/bin/python -m pip install optuna)"
        )

    rot_map = {
        "none": None,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    rot_left = rot_map[args.rot_left] if rot_map[args.rot_left] is not None else -1
    rot_right = rot_map[args.rot_right] if rot_map[args.rot_right] is not None else -1

    # After rotation: capture 640x480 becomes 480x640 (w=480, h=640)
    image_size = (args.capture_h, args.capture_w)  # (w, h)

    calib = np.load(args.calib)
    mtxL, distL = calib["mtxL"], calib["distL"]
    mtxR, distR = calib["mtxR"], calib["distR"]
    R, T = calib["R"], calib["T"]

    maps = build_rectification_maps(
        image_size=image_size,
        mtxL=mtxL,
        distL=distL,
        mtxR=mtxR,
        distR=distR,
        R=R,
        T=T,
        baseline_m=args.baseline,
    )

    print("Capturing rectified pairs...")
    pairs = _capture_rectified_pairs(
        cam_left=args.cam_left,
        cam_right=args.cam_right,
        capture_w=args.capture_w,
        capture_h=args.capture_h,
        rot_left=rot_left if rot_left != -1 else cv2.ROTATE_180,  # never used if -1
        rot_right=rot_right if rot_right != -1 else cv2.ROTATE_180,
        maps=maps,
        pairs=max(1, args.pairs),
        warmup=args.warmup,
    )

    # Re-capture with proper rotation flags (including 'none')
    pairs = _capture_rectified_pairs(
        cam_left=args.cam_left,
        cam_right=args.cam_right,
        capture_w=args.capture_w,
        capture_h=args.capture_h,
        rot_left=rot_left,
        rot_right=rot_right,
        maps=maps,
        pairs=max(1, args.pairs),
        warmup=args.warmup,
    )

    image_width = image_size[0]

    def objective(trial: Any) -> float:
        candidate = {
            "numDisparities": trial.suggest_int("numDisparities", 16, max(16, image_width - 1), step=16),
            "blockSize": trial.suggest_int("blockSize", 3, 21, step=2),
            "uniquenessRatio": trial.suggest_int("uniquenessRatio", 0, 20),
            "speckleWindowSize": trial.suggest_int("speckleWindowSize", 0, 200, step=10),
        }

        p = _sanitize_sgbm_params(candidate, image_width=image_width)
        stereo = _make_sgbm(p)

        scores: list[float] = []
        for rectL, rectR in pairs:
            disp = stereo.compute(rectL, rectR).astype(np.float32) / 16.0
            depth_z = _disparity_to_depth_z(disp, maps.Q)
            scores.append(_valid_depth_fraction(depth_z, args.z_min, args.z_max))

        score = float(np.mean(scores))

        _append_jsonl(
            args.history,
            {
                "ts": time.time(),
                "trial": int(trial.number),
                "score": score,
                "params": p,
            },
        )

        if not args.no_gui:
            disp_u8 = np.empty_like(disp, dtype=np.uint8)
            cv2.normalize(disp, disp_u8, 0, 255, cv2.NORM_MINMAX)
            cv2.imshow("optuna_disp", disp_u8)
            cv2.waitKey(1)

        return score

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    print(f"Study: {args.study}  Storage: {args.storage}")
    print(f"Trials so far: {len(study.trials)}")

    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    best_params = _sanitize_sgbm_params({k: int(v) for k, v in best.params.items()}, image_width=image_width)

    print("Best score:", best.value)
    print("Best params:", best_params)

    np.savez(
        "calib_viewer_optuna.npz",
        best_score=np.float32(best.value),
        params_json=json.dumps(best_params),
        numDisparities=np.int32(best_params["numDisparities"]),
        blockSize=np.int32(best_params["blockSize"]),
        uniquenessRatio=np.int32(best_params["uniquenessRatio"]),
        speckleWindowSize=np.int32(best_params["speckleWindowSize"]),
    )


if __name__ == "__main__":
    main()
