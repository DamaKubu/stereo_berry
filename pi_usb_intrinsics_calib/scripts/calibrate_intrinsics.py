from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Any, cast


def _parse_board(s: str) -> tuple[int, int]:
    if "x" in s:
        a, b = s.lower().split("x", 1)
        return int(a), int(b)
    if "," in s:
        a, b = s.split(",", 1)
        return int(a), int(b)
    raise ValueError("Board must be like 8x5")


def find_corners(gray: Any, board_wh: tuple[int, int]) -> tuple[bool, Any]:
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


@dataclass(frozen=True)
class CalibResult:
    rms: float
    K: NDArray[np.float64]
    dist: NDArray[np.float64]
    image_size: tuple[int, int]


def reprojection_errors(
    objpoints: list[NDArray[np.float32]],
    imgpoints: list[Any],
    rvecs: list[Any],
    tvecs: list[Any],
    K: NDArray[np.float64],
    dist: NDArray[np.float64],
) -> tuple[list[float], float]:
    per_view: list[float] = []
    total_err2 = 0.0
    total_n = 0

    for obj, img, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        err = cv2.norm(img, proj, cv2.NORM_L2)
        n = len(obj)
        rmse = float(np.sqrt((err * err) / max(n, 1)))
        per_view.append(rmse)
        total_err2 += float(err * err)
        total_n += int(n)

    total_rmse = float(np.sqrt(total_err2 / max(total_n, 1)))
    return per_view, total_rmse


def write_yaml(path: Path, K: NDArray[np.float64], dist: NDArray[np.float64], image_size: tuple[int, int]) -> None:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.write("image_width", int(image_size[0]))
        fs.write("image_height", int(image_size[1]))
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", dist)
    finally:
        fs.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="output/captures")
    ap.add_argument("--board", type=str, default="8x5")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = use all")
    ap.add_argument("--out", type=str, default="output")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    board = _parse_board(args.board)
    square = float(args.square_mm) / 1000.0  # meters

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(in_dir.glob("*.png"))
    if args.max_frames and args.max_frames > 0:
        imgs = imgs[: int(args.max_frames)]

    if not imgs:
        raise SystemExit(f"No PNG images found in {in_dir}")

    # Object points grid
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board[0], 0 : board[1]].T.reshape(-1, 2)
    objp *= square

    objpoints: list[NDArray[np.float32]] = []
    imgpoints: list[Any] = []
    image_size: tuple[int, int] | None = None

    used = 0
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if image_size is None:
            image_size = (int(img.shape[1]), int(img.shape[0]))

        ok, corners = find_corners(img, board)
        if not ok or corners is None:
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners.astype(np.float32))
        used += 1

        if args.show:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cast(Any, cv2).drawChessboardCorners(vis, board, corners, True)
            cv2.imshow("corners", vis)
            if (cv2.waitKey(50) & 0xFF) == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    if image_size is None or used < 10:
        raise SystemExit(f"Not enough valid frames. Found corners in {used} images (need >= 10).")

    # Calibration
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    rms, K0, dist0, rvecs, tvecs = cast(Any, cv2).calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags, criteria=criteria
    )

    K = np.asarray(K0, dtype=np.float64)
    dist = np.asarray(dist0, dtype=np.float64)

    per_view, total_rmse = reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    print(f"Used frames: {used}/{len(imgs)}")
    print(f"OpenCV RMS: {float(rms):.4f} px")
    print(f"Reprojection RMSE (all points): {float(total_rmse):.4f} px")
    print(f"Per-view RMSE: mean={np.mean(per_view):.4f} px  p95={np.percentile(per_view, 95):.4f} px")

    yaml_path = out_dir / "intrinsics.yaml"
    npz_path = out_dir / "intrinsics.npz"
    write_yaml(yaml_path, K, dist, image_size)
    np.savez(npz_path, K=K, dist=dist, image_size=np.array(image_size, dtype=np.int32), per_view=np.array(per_view))

    print(f"Wrote: {yaml_path}")
    print(f"Wrote: {npz_path}")

    if total_rmse > 1.0:
        print("WARNING: RMSE > 1px. Capture more diverse views + better lighting.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
