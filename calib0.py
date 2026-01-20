
import cv2
import numpy as np
import glob

# -------- SETTINGS --------
CHESSBOARD_SIZE = (8, 5)      # inner corners
SQUARE_SIZE = 0.065           # meters (measure yours)
IMAGE_SIZE = None             # auto-detected

# -------- PREP OBJECT POINTS --------
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                       0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []

left_images = sorted(glob.glob("data/left/*.jpg"))
right_images = sorted(glob.glob("data/right/*.jpg"))

# -------- FIND CHESSBOARD CORNERS --------
for l_img, r_img in zip(left_images, right_images):
    img_l = cv2.imread(l_img)
    img_r = cv2.imread(r_img)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if IMAGE_SIZE is None:
        IMAGE_SIZE = gray_l.shape[::-1]

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

# -------- MONO CALIBRATION (handles barrel distortion) --------
_, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, IMAGE_SIZE, None, None)

_, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, IMAGE_SIZE, None, None)

# -------- STEREO CALIBRATION --------
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

_, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l,
    imgpoints_r,
    mtx_l,
    dist_l,
    mtx_r,
    dist_r,
    IMAGE_SIZE,
    criteria=criteria,
    flags=flags
)

# -------- RECTIFICATION --------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, IMAGE_SIZE, R, T)

map_l1, map_l2 = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, IMAGE_SIZE, cv2.CV_16SC2)

map_r1, map_r2 = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, IMAGE_SIZE, cv2.CV_16SC2)

np.savez("stereo_calib.npz",
         mtx_l=mtx_l, dist_l=dist_l,
         mtx_r=mtx_r, dist_r=dist_r,
         R=R, T=T, Q=Q,
         map_l1=map_l1, map_l2=map_l2,
         map_r1=map_r1, map_r2=map_r2)

print("Calibration complete.")
