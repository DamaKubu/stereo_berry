import cv2
import numpy as np
import glob
import optuna
import os
import time

# Optional OpenAI integration
try:
    import openai
    USE_OPENAI = True
except ImportError:
    USE_OPENAI = False

# ------------------ CONFIG ------------------
CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE = 0.065  # meters
MIN_DISTANCE = 0.5
MAX_DISTANCE = 50.0

CAM_LEFT = "/dev/video0"
CAM_RIGHT = "/dev/video2"

DATA_LEFT = "data/data_cam1/images"
DATA_RIGHT = "data/data_cam2/images"

ROT_LEFT = cv2.ROTATE_90_CLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_COUNTERCLOCKWISE

N_TRIALS = 30  # Optuna trials
# -------------------------------------------

def load_images(folder):
    images = sorted(glob.glob(f"{folder}/*.jpg"))
    return [cv2.imread(f) for f in images]

def preprocess_images(images):
    preprocessed = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        preprocessed.append(gray)
    return preprocessed

# ---------------- INTRINSIC CALIBRATION ----------------
def calibrate_pinhole(images, win=11, max_iter=30, eps=0.001, flags=0):
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []

    for img in images:
        ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img, corners, (win,win), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps))
            imgpoints.append(corners2)

    if len(objpoints) < 3:
        raise RuntimeError("Not enough corners detected for intrinsic calibration!")

    h, w = images[0].shape
    camera_matrix = np.array([[w,0,w/2],[0,w,w/2],[0,0,1]], dtype=np.float64)
    dist_coeffs = np.zeros((5,1), dtype=np.float64)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w,h), camera_matrix, dist_coeffs,
        flags=flags)

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    return mtx, dist, mean_error

def calibrate_fisheye(images, win=11, max_iter=30, eps=0.001, flags=0):
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []

    for img in images:
        ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img, corners, (win,win), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps))
            imgpoints.append(corners2)

    if len(objpoints) < 3:
        raise RuntimeError("Not enough corners detected for fisheye calibration!")

    h, w = images[0].shape
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4,1))
    flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, (w,h), K, D,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, max_iter, eps)
    )
    # approximate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    return K, D, mean_error

def auto_calibrate_intrinsic(images):
    # Heuristic: if width >> height or spread is wide, choose fisheye
    h, w = images[0].shape
    if w/h > 1.5:
        return calibrate_fisheye(images)
    else:
        return calibrate_pinhole(images)

# ------------------ OPTUNA HYPERPARAMETER TUNING ------------------
def objective(trial, images):
    win = trial.suggest_int("win", 5, 21)
    max_iter = trial.suggest_int("max_iter", 20, 50)
    eps = trial.suggest_float("eps", 0.001, 0.01)
    flags = trial.suggest_categorical("flags", [0, cv2.CALIB_RATIONAL_MODEL])
    # Can optionally switch between pinhole/fisheye with trial
    try:
        if trial.suggest_categorical("method", ["pinhole","fisheye"])=="fisheye":
            _, _, mean_error = calibrate_fisheye(images, win=win, max_iter=max_iter, eps=eps, flags=flags)
        else:
            _, _, mean_error = calibrate_pinhole(images, win=win, max_iter=max_iter, eps=eps, flags=flags)
    except:
        mean_error = 1000  # fail-safe
    return mean_error

def tune_hyperparams(images):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, images), n_trials=N_TRIALS)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

# ------------------ STEREO CALIBRATION ------------------
def stereo_calibrate(imagesL, imagesR, mtxL, distL, mtxR, distR):
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints, imgpointsL, imgpointsR = [], [], []

    for imgL, imgR in zip(imagesL, imagesR):
        retL, cornersL = cv2.findChessboardCorners(imgL, CHESSBOARD_SIZE)
        retR, cornersR = cv2.findChessboardCorners(imgR, CHESSBOARD_SIZE)
        if retL and retR:
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

    if len(objpoints) < 3:
        raise RuntimeError("Not enough stereo pairs for calibration!")

    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL,
        mtxR, distR,
        imagesL[0].shape[::-1],
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return R, T, E, F

# ------------------ LIVE DEPTH STREAM ------------------
def depth_stream(mtxL, distL, mtxR, distR, R, T):
    capL = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    while True:
        capL.grab(); capR.grab()
        retL, frameL = capL.retrieve()
        retR, frameR = capR.retrieve()
        if not (retL and retR):
            continue

        frameL = cv2.rotate(frameL, ROT_LEFT)
        frameR = cv2.rotate(frameR, ROT_RIGHT)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        disp = stereo.compute(grayL, grayR).astype(np.float32)/16.0
        disp[disp<0]=0

        baseline = np.linalg.norm(T)
        f = mtxL[0,0]
        with np.errstate(divide='ignore'):
            Z = f*baseline/(disp+1e-6)

        median_Z = np.median(Z[Z>0])
        if median_Z < MIN_DISTANCE or median_Z > MAX_DISTANCE:
            print(f"Out of range ({median_Z:.2f} m), retrying...")
            continue

        cv2.imshow("Left", frameL)
        cv2.imshow("Right", frameR)
        cv2.imshow("Disparity", (disp/disp.max()*255).astype(np.uint8))

        key = cv2.waitKey(1)
        if key==ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

# ------------------ MAIN ------------------
def main():
    print("Loading images...")
    imagesL = preprocess_images(load_images(DATA_LEFT))
    imagesR = preprocess_images(load_images(DATA_RIGHT))

    print("Tuning hyperparameters for left camera...")
    best_paramsL = tune_hyperparams(imagesL)
    print("Tuning hyperparameters for right camera...")
    best_paramsR = tune_hyperparams(imagesR)

    print("Calibrating left camera...")
    mtxL, distL, _ = auto_calibrate_intrinsic(imagesL)
    print("Calibrating right camera...")
    mtxR, distR, _ = auto_calibrate_intrinsic(imagesR)

    print("Performing stereo calibration...")
    R, T, E, F = stereo_calibrate(imagesL, imagesR, mtxL, distL, mtxR, distR)

    np.savez("calib.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F)
    print("Calibration saved to calib.npz")

    print("Opening live depth stream...")
    depth_stream(mtxL, distL, mtxR, distR, R, T)

if __name__=="__main__":
    main()
