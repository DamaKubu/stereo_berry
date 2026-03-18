import cv2
import numpy as np
import glob
import os
import time

# ---------------- CONFIG ----------------
CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE = 0.065  # meters

DATA_LEFT = "data/intrinsicLeftCam1"
DATA_RIGHT = "data/intrinsicRightCam2"

ROT_LEFT = cv2.ROTATE_90_COUNTERCLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_CLOCKWISE

SAVE_CALIB = "calib_auto.npz"
# ---------------------------------------

def save_readable_calib(filename, mtxL, distL, mtxR, distR, R, T, E, F):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("mtxL", mtxL)
    fs.write("distL", distL)
    fs.write("mtxR", mtxR)
    fs.write("distR", distR)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.release()
    print(f"Calibration saved to {filename} (human-readable YAML)")



# 1️⃣ Load images
def load_images(folder):
    paths = sorted(glob.glob(f"{folder}/*.png"))
    return [cv2.imread(p) for p in paths]

def preprocess(imgs):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

# 2️⃣ Intrinsic calibration
def calibrate_camera(images, fisheye=False):
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []

    for img in images:
        ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

    if len(objpoints)<3:
        raise RuntimeError("Not enough corners detected!")

    h, w = images[0].shape
    if fisheye:
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4,1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, (w,h), K, D, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        # reprojection error
        total_err = 0
        for i in range(len(objpoints)):
            imgp2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            total_err += cv2.norm(imgpoints[i], imgp2, cv2.NORM_L2)/len(imgp2)
        return K, D, total_err/len(objpoints)
    else:
        camera_matrix = np.array([[w,0,w/2],[0,w,w/2],[0,0,1]], np.float64)
        dist_coeffs = np.zeros((5,1), np.float64)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w,h), camera_matrix, dist_coeffs)
        # reprojection error
        total_err = 0
        for i in range(len(objpoints)):
            imgp2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            total_err += cv2.norm(imgpoints[i], imgp2, cv2.NORM_L2)/len(imgp2)
        return mtx, dist, total_err/len(objpoints)

# 3️⃣ Stereo calibration
def stereo_calibrate(imgsL, imgsR, mtxL, distL, mtxR, distR):
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints, imgpointsL, imgpointsR = [], [], []

    for imL, imR in zip(imgsL, imgsR):
        retL, cornersL = cv2.findChessboardCorners(imL, CHESSBOARD_SIZE)
        retR, cornersR = cv2.findChessboardCorners(imR, CHESSBOARD_SIZE)
        if retL and retR:
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

    if len(objpoints)<3:
        raise RuntimeError("Not enough stereo pairs!")

    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL, mtxR, distR,
        imgsL[0].shape[::-1],
        criteria=(cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return R, T, E, F

# 4️⃣ Depth map viewer
def show_depth(mtxL, distL, mtxR, distR, R, T, camL="/dev/video2", camR="/dev/video0"):
    capL = cv2.VideoCapture(camL, cv2.CAP_V4L2)
    capR = cv2.VideoCapture(camR, cv2.CAP_V4L2)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Cannot open cameras!")

    stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=5)

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
        cv2.imshow("Left", frameL)
        cv2.imshow("Right", frameR)
        cv2.imshow("Disparity", (disp/disp.max()*255).astype(np.uint8))

        if cv2.waitKey(1)==ord("q"):
            break
    capL.release(); capR.release()
    cv2.destroyAllWindows()

# 5️⃣ Full automatic loop
def main():
    print("Loading images...")
    imgsL = preprocess(load_images(DATA_LEFT))
    imgsR = preprocess(load_images(DATA_RIGHT))

    print("Calibrating left camera...")
    mtxL, distL, errL = calibrate_camera(imgsL)
    print(f"Left reprojection error: {errL:.4f}")
    print("Calibrating right camera...")
    mtxR, distR, errR = calibrate_camera(imgsR)
    print(f"Right reprojection error: {errR:.4f}")

    print("Stereo calibration...")
    R, T, E, F = stereo_calibrate(imgsL, imgsR, mtxL, distL, mtxR, distR)

    print("Saving calibration...")
    np.savez(SAVE_CALIB, mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F)

    print("Opening depth viewer...")
    save_readable_calib("calib_auto.yml", mtxL, distL, mtxR, distR, R, T, E, F)


if __name__=="__main__":
    main()