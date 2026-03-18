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

CHESSBOARD_SIZE = (8, 5)
SQUARE_SIZE = 0.065  # meters

DATA_LEFT = "data/data_cam1/images"
DATA_RIGHT = "data/data_cam2/images"

ROT_LEFT = cv2.ROTATE_90_CLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_COUNTERCLOCKWISE

CAM_LEFT = "/dev/video0"
CAM_RIGHT = "/dev/video2"

# ---------------- INTRINSIC CALIBRATION ----------------
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

def safe_find_corners(img):
    ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE)
    if ret:
        corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
        return True, corners2
    return False, None

def calibrate_pinhole(images):
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []
    for img in images:
        found, corners = safe_find_corners(img)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(objpoints) < 3:
        raise RuntimeError("Not enough corners detected!")

    h, w = images[0].shape
    mtx_init = np.array([[w,0,w/2],[0,w,w/2],[0,0,1]], dtype=np.float64)
    dist_init = np.zeros((5,1), dtype=np.float64)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w,h), mtx_init, dist_init)
    return mtx, dist

# ------------------ STEREO CALIBRATION ------------------
def stereo_calibrate(imagesL, imagesR, mtxL, distL, mtxR, distR):
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints, imgpointsL, imgpointsR = [], [], []

    for imgL, imgR in zip(imagesL, imagesR):
        retL, cornersL = safe_find_corners(imgL)
        retR, cornersR = safe_find_corners(imgR)
        if retL and retR:
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

    if len(objpoints) < 3:
        raise RuntimeError("Not enough stereo pairs!")

    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        mtxL, distL,
        mtxR, distR,
        imagesL[0].shape[::-1],
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return R, T, E, F

# ------------------ OPENAI AUTOMATION ------------------
def llm_generate_calibration_code(prompt):
    if not USE_OPENAI:
        print("OpenAI not available, skipping LLM function generation.")
        return
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return resp.choices[0].message.content

# ------------------ LIVE DEPTH STREAM ------------------
def depth_stream(mtxL, distL, mtxR, distR, R, T):
    capL = cv2.VideoCapture(CAM_LEFT)
    capR = cv2.VideoCapture(CAM_RIGHT)
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR,
        (640,480), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (640,480), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (640,480), cv2.CV_32FC1)

    while True:
        capL.grab(); capR.grab()
        retL, frameL = capL.retrieve()
        retR, frameR = capR.retrieve()
        if not (retL and retR):
            continue

        frameL = cv2.rotate(frameL, ROT_LEFT)
        frameR = cv2.rotate(frameR, ROT_RIGHT)
        grayL = cv2.cvtColor(cv2.remap(frameL,map1x,map1y,cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(cv2.remap(frameR,map2x,map2y,cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)

        disp = stereo.compute(grayL, grayR).astype(np.float32)/16.0
        cv2.imshow("Disparity", cv2.normalize(disp,None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
        if cv2.waitKey(1)==ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

# ------------------ MAIN ------------------
def main():
    imagesL = preprocess_images(load_images(DATA_LEFT))
    imagesR = preprocess_images(load_images(DATA_RIGHT))

    print("Calibrating left camera...")
    mtxL, distL = calibrate_pinhole(imagesL)
    print("Calibrating right camera...")
    mtxR, distR = calibrate_pinhole(imagesR)

    print("Performing stereo calibration...")
    R, T, E, F = stereo_calibrate(imagesL, imagesR, mtxL, distL, mtxR, distR)
    np.savez("calib.npz", mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F)
    print("Calibration saved to calib.npz")

    print("Opening live depth stream...")
    depth_stream(mtxL, distL, mtxR, distR, R, T)

    # Optional: ask LLM to generate additional helper code
    if USE_OPENAI:
        prompt = "Generate Python function to optimize stereo calibration parameters automatically."
        code = llm_generate_calibration_code(prompt)
        print("\n--- LLM GENERATED FUNCTION ---\n", code)

if __name__=="__main__":
    main()
