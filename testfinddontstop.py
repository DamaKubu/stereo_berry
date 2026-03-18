import cv2
import numpy as np

# Cameras
CAM_LEFT = "/dev/video2"
CAM_RIGHT = "/dev/video0"

# Rotation
ROT_LEFT = cv2.ROTATE_90_COUNTERCLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_CLOCKWISE

# Load calibration
calib = np.load("calib_auto.npz")
mtxL, distL = calib["mtxL"], calib["distL"]
mtxR, distR = calib["mtxR"], calib["distR"]
R, T = calib["R"], calib["T"]

BASELINE = 0.98  # meters

# TUNED SGBM PARAMETERS
params = {'numDisparities': 320, 'blockSize': 7, 'uniquenessRatio': 10, 'speckleWindowSize': 100}

# Compute average focal length
f = (mtxL[0,0] + mtxL[1,1]) / 2

# Stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=params['numDisparities'],
    blockSize=params['blockSize'],
    uniquenessRatio=params['uniquenessRatio'],
    speckleWindowSize=params['speckleWindowSize']
)

capL = cv2.VideoCapture(CAM_LEFT, cv2.CAP_V4L2)
capR = cv2.VideoCapture(CAM_RIGHT, cv2.CAP_V4L2)

if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("Cannot open cameras!")

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
    disp[disp <= 0] = 1e-6

    # Real depth in meters
    depth = f * BASELINE / disp

    # Normalize depth for visualization
    vis = (np.clip(depth, 0.5, 10) - 0.5) / 9.5 * 255
    vis = vis.astype(np.uint8)

    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)
    cv2.imshow("Depth (m)", vis)

    if cv2.waitKey(1) == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
