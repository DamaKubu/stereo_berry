import cv2
import numpy as np

# --- Calibration (from your calib_auto.npz / YAML) ---
npz = np.load("calib_auto.npz", allow_pickle=True)
mtxL, distL = npz["mtxL"], npz["distL"]
mtxR, distR = npz["mtxR"], npz["distR"]
R, T = npz["R"], npz["T"]

# --- Rectification setup ---
w, h = 640, 480  # your capture size
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY
)
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)

# --- Stereo matcher ---
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # must be divisible by 16
    blockSize=7,         # odd 3..11
    P1=8*3*7**2,
    P2=32*3*7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2
)

# --- Open cameras ---
capL = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
capR = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)

ROT_LEFT = cv2.ROTATE_90_COUNTERCLOCKWISE
ROT_RIGHT = cv2.ROTATE_90_CLOCKWISE

while True:
    capL.grab(); capR.grab()
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()
    if not (retL and retR):
        continue

    # Rotate
    frameL = cv2.rotate(frameL, ROT_LEFT)
    frameR = cv2.rotate(frameR, ROT_RIGHT)

    # Convert to gray
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # Rectify
    rectL = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Compute disparity
    disp = stereo.compute(rectL, rectR).astype(np.float32) / 16.0
    disp[disp < 0] = 0

    # Depth in meters
    depth = cv2.reprojectImageTo3D(disp, Q)[:, :, 2]

    # Normalize disparity and depth for display
    disp_vis = (disp / (disp.max()+1e-6) * 255).astype(np.uint8)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Show
    cv2.imshow("Left", frameL)
    cv2.imshow("Right", frameR)
    cv2.imshow("Disparity", disp_vis)
    cv2.imshow("Depth (m)", depth_color)

    if cv2.waitKey(1) == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
