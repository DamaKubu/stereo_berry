# evaluate.py
import cv2
import numpy as np

def evaluate(params, calib, img_l, img_r):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=params["numDisparities"],
        blockSize=params["blockSize"],
        P1=8 * 3 * params["blockSize"] ** 2,
        P2=32 * 3 * params["blockSize"] ** 2,
        uniquenessRatio=params["uniquenessRatio"],
        speckleWindowSize=params["speckleWindowSize"],
        speckleRange=2,
        disp12MaxDiff=1,
    )

    disp = stereo.compute(img_l, img_r).astype(np.float32) / 16.0

    valid = disp > 0
    if valid.sum() == 0:
        return -1e9

    # ---- quality metrics ----
    valid_ratio = valid.mean()
    noise = disp[valid].std()

    score = (
        valid_ratio * 10.0
        - noise * 0.5
    )

    return score
