# optimize.py
import optuna
import cv2
import numpy as np
from evaluate import evaluate

# ---- load calibration ----
calib = np.load("calib.npz")

img_l = cv2.imread("left.jpg", cv2.IMREAD_GRAYSCALE)
img_r = cv2.imread("right.jpg", cv2.IMREAD_GRAYSCALE)

def objective(trial):
    params = {
        "numDisparities": trial.suggest_int("numDisparities", 64, 256, step=16),
        "blockSize": trial.suggest_int("blockSize", 3, 11, step=2),
        "uniquenessRatio": trial.suggest_int("uniquenessRatio", 5, 20),
        "speckleWindowSize": trial.suggest_int("speckleWindowSize", 0, 200),
    }

    return evaluate(params, calib, img_l, img_r)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best score:", study.best_value)
print("Best params:", study.best_params)
