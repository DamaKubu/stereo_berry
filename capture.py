import cv2
import os

# Folders
os.makedirs("data/intrinsicLeftCam1", exist_ok=True)
os.makedirs("data/intrinsicRightCam2", exist_ok=True)

# Open cameras
capL = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)  # camera 1
capR = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)  # camera 2

# Force YUYV + resolution
for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

idx = 0

while True:
        # GRAB frames first (lock in buffer)
    capL.grab()
    capR.grab()

    # Retrieve (decode) frames at the same moment
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()

    if not (retL and retR):
        continue


    if not (retL and retR):
        continue


    # Rotate
    frameR = cv2.rotate(frameR, cv2.ROTATE_90_CLOCKWISE)
    frameL = cv2.rotate(frameL, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Show
    cv2.imshow("Left (Camera1)", frameL)
    cv2.imshow("Right (Camera2)", frameR)

    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite(f"data/intrinsicLeftCam1/img_{idx:04d}.png", frameL)
        cv2.imwrite(f"data/intrinsicRightCam2/img_{idx:04d}.png", frameR)
        print("Saved pair", idx)
        idx += 1
    elif key == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
