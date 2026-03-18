import cv2
import os

CAM_DEVICE = "/dev/video2"
ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE
SAVE_FOLDER = "data_cam2/images"

os.makedirs(SAVE_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

idx = 0

while True:
    cap.grab()  # freeze frame
    ret, frame = cap.retrieve()
    if not ret:
        continue

    frame = cv2.rotate(frame, ROTATION)
    cv2.imshow("Camera2", frame)

    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite(f"{SAVE_FOLDER}/img_{idx:04d}.jpg", frame.copy())
        print(f"Saved {idx}")
        idx += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
