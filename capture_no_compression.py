import cv2

cv2.setLogLevel(3)

capL = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
capR = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)

for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

print("Opened L:", capL.isOpened())
print("Opened R:", capR.isOpened())

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    print("read:", retL, retR)

    if retL:
        cv2.imshow("Left", frameL)
    if retR:
        cv2.imshow("Right", frameR)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
