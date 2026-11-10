"""
test_camera.py -- check the USB camera on the PYNQ-Z2 (or PC) before running the detector.
  * lists /dev/video* devices
  * grabs 30 frames, prints resolution and FPS
  * saves cam_test.jpg (open it in Jupyter's file browser)

Jupyter cell:   %run test_camera.py          other camera:  %run test_camera.py 1
"""
import glob
import sys
import time

import cv2

src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print("video devices:", glob.glob("/dev/video*") or "none found (is the camera plugged in?)")

if sys.platform.startswith("linux"):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
else:
    cap = cv2.VideoCapture(src)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"ERROR: camera {src} did not open. Try another USB port or: %run test_camera.py 1")

for _ in range(5):          # let auto-exposure settle
    cap.read()
t0, n, frame = time.time(), 0, None
while n < 30:
    ok, f = cap.read()
    if ok:
        frame, n = f, n + 1
fps = n / (time.time() - t0)
cap.release()

if frame is None:
    raise SystemExit("ERROR: camera opened but gave no frames")
cv2.imwrite("cam_test.jpg", frame)
print(f"camera {src}: {frame.shape[1]}x{frame.shape[0]}, {fps:.1f} FPS -> saved cam_test.jpg")
print("CAMERA OK")
