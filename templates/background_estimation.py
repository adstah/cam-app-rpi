import numpy as np
import cv2
from camera.camera import picam2

# backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN()

while True:
    frame = picam2.capture_array()

    fgMask = backSub.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cv2.destroyAllWindows()
