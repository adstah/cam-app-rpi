import cv2
import numpy as np
from camera.helpers import encode_for_transission

def generate(camera):
    # backSub = cv2.createBackgroundSubtractorMOG2()
    backSub = cv2.createBackgroundSubtractorKNN()

    while True:
        frame = camera.capture_array()
        fgMask = backSub.apply(frame)
        yield(encode_for_transission(fgMask))

    