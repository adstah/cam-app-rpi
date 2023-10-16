import numpy as np
import cv2
from camera.camera import picam2
# from skimage import data, filters
 
# Open Video
# cap = cv2.VideoCapture('video.mp4')

frames = []
for i in range(25):
   frames.append(picam2.capture_image())
 
# Calculate the median along the time axis
# medianFrame = np.median(frames, axis=0) 
 
# Display median frame
medianFrame = np.array(frames[12])
# cv2.imshow('frame', np.array(medianFrame))
# cv2.waitKey(0)

# # Reset frame number to 0
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
# # Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

while(True):
  # Read frame
  frame = np.array(picam2.capture_image())
  # Convert current frame to grayscale
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Calculate absolute difference of current frame and 
  # the median frame
  dframe = cv2.absdiff(frame, grayMedianFrame)
  # Treshold to binarize
  th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
  # Display image
  cv2.imshow('frame', dframe)
  cv2.waitKey(20)
