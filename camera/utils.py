from camera.camera import picam2
import cv2
import numpy as np
import time
from camera.helpers import encode_for_transission

def gen_frames():  
	while True:
		# success, frame = camera.read()  # read the camera frame
		frame = picam2.capture_array()
		if False:
			break
		else:
			yield(encode_for_transission(frame))

def gen_frames_background_estimation():
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
        # the median fram
        dframe = cv2.absdiff(frame, grayMedianFrame)
        # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
        # Display image
        yield(encode_for_transission(dframe))
			
# load the COCO class names
with open('assets/object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
model = cv2.dnn.readNet(model='assets/frozen_inference_graph.pb', config='assets/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',framework='TensorFlow')

# detect objects in each frame of the video
def gen_frames_with_detection():
    while True:
        frame = picam2.capture_array()
        if frame.any():
            image = frame
            image_height, image_width, _ = image.shape
            # create blob from image
            blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
            # start time to calculate FPS
            start = time.time()
            model.setInput(blob)
            output = model.forward()       
            # end time after detection
            end = time.time()
            # calculate the FPS for current frame detection
            fps = 1 / (end-start)
            # loop over each of the detections
            for detection in output[0, 0, :, :]:
                # extract the confidence of the detection
                confidence = detection[2]
                # draw bounding boxes only if the detection confidence is above...
                # ... a certain threshold, else skip
                if confidence > .4:
                    # get the class id
                    class_id = detection[1]
                    # map the class id to the class
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    # put the class name text on the detected object
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # put the FPS text on top of the frame
                    cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            yield(encode_for_transission(image))
        else:
            break
