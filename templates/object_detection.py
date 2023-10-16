import cv2
import numpy as np
from camera.camera import picam2
import time

# load the COCO class names
with open('assets/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

print(class_names)
  
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet(model='assets/frozen_inference_graph.pb', config='assets/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework='TensorFlow')

# detect objects in each frame of the video
while True:
    frame = picam2.capture_array()
    if frame.any():
        image = frame
        image_height, image_width, _ = image.shape
        # create blob from image
        blob = cv2.dnn.blobFromImage(image=image, size=(256, 128), mean=(104, 117, 123), swapRB=True)
        
        # start time to calculate FPS
        start = time.time()
        model.setInput(blob)
        output = model.forward()
        
        # end time after detection
        end = time.time()
        
        # calculate the FPS for current frame detection
        fps = 1 / (end - start)
        
        boxes = []
        confidences = []
        class_ids = []
        
        # loop over each of the detections
        for detection in output[0, 0, :, :]:
            # extract the confidence of the detection
            confidence = detection[2]
            
            # draw bounding boxes only if the detection confidence is above...
            # ... a certain threshold, else skip
            if confidence > .3:
                class_id = detection[1]
                box_x = int(detection[3] * image_width)
                box_y = int(detection[4] * image_height)
                box_width = int(detection[5] * image_width - box_x)
                box_height = int(detection[6] * image_height - box_y)
                
                boxes.append([box_x, box_y, box_width, box_height])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Use NMS to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.1)
        
        # loop over the indices to draw bounding boxes
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = class_names[class_ids[i]-1]
            color = COLORS[class_ids[i]]
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('image', image)
        
        # out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
  
cv2.destroyAllWindows()
