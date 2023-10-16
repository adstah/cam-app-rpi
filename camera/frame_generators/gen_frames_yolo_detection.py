import cv2
import numpy as np
import time
from camera.helpers import encode_for_transission

with open('assets/object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet('assets/yolov4-tiny.weights', 'assets/yolov4-tiny.cfg')

def generate(camera):
    while True:
        frame = camera.capture_array()
        if frame.any():
            image = frame
            image_height, image_width, _ = image.shape
            
            blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(256, 128), swapRB=True, crop=False)
            model.setInput(blob)
            output_layers = model.getUnconnectedOutLayersNames()
            outputs = model.forward(output_layers)
            
            for output in outputs:
                for obj in output:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(obj[0] * image_width)
                        center_y = int(obj[1] * image_height)
                        w = int(obj[2] * image_width)
                        h = int(obj[3] * image_height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        color = COLORS[int(class_id)]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(image, class_names[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                yield(encode_for_transission(image))