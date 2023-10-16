import cv2
import numpy as np
import time
from camera.helpers import encode_for_transission

with open('assets/object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
model = cv2.dnn.readNet(model='assets/frozen_inference_graph.pb', config='assets/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',framework='TensorFlow')

def generate():
    while True:
        frame = picam2.capture_array()
        if frame.any():
            image = frame
            image_height, image_width, _ = image.shape
            blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
            start = time.time()
            model.setInput(blob)
            output = model.forward()       
            end = time.time()
            fps = 1 / (end-start)
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .4:
                    class_id = detection[1]
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            yield(encode_for_transission(image))
        else:
            break
