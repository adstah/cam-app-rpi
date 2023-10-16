import cv2
import numpy as np
from camera.camera import picam2
import time

# Load the COCO class names
with open('assets/object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')

# Get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load Tiny YOLO model
model = cv2.dnn.readNet('assets/yolov4-tiny.weights', 'assets/yolov4-tiny.cfg')

while True:
   frame = picam2.capture_array()
   if frame.any():
       image = frame
       image_height, image_width, _ = image.shape
       
       blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(256, 128), swapRB=True, crop=False)
       model.setInput(blob)
       output_layers = model.getUnconnectedOutLayersNames()
       outputs = model.forward(output_layers)
       
       # Process the outputs
       for output in outputs:
           for obj in output:
               scores = obj[5:]
               class_id = np.argmax(scores)
               confidence = scores[class_id]
               if confidence > 0.5:
                   # Object detected
                   center_x = int(obj[0] * image_width)
                   center_y = int(obj[1] * image_height)
                   w = int(obj[2] * image_width)
                   h = int(obj[3] * image_height)

                   # Rectangle coordinates
                   x = int(center_x - w / 2)
                   y = int(center_y - h / 2)

                   color = COLORS[int(class_id)]
                   cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                   cv2.putText(image, class_names[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
       
       cv2.imshow('image', image)
       if cv2.waitKey(10) & 0xFF == ord('q'):
           break
   else:
       break
  
cv2.destroyAllWindows()