from picamera2 import Picamera2
from camera.frame_generators import gen_frames, gen_frames_mobile_detection, gen_frames_move_detection, gen_frames_yolo_detection

class Camera(Picamera2):
    def __init__(self):
        super().__init__()
        self.preview_configuration.main.size = (1024,640)
        self.preview_configuration.main.format = "RGB888"
        self.preview_configuration.align()
        self.configure("preview")
        self.start()

    def gen_frames(self):
        return gen_frames.generate(self)

    def gen_frames_move_detection(self):
        return gen_frames_move_detection.generate(self)

    def gen_frames_yolo_detection(self):
        return gen_frames_yolo_detection.generate(self)


camera = Camera()