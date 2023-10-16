import cv2
from flask import Flask, render_template, Response, request
from camera.camera import camera
from constant.camera_types import CameraType

app = Flask(__name__)   

@app.route("/")         
def index():
	return "hello"

@app.route('/video-feed')
def video_feed():
	mimetype='multipart/x-mixed-replace; boundary=frame'
	query_cam_type = request.args.__getitem__('cam-type')
	print(query_cam_type, CameraType.MOVEMENT.value, CameraType.DETECTION.value)
	if query_cam_type == CameraType.BASIC.value:
		return Response(camera.gen_frames(), mimetype=mimetype)
	if query_cam_type == CameraType.DETECTION.value:
		return Response(camera.gen_frames_yolo_detection(), mimetype=mimetype)
	if query_cam_type == CameraType.MOVEMENT.value:
		return Response(camera.gen_frames_move_detection(), mimetype=mimetype)
	else:
		return "inproper cam type"

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000)
