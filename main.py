import cv2
from picamera2 import Picamera2
from flask import Flask, render_template, Response

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
while False:
	frame = picam2.capture_array()


def gen_frames():  
	while True:
		# success, frame = camera.read()  # read the camera frame
		frame = picam2.capture_array()
		if False:
			break
		else:
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

app = Flask(__name__)   

@app.route("/")         
def index():
	return "hello"

@app.route('/video-feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000)