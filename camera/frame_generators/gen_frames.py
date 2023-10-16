from camera.helpers import encode_for_transission

def generate(camera):  
	while True:
		# success, frame = camera.read()  # read the camera frame
		frame = camera.capture_array()
		if frame.any():
			yield(encode_for_transission(frame))
		else:
			break