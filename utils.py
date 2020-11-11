import numpy
import cv2
import glob
import matplotlib.pyplot as plt


def get_img_id(i):
	num_digits = len(str(i))
	id = "0" * (4 - num_digits)
	id += str(i)
	return id


def draw_3d_plot(xs, ys, zs, true_xs, true_ys, true_zs):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(xs, ys, zs, label='Estimated Trajectory', color='green')
	ax.plot(true_xs, true_ys, true_zs, label='Ground Truth', color='red')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
	plt.show()


def preprocess_images(filepath):
	out = []
	images = [cv2.imread(file, 0) for file in sorted(glob.glob(filepath))]
	for img in images:
		#img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
		#img = cv2.medianBlur(img, 3)
		# img = cv2.resize(img,)
		img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
		out.append(img)
	return out


def get_images_from_video(video_path):
	imgs = []
	cap = cv2.VideoCapture(video_path)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("No more frames.")
			break

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		imgs.append(frame_gray)
	return imgs