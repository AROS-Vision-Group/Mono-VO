import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def get_img_id(i):
	num_digits = len(str(i))
	id = "0" * (4 - num_digits)
	id += str(i)
	return id


def plot_inlier_ratio(ratios):
	plt.plot([i for i in range(len(ratios))], ratios, color='blue')
	plt.title('RANSAC inlier ratio across frames')
	plt.xlabel('# of frames')
	plt.ylabel('Inlier Ratio')
	plt.show()


def plot_drift(drift):
	plt.plot([i for i in range(len(drift))], drift, color='blue')
	plt.title('Drift (l2 distance) between estimated and GT')
	plt.xlabel('# of frames')
	plt.ylabel('Drift')
	plt.show()


def plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(xs, ys, zs, label='Estimated Trajectory', color='green')
	ax.plot(true_xs, true_ys, true_zs, label='Ground Truth', color='red')
	ax.set_title('3D trajectory')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
	plt.show()


def preprocess_images(filepath):
	out = []
	images = [cv2.imread(file, 0) for file in sorted(glob.glob(filepath))]
	for img in images:
		processed_img = img.copy()
		#img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
		# processed_img = cv2.medianBlur(processed_img, 5)
		processed_img = cv2.GaussianBlur(processed_img, (7, 7), 0)
		processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
		out.append(processed_img)
	return out


def preprocess_images2(filepath):
	images = [cv2.imread(file, 0) for file in sorted(glob.glob(filepath))]
	return images


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


def euclidean_distance(a, b):
	return np.linalg.norm(a - b)