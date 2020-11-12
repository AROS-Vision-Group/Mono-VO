import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def get_img_id(i):
	num_digits = len(str(i))
	id = "0" * (4 - num_digits)
	id += str(i)
	return id


def plot_inlier_ratio(ratios, save=True):
	plt.plot([i for i in range(len(ratios))], ratios, color='blue')
	plt.title('RANSAC inlier ratio across frames')
	plt.xlabel('# of frames')
	plt.ylabel('Inlier Ratio')
	if save:
		plt.savefig('plots/inlier_ratio.png', bbox_inches='tight')
	plt.show()


def plot_drift(drift, save=True):
	plt.plot([i for i in range(len(drift))], drift, color='blue')
	plt.title('Drift (L2 distance) between estimated and GT')
	plt.xlabel('# of frames')
	plt.ylabel('Drift')
	if save:
		plt.savefig('plots/drift.png', bbox_inches='tight')
	plt.show()


def plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs, save=True):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(xs, ys, zs, label='Estimated Trajectory', color='red')
	ax.plot(true_xs, true_ys, true_zs, label='Ground Truth', color='green')
	ax.set_title('3D trajectory')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
	if save:
		plt.savefig('plots/3d_traj.png', bbox_inches='tight')
	plt.show()


def preprocess_images(filepath, default=False, morphology=False):
	out = []
	images = [cv2.imread(file, 0) for file in sorted(glob.glob(filepath))]
	if default:
		return images
	for img in images:
		processed_img = img.copy()
		cv2.normalize(img.astype('float'), img, 0.0, 1.0, cv2.NORM_MINMAX)
		processed_img = cv2.GaussianBlur(processed_img, (7, 7), 0)
		processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
		if morphology:
			kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=(3, 3))
			# img_erosion = cv2.erode(img, kernel, iterations=1)
			processed_img = cv2.dilate(processed_img, kernel, iterations=3)
			processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel, iterations=1)
			# processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel, iterations=1)
		out.append(processed_img)
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


def euclidean_distance(a, b):
	return np.linalg.norm(a - b)