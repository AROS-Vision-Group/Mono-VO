import os
import cv2
import numpy as np
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera
from utils import preprocess_images, plot_3d_traj, plot_inlier_ratio, euclidean_distance, plot_drift, plot_rotation_erros, plot_orientation_angle
import utils
import math

# For UW simulation dataset
W = 1920
H = 1080

cam = PinholeCamera(width=float(W), height=float(H), fx=1263.1578, fy=1125, cx=960, cy=540)
vo = VisualOdometry(cam, annotations='./data/transformed_ground_truth_vol2.txt')
num_frames = len(os.listdir('data/images_v1/'))
orig_images = preprocess_images('data/images_v1/*.jpg', default=True)
images = preprocess_images('data/images_v1/*.jpg')

N = len(images)

traj = np.zeros((480, 640, 3), dtype=np.uint8)
x_orig, y_orig = 290, 200

drift = []
inlier_ratios = []
rotation_errors = []
xs, ys, zs = [], [], []
true_xs, true_ys, true_zs = [], [], []
theta_xs, theta_ys, theta_zs = [], [], []
theta_xs_true, theta_ys_true, theta_zs_true = [], [], []

display_img = np.zeros((1080, 1920, 3))

# for i, (orig_img, img) in enumerate(images):
for i, img in enumerate(images):
	vo.update(img, i)
	cur_t = vo.cur_t
	orig_img = orig_images[i]

	# Wait til 3rd image
	if i > 1:
		x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]
		inlier_ratios.append(vo.inlier_ratio)
	else:
		x, y, z = 0., 0., 0.
		vo.cur_t = np.zeros((3, 1))

	# Y-axis and Z-axis of ground truth has opposite directions of the VO estimation
	true_transf_x, true_transf_y, true_transf_z = vo.trueX, vo.trueY, vo.trueZ

	# Key point visualization
	if i > 2:
		for j, (new, old) in enumerate(zip(vo.px_cur, vo.px_ref)):
			a, b = new.ravel()
			c, d = old.ravel()
			color = [255, 100, 0]
			orig_img = cv2.circle(orig_img, (a, b), 2, color=(255, 100, 0), thickness=2, lineType=cv2.LINE_AA)

	# Calculate translation error (drift)
	d = euclidean_distance(np.array([x, y, z]), np.array([true_transf_x, true_transf_y, true_transf_z]))
	drift.append(d)

	# Calculate rotation error
	if i > 1:
		R1 = vo.trueR
		R2 = vo.cur_R

		R2_inv = np.transpose(R2)
		mul_R = R1 @ R2_inv

		theta_x = np.arctan2(R1[2, 1], R1[2, 2])
		theta_y = np.arctan2(-R1[2, 0], np.sqrt(R1[2, 1] ** 2 + R1[2, 2] ** 2))
		theta_z = np.arctan2(R1[1, 0], R1[0, 0])

		theta_x_true = np.arctan2(R2[2, 1], R2[2, 2])
		theta_y_true = np.arctan2(-R2[2, 0], np.sqrt(R2[2, 1] ** 2 + R2[2, 2] ** 2))
		theta_z_true = np.arctan2(R2[1, 0], R2[0, 0])

		theta_xs.append(theta_x)
		theta_ys.append(theta_y)
		theta_zs.append(theta_z)

		theta_xs_true.append(theta_x_true)
		theta_ys_true.append(theta_y_true)
		theta_zs_true.append(theta_z_true)

		theta_sum = np.abs(theta_x) + np.abs(theta_y) + np.abs(theta_z)
		rotation_errors.append(theta_sum)

	# 2D trajectory
	k = 30
	draw_x, draw_y = int(x * k) + x_orig, int(z * k) + y_orig
	true_x, true_y = int(true_transf_x * k) + x_orig, int(true_transf_z * k) + y_orig

	cv2.circle(traj, (true_x, true_y), 1, (0, i*(255), 0), 1, cv2.LINE_AA)
	cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, i*(255)), 1, cv2.LINE_AA)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)

	text = f"Estimated:    x={x:.3f} y={y:.3f} z={z:.3f}"
	cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	text_true = f"GroundTruth: x={true_transf_x:.3f} y={true_transf_y:.3f} z={true_transf_z:.3f}"
	cv2.putText(traj, text_true, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	cv2.namedWindow('Snake Robot Camera', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Snake Robot Camera', W, H//2)

	hstack = np.hstack((orig_img, img))
	cv2.imshow('Snake Robot Camera', hstack)

	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

	xs.append(x)
	ys.append(y)
	zs.append(z)

	true_xs.append(true_transf_x)
	true_ys.append(true_transf_y)
	true_zs.append(true_transf_z)


cv2.imwrite('plots/map.png', traj)
plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs)
plot_inlier_ratio(inlier_ratios)
plot_drift(drift)
plot_rotation_erros(rotation_errors)

plot_orientation_angle(theta_xs_true, theta_xs, 'x_angle')
plot_orientation_angle(theta_ys_true, theta_ys, 'y_angle')
plot_orientation_angle(theta_zs_true, theta_zs, 'z_angle')

print(f'-- Evaluation')
print(f'Total translation error: {np.sum(drift):.3f}')
print(f'Total rotation error: {np.sum(rotation_errors):.3f}')
print(f'Average RANSAC inlier ratio: {np.mean(inlier_ratios):.3f}')