import os
import cv2
import numpy as np
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera
from utils import preprocess_images, preprocess_images2, plot_3d_traj, plot_inlier_ratio, euclidean_distance, plot_drift



# For UW simulation dataset
W = 1920
H = 1080

cam = PinholeCamera(width=float(W), height=float(H), fx=1263.1578, fy=1125, cx=960, cy=540)
vo = VisualOdometry(cam, annotations='./data/simulation_ground_truth_poses.txt')
num_frames = len(os.listdir('data/images_v1/'))
orig_images = preprocess_images2('data/images_v1/*.jpg')
images = preprocess_images('data/images_v1/*.jpg')

traj = np.zeros((480, 640, 3), dtype=np.uint8)
x_orig, y_orig = 290, 400

origin_transformation_mtx = np.array([
	[
		-0.40673887171736184,
		0.9135444653834002,
		1.2021145039395212e-06,
		0.22304523604587842
	],
	[
		-0.14605695240041458,
		-0.0650304707959155,
		0.9871364670214371,
		-0.07497521108390426
	],
	[
		0.9017931341996291,
		0.4015065972501812,
		0.159879940814945,
		-5.6632791527651305
	],
	[
		0.0,
		0.0,
		0.0,
		1.0
	]
])

inlier_ratios = []
drift = []
xs, ys, zs = [], [], []
true_xs, true_ys, true_zs = [], [], []
display_img = np.zeros((1080, 1920, 3))

# for i, (orig_img, img) in enumerate(images):
for i, img in enumerate(images):
	vo.update(img, i)
	cur_t = vo.cur_t
	orig_img = orig_images[i]

	# Wait til 3rd image
	if i > 1:
		x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]
		# Transform ground truth coordinates
		true_point_transformed = origin_transformation_mtx @ np.array([[vo.trueX], [vo.trueY], [vo.trueZ], [1]])
		true_transf_x, true_transf_y, true_transf_z = true_point_transformed[:3]
		inlier_ratios.append(vo.inlier_ratio)
	else:
		x, y, z = 0., 0., 0.
		vo.cur_t = np.zeros((3, 1))
		true_transf_x, true_transf_y, true_transf_z = vo.trueX, vo.trueY, vo.trueZ

	# Key point visualization
	if i > 2:
		for j, (new, old) in enumerate(zip(vo.px_cur, vo.px_ref)):
			a, b = new.ravel()
			c, d = old.ravel()
			# if circle_contains(circle_center, circle_r, (a, b)):
			# color = [0, 200, 0]
			# else:
			color = [255, 100, 0]
			orig_img = cv2.circle(orig_img, (a, b), 2, color, 2)  # color[i].tolist(), -1)

	# Calculate drift
	d = euclidean_distance(np.array([x, -y, -z]), np.array([true_transf_x, true_transf_y, true_transf_z]))
	drift.append(d)

	# 2D trajectory
	k = 30
	draw_x, draw_y = int(x * k) + x_orig, -int(z * k) + y_orig
	true_x, true_y = int(true_transf_x * k) + x_orig, int(true_transf_z * k) + y_orig

	cv2.circle(traj, (draw_x, draw_y), 1, (i * 255 / 4540, 255 - i * 255 / 4540, 0), 1)
	cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 1)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
	text = "Estimated Coordinates: x=%2fm y=%2fm z=%2fm" % (x, -y, -z)
	cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	text_true = "True Coordinates: x=%2fm y=%2fm z=%2fm" % (true_transf_x, true_transf_y, true_transf_z)
	cv2.putText(traj, text_true, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	cv2.namedWindow('Snake Robot Camera', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Snake Robot Camera', W//2, H//2)

	concat = np.concatenate((orig_img, img), axis=1)  # axis=1 for horisontal concat
	cv2.imshow('Snake Robot Camera', concat)

	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

	xs.append(x)
	ys.append(-y)
	zs.append(-z)

	true_xs.append(true_transf_x)
	true_ys.append(true_transf_y)
	true_zs.append(true_transf_z)


cv2.imwrite('plots/map.png', traj)
plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs)
plot_inlier_ratio(inlier_ratios)
plot_drift(drift)

