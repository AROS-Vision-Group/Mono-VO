import os
import cv2
import numpy as np
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera
from utils import preprocess_images, plot_3d_traj, plot_inlier_ratio, euclidean_distance, plot_drift

# For UW simulation dataset
W = 1920
H = 1080

cam = PinholeCamera(width=float(W), height=float(H), fx=1263.1578, fy=1125, cx=960, cy=540)
vo = VisualOdometry(cam, annotations='./data/transformed_ground_truth.txt')
num_frames = len(os.listdir('data/images_v1/'))
orig_images = preprocess_images('data/images_v1/*.jpg', default=True)
images = preprocess_images('data/images_v1/*.jpg')

N = len(images)

traj = np.zeros((480, 640, 3), dtype=np.uint8)
x_orig, y_orig = 290, 400

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
			color = [255, 100, 0]
			orig_img = cv2.circle(orig_img, (a, b), 2, color=(255, 100, 0), thickness=2, lineType=cv2.LINE_AA)

	# Calculate drift
	d = euclidean_distance(np.array([x, -y, -z]), np.array([true_transf_x, true_transf_y, true_transf_z]))
	drift.append(d)

	# 2D trajectory
	k = 30
	draw_x, draw_y = int(x * k) + x_orig, -int(z * k) + y_orig
	true_x, true_y = int(true_transf_x * k) + x_orig, int(true_transf_z * k) + y_orig

	cv2.circle(traj, (true_x, true_y), 1, (0, i*(255), 0), 1, cv2.LINE_AA)
	cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, i*(255)), 1, cv2.LINE_AA)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)

	text = f"Estimated:    x={x:.3f} y={-y:.3f} z={-z:.3f}"
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
	ys.append(-y)
	zs.append(-z)

	true_xs.append(true_transf_x)
	true_ys.append(true_transf_y)
	true_zs.append(true_transf_z)


cv2.imwrite('plots/map.png', traj)
plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs)
plot_inlier_ratio(inlier_ratios)
plot_drift(drift)

