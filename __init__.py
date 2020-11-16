import os
import cv2
import numpy as np
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera
from utils import preprocess_images, plot_3d_traj, plot_inlier_ratio, euclidean_distance, plot_drift, plot_rotation_erros, plot_orientation_angle
import utils
from eval import Eval

# For UW simulation dataset
W = 1920
H = 1080

cam = PinholeCamera(width=float(W), height=float(H), fx=1263.1578, fy=1125, cx=960, cy=540)
vo = VisualOdometry(cam, annotations='./data/transformed_ground_truth_vol2.txt')
vo_eval = Eval(vo)

orig_images = preprocess_images('data/images_uw_denoized/*.jpg', default=True)[:200]
images = preprocess_images('data/images_uw_denoized/*.jpg', morphology=False)[:200]
N = len(images)

traj = np.zeros((480, 640, 3), dtype=np.uint8)
display_img = np.zeros((1080, 1920, 3))

for i, img in enumerate(images):
	vo.update(img, i)
	vo_eval.update()

	orig_img = orig_images[i]
	camera_traj = np.zeros((480, 640, 3), dtype=np.uint8)

	x, y, z = vo.cur_t[0][0], vo.cur_t[1][0], vo.cur_t[2][0]
	true_x, true_y, true_z = vo.true_x, vo.true_y, vo.true_z

	# For camera pose line visualization
	if i > 0:
		cur_lines = vo.cur_lines.reshape(-1, 3)
		a, b, c = cur_lines[0][0], cur_lines[0][1], cur_lines[0][2]
		distances = utils.compute_perpendicular_distance(vo.cur_points, a, b, z)
		#frame_perp_distances[i] = compute_mean_distance(distances)

	# Key point visualization
	if i > 0:
		for j, (new, old) in enumerate(zip(vo.cur_points, vo.prev_points)):
			a, b = new.ravel()
			c, d = old.ravel()
			orig_img = cv2.circle(orig_img, (int(a), int(b)), 2, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

	cv2.namedWindow('Snake Robot Camera', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Snake Robot Camera', W, H//2)

	hstack = np.hstack((orig_img, img))
	cv2.imshow('Snake Robot Camera', hstack)

	utils.visualize_2d_traj(vo, traj, camera_traj)
	cv2.waitKey(1)


cv2.imwrite('plots/map.png', traj)
plot_3d_traj(vo_eval.xs, vo_eval.ys, vo_eval.zs, vo_eval.true_xs, vo_eval.true_ys, vo_eval.true_zs)
plot_inlier_ratio(vo_eval.inlier_ratios)
plot_drift(vo_eval.translation_error)
plot_rotation_erros(vo_eval.rotation_errors)

plot_orientation_angle(vo_eval.theta_xs_true, vo_eval.theta_xs, 'x_angle')
plot_orientation_angle(vo_eval.theta_ys_true, vo_eval.theta_ys, 'y_angle')
plot_orientation_angle(vo_eval.theta_zs_true, vo_eval.theta_zs, 'z_angle')

print(f'-- Evaluation')
print(f'Total translation error: {np.sum(vo_eval.translation_error):.3f}')
print(f'Total rotation error: {np.sum(vo_eval.rotation_errors):.3f}')
print(f'Average RANSAC inlier ratio: {np.mean(vo_eval.inlier_ratios):.3f}')