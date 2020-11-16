import os
import cv2
import numpy as np
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera
from utils import preprocess_images, plot_3d_traj, plot_inlier_ratio, euclidean_distance, plot_drift, plot_rotation_erros, plot_orientation_angle
import utils
from eval import Eval
from visualizer import VO_Visualizer

# For UW simulation dataset
W = 1920
H = 1080

cam = PinholeCamera(width=float(W), height=float(H), fx=1263.1578, fy=1125, cx=960, cy=540)
vo = VisualOdometry(cam, annotations='./data/transformed_ground_truth_vol2.txt')
vo_eval = Eval(vo)
vo_visualizer = VO_Visualizer(vo, W, H)

orig_images = preprocess_images('data/images_v1/*.jpg', default=True)
images = preprocess_images('data/images_v1/*.jpg', morphology=False)
N = len(images)

for i, img in enumerate(images):
	vo.update(img, i)
	vo_eval.update()
	vo_visualizer.show(img, orig_images[i])

	x, y, z = vo.cur_t[0][0], vo.cur_t[1][0], vo.cur_t[2][0]
	true_x, true_y, true_z = vo.true_t[0][0], vo.true_t[1][0], vo.true_t[2][0]

	# For camera pose line visualization
	if i > 0:
		cur_lines = vo.cur_lines.reshape(-1, 3)
		a, b, c = cur_lines[0][0], cur_lines[0][1], cur_lines[0][2]
		distances = utils.compute_perpendicular_distance(vo.cur_points, a, b, z)
		#frame_perp_distances[i] = compute_mean_distance(distances)


cv2.imwrite('plots/map.png', vo_visualizer.traj)
plot_3d_traj(vo_eval.xs, vo_eval.ys, vo_eval.zs, vo_eval.true_xs, vo_eval.true_ys, vo_eval.true_zs)
plot_inlier_ratio(vo_eval.inlier_ratios)
plot_drift(vo_eval.translation_error)
plot_rotation_erros(vo_eval.rotation_errors)

plot_orientation_angle(vo_eval.theta_xs_true, vo_eval.theta_xs, 'x_angle')
plot_orientation_angle(vo_eval.theta_ys_true, vo_eval.theta_ys, 'y_angle')
plot_orientation_angle(vo_eval.theta_zs_true, vo_eval.theta_zs, 'z_angle')

print(f'-- Evaluation')
print(f'Total translation error: {np.mean(vo_eval.translation_error):.3f}')
print(f'Total rotation error: {np.mean(vo_eval.rotation_errors):.3f}')
print(f'Average RANSAC inlier ratio: {np.mean(vo_eval.inlier_ratios):.3f}')


rel_errors = vo_eval.calc_relative_errors()
ave_t_err, ave_r_err = vo_eval.compute_overall_err(rel_errors, reduce='sum')

print(f'Overall relative translation error: {ave_t_err:.3f}')
print(f'Overall relative rotation error: {ave_r_err:.3f}')

vo_eval.plot_relative_error(rel_errors)
