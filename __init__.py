import os
import cv2
import numpy as np
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera
from utils import preprocess_images, plot_3d_traj, plot_inlier_ratio, plot_orientation_angle
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

orig_images = preprocess_images('data/images_v2/*.jpg', default=True)
images = preprocess_images('data/images_v2/*.jpg', morphology=False)
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


vo_eval.evaluate()





