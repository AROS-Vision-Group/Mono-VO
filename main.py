import os
import cv2
import numpy as np
import yaml
import utils
from eval import Eval
from config import Config
from visualizer import VO_Visualizer
from pinhole_camera import PinholeCamera
from visual_odometry import VisualOdometry
from utils import preprocess_images, plot_3d_traj, plot_inlier_ratio, plot_orientation_angle


def run(configuration: dict):
	# Read from config.yaml

	config = Config(configuration)
	defaults = config.defaults
	H, W = defaults["H"], defaults["W"]
	pin_hole_params = defaults["PIN_HOLE_PARAMS"]

	# Initilalize
	cam = PinholeCamera(width=float(W), height=float(H), **pin_hole_params)
	vo = VisualOdometry(cam, annotations='./data/transformed_ground_truth_vol2.txt', config=config)
	vo_eval = Eval(vo, name=config.name)
	vo_visualizer = VO_Visualizer(vo, W, H)

	orig_images = preprocess_images('data/images_v1/*.jpg', default=True)[:200]
	images = preprocess_images('data/images_v1/*.jpg', morphology=True)[:200]

	# Run
	for i, img in enumerate(images):
		vo.update(img, i)
		vo_eval.update()
		vo_visualizer.show(img, orig_images[i])

	vo_eval.evaluate()


if __name__ == '__main__':
	config = yaml.load(open("config.yaml"), Loader=yaml.Loader)
	run(config)