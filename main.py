import os
import cv2
import numpy as np
from sys import argv
import yaml
import utils
from eval import Eval
from config import Config
from visualizer import VO_Visualizer
from pinhole_camera import PinholeCamera
from visual_odometry import VisualOdometry
from utils import preprocess_images


def run(configuration: dict):
	# Read from ground_truth_config.yaml
	config = Config(configuration)
	H, W = config.H, config.W
	pin_hole_params = config.pin_hole_params
	image_path = config.images
	annotations = config.annotations

	# Initialize
	cam = PinholeCamera(width=float(W), height=float(H), **pin_hole_params)
	vo = VisualOdometry(cam, annotations=annotations, config=config)
	vo_eval = Eval(vo, config)
	vo_visualizer = VO_Visualizer(vo, W, H)

	# Fetch and initialize preprocessing of images
	orig_images = preprocess_images(image_path, default=True)[:175]
	preprocessed_images = preprocess_images(image_path, default=True)[:175]  # morphology=config.toggle_morphology)[:175]

	# Run
	for i, img in enumerate(preprocessed_images):
		vo.update(img, i)
		vo_eval.update()
		vo_visualizer.show(img, orig_images[i])

	# Evaluate
	vo_eval.evaluate(traj=vo_visualizer.traj)


if __name__ == '__main__':
	scenario = str(argv[2]).upper()
	assert scenario in ['UW', 'GT', 'CYCLES'], f"Scenario must be one of ['UW', 'GT', 'CYCLES'], but received {scenario}"
	config_paths = {'UW': 'eevee_config.yaml',
					'GT': 'ground_truth_config.yaml',
					'CYCLES': 'cycles_config.yaml'}

	#config_path = "eevee_config.yaml" if scenario == 'UW' else 'ground_truth_config.yaml'
	config_path = config_paths[scenario]
	cfg = yaml.load(open(config_path), Loader=yaml.Loader)
	run(cfg)