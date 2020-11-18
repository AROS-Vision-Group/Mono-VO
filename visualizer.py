import cv2
import numpy as np
import utils


class VO_Visualizer:
	def __init__(self, vo, W, H):
		self.vo = vo
		self.traj = np.zeros((480, 640, 3), dtype=np.uint8)
		self.W = W
		self.H = H

	def visualize_key_points(self, orig_img):
		if self.vo.frame_id > 0:
			for j, (new, old) in enumerate(zip(self.vo.cur_points, self.vo.cur_points)):
				a, b = new.ravel()
				c, d = old.ravel()
				orig_img = cv2.circle(orig_img, (int(a), int(b)), 2, color=(255, 255, 0), thickness=2,
									  lineType=cv2.LINE_AA)

	def show_frame_nr(self, orig_img):
		text = f" {self.vo.frame_id}"
		cv2.putText(orig_img, text, (920, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, 8)

	def show(self, img, orig_img):
		cv2.namedWindow('Snake Robot Camera', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Snake Robot Camera', self.W, self.H // 2)

		self.visualize_key_points(orig_img)

		self.show_frame_nr(orig_img)

		hstack = np.hstack((orig_img, img))
		cv2.imshow('Snake Robot Camera', hstack)

		self.visualize_2d_traj()
		cv2.waitKey(1)

	def visualize_2d_traj(self, x_orig=290, y_orig=400):
		camera_traj = np.zeros((480, 640, 3), dtype=np.uint8)
		vo = self.vo
		i = self.vo.frame_id
		x, y, z = vo.cur_t[0][0], vo.cur_t[1][0], vo.cur_t[2][0]
		true_x, true_y, true_z = self.vo.true_t[0][0], self.vo.true_t[1][0], self.vo.true_t[2][0]

		# 2D trajectory
		k = 30
		draw_x, draw_y = int(x * k) + x_orig, -int(z * k) + y_orig
		draw_true_x, draw_true_y = int(true_x * k) + x_orig, -int(true_z * k) + y_orig

		cv2.circle(self.traj, (draw_true_x, draw_true_y), 1, (0, i * (255), 0), 1, cv2.LINE_AA)
		cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 0, i * (255)), 1, cv2.LINE_AA)
		cv2.rectangle(self.traj, (10, 20), (600, 60), (0, 0, 0), -1)

		text = f"Estimated:    x={x:.3f} y={y:.3f} z={z:.3f}"
		cv2.putText(self.traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
		text_true = f"GroundTruth: x={true_x:.3f} y={true_y:.3f} z={true_z:.3f}"
		cv2.putText(self.traj, text_true, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

		# Camera viewport lines
		l = 20
		x_x, x_y = vo.cur_R[0][0] * l, vo.cur_R[0][2] * l
		z_x, z_y = -(vo.cur_R[2][0]) * l, -(vo.cur_R[2][2]) * l

		x_x, x_y = utils.rotate_around(x_x, x_y, draw_x, draw_y, -45)
		z_x, z_y = utils.rotate_around(z_x, z_y, draw_x, draw_y, -45)

		cv2.line(camera_traj, (draw_x, draw_y), (int(x_x), int(x_y)), (255, 255, 255), 1, cv2.LINE_AA)
		cv2.line(camera_traj, (draw_x, draw_y), (int(z_x), int(z_y)), (255, 255, 255), 1, cv2.LINE_AA)
		cv2.line(camera_traj, (int(x_x), int(x_y)), (int(z_x), int(z_y)), (255, 255, 255), 2, cv2.LINE_AA)

		combined = cv2.add(self.traj, camera_traj)
		cv2.imshow('Trajectory', combined)

