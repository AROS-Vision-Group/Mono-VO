import utils
import numpy as np


class Eval:

	""" Class for evaluating feature detectors, descriptors, tracking and matching methods used in a VO pipeline """
	def __init__(self, vo):
		self.vo = vo
		self.inlier_ratios = []
		self.translation_error = []
		self.rotation_errors = []
		self.xs, self.ys, self.zs = [], [], []
		self.true_xs, self.true_ys, self.true_zs = [], [], []
		self.theta_xs, self.theta_ys, self.theta_zs = [], [], []
		self.theta_xs_true, self.theta_ys_true, self.theta_zs_true = [], [], []
		self.rotation_errors = []

	def compute_translation_error(self):
		""" Update estimated and true coordinates and compute translation error (drift)"""
		x, y, z = self.vo.cur_t[0][0], self.vo.cur_t[1][0], self.vo.cur_t[2][0]
		trueX, trueY, trueZ = self.vo.trueX, self.vo.trueY, self.vo.trueZ
		self.xs.append(x)
		self.ys.append(y)
		self.zs.append(z)

		self.true_xs.append(trueX)
		self.true_ys.append(trueY)
		self.true_zs.append(trueZ)

		d = utils.euclidean_distance(np.array([x, y, z]), np.array([trueX, trueY, trueZ]))
		self.translation_error.append(d)

	def compute_rotation_error(self):
		""" Compute rotation error """
		if self.vo.frame_id > 1:
			R1 = self.vo.cur_R
			R2 = self.vo.trueR

			R2_inv = np.transpose(R2)
			mul_R = R1 @ R2_inv

			theta_x = np.arctan2(R1[2, 1], R1[2, 2])
			theta_y = np.arctan2(-R1[2, 0], np.sqrt(R1[2, 1] ** 2 + R1[2, 2] ** 2))
			theta_z = np.arctan2(R1[1, 0], R1[0, 0])

			theta_x_true = np.arctan2(R2[2, 1], R2[2, 2])
			theta_y_true = np.arctan2(-R2[2, 0], np.sqrt(R2[2, 1] ** 2 + R2[2, 2] ** 2))
			theta_z_true = np.arctan2(R2[1, 0], R2[0, 0])

			self.theta_xs.append(theta_x)
			self.theta_ys.append(theta_y)
			self.theta_zs.append(theta_z)

			self.theta_xs_true.append(theta_x_true)
			self.theta_ys_true.append(theta_y_true)
			self.theta_zs_true.append(theta_z_true)

			theta_sum = np.abs(theta_x) + np.abs(theta_y) + np.abs(theta_z)
			self.rotation_errors.append(theta_sum)

	def update(self):
		self.compute_translation_error()
		self.compute_rotation_error()
		if self.vo.frame_id > 0:
			self.inlier_ratios.append(self.vo.inlier_ratio)



