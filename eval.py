import utils
import numpy as np
import matplotlib.pyplot as plt


class Eval:

	""" Class for evaluating feature detectors, descriptors, tracking and matching methods used in a VO pipeline """
	def __init__(self, vo):
		self.vo = vo
		self.vo_poses = []
		self.gt_poses = []
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
		true_x, true_y, true_z = self.vo.true_t[0][0], self.vo.true_t[1][0], self.vo.true_t[2][0]
		self.xs.append(x)
		self.ys.append(y)
		self.zs.append(z)

		self.true_xs.append(true_x)
		self.true_ys.append(true_y)
		self.true_zs.append(true_z)

		d = utils.euclidean_distance(np.array([x, y, z]), np.array([true_x, true_y, true_z]))
		self.translation_error.append(d)

	def get_relative_poses(self, poses_type):
		""" Calculates the relative poses from the absolute ones

		:param poses_type: groundt truth ('gt') or estimated ('vo')
		:return: rel_poses: list of poses, each relative to the previous pose
		"""
		assert poses_type == 'vo' or poses_type == 'gt'

		abs_poses = self.vo_poses if poses_type == 'vo' else self.gt_poses
		rel_poses = []
		for i in range(len(abs_poses)-1):
			pose1, pose2 = abs_poses[i], abs_poses[i+1]
			rel_poses.append(np.dot(
				np.linalg.inv(pose1),
				pose2)
			)

		return rel_poses

	def get_rotation_error(self, pose_error):
		"""Compute rotation error
		Args:
			pose_error (4x4 array): relative pose error
		Returns:
			rot_error (float): rotation error
		"""
		a = pose_error[0, 0]
		b = pose_error[1, 1]
		c = pose_error[2, 2]
		d = 0.5 * (a + b + c - 1.0)
		rot_error = np.arccos(max(min(d, 1.0), -1.0))
		return rot_error

	def get_translation_error(self, pose_error):
		"""Compute translation error
		Args:
			pose_error (4x4 array): relative pose error
		Returns:
			trans_error (float): translation error
		"""
		dx = pose_error[0, 3]
		dy = pose_error[1, 3]
		dz = pose_error[2, 3]
		trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
		return trans_error

	def calc_relative_errors(self):
		""" Calculate relative erros across frames

		:return: err (list list): [rotation error, translation error]
		"""
		err = []

		rel_vo_poses = self.get_relative_poses('vo')
		rel_gt_poses = self.get_relative_poses('gt')

		for vo_pose, gt_pose in zip(rel_vo_poses, rel_gt_poses):
			# compute rotation and translation error
			pose_error = np.dot(
				np.linalg.inv(vo_pose),
				gt_pose
			)

			r_err = self.get_rotation_error(pose_error)
			t_err = self.get_translation_error(pose_error)

			err.append([r_err, t_err])

		return err

	def compute_overall_err(self, err, reduce='average'):
		"""Compute average or summed translation & rotation errors
		Args:
			seq_err (list list): [[r_err, t_err],[r_err, t_err],...]
				- r_err (float): rotation error
				- t_err (float): translation error
		Returns:
			ave_t_err (float): average translation error
			ave_r_err (float): average rotation error
		"""

		print(np.mean(np.array(err)))
		print(np.sum(np.array(err)))
		t_err = 0
		r_err = 0

		seq_len = len(err)

		if seq_len > 0:
			for item in err:
				r_err += item[0]
				t_err += item[1]
			if reduce == 'average':
				r_err = r_err / seq_len
				t_err = t_err / seq_len
			return r_err, t_err
		else:
			return 0, 0

	def compute_rotation_error(self):
		""" Compute rotation error """
		if self.vo.frame_id > 1:
			R1 = self.vo.cur_R
			R2 = self.vo.true_R

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

	def plot_relative_error(self, err, save=True):
		a = np.array(err)
		r_err, t_err = a[:, 0], a[:, 1]

		plt.plot([i for i in range(len(r_err))], r_err, color='blue')
		plt.title('Relative rotation error across frames')
		plt.xlabel('frame #')
		plt.ylabel('deg')
		if save:
			plt.savefig('plots/relative_rotation_error.png', bbox_inches='tight')
		plt.show()

		plt.plot([i for i in range(len(t_err))], t_err, color='blue')
		plt.title('Relative translation error across frames')
		plt.xlabel('frame #')
		plt.ylabel('%')
		if save:
			plt.savefig('plots/relative_translation_error.png', bbox_inches='tight')
		plt.show()


	def add_pose(self):
		hom_row = np.array([[0, 0, 0, 1]])

		vo_pose = np.hstack((self.vo.cur_R, self.vo.cur_t))  # 3x4
		vo_pose = np.concatenate((vo_pose, hom_row), axis=0)  # Add homogeneous part to make 4x4

		gt_pose = np.hstack((self.vo.true_R, self.vo.true_t))  # 3x4
		gt_pose = np.concatenate((gt_pose, hom_row), axis=0)  # Add homogeneous part to make 4x4

		self.vo_poses.append(vo_pose)
		self.gt_poses.append(gt_pose)

	def update(self):
		self.add_pose()

		self.compute_translation_error()
		self.compute_rotation_error()
		if self.vo.frame_id > 0:
			self.inlier_ratios.append(self.vo.inlier_ratio)



