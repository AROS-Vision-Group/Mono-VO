import utils
import numpy as np
from utils import init_logger


class Eval:
    """ Class for evaluating feature detectors, descriptors, tracking and matching methods used in a VO pipeline """

    def __init__(self, vo, config):
        self.config = config
        self.vo = vo
        self.name = config.name
        self.vo_poses = []
        self.gt_poses = []
        self.inlier_ratios = []
        self.runtimes = []

    @staticmethod
    def get_relative_poses(abs_poses):
        """ Calculates the relative poses from the absolute ones

        :param abs_poses: absolute poses (4x4 array)
        :return: rel_poses: list of poses, each relative to the previous pose
        """

        rel_poses = []
        for i in range(len(abs_poses) - 1):
            pose1, pose2 = abs_poses[i], abs_poses[i + 1]
            rel_poses.append(np.dot(
                np.linalg.inv(pose1),
                pose2)
            )

        return rel_poses

    @staticmethod
    def rotation_error(pose_error):
        """Compute rotation error

        :param pose_error: relative pose error (4x4 array)
        :return: rot_error: rotation error (float)
        """

        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    @staticmethod
    def translation_error(pose_error):
        """Compute translation error

        :param pose_error: relative pose error (4x4 array)
        :return: trans_error: translation error as euclidean distance (float)
        """

        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        return trans_error

    @staticmethod
    def compute_orientations(poses):
        """ Compute orientation (yaw, pitch, roll) for each pose """
        yaw_list, pitch_list, roll_list = [], [], []
        for pose in poses:
            R = pose[:3, :3]
            roll, pitch, yaw = utils.rotation_matrix_to_euler_angles(R)
            yaw_list.append(yaw)
            pitch_list.append(pitch)
            roll_list.append(roll)

        return np.array(yaw_list) * 180/np.pi, np.array(pitch_list) * 180/np.pi, np.array(roll_list) * 180/np.pi

    @staticmethod
    def compute_ATE(gt_poses, vo_poses):
        """ Compute RMSE of ATE (Absolute Trajectory Error)

        :param gt_poses: ground-truth poses (4x4 array)
        :param vo_poses: estimated poses (4x4 array)
        :return: ate: absolute trajectory error (float)
        """
        errors = []

        for gt_pose, vo_pose in zip(gt_poses, vo_poses):
            gt_xyz = gt_pose[:3, 3]
            vo_xyz = vo_pose[:3, 3]

            align_err = gt_xyz - vo_xyz
            errors.append(np.sqrt(np.sum(align_err ** 2)))

        ate = np.sqrt(np.mean(np.array(errors)))
        return ate

    def compute_errors(self, gt_poses, vo_poses):
        """ Calculate translation and rotation errors

        :param gt_poses: ground-truth poses (4x4 array)
        :param vo_poses: estimated poses (4x4 array)
        :return: rotation errors, translation errors
        """

        trans_errors = []
        rot_errors = []

        for gt_rel, vo_rel in zip(gt_poses, vo_poses):
            rel_error = np.dot(
                np.linalg.inv(gt_rel),
                vo_rel
            )
            trans_errors.append(self.translation_error(rel_error))
            rot_errors.append(self.rotation_error(rel_error))

        return np.array(rot_errors) * 180/np.pi, np.array(trans_errors)

    def compute_AOE(self, gt_poses, vo_poses):
        """ Compute Absolute Orientation Error

        :param gt_poses: ground-truth poses (4x4 array)
        :param vo_poses: estimated poses (4x4 array)
        :return: aoe: absolute orientation error (float)
        """
        rot_errors = []
        for gt_pose, vo_pose in zip(gt_poses, vo_poses):
            pose_error = np.dot(
                np.linalg.inv(gt_pose),
                vo_pose
            )
            rot_errors.append(self.rotation_error(pose_error))

        return np.array(rot_errors)

    def add_pose(self):
        hom_row = np.array([[0, 0, 0, 1]])

        vo_pose = np.hstack((self.vo.cur_R, self.vo.cur_t))  # 3x4
        vo_pose = np.concatenate((vo_pose, hom_row), axis=0)  # Add homogeneous part to make 4x4

        gt_pose = np.hstack((self.vo.true_R, self.vo.true_t))  # 3x4
        gt_pose = np.concatenate((gt_pose, hom_row), axis=0)  # Add homogeneous part to make 4x4

        self.vo_poses.append(vo_pose)
        self.gt_poses.append(gt_pose)

    def get_positions(self, poses):
        xs, ys, zs = [], [], []
        for pose in poses:
            xs.append(pose[0, 3])
            ys.append(pose[1, 3])
            zs.append(pose[2, 3])

        return np.array(xs), np.array(ys), np.array(zs)

    def compute_orientation_error(self, orientations, true_orientations):
        """ Method that calculates the orientation error.
        Fixes the fact that the angles turns negative after reaching 180 (170 -> 175 -> 180 -> -179 -> -175...)
        So if estimated angle is 175 and the true angle is -175, the actual error is 10 degrees, and not 360 degrees
        """
        ori = orientations.copy()
        true_ori = true_orientations.copy()
        ori[ori < 0] *= -1
        true_ori[true_ori < 0] *= -1

        return ori - true_ori


    def update(self):
        self.add_pose()
        self.runtimes.append(self.vo.cur_runtime)

        if self.vo.frame_id > 0:
            self.inlier_ratios.append(self.vo.inlier_ratio)

    def evaluate(self, traj=None):

        result = {'name': self.name}

        # Create result directory
        scenario = "ground_truth" if self.config.dataset_scenario == 'GT' else 'eevee'
        res_dir = f'./results/images_v{self.config.dataset_version}/{scenario}/{self.name}'
        plot_path = f'{res_dir}/plots/'
        log_path = f'{res_dir}/logs/'

        utils.create_dir(res_dir)
        utils.create_dir(plot_path)
        utils.create_dir(log_path)

        # Absolute poses
        gt_poses, vo_poses = self.gt_poses, self.vo_poses

        # Relative poses
        rel_gt_poses, rel_vo_poses = self.get_relative_poses(gt_poses), self.get_relative_poses(vo_poses)

        # ---- Absolute Errors ----
        rot_errors, trans_errors = self.compute_errors(gt_poses, vo_poses)

        # Absolute Trajectory Error (ATE)
        ate = np.sqrt(np.mean(np.array(trans_errors) ** 2))  # Root Mean Squared Error
        result['ate'] = ate

        # Absolute Orientation Error (AOE)
        aoe = np.mean(rot_errors)
        result['aoe'] = aoe

        # ---- Relative Errors ----
        rel_rot_errors, rel_trans_errors = self.compute_errors(rel_gt_poses, rel_vo_poses)

        # Relative Trajectory Error (RTE)
        rte = np.sqrt(np.mean(np.array(rel_trans_errors) ** 2))
        result['rte'] = rte

        # Relative Rotation Error (RRE)
        rre = np.mean(rel_rot_errors)
        result['rre'] = rre

        # ---- Yaw, Pitch, Roll ----
        yaw, pitch, roll = self.compute_orientations(vo_poses)
        true_yaw, true_pitch, true_roll = self.compute_orientations(gt_poses)

        rel_yaw, rel_pitch, rel_roll = self.compute_orientations(rel_vo_poses)
        true_rel_yaw, true_rel_pitch, true_rel_roll = self.compute_orientations(rel_gt_poses)

        #yaw_errors, pitch_errors, roll_errors = yaw - true_yaw, pitch - true_pitch, roll - true_roll
        yaw_errors = self.compute_orientation_error(yaw, true_yaw)
        pitch_errors = self.compute_orientation_error(pitch, true_pitch)
        roll_errors = self.compute_orientation_error(roll, true_roll)

        inlier_ratios = np.mean(self.inlier_ratios)
        result['inlier_ratio'] = inlier_ratios

        # ---- Processing Time ----
        runtime = np.mean(self.runtimes)
        result['runtime'] = runtime

        # -------------------------
        xs, ys, zs = self.get_positions(vo_poses)
        true_xs, true_ys, true_zs = self.get_positions(gt_poses)
        x_errors, y_errors, z_errors = xs - true_xs, ys - true_ys, zs - true_zs

        # ---- PLOTS ----
        utils.plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs,
                           save_path=f'{plot_path}/3d_traj.png')

        utils.plot_translation_error(trans_errors,
                                     title="Absolute Translation Error",
                                     save_path=f'{plot_path}/translation_error.png')
        utils.plot_rotation_erros(rot_errors,
                                  title='Absolute Rotation Error',
                                  save_path=f'{plot_path}/rotation_error.png')

        utils.plot_translation_error(rel_trans_errors,
                                     title="Relative Translation Error",
                                     save_path=f'{plot_path}/rel_translation_error.png')
        utils.plot_rotation_erros(rel_rot_errors,
                                  title='Relative Rotation Error',
                                  save_path=f'{plot_path}/rel_rotation_error.png')
        utils.plot_inlier_ratio(self.inlier_ratios,
                                save_path=f'{plot_path}/inlier_ratio.png')

        utils.plot_position_error(x_errors, y_errors, z_errors, save_path=f'{plot_path}/position_errors.png')
        utils.plot_orientation_error(yaw_errors, pitch_errors, roll_errors,
                                     save_path=f"{plot_path}/orientation_errors.png")

        utils.plot_orientation_angle(true_yaw, yaw, 'yaw',
                                     title='Yaw across frames',
                                     save_path=f'{plot_path}/yaw.png')
        utils.plot_orientation_angle(true_pitch, pitch, 'pitch',
                                     title='Pitch across frames',
                                     save_path=f'{plot_path}/pitch.png')
        utils.plot_orientation_angle(true_roll, roll, 'roll',
                                     title='Roll across frames',
                                     save_path=f'{plot_path}/roll.png')

        utils.plot_orientation_angle(true_rel_yaw, rel_yaw, 'rel_yaw',
                                     title='Relative yaw across frames',
                                     save_path=f'{plot_path}/rel_yaw.png')
        utils.plot_orientation_angle(true_rel_pitch, rel_pitch, 'rel_pitch',
                                     title='Relative pitch across frames',
                                     save_path=f'{plot_path}/rel_pitch.png')
        utils.plot_orientation_angle(true_rel_roll, rel_roll, 'rel_roll',
                                     title='Relative roll across frames',
                                     save_path=f'{plot_path}/rel_roll.png')
        if traj is not None:
            import cv2
            cv2.imwrite(f'{plot_path}/2d_live_traj.png', traj)

        # Initialize logger
        init_logger(log_path)

        # Write to log
        utils.pretty_log(config=self.config, results=result)
