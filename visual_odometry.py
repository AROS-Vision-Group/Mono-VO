import cv2
import time
import detector
import numpy as np
from config import Config
from pinhole_camera import PinholeCamera
from detector import DetectorDescriptorInterface
from point_correspondence import OpticalFlowTracker, FLANN_Matcher


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1200


class VisualOdometry:
    def __init__(self,
                 cam: PinholeCamera,
                 annotations: list,
                 config: Config):

        self.frame_stage = 0
        self.frame_id = 0
        self.cam = cam
        self.cur_frame = None
        self.prev_frame = None

        self.prev_t = None
        self.cur_t = np.zeros((3, 1))
        self.cur_R = np.eye(3)

        self.true_t = np.zeros((3, 1))
        self.true_R = np.eye(3)

        self.prev_points = None
        self.cur_points = None
        self.prev_desc = None
        self.cur_desc = None

        self.all_cur_desc = None
        self.all_prev_desc = None

        self.cur_lines = None
        self.inlier_ratio = 0
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        
        self.cur_runtime = 0

        self.detector = config.detector
        self.detector.set_extractor(config.extractor)

        self.correspondence_method = config.correspondence_method
        if self.correspondence_method == 'tracking':
            self.point_corr_computer = OpticalFlowTracker(config.lk_params)
        else:
            self.point_corr_computer = FLANN_Matcher(config.flann_params)

        with open(annotations) as f:
            self.annotations = f.readlines()
	def get_absolute_scale(self, frame_id):
		xi, yi, zi = 3, 7, 11
		ss = self.annotations[frame_id - 1].strip().split()
		x_prev = float(ss[xi])
		y_prev = float(ss[yi])
		z_prev = float(ss[zi])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[xi])
		y = float(ss[yi])
		z = float(ss[zi])
		self.true_t = np.array([[x], [y], [z]])
		#self.true_x, self.true_y, self.true_z = x, y, z

		r11, r12, r13 = float(ss[0]), float(ss[1]), float(ss[2])
		r21, r22, r23 = float(ss[4]), float(ss[5]), float(ss[6])
		r31, r32, r33 = float(ss[8]), float(ss[9]), float(ss[10])
		self.true_R = np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape((3, 3))

		return np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)

	def get_relative_scale(self):
		"""
		Triangulate 3-D points X_(k-1) and X_k from current and previous frame to get relative scale
		:return: relative scale of translation between previous and current frame
		"""
		raise NotImplementedError("Relative Scale Method not implemted yet.")

	def process_initial_frame(self):
		self.prev_points = self.detector.get_keypoints(self.cur_frame)
		if self.correspondence_method == 'matching':
			self.prev_points, self.prev_desc = self.detector.get_descriptors(self.cur_frame, self.prev_points)
			self.all_prev_desc = self.prev_desc

		self.prev_points = np.array([x.pt for x in self.prev_points], dtype=np.float32)
		self.all_px_ref = self.prev_points
		self.frame_stage = STAGE_DEFAULT_FRAME

	def process_frame(self, frame_id):

		if self.correspondence_method == 'tracking':
			self.prev_points, self.cur_points = self.point_corr_computer.get_corresponding_points(img_ref=self.prev_frame,
																								  img_cur=self.cur_frame,
																								  px_ref=self.prev_points)
		else:
			self.cur_points = self.detector.get_keypoints(self.cur_frame)
			self.cur_points, self.cur_desc = self.detector.get_descriptors(self.cur_frame, self.cur_points)
			self.cur_points = np.array([x.pt for x in self.cur_points], dtype=np.float32)
			temp_px = self.cur_points
			temp_des = self.cur_desc

			self.prev_points, self.cur_points = self.point_corr_computer.get_corresponding_points(px_ref=self.all_px_ref,
																								  px_cur=self.cur_points,
																								  des_ref=self.all_prev_desc,
																								  des_cur=self.cur_desc)
			self.all_px_ref = temp_px
			self.all_prev_desc = temp_des

		E, mask = cv2.findEssentialMat(self.cur_points, self.prev_points, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
									   prob=0.999, threshold=1.0)

		self.inlier_ratio = np.sum(mask) / (len(mask) + 1)
		self.cur_lines = cv2.computeCorrespondEpilines(self.prev_points.reshape(-1, 1, 2), 2, E)

		_, R, t, mask = cv2.recoverPose(E, self.cur_points, self.prev_points, focal=self.focal, pp=self.pp, mask=mask)

		absolute_scale = self.get_absolute_scale(frame_id)
		if absolute_scale > 0.01:
			self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
			self.cur_R = R.dot(self.cur_R)

		if self.correspondence_method == 'tracking' and self.prev_points.shape[0] < kMinNumFeature: #or frame_id % 50 == 0:
			self.cur_points = self.detector.get_keypoints(self.cur_frame)
			self.cur_points = np.array([x.pt for x in self.cur_points], dtype=np.float32)

		self.prev_points = self.cur_points
		self.prev_desc = self.cur_desc

	def update(self, img, frame_id):
		assert (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width), \
			"Frame: provided image has not the same size as the camera model or image is not grayscale"

		start_time = time.time()
		self.cur_frame = img
		if self.frame_stage == STAGE_DEFAULT_FRAME:
			self.process_frame(frame_id)
		elif self.frame_stage == STAGE_FIRST_FRAME:
			self.process_initial_frame()
		self.prev_frame = self.cur_frame
		self.frame_id = frame_id
		self.cur_runtime = time.time() - start_time
