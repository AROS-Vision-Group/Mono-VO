import cv2


class DetectorDescriptorInterface:

	def __init__(self, des_extractor, as_extractor):
		self.frame = None
		self.detector = None
		self.des_extractor = des_extractor
		self.as_extractor = as_extractor
		self.kp = None
		self.des = None

	def get_keypoints(self, frame):
		"""
		Takes in a frame and detects key points

		:param frame: frame from which to detect key points from
		:return: frame location of detected points, shape: (numberOfKeypoints, 2)
		"""
		pass

	def get_descriptors(self, frame, kp):
		"""
		If applicable for this detector, outputs the descriptors for each detected key point.
		Otherwise, return None

		:param frame: frame from which to detect key points from
		:param kp: key points to compute descritors frome
		:return: descriptors of detected points, shape: (numberOfKeypoints, descriptorDimension)
		"""
		pass


class HarrisDetector(DetectorDescriptorInterface):
	def __init__(self, des_extractor=None):
		super().__init__(des_extractor, as_extractor=False)


class ShiTomasiDetector(DetectorDescriptorInterface):
	def __init__(self, des_extractor=None):
		super().__init__(des_extractor, as_extractor=False)

	def get_keypoints(self, frame):
		corners = cv2.goodFeaturesToTrack(frame,
										  maxCorners=5000,
										  qualityLevel=0.1,
										  minDistance=10,
										  blockSize=10)

		kp = [cv2.KeyPoint(c[0][1], c[0][1], 1.0) for c in corners]  # np.squeeze(corners)]

		return kp

	def get_descriptors(self, frame, kp):
		if self.des_extractor is None:
			raise NotImplementedError("No descriptor extractor specified.")

		return self.des_extractor.get_descriptors(frame, kp)


class FAST_Detector(DetectorDescriptorInterface):
	def __init__(self, des_extractor=None):
		super().__init__(des_extractor, as_extractor=False)
		self.detector = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)

	def get_keypoints(self, frame):
		kp = self.detector.detect(frame)
		return kp

	def get_descriptors(self, frame, kp):
		if self.des_extractor is None:
			raise NotImplementedError("No descriptor extractor specified.")

		return self.des_extractor.get_descriptors(frame, kp)


class SIFT(DetectorDescriptorInterface):
	def __init__(self, des_extractor=None, as_extractor=False):
		super().__init__(des_extractor, as_extractor)
		self.detector = cv2.SIFT_create()

	def get_keypoints(self, frame):
		kp, des = self.detector.detectAndCompute(frame, None)
		print(des.shape)
		self.des = des
		return kp

	def get_descriptors(self, frame, kp):
		if self.des_extractor is None:
			if self.as_extractor:
				kp, des = self.detector.compute(frame, kp)
				return kp, des
			else:
				return kp, self.des

		return self.des_extractor.get_descriptors(frame, kp)


class SURF(DetectorDescriptorInterface):
	def __init__(self, des_extractor=None, as_extractor=False):
		super().__init__(des_extractor, as_extractor)
		self.detector = cv2.xfeatures2d.SURF_create()

	def get_keypoints(self, frame):
		kp, des = self.detector.detectAndCompute(frame, None)
		self.des = des
		return kp

	def get_descriptors(self, frame, kp):
		if self.des_extractor is None:
			if self.as_extractor:
				kp, des = self.detector.compute(frame, None)
				return kp, des
			else:
				return kp, self.des

		return self.des_extractor.get_descriptors(frame, kp)


class ORB(DetectorDescriptorInterface):
	def __init__(self, des_extractor=None, as_extractor=False):
		super().__init__(des_extractor, as_extractor)
		self.detector = cv2.ORB_create(nfeatures=5000)

	def get_keypoints(self, frame):
		kp, des = self.detector.detectAndCompute(frame, None)
		self.des = des
		return kp

	def get_descriptors(self, frame, kp):
		if self.des_extractor is None:
			if self.as_extractor:
				kp, des = self.detector.compute(frame, kp)
				return kp, des
			else:
				return kp, self.des

		return self.des_extractor.get_descriptors(frame, kp)


class BRIEF_Extractor(DetectorDescriptorInterface):
	def __init__(self):
		super().__init__(des_extractor=None, as_extractor=True)
		self.des_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

	def get_descriptors(self, frame, kp):
		kp, des = self.des_extractor.compute(frame, kp)
		print(des.shape)
		return kp, des

