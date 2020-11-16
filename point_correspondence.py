import cv2
import numpy as np


class PointCorrespondenceFinderInterface:
	def __init__(self, params=None):
		self.params = params

	def get_corresponding_points(self, img_ref=None, img_cur=None, px_ref=None, px_cur=None, des_ref=None, des_cur=None):
		"""

		:param img_ref: previous image
		:param img_cur: current image
		:param px_ref: key point detected/tracked in previous image
		:param px_cur: key points detected/tracked in current image (for matching-based methods)
		:param des_ref: descriptors corresponding to px_ref (for matching-based methods)
		:param des_cur: descriptors corresponding to px_cur (for matching-based methods)
		:return: point correspondences between previous and current image
		"""
		raise NotImplementedError('Method "get_corresponding_points" not implemented.')


class OpticalFlowTracker(PointCorrespondenceFinderInterface):
	def __init__(self, params=None):
		super().__init__(params)

	def get_corresponding_points(self, img_ref=None, img_cur=None, px_ref=None, px_cur=None, des_ref=None, des_cur=None):
		assert (img_ref is not None and img_cur is not None and px_ref is not None)

		kp2, st, err = cv2.calcOpticalFlowPyrLK(img_ref, img_cur, px_ref, None,
												**self.params)  # shape: [k,2] [k,1] [k,1]
		st = st.reshape(st.shape[0])

		kp1 = px_ref[st == 1]
		kp2 = kp2[st == 1]

		return kp1, kp2


class FLANN_Matcher(PointCorrespondenceFinderInterface):
	def __init__(self, params=None):
		super().__init__(params)

	def get_corresponding_points(self, img_ref=None, img_cur=None, px_ref=None, px_cur=None, des_ref=None, des_cur=None):
		assert (px_ref is not None and px_cur is not None and des_ref is not None and des_cur is not None)

		index_params = self.params['index_params']
		search_params = self.params['search_params']

		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des_ref, des_cur, k=2)

		# Discard bad matches, ratio test as per Lowe's paper
		good_matches = list(filter(lambda x: x[0].distance < 0.7 * x[1].distance,
								   matches))
		good_matches = [good_matches[i][0] for i in range(len(good_matches))]

		kp1 = []
		kp2 = []
		for m in good_matches:
			kp1.append(px_ref[m.queryIdx])
			kp2.append(px_cur[m.trainIdx])

		kp1 = np.array(kp1)
		kp2 = np.array(kp2)

		return kp1, kp2
