
import cv2
import numpy as np
import utils
import os
import matplotlib.pyplot as plt


def feature_matching(px_ref, px_cur, des_ref, des_cur):
	"""
	Finding point correspondences between previous and current image using Brute-Force or FLANN Matcher

	:param px_ref: key points detected/tracked in previous image
	:param px_cur: key points detected/tracked in current image
	:param des_ref: descriptors corresponding to px_ref
	:param des_cur: descriptors corresponding to px_cur
	:return: point correspondences between previous and current image
	"""

	print('-'*30)
	print(f'# of point in ref: {len(px_ref)}')
	print(f'# of descs in ref: {len(des_ref)}')

	print(f'# of point in cur: {len(px_cur)}')
	print(f'# of descs in cur: {len(des_cur)}')

	#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
	#matches = bf.knnMatch(des_ref, des_cur, k=2)

	FLANN_INDEX_LSH = 6
	index_params = dict(algorithm=FLANN_INDEX_LSH,
						table_number=6,  # 12
						key_size=12,  # 20
						multi_probe_level=1)  # 2
	search_params = dict(checks=50)  # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.match(des_ref, des_cur)  # k=2 to apply ratio test

	#good_matches = matches
	# create BFMatcher object
	#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# matches = bf.match(des1, des2)

	good_matches = sorted(matches, key=lambda x: x.distance)[:50]

	# Discard bad matches, ratio test as per Lowe's paper
	#good_matches = list(filter(lambda x: x[0].distance < 0.7 * x[1].distance,
	 #                     matches))
	#good_matches = [good_matches[i][0] for i in range(len(good_matches))]

	#good_matches = []
	#for (m, n) in matches:
	#	if m.distance < 0.75 * n.distance:
	#		good_matches.append([m])

	return good_matches

	# kp1 = []
	# kp2 = []
	# for m in good_matches:
	#     kp1.append(px_ref[m.trainIdx])
	#     kp2.append(px_cur[m.queryIdx])
	# #(tp, qp) = np.float32((tp, qp))
	#
	#
	# return np.array(kp1), np.array(kp2)


if __name__ == '__main__':
	# Initiate SIFT detector
	sift = cv2.ORB_create()
	num_frames = len(os.listdir('data/images_v1/'))
	orig_images = utils.preprocess_images('data/images_v1/*.jpg', default=True)
	images = utils.preprocess_images('data/images_v1/*.jpg')

	img1 = images[0]
	img2 = images[10]

	# # find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	print(len(kp1), len(des1))
	print(len(kp2), len(des2))

	good_matches = feature_matching(kp1, kp2, des1, des2)
	# cv2.drawMatchesKnn expects list of lists as matches.
	img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

	plt.imshow(img3)
	plt.show()


	#img1 = cv2.imread('box.png',cv2.IMREAD_GRAYSCALE)          # queryImage
	#img2 = cv2.imread('box_in_scene.png',cv2.IMREAD_GRAYSCALE) # trainImage


	# Initiate ORB detector
	# orb = cv2.ORB_create()
	# # find the keypoints and descriptors with ORB
	# kp1, des1 = orb.detectAndCompute(img1,None)
	# kp2, des2 = orb.detectAndCompute(img2,None)
	#
	# # create BFMatcher object
	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# # Match descriptors.
	# matches = bf.match(des1,des2)
	# # Sort them in the order of their distance.
	# matches = sorted(matches, key=lambda x:x.distance)
	# # Draw first 10 matches.
	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:200],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	# plt.imshow(img3),plt.show()
