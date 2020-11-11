import numpy as np
import cv2
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera
import os, os.path
import sys


def get_img_id(i):
	num_digits = len(str(i))
	id = "0" * (4 - num_digits)
	id += str(i)
	return id


# For kitty dataset
# cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
# vo = VisualOdometry(cam, './data/00/00.txt')
# num_frames = 3000 #len(os.listdir('./data/00/image_0/'))
# imgs = [cv2.imread(f'./data/00/image_0/'+str(i).zfill(6)+'.png', 0) for i in range(num_frames)]

# For UW simulation dataset
cam = PinholeCamera(width=1920.0, height=1080.0, fx=1263.1578, fy=1125, cx=960, cy=540)
vo = VisualOdometry(cam, annotations='./data/simulation_ground_truth_poses.txt')
num_frames = len(os.listdir('./data/images/'))
imgs = [cv2.imread(f'./data/images/ground_truth{get_img_id(i + 1)}.jpg', 0) for i in range(num_frames)]

traj = np.zeros((600, 600, 3), dtype=np.uint8)
x_orig, y_orig = 290, 200

origin_transformation_mtx = np.array([
	[
		-0.40673887171736184,
		0.9135444653834002,
		1.2021145039395212e-06,
		0.22304523604587842
	],
	[
		-0.14605695240041458,
		-0.0650304707959155,
		0.9871364670214371,
		-0.07497521108390426
	],
	[
		0.9017931341996291,
		0.4015065972501812,
		0.159879940814945,
		-5.6632791527651305
	],
	[
		0.0,
		0.0,
		0.0,
		1.0
	]
])

"""
print(origin_transformation_mtx.shape)
x0 = np.array([[5.1868767738342285], [2.065206527709961], [0.9794552326202393], [1]])
x0 = np.array([ [-0.40673887729644775, -0.14605696499347687, 0.9017932415008545, 5.1868767738342285],
				[0.9135444760322571, -0.06503047794103622, 0.4015066623687744, 2.065206527709961],
				[1.2021145039398107e-06, 0.9871366024017334, 0.15987995266914368, 0.9794552326202393],
				[0, 0, 0, 1]
			  ])
print(x0.shape)
print(origin_transformation_mtx @ x0)
"""

# cv2.circle(traj, (x_orig, y_orig), 3, (0, 255, 255), 1)

for i, img in enumerate(imgs):

	vo.update(img, i)
	cur_t = vo.cur_t

	if i > 1:
		x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]
		# Transform ground truth coordinates
		true_point_transformed = origin_transformation_mtx @ np.array([[vo.trueX], [vo.trueY], [vo.trueZ], [1]])
		true_transf_x, true_transf_y, true_transf_z = true_point_transformed[:3]
	else:
		x, y, z = 0., 0., 0.
		vo.cur_t = np.zeros((3, 1))
		true_transf_x, true_transf_y, true_transf_z = vo.trueX, vo.trueY, vo.trueZ

	print('-' * 30)
	print(x, y, z)
	print(true_transf_x, true_transf_y, true_transf_z)

	# For UW simulation dataset
	k = 30
	draw_x, draw_y = int(x * k) + x_orig, -int(z * k) + y_orig
	# true_x, true_y = int(vo.trueY*10)+x_orig, int(vo.trueX*10)+y_orig
	true_x, true_y = int(true_transf_x * k) + x_orig, int(true_transf_z * k) + y_orig

	# For kitty dataset
	#draw_x, draw_y = int(x) + 290, int(z) + 90
	#true_x, true_y = int(vo.trueX) + 290, int(vo.trueZ) + 90

	cv2.circle(traj, (draw_x, draw_y), 1, (i * 255 / 4540, 255 - i * 255 / 4540, 0), 1)
	cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 1)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
	text = "Estimated Coordinates: x=%2fm y=%2fm z=%2fm" % (x, -y, -z)
	cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	text_true = "True Coordinates: x=%2fm y=%2fm z=%2fm" % (true_transf_x, true_transf_y, true_transf_z)
	cv2.putText(traj, text_true, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	cv2.imshow('Snake Robot Camera', img)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

cv2.imwrite('map.png', traj)
