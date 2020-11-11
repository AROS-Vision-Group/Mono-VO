import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from visual_odometry import VisualOdometry
from pinhole_camera import PinholeCamera


def get_img_id(i):
	num_digits = len(str(i))
	id = "0" * (4 - num_digits)
	id += str(i)
	return id


def draw_3d_plot(xs, ys, zs, true_xs, true_ys, true_zs):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(xs, ys, zs, label='Estimated Trajectory', color='green')
	ax.plot(true_xs, true_ys, true_zs, label='Ground Truth', color='red')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
	plt.show()


def preprocess_images(filepath):
	out = []
	images = [cv2.imread(file, 0) for file in sorted(glob.glob(filepath))]
	for img in images:
		#img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
		#img = cv2.medianBlur(img, 3)
		img = cv2.resize(img,)
		img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		out.append(img)
	return out

# For UW simulation dataset
cam = PinholeCamera(width=1920.0, height=1080.0, fx=1263.1578, fy=1125, cx=960, cy=540)
vo = VisualOdometry(cam, annotations='./data/simulation_ground_truth_poses.txt')
num_frames = len(os.listdir('data/images_misty/'))
images = preprocess_images('data/images_misty/*.jpg')

traj = np.zeros((480, 640, 3), dtype=np.uint8)
x_orig, y_orig = 290, 400

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

xs, ys, zs = [], [], []
true_xs, true_ys, true_zs = [], [], []
for i, img in enumerate(images):
	vo.update(img, i)
	cur_t = vo.cur_t

	# Wait til 3rd image
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
	true_x, true_y = int(true_transf_x * k) + x_orig, int(true_transf_z * k) + y_orig


	cv2.circle(traj, (draw_x, draw_y), 1, (i * 255 / 4540, 255 - i * 255 / 4540, 0), 1)
	cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 1)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
	text = "Estimated Coordinates: x=%2fm y=%2fm z=%2fm" % (x, -y, -z)
	cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	text_true = "True Coordinates: x=%2fm y=%2fm z=%2fm" % (true_transf_x, true_transf_y, true_transf_z)
	cv2.putText(traj, text_true, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	cv2.namedWindow('Snake Robot Camera', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Snake Robot Camera', 640, 480)
	cv2.imshow('Snake Robot Camera', img)

	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

	xs.append(x)
	ys.append(-y)
	zs.append(-z)

	true_xs.append(true_transf_x)
	true_ys.append(true_transf_y)
	true_zs.append(true_transf_z)

cv2.imwrite('map.png', traj)

draw_3d_plot(xs, ys, zs, true_xs, true_ys, true_zs)


