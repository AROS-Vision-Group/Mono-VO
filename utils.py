import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math
from scipy.spatial.kdtree import KDTree


def get_img_id(i):
	num_digits = len(str(i))
	id = "0" * (4 - num_digits)
	id += str(i)
	return id


def plot_inlier_ratio(ratios, save=True):
	plt.plot([i for i in range(len(ratios))], ratios, color='blue')
	plt.title('RANSAC inlier ratio across frames')
	plt.xlabel('frame #')
	plt.ylabel('Inlier Ratio')
	if save:
		plt.savefig('plots/inlier_ratio.png', bbox_inches='tight')
	plt.show()


def plot_drift(drift, save=True):
	plt.plot([i for i in range(len(drift))], drift, color='blue')
	plt.title('Drift (L2 distance) between estimated and GT')
	plt.xlabel('frame #')
	plt.ylabel('Drift')
	if save:
		plt.savefig('plots/drift.png', bbox_inches='tight')
	plt.show()


def plot_orientation_angle(theta_true, theta_hat, angle_name, save=True):
	plt.plot([i for i in range(len(theta_true))], theta_true, label=f'{angle_name}_true')
	plt.plot([i for i in range(len(theta_hat))], theta_hat, label=f'{angle_name}_estimated')
	plt.title(f'{angle_name} value across frame')
	plt.xlabel('frame #')
	plt.legend()
	if save:
		plt.savefig(f'plots/orientation_{angle_name}.png', bbox_inches='tight')
	plt.show()


def plot_rotation_erros(rot_errors, save=True):
	plt.plot([i for i in range(len(rot_errors))], rot_errors, color='blue')
	plt.title('Rotation error across frames')
	plt.xlabel('frame #')
	plt.ylabel('Angle (absolute value)')
	if save:
		plt.savefig('plots/rotation_error.png', bbox_inches='tight')
	plt.show()


def visualize_2d_traj(vo, traj, camera_traj, x_orig=290, y_orig=400):
	i = vo.frame_id
	x, y, z = vo.cur_t[0][0], vo.cur_t[1][0], vo.cur_t[2][0]
	trueX, trueY, trueZ = vo.true_x, vo.true_y, vo.true_z

	# 2D trajectory
	k = 30
	draw_x, draw_y = int(x * k) + x_orig, -int(z * k) + y_orig
	true_x, true_y = int(trueX * k) + x_orig, -int(trueZ * k) + y_orig

	cv2.circle(traj, (true_x, true_y), 1, (0, i * (255), 0), 1, cv2.LINE_AA)
	cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, i * (255)), 1, cv2.LINE_AA)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)

	text = f"Estimated:    x={x:.3f} y={y:.3f} z={z:.3f}"
	cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
	text_true = f"GroundTruth: x={trueX:.3f} y={trueY:.3f} z={trueZ:.3f}"
	cv2.putText(traj, text_true, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

	# Camera viewport lines
	l = 20
	x_x, x_y = vo.cur_R[0][0] * l, vo.cur_R[0][2] * l
	z_x, z_y = -(vo.cur_R[2][0]) * l, -(vo.cur_R[2][2]) * l

	x_x, x_y = rotate_around(x_x, x_y, draw_x, draw_y, -45)
	z_x, z_y = rotate_around(z_x, z_y, draw_x, draw_y, -45)

	cv2.line(camera_traj, (draw_x, draw_y), (int(x_x), int(x_y)), (0, 255, 255), 1, cv2.LINE_AA)
	cv2.line(camera_traj, (draw_x, draw_y), (int(z_x), int(z_y)), (255, 255, 0), 1, cv2.LINE_AA)

	combined = cv2.add(traj, camera_traj)
	cv2.imshow('Trajectory', combined)


def make_2d_traj(traj, vo, draw_x, draw_y):
	x, y, z = vo.x, vo.y, vo.z
	trueX, trueY, trueZ = vo.true_x, vo.true_y, vo.true_z

	k = 30
	draw_x, draw_y = int(x * k) + x_orig, -int(z * k) + y_orig
	true_x, true_y = int(trueX * k) + x_orig, -int(trueZ * k) + y_orig

	cv2.circle(traj, (true_x, true_y), 1, (0, 255, 0), 1, cv2.LINE_AA)
	cv2.circle(traj, (draw_x, draw_y), 1, (0, 0, 255), 1, cv2.LINE_AA)
	cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)

	text = f"Estimated:    x={x:.3f} y={y:.3f} z={z:.3f}"
	cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
	text_true = f"GroundTruth: x={trueX:.3f} y={trueY:.3f} z={trueZ:.3f}"
	cv2.putText(traj, text_true, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)


def plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs, save=True):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(xs, ys, zs, label='Estimated Trajectory', color='red')
	ax.plot(true_xs, true_ys, true_zs, label='Ground Truth', color='green')
	ax.set_title('3D trajectory')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.legend()
	if save:
		plt.savefig('plots/3d_traj.png', bbox_inches='tight')
	plt.show()


def compute_perpendicular_distance(points, a, b, c):
	perp_dist = []
	for point in points:
		#print(point)
		x, y = point[0], point[1]
		perp_dist.append(np.abs((a*x + b*y + c)/np.sqrt(a**2 + b**2)))
	return perp_dist


def draw_lines(frame, lines, pts1, pts2):
	r, c = frame.shape

	for r, pt1, pt2 in zip(lines, pts1, pts2):
		color = tuple(np.random.randint(0, 255, 3).tolist())

		x0, y0 = map(int, [0, -r[2] / r[1]])
		x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

		frame = cv2.line(frame, (x0, y0), (x1, y1), color, 1)
		frame = cv2.circle(frame, tuple(pt1), 5, color, -1)

		plt.imshow(frame)
		plt.show()

	return frame

def adjust_gamma(image, gamma=1.0):
    # source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    img = image.copy()
    inverted_gamma = 1.0 / gamma
    look_up_table = np.array([((i / 255.0) ** inverted_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img , look_up_table)

def preprocess_images(filepath, default=False, morphology=False):
	out = []
	images = [cv2.imread(file, 0) for file in sorted(glob.glob(filepath))]
	if default:
		return images
	for img in images:
		processed_img = img.copy()
		cv2.normalize(img.astype('float'), img, 0.0, 1.0, cv2.NORM_MINMAX)
		if morphology:
			processed_img = cv2.GaussianBlur(processed_img, (7, 7), 0)
			processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
			kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=(3, 3))
			# img_erosion = cv2.erode(img, kernel, iterations=1)
			processed_img = cv2.dilate(processed_img, kernel, iterations=3)
			processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel, iterations=1)
			# processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel, iterations=1)
		else:
			processed_img = cv2.GaussianBlur(processed_img, (7, 7), 0)
			processed_img = adjust_gamma(processed_img, 1.5)
			cv2.equalizeHist(processed_img, processed_img)
			#processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
			processed_img = processed_img
		out.append(processed_img)
	return out


def get_images_from_video(video_path):
	imgs = []
	cap = cv2.VideoCapture(video_path)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("No more frames.")
			break

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		imgs.append(frame_gray)
	return imgs


def euclidean_distance(a, b):
	return np.linalg.norm(a - b)


def rotate_around(px, py, cx, cy, angle):
	s = np.sin(angle)
	c = np.cos(angle)
	x_new = px * c - py * s
	y_new = px * s + py * c
	px = x_new + cx
	py = y_new + cy

	return px, py


def euler_angles_to_rotation_matrix(theta):
	""" Calculates Rotation Matrix given euler angles."""
	R_x = np.array([[1, 0, 0],
					[0, math.cos(theta[0]), -math.sin(theta[0])],
					[0, math.sin(theta[0]), math.cos(theta[0])]
					])

	R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
					[0, 1, 0],
					[-math.sin(theta[1]), 0, math.cos(theta[1])]
					])

	R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
					[math.sin(theta[2]), math.cos(theta[2]), 0],
					[0, 0, 1]
					])

	R = np.dot(R_z, np.dot(R_y, R_x))

	return R


def is_rotation_matrix(R):
	""" Checks if a matrix is a valid rotation matrix."""
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	print(n)
	return n < 1e-6


def rotation_matrix_to_euler_angles(R):
	""" Calculates rotation matrix to euler angles"""
	assert (is_rotation_matrix(R))

	sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
	singular = sy < 1e-6

	if not singular:
		x = math.atan2(R[2, 1], R[2, 2])
		y = math.atan2(-R[2, 0], sy)
		z = math.atan2(R[1, 0], R[0, 0])
	else:
		x = math.atan2(-R[1, 2], R[1, 1])
		y = math.atan2(-R[2, 0], sy)
		z = 0

	return np.array([x, y, z])


def KDT_NMS(kps, descs=None, r=15, k_max=20):
    """ Use kd-tree to perform local non-maximum suppression of key-points
    kps - key points obtained by one of openCVs 2d features detectors (SIFT, SURF, AKAZE etc..)
    r - the radius of points to query for removal
    k_max - maximum points retreived in single query
    """
    # sort by score to keep highest score features in each locality
    neg_responses = [-kp.response for kp in kps]
    order = np.argsort(neg_responses)
    kps = np.array(kps)[order].tolist()

    # create kd-tree for quick NN queries
    data = np.array([list(kp.pt) for kp in kps])
    kd_tree = KDTree(data)

    # perform NMS using kd-tree, by querying points by score order,
    # and removing neighbors from future queries
    N = len(kps)
    removed = set()
    for i in range(N):
        if i in removed:
            continue

        dist, inds = kd_tree.query(data[i,:],k=k_max,distance_upper_bound=r)
        for j in inds:
            if j>i:
                removed.add(j)

    kp_filtered = [kp for i,kp in enumerate(kps) if i not in removed]
    descs_filtered = None
    if descs is not None:
        descs = descs[order]
        descs_filtered = np.array([desc for i,desc in enumerate(descs) if i not in removed])
    print('Filtered',len(kp_filtered),'of',N)
    return kp_filtered, descs_filtered

