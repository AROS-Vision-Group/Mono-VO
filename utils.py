import os
import cv2
import glob
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.kdtree import KDTree


def get_img_id(i):
    num_digits = len(str(i))
    id = "0" * (4 - num_digits)
    id += str(i)
    return id


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def plot_inlier_ratio(ratios, save=True, save_path="", show=True):
    plt.plot([i for i in range(len(ratios))], ratios, color='blue')
    plt.title('RANSAC inlier ratio across frames')
    plt.xlabel('frame #')
    plt.ylabel('Inlier Ratio')
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def plot_translation_error(drift, title="", save_path="", save=True, show=True):
    plt.plot([i for i in range(len(drift))], drift, color='blue')
    plt.title(title)
    plt.xlabel('frame #')
    plt.ylabel('Translation Error (m)')
    if save:
        plt.savefig(f'{save_path}', bbox_inches='tight')
    if show:
        plt.show()


def plot_rotation_erros(rot_errors, title="", save_path="", save=True, show=True):
    plt.plot([i for i in range(len(rot_errors))], rot_errors, color='blue')
    plt.title(title)
    plt.xlabel('frame #')
    plt.ylabel('Rotation Error (deg)')
    if save:
        plt.savefig(f'{save_path}', bbox_inches='tight')
    if show:
        plt.show()


def plot_position_error(x_errors, y_errors, z_errors, save_path="", save=True, show=True):
    plt.plot([i for i in range(len(x_errors))], x_errors, label='x', color='red')
    plt.plot([i for i in range(len(y_errors))], y_errors, label='y', color='green')
    plt.plot([i for i in range(len(z_errors))], z_errors, label='z', color='blue')
    plt.title('Position Error across frames')
    plt.xlabel('frame #')
    plt.ylabel('Position Error [m]')
    plt.legend()
    if save:
        plt.savefig(f'{save_path}', bbox_inches='tight')
    if show:
        plt.show()


def plot_orientation_error(yaw_errors, pitch_errors, roll_errors, save_path="", save=True, show=True):
    plt.plot([i for i in range(len(yaw_errors))], yaw_errors, label='yaw', color='red')
    plt.plot([i for i in range(len(pitch_errors))], pitch_errors, label='pitch', color='green')
    plt.plot([i for i in range(len(roll_errors))], roll_errors, label='roll', color='blue')
    plt.title('Orientation Error across frames')
    plt.xlabel('frame #')
    plt.ylabel('Orientation Error [deg]')
    plt.legend()
    if save:
        plt.savefig(f'{save_path}', bbox_inches='tight')
    if show:
        plt.show()


def plot_orientation_angle(theta_true, theta_hat, angle_name, title="", save_path="", save=True, show=True):
    plt.plot([i for i in range(len(theta_true))], theta_true, label=f'{angle_name}_true', color='green')
    plt.plot([i for i in range(len(theta_hat))], theta_hat, label=f'{angle_name}_estimated', color='red')
    plt.title(title)
    plt.xlabel('frame #')
    plt.ylabel('Orientation Angle [deg]')
    plt.legend()
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def plot_2d_traj(traj, save=True, save_path="", show=True):
    # fig = plt.figure()
    plt.imshow(traj)
    # plt.plot(xs, zs, label='Estimated', color='red')
    # plt.plot(true_xs, true_zs, label='Ground Truth', color='green')
    plt.title('2D trajectory (from above)')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.legend()
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def plot_3d_traj(xs, ys, zs, true_xs, true_ys, true_zs, save=True, save_path="", show=True):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='Estimated', color='red')
    ax.plot(true_xs, true_ys, true_zs, label='Ground Truth', color='green')
    ax.set_title('3D trajectory')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def compute_perpendicular_distance(points, a, b, c):
    perp_dist = []
    for point in points:
        # print(point)
        x, y = point[0], point[1]
        perp_dist.append(np.abs((a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)))
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
    return cv2.LUT(img, look_up_table)


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
            processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 7, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=(3, 3))
            processed_img = cv2.dilate(processed_img, kernel, iterations=3)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            processed_img = cv2.GaussianBlur(processed_img, (7, 7), 0)
            processed_img = cv2.adaptiveThreshold(processed_img,
                                                  maxValue=255,
                                                  adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  thresholdType=cv2.THRESH_BINARY,
                                                  blockSize=7,
                                                  C=2)
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

        dist, inds = kd_tree.query(data[i, :], k=k_max, distance_upper_bound=r)
        for j in inds:
            if j > i:
                removed.add(j)

    kp_filtered = [kp for i, kp in enumerate(kps) if i not in removed]
    descs_filtered = None
    if descs is not None:
        descs = descs[order]
        descs_filtered = np.array([desc for i, desc in enumerate(descs) if i not in removed])
    print('Filtered', len(kp_filtered), 'of', N)
    return kp_filtered, descs_filtered


def init_logger(filepath):
    """
    Initialize logger settings
    :return: None
    """
    from datetime import datetime
    NOW = datetime.now().strftime("%m%d%y_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)-5.5s] %(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(f"{filepath}log_{NOW}.txt", mode="w"),
            logging.StreamHandler()
        ])

def pretty_log(config, results):
    if config.correspondence_method == 'matching':
        tracker = 'FLANN matcher'
        tracker_params = config.flann_params
    else:
        tracker = 'Lucas-Kanade Optical Flow'
        tracker_params = config.lk_params

    output =    f"\nExperiment:\t{results['name']}" \
                f"\nAbsolute Trajectory Error(ATE)[m]:\t{results['ate']:.4f}" \
                f"\nAbsolute Orientation Error(AOE)[deg]:\t{results['aoe']:.4f}" \
                f"\nRelative Trajectory Error(RTE)[m]:\t{results['rte']:.4f}" \
                f"\nRelative Rotation Error(RRE)[deg]:\t{results['rre']:.4f}" \
                f"\nRANSAC inlier ratio:\t{results['inlier_ratio']:.4f}" \
                f"\nRuntime:\t{results['runtime']:.4f}" \
                f"\n\nParams for {config.detector} detector:" \
                f"\n{config.detector_params}"

    if config.extractor is not None:
        output += f"\nParams for {config.extractor} extractor: " \
                  f"\n{config.extractor_params}"

    output +=   f"\nParams for {tracker}:" \
                f"\n{tracker_params}"

    logging.info(output)
