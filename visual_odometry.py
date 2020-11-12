import numpy as np
import cv2
import detector


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1200

lk_params = dict(winSize=(21, 21),
                maxLevel=6,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# lk_params = dict(winSize=(21, 21),
#                  maxLevel=5,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
#                  #flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
#                  minEigThreshold=1e-5)

# Params for ShiTomasi corner detection
# feature_params = dict(maxCorners=300,
#                      qualityLevel=0.03,
#                      minDistance=50,
#                      blockSize=10)


def feature_tracking(image_ref, image_cur, px_ref):
    """
    Finding point correspondences between previous and current image using Lucas-Kanade Optical Flow

    :param image_ref: previous image
    :param image_cur: current images
    :param px_ref: key points detected/tracked in previous image
    :return: point correspondences between previous and current image
    """

    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  # shape: [k,2] [k,1] [k,1]
    st = st.reshape(st.shape[0])

    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    # idxs = np.random.choice(kp1.shape[0], min(kp1.shape[0], 10000), replace=False)
    # kp1 = kp1[idxs]
    # kp2 = kp2[idxs]

    print(kp1.shape)
    print(kp2.shape)
    print(type(kp1[0][0]))
    return kp1, kp2


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
    matches = flann.knnMatch(des_ref, des_cur, k=2)  # k=2 to apply ratio test

    # Ratio Test as per Lowe's paper
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)

    # discard bad matches, ratio test as per Lowe's paper
    good_matches = list(filter(lambda x: x[0].distance < 0.7 * x[1].distance,
                          matches))
    good_matches = [good_matches[i][0] for i in range(len(good_matches))]
    # print(good_matches)

    kp1 = []
    kp2 = []
    for mat in good_matches:
        kp1.append(px_ref[mat.queryIdx])
        kp2.append(px_cur[mat.trainIdx])

    kp1 = np.array(kp1)
    kp2 = np.array(kp2)

    return kp1, kp2


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.des_ref = None
        self.des_cur = None
        self.inlier_ratio = 0
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.correspondance_method = 'matching'
        #brief_extractor = detector.BRIEF_extractor()
        #self.detector = detector.SIFT(brief_extractor)
        self.detector = detector.ORB()
        #self.detector = detector.FAST_Detector()
        #self.detector = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        #self.detector = detector.Harris()
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
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)

    def get_relative_scale(self):
        """
        Triangulate 3-D points X_(k-1) and X_k from current and previous frame to get relative scale
        :return: relative scale of translation between previous and current frame
        """
        point_distances_prev = np.array([np.sqrt((self.px_ref[i + 1][0] - self.px_ref[i][0])**2) for i in range(len(self.px_ref) - 1)])
        point_distances_cur = np.array([np.sqrt((self.px_cur[i + 1][0] - self.px_cur[i][0])**2) for i in range(len(self.px_cur) - 1)])
        # z_change = np.linalg.norm(feature_distances_new, feature_distances_old)
        rel_scale = np.median(point_distances_prev / point_distances_cur)
        return rel_scale

    def process_initial_frame(self):
        self.px_ref = self.detector.get_keypoints(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        if self.correspondance_method == 'matching':
            self.des_ref = self.detector.get_descriptors(self.new_frame, self.px_ref)

        self.frame_stage = STAGE_SECOND_FRAME

    def process_second_frame(self):
        if self.correspondance_method == 'tracking':
            self.px_ref, self.px_cur = feature_tracking(self.last_frame, self.new_frame, self.px_ref)
        else:
            self.px_cur = self.detector.get_keypoints(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            self.des_cur = self.detector.get_descriptors(self.new_frame, self.px_cur)
            self.px_ref, self.px_cur = feature_matching(self.px_ref, self.px_cur, self.des_ref, self.des_cur)

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        self.inlier_ratio = np.sum(mask) / (len(mask)+1)

        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, mask=mask)

        self.px_ref = self.px_cur
        self.des_ref = self.des_cur
        self.frame_stage = STAGE_DEFAULT_FRAME

    def process_frame(self, frame_id):
        if self.correspondance_method == 'tracking':
            self.px_ref, self.px_cur = feature_tracking(self.last_frame, self.new_frame, self.px_ref)
        else:
            self.px_cur = self.detector.get_keypoints(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            self.des_cur = self.detector.get_descriptors(self.new_frame, self.px_cur)
            self.px_ref, self.px_cur = feature_matching(self.px_ref, self.px_cur, self.des_ref, self.des_cur)

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        self.inlier_ratio = np.sum(mask) / (len(mask) + 1)

        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, mask=mask)

        # relative_scale = self.get_relative_scale()
        absolute_scale = self.get_absolute_scale(frame_id)

        if absolute_scale > 0.01:
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)

        if self.correspondance_method == 'tracking' and self.px_ref.shape[0] < kMinNumFeature: #or frame_id % 50 == 0:
            self.px_cur = self.detector.get_keypoints(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)

        self.px_ref = self.px_cur
        self.des_ref = self.des_cur

    def update(self, img, frame_id):
        assert (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width), \
            "Frame: provided image has not the same size as the camera model or image is not grayscale"

        self.new_frame = img
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.process_frame(frame_id)
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.process_second_frame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.process_initial_frame()
        self.last_frame = self.new_frame
