import numpy as np
import cv2
import detector
from point_correspondence import OpticalFlowTracker, FLANN_Matcher


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1200

# --- Detection/Descriptor Params----
FAST_PARAMS = dict(
    threshold=10,
    nonmaxSuppression=True
)

ORB_PARAMS = dict(
    nfeatures=5000
)

BRIEF_PARAMS = dict(
    use_orientation=True
)

# --- KLT Tracker Params---
LK_PARAMS = dict(winSize=(21, 21),
                maxLevel=6,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# --- FLANN matcher Params---
FLANN_PARAMS = {
    'index_params': dict(algorithm=6,
                         table_number=6,  # 12
                         key_size=12,  # 20
                         multi_probe_level=1),
    'search_params': dict(checks=50)
}

# (For SIFT to work)
# FLANN_PARAMS = {
#     'index_params': dict(algorithm=0,
#                          trees=5),
#     'search_params': dict(checks=50)
# }



class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None

        self.prev_t = None
        self.cur_R = None
        self.cur_t = None

        self.px_ref = None
        self.px_cur = None
        self.des_ref = None
        self.des_cur = None

        self.all_des_cur = None
        self.all_des_ref = None

        self.lines_cur = None
        self.inlier_ratio = 0
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.trueR = np.zeros((3, 3))

        brief_extractor = detector.BRIEF_Extractor(**BRIEF_PARAMS)
        #orb_extractor = detector.ORB(as_extractor=True)
        #sift_extractor = detector.SIFT(as_extractor=True)

        #self.detector = detector.SIFT()
        #self.detector = detector.ORB(des_extractor=brief_extractor, **ORB_PARAMS)
        #self.detector = detector.CenSurE_Detector(des_extractor=brief_extractor)
        #self.detector = detector.AKAZE()
        self.detector = detector.FAST_Detector(brief_extractor, **FAST_PARAMS)
        #self.detector = detector.FAST_Detector(**FAST_PARAMS)

        self.correspondence_method = 'tracking'
        if self.correspondence_method == 'tracking':
            self.point_corr_computer = OpticalFlowTracker(LK_PARAMS)
        else:
            self.point_corr_computer = FLANN_Matcher(FLANN_PARAMS)

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

        r11, r12, r13 = float(ss[0]), float(ss[1]), float(ss[2])
        r21, r22, r23 = float(ss[4]), float(ss[5]), float(ss[6])
        r31, r32, r33 = float(ss[8]), float(ss[9]), float(ss[10])
        self.trueR = np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape((3, 3))

        return np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)

    def get_relative_scale(self):
        """
        Triangulate 3-D points X_(k-1) and X_k from current and previous frame to get relative scale
        :return: relative scale of translation between previous and current frame
        """
        raise NotImplementedError("Relative Scale Method not implemted yet.")

    def process_initial_frame(self):
        self.px_ref = self.detector.get_keypoints(self.new_frame)
        if self.correspondence_method == 'matching':
            self.px_ref, self.des_ref = self.detector.get_descriptors(self.new_frame, self.px_ref)
            self.all_des_ref = self.des_ref

        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.all_px_ref = self.px_ref
        self.frame_stage = STAGE_DEFAULT_FRAME

    def process_frame(self, frame_id):
        if self.correspondence_method == 'tracking':
            self.px_ref, self.px_cur = self.point_corr_computer.get_corresponding_points(img_ref=self.last_frame,
                                                                                         img_cur=self.new_frame,
                                                                                         px_ref=self.px_ref)
        else:
            self.px_cur = self.detector.get_keypoints(self.new_frame)
            self.px_cur, self.des_cur = self.detector.get_descriptors(self.new_frame, self.px_cur)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
            temp_px = self.px_cur
            temp_des = self.des_cur

            self.px_ref, self.px_cur = self.point_corr_computer.get_corresponding_points(px_ref=self.all_px_ref,
                                                                                         px_cur=self.px_cur,
                                                                                         des_ref=self.all_des_ref,
                                                                                         des_cur=self.des_cur)
            self.all_px_ref = temp_px
            self.all_des_ref = temp_des

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)

        self.inlier_ratio = np.sum(mask) / (len(mask) + 1)
        self.lines_cur = cv2.computeCorrespondEpilines(self.px_ref.reshape(-1, 1, 2), 2, E)

        if frame_id == 1:
            _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp,
                                                              mask=mask)
        else:
            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, mask=mask)

            absolute_scale = self.get_absolute_scale(frame_id)
            if absolute_scale > 0.01:
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)

            if self.correspondence_method == 'tracking' and self.px_ref.shape[0] < kMinNumFeature: #or frame_id % 50 == 0:
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
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.process_initial_frame()
        self.last_frame = self.new_frame
