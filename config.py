from sys import argv
import detector
import cv2


class DefaultConfig:
    def __init__(self, configuration):
        defaults = configuration["Defaults"]


class Config(DefaultConfig):
    def __init__(self, configuration):
        super().__init__(configuration)
        self.detector = None
        self.extractor = None
        self.defaults = configuration["Defaults"]
        self.experiment_index = str(argv[1]).upper()
        self.experiment = configuration[self.experiment_index]

        try:
            self.name = self.experiment['name']
            self.correspondence_method = self.experiment["correspondence_method"]
            self.detector_params = self.experiment["detector_params"]
            self.extractor_params = self.experiment["extractor_params"]
            self.k_min_features = self.experiment["k_min_features"]
            self.flann_params = self.experiment["flann_params"]
            self.parse_lk_params(self.experiment["lk_params"])
            self.parse_detector(self.experiment["detector"])
            self.parse_extractor(self.experiment["extractor"])
        except AttributeError:
            pass


    def parse_lk_params(self, lk_parmas_dict):
        try:
            for key, values in lk_parmas_dict.items():
                if key.upper() == "CRITERIA":
                    criteria_one = values[0]
                    criteria_two = values[1]
                    max_iterations = values[2]
                    epsilon = values[3]
                    lk_parmas_dict["criteria"] = (criteria_one | criteria_two, max_iterations, epsilon)
            self.lk_params = lk_parmas_dict
        except AttributeError:
            # raise AttributeError("No LK Params.")
            pass

    def parse_detector(self, detector_string):
        if detector_string.upper() == "FAST":
            self.detector = detector.FAST_Detector(**self.detector_params)
        elif detector_string.upper() == "HARRIS":
            self.detector = detector.HarrisDetector()
        elif detector_string.upper() == "CENSURE":
            self.detector = detector.CenSurE_Detector(**self.detector_params)
        elif detector_string.upper() == "SIFT":
            self.detector = detector.SIFT(**self.detector_params)
        elif detector_string.upper() == "SURF":
            self.detector = detector.SURF(**self.detector_params)
        elif detector_string.upper() == "ORB":
            self.detector = detector.ORB(**self.detector_params)
        elif detector_string.upper() == "AKAZE":
            self.detector = detector.AKAZE(**self.detector_params)
        else:
            raise ModuleNotFoundError(f"No detector <{self.detector}> found.")

    def parse_extractor(self, extractor_string):
        if extractor_string.upper() == "SIFT":
            self.extractor = detector.SIFT(**self.extractor_params)
        elif extractor_string.upper() == "SURF":
            self.extractor = detector.SURF(**self.extractor_params)
        elif extractor_string.upper() == "ORB":
            self.extractor = detector.ORB(**self.extractor_params)
        elif extractor_string.upper() == "AKAZE":
            self.extractor = detector.AKAZE(**self.extractor_params)
        elif extractor_string.upper() == "BRIEF":
            self.extractor = detector.BRIEF_Extractor(**self.extractor_params)
        else:
            raise ModuleNotFoundError(f"No descriptor extractor <{self.detector}> found.")
