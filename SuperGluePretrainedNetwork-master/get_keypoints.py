from pathlib import Path
import cv2
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)


# Superpoint Params
NMS_RADIUS = 4
KEYPOINT_THRESHOLD = 0.005
MAX_KEYPOINTS = -1

# Superglue Params
SUPERGLUE = "indoor"
SINKHORN_ITERATIONS = 20
MATCH_THRESHOLD = 0.2

fromMain = False

class Shaper:
    def __init__(self, shape):
        self.shape = (0, 0, shape[0], shape[1])

class SuperGlue:
    def __init__(self):
        self.config = {
            'superpoint': {
                'nms_radius': NMS_RADIUS,
                'keypoint_threshold': KEYPOINT_THRESHOLD,
                'max_keypoints': MAX_KEYPOINTS
            },
            'superglue': {
                'weights': SUPERGLUE,
                'sinkhorn_iterations': SINKHORN_ITERATIONS,
                'match_threshold': MATCH_THRESHOLD,
            }
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matching = Matching(self.config).eval().to(self.device)

    #returns a dictionary of keypoints, scores, descriptors
    def getKeypointsAndDescriptors(self, frame):
        if not fromMain:
            frame = frame[0]
        frame_tensor = frame2tensor(frame, self.device)
        last_data = self.matching.superpoint({'image': frame_tensor})
        kp_list = last_data["keypoints"][0].tolist()
        for i in range(len(kp_list)):
            for j in range(2):
                kp_list[i][j] = int(kp_list[i][j])
        return [kp_list, last_data]
    
    def getMatches(self, im1_data, im2_data, shape):
        im1_data = {**{k+'0': v for k, v in im1_data.items()}}
        im2_data = {**{k+'1': v for k, v in im2_data.items()}}
        data = {**im1_data, **im2_data, "image0": Shaper(shape), "image1": Shaper(shape)}
        return self.matching(data)


if __name__ == "__main__":
    fromMain = True
    grayim1 = cv2.imread("/home/amons/ORB_SLAM3/SuperGluePretrainedNetwork-master/assets/freiburg_sequence/1341847980.722988.png", 0)
    grayim2 = cv2.imread("/home/amons/ORB_SLAM3/SuperGluePretrainedNetwork-master/assets/freiburg_sequence/1341847981.726650.png", 0)

    sp = SuperGlue()

    im1_data = sp.getKeypointsAndDescriptors(grayim1)[1]
    im2_data = sp.getKeypointsAndDescriptors(grayim2)[1]
    print(im2_data["descriptors"][0].shape)
    print(im2_data["keypoints"][0].shape)

    matches = sp.getMatches(im1_data, im2_data, (480, 640))

    # print(matches)
