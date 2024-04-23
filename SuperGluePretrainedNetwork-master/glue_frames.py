#! /usr/bin/env python3


from pathlib import Path
import cv2
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

# Superpoint Params
NMS_RADIUS = 4
KEYPOINT_THRESHOLD = 0.005
MAX_KEYPOINTS = -1

# Superglue Params
SUPERGLUE = "indoor"
SINKHORN_ITERATIONS = 20
MATCH_THRESHOLD = 0.2

# Other Params
INPUT_PATH = "assets/freiburg_sequence/" # 0 = camera
OUTPUT_PATH = "hello_hoe.txt"
RESIZE = [640, 480]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
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
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(INPUT_PATH, RESIZE, 1,
                       ['*.png', '*.jpg', '*.jpeg'], 10000000)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor

    
    f = open(OUTPUT_PATH, 'w')
    np.set_printoptions(suppress=True)
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            break

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})

        f.write(np.array2string(last_data['keypoints0'][0].cpu().numpy()[:,0], separator=", ", max_line_width=np.inf, precision=2)[1:-1] + "\n")
        f.write(np.array2string(last_data['keypoints0'][0].cpu().numpy()[:,1], separator=", ", max_line_width=np.inf, precision=2)[1:-1] + "\n")

        f.write(np.array2string(pred['keypoints1'][0].cpu().numpy()[:,0], separator=", ", max_line_width=np.inf, precision=2)[1:-1] + "\n")
        f.write(np.array2string(pred['keypoints1'][0].cpu().numpy()[:,1], separator=", ", max_line_width=np.inf, precision=2)[1:-1] + "\n")

        f.write(np.array2string(pred['matches0'][0].cpu().numpy(), separator=", ", max_line_width=np.inf, precision=2)[1:-1] + "\n")
        f.write(np.array2string(pred['matching_scores0'][0].cpu().numpy(), separator=", ", max_line_width=np.inf, precision=2)[1:-1] + "\n")

        last_data = matching.superpoint({'image': frame_tensor})
        last_data = {k+'0': last_data[k] for k in keys}
        last_data['image0'] = frame_tensor


    f.close()
    cv2.destroyAllWindows()
    vs.cleanup()
