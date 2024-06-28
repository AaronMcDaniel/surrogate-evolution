import torch
import numpy as np
import pytest
import os
import sys
import torchvision.ops as ops
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
import utils as u

# # test 1: Many predictions, fewer true boxes
# pred_boxes = torch.tensor([
# [0, 0, 10, 10, 0.9],
# [1, 1, 9, 9, 0.85],
# [15, 15, 25, 25, 0.95],
# [30, 30, 40, 40, 0.8],
# [35, 35, 45, 45, 0.75],
# [30, 30, 40, 40, 0.7],
# [50, 50, 60, 60, 0.6],
# [0, 0, 5, 5, 0.95],
# [1, 1, 5, 5, 0.6]
# ])
# true_boxes = torch.tensor([
#     [0, 0, 10, 10],
#     [30, 30, 40, 40],
#     [50, 50, 60, 60],
#     [0, 0, 5, 5]
# ])

# iou_matrix = torch.tensor([
#         [1.0000, 0.8100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.2500],
#         [0.0000, 0.0000, 0.0471, 1.0000, 0.5104, 1.0000, 0.0833, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0833, 0.1905, 0.0833, 1.0000, 0.0000, 0.0000], 
#         [0.2500, 0.1778, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.4706]
# ])

# iou_thresh = 0
# conf_thresh = .6

# matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh)

# # test 2: Multiple predictions, one true box
# pred_boxes = torch.tensor([
#     [0, 0, 10, 10, 0.9],
#     [1, 1, 9, 9, 0.85],
#     [2, 2, 8, 8, 0.95],
#     [3, 3, 7, 7, 0.8]
# ], dtype=torch.float32)

# true_boxes = torch.tensor([
#     [0, 0, 10, 10]
# ], dtype=torch.float32)

# iou_thresh = 0.9
# conf_thresh = 0.6

# matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh)

# # test 3: One prediction, multiple true boxes
# pred_boxes = torch.tensor([
#         [5, 5, 10, 10, 0.95]
#     ], dtype=torch.float32)

# true_boxes = torch.tensor([
#     [5, 5, 10, 10],
#     [6, 6, 8, 8],
#     [7, 7, 6, 6]
# ], dtype=torch.float32)

# iou_thresh = 0.2
# conf_thresh = 0.5

# matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh)

# test 4: Multiple predictions with high IoUs, true boxes with high IoUs
# changing IoU threshold here greatly varies the outcome
pred_boxes = torch.tensor([
        [0, 0, 10, 10, 0.9],
        [1, 1, 9, 9, 0.85],
        [2, 2, 8, 8, 0.95]
    ], dtype=torch.float32)

true_boxes = torch.tensor([
    [0, 0, 10, 10],
    [1, 1, 9, 9]
], dtype=torch.float32)

iou_thresh = 0.1  # High IoU threshold
conf_thresh = 0.7

matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "train")

# # test 5: complex matching scenario
# pred_boxes = torch.tensor([
#         [0, 0, 10, 10, 0.9],   # High confidence, should match
#         [1, 1, 10, 10, 0.85],  # High confidence, but overlaps with the first box
#         [5, 5, 15, 15, 0.7],   # Lower confidence, overlaps with the above two
#         [20, 20, 10, 10, 0.95],# High confidence, should match
#         [21, 21, 10, 10, 0.9], # High confidence, but overlaps with the above one
#         [22, 22, 10, 10, 0.75],# Lower confidence, overlaps with the above two
#         [50, 50, 10, 10, 0.6], # Below confidence threshold, should be ignored
#         [70, 70, 10, 10, 0.92],# High confidence, should match
#         [71, 71, 10, 10, 0.88],# High confidence, but overlaps with the above one
#         [72, 72, 10, 10, 0.8], # Lower confidence, overlaps with the above two
#     ], dtype=torch.float32)

# true_boxes = torch.tensor([
#     [0, 0, 10, 10],
#     [20, 20, 10, 10],
#     [70, 70, 10, 10],
#     [71, 71, 10, 10],
#     [72, 72, 10, 10]
# ], dtype=torch.float32)

# iou_thresh = 0.5
# conf_thresh = 0.7

# final_matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh)

# expected_matches = {
#     0: (0, 1.0),  # First true box matches with first pred box (highest confidence)
#     1: (3, 1.0),  # Second true box matches with fourth pred box
#     2: (7, 1.0),  # Third true box matches with seventh pred box
#     3: (8, 0.81), # Fourth true box matches with eighth pred box
#     4: (9, 0.69)  # Fifth true box matches with ninth pred box
# }

# expected_fp = [
#     [1, 1, 10, 10, 0.85],
#     [5, 5, 15, 15, 0.7],
#     [21, 21, 10, 10, 0.9],
#     [22, 22, 10, 10, 0.75]
# ]

# expected_fn = []