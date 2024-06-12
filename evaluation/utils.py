import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import criterion as c
import torch
import torchvision.ops as ops

# takes in pred_boxes tensor of shape (N, 4) and true_boxes of shape (M, 4), and threshold for detection
def match_boxes(pred_boxes, true_boxes, det_thresh):
    # returns IoU tensor of shape (N, M)
    # each element [i, j] in the matrix represents the IoU between the i-th predicted box and the j-th true box
    iou_matrix = ops.box_iou(pred_boxes, true_boxes)

    # hold the matched prediction/true bboxes, equivalent to true positive (lists must be same length)
    matched_preds = []
    matched_truths = []
    # hold the false positive bboxes
    fp = []
    # hold the false negative bboxes
    fn = []

    # matches is a tensor containing pairs of indices where the IoU between predicted and true boxes exceeds the threshold
    matches = (iou_matrix >= det_thresh).nonzero(as_tuple=False)

    # nume1() returns the number of elements in the tensor to check if there are any matches
    if matches.nume1() > 0:

        # extracts the unique indices of matched predicted boxes and makes a list
        pred_matched = matches[:, 0].unique().tolist()
        # extracts the unique indices of matched true boxes and makes a list
        true_matched = matches[:, 1].unique().tolist()

        for match in matches:
            # unpacks indices of predicted and true match
            pred_i, true_i = match

            # gets associated bboxes
            matched_preds.append(pred_boxes[pred_i])
            matched_truths.append(true_boxes[true_i])
        
        # iterates through the predicted boxes
        for i in range(pred_boxes.shape[0]):
            # if the predicted box does not have corresponding true match above the threshold, it is a false positive
            if i not in pred_matched:
                fp.append(pred_boxes[i])
        
        # iterates through the true boxes
        for i in range(true_boxes.shape[0]):
            # if the true box does not have corresponding predicted match above the threshold, it is a false negative
            if i not in true_matched:
                fn.append(true_boxes[i])
    else:
        # in the case no matches are found, all predictions are false positives, and all truths are false negatives
        fp = pred_boxes
        fn = true_boxes
    
    return matched_preds, matched_truths, fp, fn

# precision quantifies the accuracy of good predictions made by the model
# takes in the number of true positives and the number of false negatives
def precision(tp, fp):
    return tp / (tp + fp)

# recall quantifies the completeness of the objects detected in the image
# takes in the number of true positives and the number of false negatives
def recall(tp, fn):
    return tp / (tp + fn)

# f1-score gives balanced measure of model's performance based on precision and recall
# takes in number of true positives, number of false negatives, and number of false positives
def f1_score(tp, fn, fp):
    precision = precision(tp, fp)
    recall = recall(tp, fn)
    return (2 * precision * recall) / (precision + recall)

def mAP()
