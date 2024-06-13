import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import criterion as c
import torch
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment

# takes in pred_boxes tensor of shape (N, 5) and true_boxes of shape (M, 4), iou threshold, and confidence threshold
def match_boxes1(pred_boxes, true_boxes, iou_thresh, conf_thresh):

    # immediately filters any predictions that do not pass the confidence threshold
    pred_boxes = pred_boxes[pred_boxes[:, 4] >= conf_thresh]

    if pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
        return [], [], pred_boxes.tolist(), true_boxes.tolist()

    # returns IoU tensor of shape (N, M)
    # each element [i, j] in the matrix represents the IoU between the i-th predicted box and the j-th true box
    matrix = iou_matrix(pred_boxes[:, :4], true_boxes)

    # hungarian algorithm maximizes IoU between matched pairs of predicted and true boxes
    # convert IoU matrix to cost matrix by subtracting 1
    cost_matrix = 1 - matrix.detach().cpu().numpy()
    # subtracts the smallest value in each row/col from values in row/col
    # identifies prime 0s, so that exactly one element from each row and each column
    # algorithm ensures 1-to-1 matching and returns lists of indices indicating the optimal matching pairs
    # should match pairs with the highest IoUs and prevent any prediction/truth from being used in more than one match
    pred_indices, true_indices = linear_sum_assignment(cost_matrix)

    # initialize data structures to hold matches, fps, fns
    matched_preds = []
    matched_truths = []
    fp = []
    fn = []

    # sets used to track predictions that have already been matched
    matched_preds_indices = set()
    matched_truths_indices = set()

    for pi, ti in zip(pred_indices, true_indices):
        if matrix[pi, ti] >= iou_thresh:
            matched_preds.append(pred_boxes[pi])
            matched_truths.append(true_boxes[ti])
            matched_preds_indices.add(pi)
            matched_truths_indices.add(ti)

    for pi in range(pred_boxes.shape[0]):
        if pi not in matched_preds_indices:
            fp.append(pred_boxes[pi])
    
    for ti in range(true_boxes.shape[0]):
        if ti not in matched_truths_indices:
            fn.append(true_boxes[ti])

    return matched_preds, matched_truths, fp, fn

def match_boxes2(pred_boxes, true_boxes, iou_thresh, conf_thresh):
    pred_boxes = pred_boxes[pred_boxes[:, 4] >= conf_thresh]
    to_remove = set()
    for i in range(pred_boxes.shape[0]):
        for j in range(i + 1, pred_boxes.shape[0]):
            iou_score = iou(pred_boxes[i, :4], pred_boxes[j, :4])
            if iou_score > 0:
                if pred_boxes[i, 4] > pred_boxes[j, 4]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    
    pred_boxes = [pred_boxes[i] for i in range(pred_boxes.shape[0]) if i not in to_remove]
    if len(pred_boxes) > 0:
        pred_boxes = torch.stack(pred_boxes)
    else:
        pred_boxes = torch.empty((0, 5))

    # matrix = iou_matrix(pred_boxes, true_boxes)
    matrix = iou_matrix(pred_boxes[:, :4], true_boxes)

    matches = []
    # add largest iou pairs to matches
    for i in range(matrix.shape[0]):
        max_iou, max_j = matrix[i].max(0)
        if (max_iou >= iou_thresh):
            matches.append((i, max_j.item(), max_iou.item()))
    
    # sort matches by IoU in descending order
    matches.sort(key=lambda x: x[2], reverse=True)
    matched_preds = []
    matched_truths = []
    used_preds = set()
    used_truths = set()

    for pred_idx, true_idx, iou_score in matches:
        if true_idx in used_truths:
            # if the true box already has a match, continue to the next best prediction
            continue
        if pred_idx not in used_preds and true_idx not in used_truths:
            matched_preds.append(pred_boxes[pred_idx])
            matched_truths.append(true_boxes[true_idx])
            used_preds.add(pred_idx)
            used_truths.add(true_idx)


    # Identify false positives and false negatives
    fp = [pred_boxes[i] for i in range(pred_boxes.shape[0]) if i not in used_preds]
    fn = [true_boxes[i] for i in range(true_boxes.shape[0]) if i not in used_truths]

    return matched_preds, matched_truths, fp, fn

# OLD IMPLEMENTATION: Greedy algorithm, matches pairs based on current best IoU, without regard to solution that could come later
# ensures that truth-prediction matches are 1-to-1

    # # iterate through predicted boxes
    # for pi in range(pred_boxes.shape[0]):

    #     # don't consider predictions that don't meet the confidence threshold
    #     if (pred_boxes[pi][4] < conf_thresh):
    #         continue

    #     # keep track of the best matching true bbox and iou
    #     best_iou = 0
    #     best_ti = -1
        
    #     # iterate through true boxes
    #     for ti in range(true_boxes.shape[0]):

    #         # do not consider true boxes that have already been matched
    #         if ti not in matched_truths_indices:

    #             # retrieve iou from matrix
    #             iou = iou_matrix[pi, ti].item()

    #             # check that the iou meets the threshold and is better than the best_iou
    #             if iou >= iou_thresh and iou > best_iou:
    #                 best_ti = ti
    #                 best_iou = iou
        
    #     # if a true bbox match was found, add the matches, and record the indices
    #     if best_ti != -1:
    #         matched_preds.append(pred_boxes[pi])
    #         matched_truths.append(true_boxes[best_ti])
    #         matched_preds_indices.add(pi)
    #         matched_truths_indices.add(best_ti)

    # for pi in range(pred_boxes.shape[0]):
    #     if pi not in matched_preds_indices:
    #         fp.append(pred_boxes[pi])
    
    # for ti in range(true_boxes.shape[0]):
    #     if ti not in matched_truths_indices:
    #         fn.append(true_boxes[ti])

    # return matched_preds, matched_truths, fp, fn

    # # # matches is a tensor containing pairs of indices where the IoU between predicted and true boxes exceeds the threshold
    # # matches = (iou_matrix >= iou_thresh).nonzero(as_tuple=False)

    # # # nume1() returns the number of elements in the tensor to check if there are any matches
    # # if matches.nume1() > 0:

# OLD IMPLEMENTATION: does not filter duplicates properly

    #     # extracts the unique indices of matched predicted boxes and makes a list
    #     pred_matched = matches[:, 0].unique().tolist()
    #     # extracts the unique indices of matched true boxes and makes a list
    #     true_matched = matches[:, 1].unique().tolist()

    #     for match in matches:
    #         # unpacks indices of predicted and true match
    #         pred_i, true_i = match

    #         # gets associated bboxes
    #         matched_preds.append(pred_boxes[pred_i])
    #         matched_truths.append(true_boxes[true_i])
        
    #     # iterates through the predicted boxes
    #     for i in range(pred_boxes.shape[0]):
    #         # if the predicted box does not have corresponding true match above the threshold, it is a false positive
    #         if i not in pred_matched:
    #             fp.append(pred_boxes[i])
        
    #     # iterates through the true boxes
    #     for i in range(true_boxes.shape[0]):
    #         # if the true box does not have corresponding predicted match above the threshold, it is a false negative
    #         if i not in true_matched:
    #             fn.append(true_boxes[i])
    # else:
    #     # in the case no matches are found, all predictions are false positives, and all truths are false negatives
    #     fp = pred_boxes
    #     fn = true_boxes
    
    # return matched_preds, matched_truths, fp, fn

# precision quantifies the accuracy of good predictions made by the model
# takes in the number of true positives and the number of false negatives
def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# recall quantifies the completeness of the objects detected in the image
# takes in the number of true positives and the number of false negatives
def recall(tp, fn):
    return tp / (tp + fn) if (tp + np) > 0 else 0.0

# f1-score gives balanced measure of model's performance based on precision and recall
# takes in number of true positives, number of false negatives, and number of false positives
def f1_score(tp, fn, fp):
    precision = precision(tp, fp)
    recall = recall(tp, fn)
    return (2 * precision * recall) / (precision + recall)

# precision-recall curve shows trade-off between precision and recall at different iou thresholds
# takes in predictions and ground-truths
def precision_recall_curve(pred_boxes, true_boxes):
    precisions = []
    recalls = []

    # step thresholds by 0.1
    thresholds = np.arrange(start=0.0, stop=1.1, step=0.1)
    fixed_conf_thresh = .5

    # calculate the confusian matrix at the lowest iou threshold
    # match_boxes has to be correct
    matched_preds, matched_truths, fp, fn = match_boxes1(pred_boxes, true_boxes, thresh, fixed_conf_thresh)
    init_tp = len(matched_preds)
    init_fp = len(fp)
    init_fn = len(fn)

    for thresh in thresholds:
        
        # calculate new tp according to new threshold
        curr_tp = 0
        for p, t in zip(matched_preds, matched_truths):
            if (iou(p[:4], t) >= thresh):
                curr_tp += 1
        
        # recalculate fp and fn based on changed tp
        new_fp = init_fp + (init_tp - curr_tp)
        new_fn = init_fn + (init_tp - curr_tp)

        # calculate precison and recall using tp, fp, fn
        p = precision(curr_tp, new_fp)
        r = recall(curr_tp, new_fn)

        # add the new data points
        precisions.append(p)
        recalls.append(r)

    # return points on curve
    return precisions, recalls

# AP is the weighted-sum of precisions at each threshold where the weight is increase in recall
# takes in predicted and true bboxes
def AP(pred_boxes, true_boxes):

    # get precisions and recalls from the predicted and true bboxes
    precisions, recalls = precision_recall_curve(pred_boxes, true_boxes)

    # ensures curve ends at precision = 1 when recall = 0
    precisions.append(1)
    # ensures recall curve starts at 0
    recalls.append(0)
    
    # converts to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # computes the difference in recall between successive thresholds
    # multiplies delta-recall by precsion value at the starting point of recall segment (left reimann-sum)
    # sums weighted precision values to get the AUC of precision-recall curve
    AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])

    return AP

# useless until classes are considered
# calculates the mean of the APs for each class
# 
def mAP(pred_boxes, true_boxes):
    return None

def iou(pred_boxes, true_boxes):
    # if 1D tensor, converts shape from [4] to [1, 4]
    # can now pass in 1D and 2D tensors
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if true_boxes.dim() == 1:
        true_boxes = true_boxes.unsqueeze(0)

    pred_boxes = convert_boxes_to_x1y1x2y2(pred_boxes)
    true_boxes = convert_boxes_to_x1y1x2y2(true_boxes)

    return ops.box_iou(pred_boxes, true_boxes)

def giou(pred_boxes, true_boxes):
    # if 1D tensor, converts shape from [4] to [1, 4]
    # can now pass in 1D and 2D tensors
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if true_boxes.dim() == 1:
        true_boxes = true_boxes.unsqueeze(0)

    # convert boxes from [x, y, w, h] to [x1. y1, x2, y2]
    pred_boxes_x1y1x2y2 = convert_boxes_to_x1y1x2y2(pred_boxes) 
    true_boxes_x1y1x2y2 = convert_boxes_to_x1y1x2y2(true_boxes)

     # normalize predicted boxes to ensure x2 > x1 and y2 > y1
    pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2 = normalize_boxes(pred_boxes_x1y1x2y2)
    # set true boxes
    true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2 = true_boxes_x1y1x2y2[:, 0], true_boxes_x1y1x2y2[:, 1], true_boxes_x1y1x2y2[:, 2], true_boxes_x1y1x2y2[:, 3]

    # calculate predicted box area based on normalized predicted boxes
    pred_area = calc_box_area(pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2)
    # calculate ground-truth box area without normalization
    true_area = calc_box_area(true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2)
    
    # calculate the intersection coordinates
    inter_x1 = torch.max(pred_boxes_x1, true_boxes_x1)
    inter_y1 = torch.max(pred_boxes_y1, true_boxes_y1)
    inter_x2 = torch.min(pred_boxes_x2, true_boxes_x2)
    inter_y2 = torch.min(pred_boxes_y2, true_boxes_y2)
    
    # calculate the intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # calculate the union area
    union_area = pred_area + true_area - inter_area
    
    # calculate IoU
    iou = inter_area / union_area
    
    # calculate the coordinates of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes_x1, true_boxes_x1)
    enclose_y1 = torch.min(pred_boxes_y1, true_boxes_y1)
    enclose_x2 = torch.max(pred_boxes_x2, true_boxes_x2)
    enclose_y2 = torch.max(pred_boxes_y2, true_boxes_y2)
    
    # calculate the area of the smallest enclosing box
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # calculate GIoU
    giou = iou - (enclose_area - union_area) / enclose_area
    
    return giou


def diou(pred_boxes, true_boxes):
    # if 1D tensor, converts shape from [4] to [1, 4]
    # can now pass in 1D and 2D tensors
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if true_boxes.dim() == 1:
        true_boxes = true_boxes.unsqueeze(0)

    # convert boxes from [x, y, w, h] to [x1. y1, x2, y2]
    pred_boxes_x1y1x2y2 = convert_boxes_to_x1y1x2y2(pred_boxes)
    true_boxes_x1y1x2y2 = convert_boxes_to_x1y1x2y2(true_boxes)

    # normalize predicted boxes to ensure x2 > x1 and y2 > y1
    pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2 = normalize_boxes(pred_boxes_x1y1x2y2)
    # set true boxes
    true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2 = true_boxes_x1y1x2y2[:, 0], true_boxes_x1y1x2y2[:, 1], true_boxes_x1y1x2y2[:, 2], true_boxes_x1y1x2y2[:, 3]
    # true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2 = normalize_boxes(true_boxes)
    
    # calculate predicted box area based on normalized predicted boxes
    pred_area = calc_box_area(pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2)
    # calculate ground-truth box area without normalization
    true_area = calc_box_area(true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2)
    
    # calculate the intersection coordinates
    inter_x1 = torch.max(pred_boxes_x1, true_boxes_x1)
    inter_y1 = torch.max(pred_boxes_y1, true_boxes_y1)
    inter_x2 = torch.min(pred_boxes_x2, true_boxes_x2)
    inter_y2 = torch.min(pred_boxes_y2, true_boxes_y2)
    
    # calculate the intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # calculate the union area
    union_area = pred_area + true_area - inter_area
    
    # calculate IoU
    iou = inter_area / union_area
    # print(f"IoU: {iou}")

    # calculate the coordinates of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes_x1, true_boxes_x1)
    enclose_y1 = torch.min(pred_boxes_y1, true_boxes_y1)
    enclose_x2 = torch.max(pred_boxes_x2, true_boxes_x2)
    enclose_y2 = torch.max(pred_boxes_y2, true_boxes_y2)
    
    # calculate the area of the smallest enclosing box
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    # print(f"c^2: {enclose_diagonal}")

    # calculate box centers for predicted and ground-truth
    pred_c1, pred_c2 = calc_box_center(pred_boxes)
    # [tensor.detach().cpu().numpy() for tensor in calc_box_center(pred_boxes)]
    true_c1, true_c2 = calc_box_center(true_boxes)
    # print(f"Predicted center: {pred_c1, pred_c2}")
    # print(f"True center: {true_c1, true_c2}")
    # calculate the euclidian distance between the predicted center and the ground-truth center
    center_distance_squared = (pred_c1 - true_c1) ** 2 + (pred_c2 - true_c2) ** 2
    # print(f"d^2: {center_distance_squared}")

    # calculate DIoU as IoU - (d^2) / (C^2)
    diou = iou - (center_distance_squared / enclose_diagonal)
    # print(f"DIoU: {diou}")

    return diou

# takes in boxes in format [x, y, w, h] and converts to [x1, y1, x2, y2]
def convert_boxes_to_x1y1x2y2(boxes):
    # adds w, h to x, y coordinate to obtain x2, y2
    converted_boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], dim=1)
    return converted_boxes

# takes in boxes in format [x1, y1, x2, y2] and ensures that x2 > x1, and y2 > y1
def normalize_boxes(boxes):
    boxes_x1 = torch.min(boxes[:, 0], boxes[:, 2])
    boxes_y1 = torch.min(boxes[:, 1], boxes[:, 3])
    boxes_x2 = torch.max(boxes[:, 0], boxes[:, 2])
    boxes_y2 = torch.max(boxes[:, 1], boxes[:, 3])
    return boxes_x1, boxes_y1, boxes_x2, boxes_y2

# calculates box area based on bottom-left and top-right bbox coordinates
def calc_box_area(x1, y1, x2, y2):
    return torch.abs(x2 - x1) * torch.abs(y2 - y1)

# takes in a box tensor of format [x, y, w, h]
def calc_box_center(boxes):
    c1 = boxes[:, 0] + (boxes[:, 2] / 2.0)
    c2 = boxes[:, 1] + (boxes[:, 3] / 2.0)
    return c1, c2

def iou_matrix(pred_boxes, true_boxes, iou_type="iou"):
    if iou_type == "diou":
        return torch.tensor([[diou(pred[:4], true) for true in true_boxes] for pred in pred_boxes])
    elif iou_type == "giou":
        return torch.tensor([[giou(pred[:4], true) for true in true_boxes] for pred in pred_boxes])
    else:
        return torch.tensor([[iou(pred[:4], true) for true in true_boxes] for pred in pred_boxes])
    



