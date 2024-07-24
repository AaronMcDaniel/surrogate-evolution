import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torchvision.ops as ops
import heapq
import math
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# used to match predictions and truths based on thresholds and compute the confusion matrix
# note: could also be implemented with a balanced BST (from sortedcontainers) instead of heap
def match_boxes(pred_boxes, true_boxes, iou_thresh=0.3, conf_thresh=0.5, mode="val", iou_type="ciou"):
    if pred_boxes.dim() != 2 or pred_boxes.size(1) < 5:
        raise ValueError("pred_boxes should be a 2D tensor with at least 5 columns")
    
    # train mode is used for loss calculation and factors in all predictions
    if mode =="train":
        matrix = iou_matrix(pred_boxes[:, :4], true_boxes, iou_type)
        matches = {}

        if matrix.numel() == 0:
            return matches

        # for train, only care about what the max iou of each column is
        # each prediction has a paired truth, we don't care about duplicates
        for i in range(matrix.shape[1]):
            # take the maximum value of each column
            max_iou, max_j = matrix[:, i].max(0)
            matches[true_boxes[max_j]] = (pred_boxes[i], max_iou)

        return matches

    # validation mode filters predictions based on confidence, NMS, iou threshold, ect.
    elif mode == "val":
        # remove any predictions that fall below confidence threshold
        pred_boxes = pred_boxes[pred_boxes[:, 4] >= conf_thresh]

        # apply NMS to the predictions, specify threshold and NMS implementaton 
        pred_boxes = non_max_suppresion(pred_boxes, iou_thresh, 3, iou_type)
        if pred_boxes.size(0) == 0:
            return {}, [], true_boxes.tolist()

        # matrix = iou_matrix(pred_boxes, true_boxes)
        matrix = iou_matrix(pred_boxes[:, :4], true_boxes, iou_type)
        # print(f"CIoU matrix: {matrix}")

        if matrix.numel() == 0:
            # print("IoU matrix is empty")
            return {}, pred_boxes.tolist(), true_boxes.tolist()

        # dictionary to hold matches where key = true bbox index and value = tuple(prediction index, iou)
        init_matches = {}
        # iterate through matrix rows (ie true boxes)
        for i in range(matrix.shape[0]):

            # take the max iou from each row, which represents the highest iou between truth and prediction, record indices
            max_iou, max_j = matrix[i].max(0)

            # ensure max is above or equal to threshold
            if (max_iou >= iou_thresh): 
                init_matches[i] = (max_j.item(), max_iou)

        # note: matches should have at most M entries, where M is the number of true bounding boxes from the iou matrix and the number of rows
        # there are no duplicate truths because the max was taken from each row
        # initialize sets to hold used prediction and truth indices
        used_preds = set()
        used_truths = set()

        # convert matches dict to a list of tuples and create max-heap based on negated IoU values
        heap = [(-iou, ti, pi) for ti, (pi, iou) in init_matches.items()]
        heapq.heapify(heap)
        final_matches = {}

        # iterate through heap
        while heap:

            # process matches with highest ious first
            neg_iou, ti, pi = heapq.heappop(heap)
            iou_score = -neg_iou

            # case: when a prediction has already been mapped to a different truth
            if pi in used_preds:
            
                # normally, it would be necessary to retrieve iou of other truth-prediction match from matches
                # but it is guarenteed the other match has a higher iou because of sorting
                # initialize a max iou and new prediction index
                new_iou = 0
                new_pi = -1

                # iterate through row in matrix
                for j in range(matrix[ti].shape[0]):

                    # check if the prediction is already used in a higher-iou match
                    if j not in used_preds and matrix[ti][j] >= iou_thresh:
                        if matrix[ti][j] > new_iou:
                            new_iou = matrix[ti][j]
                            new_pi = j
                    
                if new_pi == -1:
                    # if there is no other valid match (not a duplicate and iou >= thresh), then truth is a false neg
                    continue
                else:
                    # new valid match was found, push the new match onto the heap
                    # heap will sort itself to maintain order
                    heapq.heappush(heap, (-new_iou, ti, new_pi))
            else:
                # add final match and indices to used sets
                used_preds.add(pi)
                used_truths.add(ti)
                final_matches[true_boxes[ti]] = (pred_boxes[pi], iou_score)

        # identify false positives and false negatives
        fp = [pred_boxes[i] for i in range(pred_boxes.shape[0]) if i not in used_preds]
        fn = [true_boxes[i] for i in range(true_boxes.shape[0]) if i not in used_truths]
        return final_matches, fp, fn
    

# precision quantifies the accuracy of good predictions made by the model
# takes in the number of true positives and the number of false negatives
def precision(tp, fp):
    try:
        if tp + fp == 0:
            return 0.0
        pre = tp / (tp + fp)
    except ZeroDivisionError as e:
        pre = 0.0
    return pre


# recall quantifies the completeness of the objects detected in the image
# takes in the number of true positives and the number of false negatives
def recall(tp, fn):
    try:
        if tp + fn == 0:
            return 0.0
        rec = tp / (tp + fn)
    except ZeroDivisionError as e:
        rec = 0.0
    return rec


# f1-score gives balanced measure of model's performance based on precision and recall
# takes in number of true positives, number of false negatives, and number of false positives
def f1_score(tp, fn, fp):
    pre= precision(tp, fp)
    rec = recall(tp, fn)
    try: 
        if pre + rec == 0.0:
            return 0.0, pre, rec
        f1 = 2 * (pre * rec) / (pre + rec)
    except ZeroDivisionError as e:
        f1 = 0.0
    return f1, pre, rec


# precision-recall curve shows trade-off between precision and recall at different confidence thresholds
# takes in list of prediction confidences, their associated confusion_status (true pos or false pos), and number of total true objects
# src: https://zihaogeng.medium.com/how-to-evaluate-an-object-detection-model-iou-precision-recall-and-map-f7cc12e0dcf6
def precision_recall_curve(confidences, confusion_status, num_labels):
    # convert lists to numpy arrays for sorting
    confidences = np.array(confidences)
    confusion_status = np.array(confusion_status)

    # sort lists by descending confidence
    sorted_indices = np.argsort(-confidences)
    confidences = confidences[sorted_indices]
    confusion_status = confusion_status[sorted_indices]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    prev_conf = None

    for i, is_tp in enumerate(confusion_status):
        # determine if new prediction instance is a true positive or false negative
        # note: all other predictions automatically filtered by current confidence threshold
        if is_tp: 
            tp += 1
        else:
            fp += 1
        
        # don't add new point on the curve if confidence is same as previous threshold
        if prev_conf is None or confidences[i] != prev_conf:
            pre = tp / (tp + fp + 1e-9)
            rec = tp / (num_labels + 1e-9)
            precisions.append(pre)
            recalls.append(rec)
        prev_conf = confidences[i]
    return np.array(precisions), np.array(recalls)


# AP is the weighted-sum of precisions at each threshold where the weight is increase in recall
# takes in precision and recall curves as lists
# src: https://github.com/rbgirshick/py-faster-rcnn.
def AP(precision, recall, case=2):
    # adds sentinel values to the beginning and end to handle edge cases
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))

    # computes precision envelope to ensure precision does not decrease as recall increases
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # 101-point interpolation from COCO
    if case == 1:
        # generates 101 equally space points between 0 and 1 at which precision will be interpolated
        x = np.linspace(0, 1, 101)

        # calculates AUC under interpolated precision-recall curve using trapezoidal rule
        AP = np.trapz(np.interp(x, recall, precision), x)

    # continuous integration for AUC
    elif case == 2:

        # finds indices where recall changes
        x = np.where(recall[1:] != recall[:-1])[0]

        # sums area under each segment where recall changes
        AP = np.sum((recall[x + 1] - recall[x]) * precision[x + 1])
    return AP


def plot_PR_curve(precisions, recalls, ap, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,'pr_curve.png')
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)


# takes in pred_boxes -> [left, top, width, height, conf] and true_boxes -> [top, left, width, height]
def iou(pred_boxes, true_boxes):
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
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
    
    # calculate the union area
    union_area = pred_area + true_area - inter_area
    
    # handle division by zero by setting IoU to 0 if union area is 0
    iou = torch.where(union_area > 0, inter_area / union_area, torch.zeros_like(union_area))
    return iou


# src: https://arxiv.org/pdf/1902.09630.pdf
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
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
    
    # calculate the union area
    union_area = pred_area + true_area - inter_area

    # handle division by zero by setting IoU to 0 if union area is 0
    iou = torch.where(union_area > 0, inter_area / union_area, torch.zeros_like(union_area))
    
    # calculate the coordinates of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes_x1, true_boxes_x1)
    enclose_y1 = torch.min(pred_boxes_y1, true_boxes_y1)
    enclose_x2 = torch.max(pred_boxes_x2, true_boxes_x2)
    enclose_y2 = torch.max(pred_boxes_y2, true_boxes_y2)
    
    # calculate the area of the smallest enclosing box
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # calculate GIoU, handle division by zero
    g = torch.where(enclose_area > 0, ((enclose_area - union_area) / enclose_area), torch.zeros_like(iou))
    giou = iou - g
    return giou


# src: https://arxiv.org/abs/1911.08287v1
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
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
    
    # calculate the union area
    union_area = pred_area + true_area - inter_area
    
    # calculate IoU
    # handle division by zero by setting IoU to 0 if union area is 0
    iou = torch.where(union_area > 0, inter_area / union_area, torch.zeros_like(union_area))

    # calculate the coordinates of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes_x1, true_boxes_x1)
    enclose_y1 = torch.min(pred_boxes_y1, true_boxes_y1)
    enclose_x2 = torch.max(pred_boxes_x2, true_boxes_x2)
    enclose_y2 = torch.max(pred_boxes_y2, true_boxes_y2)
    
    # calculate the area of the smallest enclosing box
    enclose_diagonal = torch.clamp((enclose_x2 - enclose_x1), min=0) ** 2 + torch.clamp((enclose_y2 - enclose_y1), min=0) ** 2

    # calculate box centers for predicted and ground-truth
    pred_c1, pred_c2 = calc_box_center(pred_boxes)
    true_c1, true_c2 = calc_box_center(true_boxes)

    # calculate the euclidian distance between the predicted center and the ground-truth center
    center_distance_squared = (pred_c1 - true_c1) ** 2 + (pred_c2 - true_c2) ** 2

    # calculate DIoU as IoU - (d^2) / (C^2), handle zero enclosing diagonal case
    u = torch.where(enclose_diagonal > 0, center_distance_squared / enclose_diagonal, torch.zeros_like(center_distance_squared))
    diou = iou - u
    return diou


# src: https://arxiv.org/abs/1911.08287v1
def ciou(pred_boxes, true_boxes):
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if true_boxes.dim() == 1:
        true_boxes = true_boxes.unsqueeze(0)

    # save widths and heights for v and alpha calculations
    pred_box_width = pred_boxes[:, 2]
    pred_box_height = pred_boxes[:, 3]
    true_box_width = true_boxes[:, 2]
    true_box_height = true_boxes[:, 3]

    # convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
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
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
    
    # calculate the union area
    union_area = pred_area + true_area - inter_area
    
    # calculate IoU
    # handle division by zero by setting IoU to 0 if union area is 0
    iou = torch.where(union_area > 0, inter_area / union_area, torch.zeros_like(union_area))

    # calculate the coordinates of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes_x1, true_boxes_x1)
    enclose_y1 = torch.min(pred_boxes_y1, true_boxes_y1)
    enclose_x2 = torch.max(pred_boxes_x2, true_boxes_x2)
    enclose_y2 = torch.max(pred_boxes_y2, true_boxes_y2)
    
    # calculate the area of the smallest enclosing box
    enclose_diagonal = torch.clamp((enclose_x2 - enclose_x1), min=0) ** 2 + torch.clamp((enclose_y2 - enclose_y1), min=0) ** 2

    # calculate box centers for predicted and ground-truth
    pred_c1, pred_c2 = calc_box_center(pred_boxes)
    true_c1, true_c2 = calc_box_center(true_boxes)

    # calculate the euclidian distance between the predicted center and the ground-truth center
    center_distance_squared = (pred_c1 - true_c1) ** 2 + (pred_c2 - true_c2) ** 2
    u = torch.where(enclose_diagonal > 0, center_distance_squared / enclose_diagonal, torch.zeros_like(center_distance_squared))
    # v measures the consistency of aspect ratios for bboxes
    true_wh_ratio = torch.where(true_box_height > 0, (true_box_width / true_box_height), torch.zeros_like(true_box_width))
    pred_wh_ratio = torch.where(pred_box_height > 0, (pred_box_width / pred_box_height), torch.zeros_like(pred_box_width))
    v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(true_wh_ratio) - torch.atan(pred_wh_ratio)), 2)

    # alpha is a positive trade-off parameter, must be calculated without tracking gradients
    with torch.no_grad():
        denom_alpha = 1 - iou + v
        alpha = torch.where(denom_alpha > 0, (v / denom_alpha), torch.zeros_like(v))

    # calculate ciou
    ciou = iou - u - alpha * v
    return ciou


# takes in boxes in format [x, y, w, h] and converts to [x1, y1, x2, y2]
def convert_boxes_to_x1y1x2y2(boxes):
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)

    # adds w, h to x, y coordinate to obtain x2, y2
    converted_boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], dim=1)
    return converted_boxes


# takes in boxes in format [x1, y1, x2, y2] and converts to [x, y, w, h]
def convert_boxes_to_xywh(boxes):
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    converted_boxes = torch.cat([boxes[:, :2], w.unsqueeze(1), h.unsqueeze(1)], dim=1)
    return converted_boxes


# takes in boxes in format [x1, y1, x2, y2] and ensures that x2 > x1, and y2 > y1
def normalize_boxes(boxes):
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    boxes_x1 = torch.min(boxes[:, 0], boxes[:, 2])
    boxes_y1 = torch.min(boxes[:, 1], boxes[:, 3])
    boxes_x2 = torch.max(boxes[:, 0], boxes[:, 2])
    boxes_y2 = torch.max(boxes[:, 1], boxes[:, 3])
    return boxes_x1, boxes_y1, boxes_x2, boxes_y2


def process_image(img):
    # normalize grey-scale between [0, 1]
    img = (img.float() / 255.0)
    # permute so format is [C, H, W]
    img = img.permute(2, 0, 1)
    return img


# takes in boxes in either format [x1, y1, x2, y2] or [left, top, w, h]
def norm_box_scale(boxes, img_width=2448, img_height=2048):
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    boxes = boxes.float()
    boxes[:, 0] /= img_width
    boxes[:, 2] /= img_width
    boxes[:, 1] /= img_height
    boxes[:, 3] /= img_height
    return boxes

# takes in boxes of dimension [x, y, w, h]
def clean_zero_dim(pred_boxes, img_width=2448, img_height=2048):
    x = pred_boxes[:, 0]
    y = pred_boxes[:, 1]
    w = pred_boxes[:, 2]
    h = pred_boxes[:, 3]
    w = torch.where(w == 0, 1 / img_width, w)
    h = torch.where(h == 0, 1 / img_height, h)
    new_pred_boxes = torch.stack((x, y, w, h), dim=1)
    return new_pred_boxes

# calculates box area based on x1y1x2y2 format
def calc_box_area(x1, y1, x2, y2):
    return torch.abs(x2 - x1) * torch.abs(y2 - y1)


# takes in a box tensor of format [x, y, w, h]
def calc_box_center(boxes):
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    # Will this break it, needs to be a tensor
    c1 = boxes[:, 0] + (boxes[:, 2] / 2.0)
    c2 = boxes[:, 1] + (boxes[:, 3] / 2.0)

    return c1, c2


def scale_boxes(boxes, img_width=2448, img_height=2048):
    boxes = copy.deepcopy(boxes)
    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    boxes = boxes.float()
    boxes[:, 0] *= img_width
    boxes[:, 2] *= img_width
    boxes[:, 1] *= img_height
    boxes[:, 3] *= img_height
    return boxes


# calculates squared euclidean distance between two batches of points
# pts1 & pts2 = tensor of shape (N, 2) where N is number of points
def calc_euclidean_squared(pts1, pts2):
    if pts1.dim() == 1:
        pts1 = pts1.unsqueeze(0)
    if pts2.dim() == 1:
        pts2 = pts2.unsqueeze(0)
    x1, y1, x2, y2 = pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]
    # returns tensor of shape (N,)
    return ((y2 - y1)**2 + (x2 - x1)**2)   


def iou_matrix(pred_boxes, true_boxes, iou_type="iou"):
    # note: torchvision calculates predictions as rows (ie for true in true_boxes] for pred in pred_boxes]))
    if iou_type == "diou":
        return torch.tensor([[diou(pred[:4], true) for pred in pred_boxes] for true in true_boxes], device=device)
    elif iou_type == "giou":
        return torch.tensor([[giou(pred[:4], true) for pred in pred_boxes] for true in true_boxes], device=device)
    elif iou_type == "ciou":
        return torch.tensor([[ciou(pred[:4], true) for pred in pred_boxes] for true in true_boxes], device=device)
    elif iou_type == "iou":
        pred_boxes = convert_boxes_to_x1y1x2y2(pred_boxes[:, :4])
        true_boxes = convert_boxes_to_x1y1x2y2(true_boxes)
        return ops.box_iou(true_boxes, pred_boxes)


# function applies non-max suppression to a set of predicted bounding boxes
# takes in an int representing the case and switches implementation based on that
def non_max_suppresion(pred_boxes, iou_thresh, case=1, iou_type="ciou"):
    
    # original custom implementation
    if case == 1:
        # initialize set to hold indices that need to be removed for NMS
        to_remove = set()

        # iterate through predictions for NMS step
        for i in range(pred_boxes.shape[0]):

            # calculate iou between predictions to see if there is overlap
            for j in range(i + 1, pred_boxes.shape[0]):
                # took out .item()
                iou_score = eval(iou_type + "(pred_boxes[i, :4], pred_boxes[j, :4])")

                # if overlap, add index with lower confidence to removal set
                if iou_score > iou_thresh:
                    if pred_boxes[i, 4] > pred_boxes[j, 4]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

        # remove predictions based on NMS step
        pred_boxes = [pred_boxes[i] for i in range(pred_boxes.shape[0]) if i not in to_remove]
        return torch.stack(pred_boxes) if len(pred_boxes) > 0 else torch.empty((0, 5))

    # similar custom implementation but sorts predictions by confidence first
    elif case == 2:
        # extract confidence scores and sort them
        scores = pred_boxes[:, 4]
        sorted_indices = scores.argsort(descending=True)

        # initialize list to hold indices of kept predictions
        keep = []
        # iterate until sorted list is empty
        while sorted_indices.numel() > 0:
            # keep index of score with highest confidence 
            i = sorted_indices[0].item()
            keep.append(i)

            # initialize list to hold indices that must be removed
            to_remove = []

            # remove the kept index
            sorted_indices = sorted_indices[1:]

            # iterate through the rest of the predictions from high-conf to low-conf
            for j in sorted_indices:
                # calculate iou between current prediction pair
                iou_score = eval(iou_type + "(pred_boxes[i, :4], pred_boxes[j, :4])")

                # if iou exceeds threshold, remove prediction
                if iou_score >= iou_thresh:
                    to_remove.append(j.item())
            
            # update sorted_indices so that removed indices are not present
            sorted_indices = torch.tensor([idx.item() for idx in sorted_indices if idx.item() not in to_remove], dtype=torch.long)

        if not all(0 <= idx < pred_boxes.size(0) for idx in keep):
            raise IndexError("Some indices in 'keep' are out of bounds for pred_boxes")

        # return kept list as a tensor containing final predicted boxes
        return pred_boxes[keep] if keep else torch.empty((0, 5))

    # torchvision implementation of nms
    elif case == 3:
        # retrieve confidence scores
        scores = pred_boxes[:, 4]

        # retrieve normalized/converted boxes for torchvision ops
        boxes = convert_boxes_to_x1y1x2y2(pred_boxes[:, :4])

        # apply torchvision nms
        keep = ops.nms(boxes, scores, iou_thresh)
        return pred_boxes[keep]
    

# function should take in predicted boxes and confidence scores and concatenate scores on the end of each box tensor
# pred_boxes is 2D tensor of [N, 4] dim, and scores is a 1D tensor of [N] dim of confidences corresponding to each bounding box
# each box represented as [left, top, width height]
# returns pred_boxes as a 2D tensor of [N, 5] dim, where each box is represented as [l, t, w, h, conf]
def cat_scores(pred_boxes, scores):
    # reshape scores tensor to be [N, 1]
    scores = scores.view(-1, 1)
    # concatenate tensor along the last dimension
    return torch.cat((pred_boxes, scores), dim=1)