import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment
import heapq
from sortedcontainers import SortedListWithKey
import math
import pandas as pd

# used to match predictions and truths based on thresholds and compute the confusion matrix
# note: could also be implemented with a balanced BST (from sortedcontainers) instead of heap
def match_boxes(pred_boxes, true_boxes, iou_thresh=0.3, conf_thresh=0.5, mode="val", iou_type="iou"):

    # train mode is used for loss calculation and factors in all predictions
    if mode =="train":
        matrix = iou_matrix(pred_boxes[:, :4], true_boxes, iou_type)
        print(matrix)
        matches = {}

        # for train, only care about what the max iou of each column is
        # each prediction has a paired truth, we don't care about duplicates
        for i in range(matrix.shape[1]):

            # take the maximum value of each column
            max_iou, max_j = matrix[:, i].max(0)
            matches[true_boxes[max_j.item()]] = (pred_boxes[i], max_iou.item())

        print(matches)

        return matches

    # validation mode filters predictions based on confidence, NMS, iou threshold, ect.
    elif mode == "val":

        # remove any predictions that fall below confidence threshold
        pred_boxes = pred_boxes[pred_boxes[:, 4] >= conf_thresh]

        # apply NMS to the predictions, specify threshold and NMS implementaton 
        pred_boxes = non_max_suppresion(pred_boxes, iou_thresh, 1, iou_type)
        print(f"Predictions after NMS: {pred_boxes}")

        # matrix = iou_matrix(pred_boxes, true_boxes)
        matrix = iou_matrix(pred_boxes[:, :4], true_boxes, iou_type)
        print(f"IoU matrix: {matrix}")

        # dictionary to hold matches where key = true bbox index and value = tuple(prediction index, iou)
        init_matches = {}
        # iterate through matrix rows (ie true boxes)
        for i in range(matrix.shape[0]):

            # take the max iou from each row, which represents the highest iou between truth and prediction, record indices
            max_iou, max_j = matrix[i].max(0)

            # ensure max is above or equal to threshold
            if (max_iou >= iou_thresh): 
                init_matches[i] = (max_j.item(), max_iou.item())

        print(f"Initial matches: {init_matches}")

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

        print(f"Final matches: {final_matches}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")

        return final_matches, fp, fn

# takes in pred_boxes tensor of shape (N, 5) and true_boxes of shape (M, 4), iou threshold, and confidence threshold
def match_boxes_alt(pred_boxes, true_boxes, iou_thresh, conf_thresh):

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
    matches = {}

    # sets used to track predictions that have already been matched
    used_preds = set()
    used_truths = set()

    # iterate through indices returned by hunagrian algorithm
    for pi, ti in zip(pred_indices, true_indices):

        # calculate ious and ensure above threshold before adding to matches
        iou = matrix[pi, ti]
        if iou >= iou_thresh:
            matches[ti] = (pi, iou)
            used_preds.add(pi)
            used_truths.add(ti)

    # identify false positives and false negatives
    fp = [pred_boxes[i] for i in range(pred_boxes.shape[0]) if i not in used_preds]
    fn = [true_boxes[i] for i in range(true_boxes.shape[0]) if i not in used_truths]

    return matches, fp, fn

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
    return ((2 * precision * recall) / (precision + recall)), precision, recall

# precision-recall curve shows trade-off between precision and recall at different iou thresholds
# takes in predictions, ground-truths, and fixed confidence threshold
def precision_recall_curve(pred_boxes, true_boxes, conf_thresh):
    precisions = []
    recalls = []

    # step thresholds by 0.1
    # TO-DO: Scale threshold by true positive prediction
    thresholds = np.arrange(start=0.0, stop=1.1, step=0.1)

    # iterates through thresholds
    for thresh in thresholds:
        
        # match boxes must be calculated at every single threshold due to NMS step
        matches, fps, fns = match_boxes(pred_boxes,true_boxes, thresh, conf_thresh)

        # get confusion matrix for current threshold
        tp = len(matches)
        fp = len(fps)
        fn = len(fns)

        # calculate precison and recall using tp, fp, fn
        p = precision(tp, fp)
        r = recall(tp, fn)

        # add the new data points
        precisions.append(p)
        recalls.append(r)

    # return points on curve
    return precisions, recalls

# AP is the weighted-sum of precisions at each threshold where the weight is increase in recall
# takes in precision and recall curves as lists
# src: https://github.com/rbgirshick/py-faster-rcnn.
def AP(precision, recall, case=1):

    # first convert to numpy array
    precision = np.array(precision)
    recall = np.array(recall)

    # adds sentinel values to the beginning and end to handle edge cases
    precision = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    recall = np.concatenate(([0.], precision, [0.]))
    
    # computes precision envelope to ensure precision does not decrease as recall increases
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))

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

# initializes dataframes with columns for each metric and rows for each epoch
def create_metrics_df(num_epochs):
    return pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'epoch_loss': [None] * num_epochs,
    'precision': [None] * num_epochs,
    'recall': [None] * num_epochs,
    'f1_score': [None] * num_epochs,
    'average_precision': [None] * num_epochs,
    'true_positives': [None] * num_epochs,
    'false_positives': [None] * num_epochs,
    'false_negatives': [None] * num_epochs
})

# updates metrics dataframe with metrics for a single epoch
def log_epoch_metrics(metrics_df, epoch, epoch_metrics):
    metrics_df.loc[metrics_df['epoch'] == epoch, 'epoch_loss'] = epoch_metrics['epoch_loss']
    metrics_df.loc[metrics_df['epoch'] == epoch, 'precision'] = epoch_metrics['precision']
    metrics_df.loc[metrics_df['epoch'] == epoch, 'recall'] = epoch_metrics['recall']
    metrics_df.loc[metrics_df['epoch'] == epoch, 'f1_score'] = epoch_metrics['f1_score']
    metrics_df.loc[metrics_df['epoch'] == epoch, 'average_precision'] = epoch_metrics['average_precision']
    metrics_df.loc[metrics_df['epoch'] == epoch, 'true_positives'] = epoch_metrics['true_positives']
    metrics_df.loc[metrics_df['epoch'] == epoch, 'false_positives'] = epoch_metrics['false_positives']
    metrics_df.loc[metrics_df['epoch'] == epoch, 'false_negatives'] = epoch_metrics['false_negatives']


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
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # calculate the union area
    union_area = pred_area + true_area - inter_area
    
    # calculate IoU
    iou = inter_area / union_area

    return iou
    # return ops.box_iou(pred_boxes, true_boxes)

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

# src: https://arxiv.org/abs/1911.08287v1
def ciou(pred_boxes, true_boxes):
    # if 1D tensor, converts shape from [4] to [1, 4]
    # can now pass in 1D and 2D tensors
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if true_boxes.dim() == 1:
        true_boxes = true_boxes.unsqueeze(0)

    # save widths and heights for v and alpha calculations
    pred_box_width = pred_boxes[:, 2]
    pred_box_height = pred_boxes[:, 3]
    true_box_width = true_boxes[:, 2]
    true_box_height = true_boxes[:, 3]

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

    # calculate the coordinates of the smallest enclosing box
    enclose_x1 = torch.min(pred_boxes_x1, true_boxes_x1)
    enclose_y1 = torch.min(pred_boxes_y1, true_boxes_y1)
    enclose_x2 = torch.max(pred_boxes_x2, true_boxes_x2)
    enclose_y2 = torch.max(pred_boxes_y2, true_boxes_y2)
    
    # calculate the area of the smallest enclosing box
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

    # calculate box centers for predicted and ground-truth
    pred_c1, pred_c2 = calc_box_center(pred_boxes)
    true_c1, true_c2 = calc_box_center(true_boxes)

    # calculate the euclidian distance between the predicted center and the ground-truth center
    center_distance_squared = (pred_c1 - true_c1) ** 2 + (pred_c2 - true_c2) ** 2

    # v measures the consistency of aspect ratios for bboxes
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(true_box_width / true_box_height) - torch.atan(pred_box_width / pred_box_height), 2)

    # alpha is a positive trade-off parameter
    alpha = v / (1 - iou + v + 1e-16)

    # calculate ciou
    ciou = iou - (center_distance_squared / enclose_diagonal + v * alpha)
    return ciou


# takes in boxes in format [x, y, w, h] and converts to [x1, y1, x2, y2]
def convert_boxes_to_x1y1x2y2(boxes):
    # adds w, h to x, y coordinate to obtain x2, y2
    converted_boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], dim=1)
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
    # Will this break it, needs to be a tensor
    c1 = boxes[:, 0] + (boxes[:, 2] / 2.0)
    c2 = boxes[:, 1] + (boxes[:, 3] / 2.0)
    return c1, c2

def iou_matrix(pred_boxes, true_boxes, iou_type="iou"):
    # note: torchvision calculates predictions as rows (ie for true in true_boxes] for pred in pred_boxes]))
    if iou_type == "diou":
        return torch.tensor([[diou(pred[:4], true) for pred in pred_boxes] for true in true_boxes])
    elif iou_type == "giou":
        return torch.tensor([[giou(pred[:4], true) for pred in pred_boxes] for true in true_boxes])
    elif iou_type == "ciou":
        return torch.tensor([[ciou(pred[:4], true) for pred in pred_boxes] for true in true_boxes])
    elif iou_type == "iou":
        return torch.tensor([[iou(pred[:4], true) for pred in pred_boxes] for true in true_boxes])

# function applies non-max suppression to a set of predicted bounding boxes
# takes in an int representing the case and switches implementation based on that
def non_max_suppresion(pred_boxes, iou_thresh, case=1, iou_type="iou"):
    
    # original custom implementation
    if case == 1:

        # initialize set to hold indices that need to be removed for NMS
        to_remove = set()

        # iterate through predictions for NMS step
        for i in range(pred_boxes.shape[0]):

            # calculate iou between predictions to see if there is overlap
            for j in range(i + 1, pred_boxes.shape[0]):
                iou_score = eval(iou_type + "(pred_boxes[i, :4], pred_boxes[j, :4]).item()")

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
                iou_score = eval(iou_type + "(pred_boxes[i, :4], pred_boxes[j, :4]).item()")

                # if iou exceeds threshold, remove prediction
                if iou_score >= iou_thresh:
                    to_remove.append(j)
            
            # update sorted_indices so that removed indices are not present
            sorted_indices = torch.tensor([idx for idx in sorted_indices if idx not in to_remove], dtype=torch.long)

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

# pred_boxes = torch.tensor([
#     [3, 3, 4, 4],
#     [4, 4, 1, 1],
#     [9, 9, 2, 2]
# ], dtype=torch.float32)

# scores = torch.tensor([0.3, 0.4, 0.8], dtype=torch.float32)

# pred_boxes = cat_scores(pred_boxes, scores)

# res = non_max_suppresion(pred_boxes, 0.5, 1)
# print(res)