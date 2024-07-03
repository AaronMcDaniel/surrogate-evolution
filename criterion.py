import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import torchvision.ops as ops
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
import utils as u
    
# takes in matches for a single image along with tensor of dim [7] representing the weights for each iou function and the iou function used to make matches
# returns tensor of dim [8], which is [iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss, weighted_sum_losses]
def compute_weighted_loss(matches, loss_weights, iou_used_to_match="ciou"):
    loss_types = ["iou", "giou", "diou", "ciou", "center", "size", "obj"]

    # ensure weights tensor is compatible for dotting
    loss_weights = loss_weights.to(torch.float32)
    result_losses = torch.zeros(8, dtype=torch.float32)

    for i, loss_type in enumerate(loss_types):

        # if current function was used to make matches, just sum the iou_score in matches
        if loss_type == iou_used_to_match:
            prev_calc_loss = torch.tensor([1.0 - iou_score for truth, (pred, iou_score) in matches.items()], dtype=torch.float32)
            result_losses[i] = torch.sum(prev_calc_loss)
        else:
            # calculate loss with specified function
            loss_i = eval(loss_type + "_loss(matches)")
            # place loss averaged across predictions in the result losses tensor
            result_losses[i] = loss_i

    # place weigthed-sum of iou loss functions in the last index of the result tensor
    result_losses[7] = torch.dot(loss_weights, result_losses[:7])
    return result_losses

def compute_weighted_loss_alt(matches, loss_weights, iou_used_to_match="ciou"):
    loss_types = ["ciou", "center", "size", "obj"]
    loss_weights = loss_weights.to(torch.float32)
    result_losses = torch.zeros(5, dtype=torch.float32)
    for i, loss_type in enumerate(loss_types):
        if loss_type == iou_used_to_match:
            prev_calc_loss = torch.tensor([1.0 - iou_score for truth, (pred, iou_score) in matches.items()], dtype=torch.float32)
            result_losses[i] = torch.sum(prev_calc_loss)
        else:
            loss_i = eval(loss_type + "_loss(matches)")
            result_losses[i] = loss_i
    result_losses[4] = torch.dot(loss_weights, result_losses[:4])
    return result_losses

# takes in matches dict and calculates L2 norm (MSE) between true and predicted bbox area
def size_loss(matches):
    MSELoss = nn.MSELoss()
    true_areas = torch.zeros(len(matches), dtype=torch.float32)
    pred_areas = torch.zeros(len(matches), dtype=torch.float32)

    for i, (truth, (pred, iou_score)) in enumerate(matches.items()):
        px1, py1, px2, py2 = u.normalize_boxes(u.convert_boxes_to_x1y1x2y2(pred))
        pa = u.calc_box_area(px1, py1, px2, py2)
        tx1, ty1, tx2, ty2 = u.normalize_boxes(u.convert_boxes_to_x1y1x2y2(truth))
        ta = u.calc_box_area(tx1, ty1, tx2, ty2)
        pred_areas[i] = pa
        true_areas[i] = ta
    try: 
        loss = MSELoss(pred_areas, true_areas)
    except RuntimeError as e:
        loss = 1000000
    return loss

# objectness loss function used to regress model predicted confidence and iou with BCE loss
# for detections, model confidence should equal iou, for non-detections, model confidence should equal 0
def obj_loss(matches, iou_thresh=0.0):
    BCEobj = nn.BCELoss()
    # extract model's predicted confidence scores
    pred_obj = torch.tensor([pred[4] for truth, (pred, iou_score) in matches.items()], dtype=torch.float32)
    true_obj = torch.zeros(len(matches), dtype=torch.float32)

    if matches:
        indices = torch.tensor([i for i, match in enumerate(matches)], dtype=torch.long)
        # iou_scores = torch.tensor([match[1] for match in matches.values()], dtype=torch.float32)
        iou_scores = torch.tensor([iou_score for _, (_, iou_score) in matches.items()], dtype=torch.float32)
        # print(f"iou-scores in obj_loss: {iou_scores}")
        true_obj[indices] = iou_scores.clamp(min=iou_thresh, max=1.0)

    pred_obj = pred_obj.clamp(min=0.0, max=1.0)
    true_obj = true_obj.clamp(min=0.0, max=1.0)

    try:
        loss = BCEobj(pred_obj, true_obj)
    except RuntimeError as e:
        loss = torch.tensor(1.0, requires_grad=True, device=pred_obj.device)
    return loss

# takes in matches dict and calculates L2 norm (MSE) of actual and predicted bbox centers
def center_loss(matches):
    MSELoss = nn.MSELoss()
    # pred_centers = torch.zeros((len(matches), 2), dtype=torch.float32)
    # true_centers = torch.zeros((len(matches), 2), dtype=torch.float32)
    pred_centers = []
    true_centers = []

    for i, (truth, (pred, iou_score)) in enumerate(matches.items()):
        pred_center_x, pred_center_y = u.calc_box_center(pred)
        true_center_x, true_center_y = u.calc_box_center(truth)
        
        pred_center = torch.tensor([pred_center_x, pred_center_y], dtype=torch.float32, device=pred.device)
        true_center = torch.tensor([true_center_x, true_center_y], dtype=torch.float32, device=truth.device)
        pred_centers.append(pred_center)
        true_centers.append(true_center)

    pred_centers = torch.stack(pred_centers)
    true_centers = torch.stack(true_centers)

    valid_mask = ~torch.isnan(pred_centers).any(dim=1) & ~torch.isnan(true_centers).any(dim=1)
    pred_centers = pred_centers[valid_mask]
    true_centers = true_centers[valid_mask]

    try:
        loss = MSELoss(pred_centers, true_centers)
    except:
        loss = 1000000
    return loss

# takes in matches dict, calculates iou loss between matched pairs, and returns sum
def iou_loss(matches):
    loss = 0.0
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.iou(pi, ti)
        loss += (1 - new_iou)
    if torch.isnan(loss).any(dim=1):
        loss = 1000000
    return loss

# takes in matches dict, calculates giou loss between matched pairs, and returns sum
def giou_loss(matches):
    loss = 0.0
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.giou(pi, ti)
        loss += (1 - new_iou)

    if torch.isnan(loss).any(dim=1):
        loss = 1000000
    return loss

# takes in matches dict, calculates diou loss between matched pairs, and returns sum
def diou_loss(matches):
    loss = 0.0
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.diou(pi, ti)
        loss += (1 - new_iou)
    if torch.isnan(loss).any(dim=1):
        loss = 1000000
    return loss

# takes in matches dict, calculates ciou loss between matched pairs, and returns sum
def ciou_loss(matches):
    loss = 0.0
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.ciou(pi, ti)
        loss += (1 - new_iou)
    if torch.isnan(loss).any(dim=1):
        loss = 1000000
    return loss

# old implementation of center loss where matches are made using euclidean distance
def center_loss_alt(pred_boxes, true_boxes):
    MSELoss = nn.MSELoss()
    # get centers and calculate squared euclidean distance matrix
    pred_centers = torch.stack(u.calc_box_center(pred_boxes), dim=1)
    true_centers = torch.stack(u.calc_box_center(true_boxes), dim=1)
    num_pred = pred_centers.shape[0]
    num_true = true_centers.shape[0]

    distance_matrix = torch.zeros((num_pred, num_true), device=pred_boxes.device)
    for i in range(num_pred):
        for j in range(num_true):
            distance_matrix[i, j] = u.calc_euclidean_squared(pred_centers[i], true_centers[j])

    # make matches based on the minimum squared euclidean distance
    matched_true_indices = torch.zeros(num_pred, dtype=torch.long)
    for i in range(num_pred):
        min_dist, j = distance_matrix[i].min(0)
        matched_true_indices[i] = j  
      
    matched_true_centers = true_centers[matched_true_indices]
    loss = MSELoss(pred_centers, matched_true_centers)
    return torch.tensor(loss, dtype=torch.float32) 

class ComboLoss(nn.Module):
    # loss functions and combination weights determined by the config file and genome
    def __init__(self, bbox_loss, cls_loss, bbox_weight, cls_weight):
        super(ComboLoss, self).__init__()
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight
        self.cls_loss = cls_loss
        self.bbox_loss = bbox_loss

    def forward(self, pred_boxes, pred_labels, true_boxes, true_labels):
       combo_loss = self.cls_loss * self.cls_loss(pred_labels, true_labels) + self.bbox_weight * self.bbox_loss(pred_boxes, true_boxes)
       return combo_loss

# # # testing
# pred_boxes = torch.tensor([
#         [0, 0, 10, 10, 0.9],
#         [1, 1, 9, 9, 0.85],
#         [2, 2, 8, 8, 0.95]
#     ], dtype=torch.float32)

# true_boxes = torch.tensor([
#     [0, 0, 10, 10],
#     [1, 1, 9, 9]
# ], dtype=torch.float32)

# # loss_weights = F.softmax(torch.rand(7), dim=0)
# loss_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1])
# matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", "ciou")
# weighted_loss = compute_weighted_loss(matches, loss_weights, "ciou")
# iou = iou_loss(matches)
# diou = diou_loss(matches)
# giou = giou_loss(matches)
# ciou = ciou_loss(matches)
# obj = obj_loss(matches)
# center = center_loss(matches)
# size = size_loss(matches)
# expected_result = iou + diou + giou + ciou + obj + center + size

# print(f"Loss Tensor: {weighted_loss}")
# print(f"Weighted Sum Loss: {weighted_loss[7].item()}")
# print(f"Expected Weighted Sum: {expected_result.item()}")
# print(f"IoU Loss: {iou.item()}")
# print(f"DIoU Loss: {diou.item()}")
# print(f"GIoU Loss: {giou.item()}")
# print(f"CIoU Loss: {ciou.item()}")
# print(f"Objectness Loss: {obj.item()}")
# print(f"Center Loss: {center.item()}")
# print(f"Size Loss: {size.item()}")