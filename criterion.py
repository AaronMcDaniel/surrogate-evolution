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
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# takes in matches for a single image along with tensor of dim [7] representing the weights for each iou function and the iou function used to make matches
# returns tensor of dim [8], which is [iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss, weighted_sum_losses]
def compute_weighted_loss(matches, pred_boxes, true_boxes, loss_weights, iou_used_to_match="ciou"):
    num_preds = len(pred_boxes)
    num_truths = len(true_boxes)
    loss_types = ["iou", "giou", "diou", "ciou", "center", "size", "obj"]
    loss_weights = loss_weights.to(torch.float32).to(device)
    result_losses = []
    if num_preds == 0: # edge case this isn't supposed to happen but is happening for some fcos models where num_pred is 0
        result_losses = [1, 1, 1, 1, 1, 1, 1]
        result_losses = [torch.tensor(loss, requires_grad=True, device=device, dtype=torch.float32) for loss in result_losses]
    elif num_preds > 0 and num_truths == 0:
        confs = pred_boxes[:, 4]
        no_truths = True
        result_losses.append(iou_loss(matches, no_truths=no_truths, num_preds=num_preds))
        result_losses.append(giou_loss(matches, no_truths=no_truths, num_preds=num_preds))
        result_losses.append(diou_loss(matches, no_truths=no_truths, num_preds=num_preds))
        result_losses.append(ciou_loss(matches, no_truths=no_truths, num_preds=num_preds))
        result_losses.append(center_loss(matches, no_truths=no_truths))
        result_losses.append(size_loss(matches, no_truths=no_truths))
        result_losses.append(obj_loss(matches, no_truths=no_truths, confs=confs))
    else:
        for i, loss_type in enumerate(loss_types):

            # if current function was used to make matches, just sum the iou_score in matches
            if loss_type == iou_used_to_match:
                prev_calc_loss = torch.stack([1.0 - iou_score for truth, (pred, iou_score) in matches.items()]).to(device)
                loss_i = torch.sum(prev_calc_loss)
            else:
                # calculate loss with specified function
                loss_i = eval(loss_type + "_loss(matches)")
                # place loss averaged across predictions in the result losses tensor
            result_losses.append(loss_i)

    result_losses = torch.stack(result_losses)

    # place weighted-sum of iou loss functions in the last index of the result tensor
    weighted_sum_losses = torch.dot(loss_weights, result_losses[:7])
    result_losses = torch.cat((result_losses, weighted_sum_losses.unsqueeze(0)))

    result_losses = torch.where(torch.isnan(result_losses), torch.tensor(1000000.0, dtype=torch.float32, device=result_losses.device, requires_grad=True), result_losses)
    return result_losses


# takes in matches dict and calculates L2 norm (MSE) between true and predicted bbox area
def size_loss(matches, no_truths=False):
    if no_truths:
        return torch.tensor(1000000.0, dtype=torch.float32, device=device, requires_grad=True)

    MSELoss = nn.MSELoss()
    true_areas = []
    pred_areas = []

    for i, (truth, (pred, iou_score)) in enumerate(matches.items()):
        px1, py1, px2, py2 = u.normalize_boxes(u.convert_boxes_to_x1y1x2y2(pred))
        pa = u.calc_box_area(px1, py1, px2, py2)
        tx1, ty1, tx2, ty2 = u.normalize_boxes(u.convert_boxes_to_x1y1x2y2(truth))
        ta = u.calc_box_area(tx1, ty1, tx2, ty2)
        pred_areas.append(pa)
        true_areas.append(ta)

    if not pred_areas and not true_areas:
        return torch.tensor(0.0, requires_grad=True, device=device, dtype=torch.float32)

    pred_areas = torch.stack(pred_areas)
    true_areas = torch.stack(true_areas)

    # Normalize predicted areas by true areas
    normalized_pred_areas = pred_areas / true_areas

    # Create a tensor of ones with the same shape as normalized_pred_areas
    target = torch.ones_like(normalized_pred_areas)

    try: 
        return MSELoss(normalized_pred_areas, target) * 0.1
    except RuntimeError as e:
        return torch.tensor(1000000.0, requires_grad=True, device=device, dtype=torch.float32)


# objectness loss function used to regress model predicted confidence and iou with BCE loss
# for detections, model confidence should equal iou, for non-detections, model confidence should equal 0
def obj_loss(matches, iou_thresh=0.0, no_truths=False, confs=None):
    BCEobj = nn.BCELoss()
    if no_truths and confs is not None:
        # if predictions on absent true label, use confidence and tensor of 0s for BCELoss
        pred_obj = confs
        true_obj = torch.zeros_like(pred_obj, dtype=torch.float32, device=device, requires_grad=True)
    else:
        pred_obj = []
        true_obj = []

        for truth, (pred, iou_score) in matches.items():
            pred_obj.append(pred[4])
            true_obj.append(iou_score.to(device))
        
        if pred_obj:
            pred_obj = torch.stack(pred_obj).to(torch.float32)
        else:
            pred_obj = torch.tensor([], dtype=torch.float32, requires_grad=True, device=device)
            
        if true_obj:
            true_obj = torch.stack(true_obj).to(torch.float32)
        else:
            true_obj = torch.tensor([], dtype=torch.float32, requires_grad=True, device=device)

        true_obj = torch.sigmoid(true_obj)

    try:
        loss = BCEobj(pred_obj, true_obj) * 100
    except RuntimeError as e:
        loss = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=device) * 100
    return loss


# takes in matches dict and calculates L2 norm (MSE) of actual and predicted bbox centers
def center_loss(matches, no_truths=False):
    if no_truths:
        # punish predictions on an absent label
        loss = torch.tensor(1000000.0, dtype=torch.float32, requires_grad=True, device=device)
        return loss
    
    MSELoss = nn.MSELoss()
    pred_centers = []
    true_centers = []

    for i, (truth, (pred, iou_score)) in enumerate(matches.items()):
        pred_center_x, pred_center_y = u.calc_box_center(pred)
        true_center_x, true_center_y = u.calc_box_center(truth)
        
        pred_center = torch.tensor([pred_center_x, pred_center_y], dtype=torch.float32, device=pred.device, requires_grad=True)
        true_center = torch.tensor([true_center_x, true_center_y], dtype=torch.float32, device=truth.device, requires_grad=True)
        pred_centers.append(pred_center)
        true_centers.append(true_center)

    if not pred_centers and not true_centers:
        # can only be case where there are no predictions and no labels so loss should be 0
        return torch.tensor(0.0, requires_grad=True, device=device, dtype=torch.float32)

    pred_centers = torch.stack(pred_centers)
    true_centers = torch.stack(true_centers)

    valid_mask = ~torch.isnan(pred_centers).any(dim=1) & ~torch.isnan(true_centers).any(dim=1)
    pred_centers = pred_centers[valid_mask]
    true_centers = true_centers[valid_mask]

    try:
        return MSELoss(pred_centers, true_centers) * 10
    except Exception as e:
        
        return torch.tensor(1000000.0, dtype=torch.float32, requires_grad=True, device=device)
    

# takes in matches dict, calculates iou loss between matched pairs, and returns sum
def iou_loss(matches, no_truths=False, num_preds=None):
    if no_truths and num_preds is not None:
        # worst-case iou loss = 1, so loss should equal 1 * num_preds
        return torch.tensor(1.0 * num_preds, requires_grad=True, device=device, dtype=torch.float32)
    
    losses = []
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.iou(pi, ti)
        losses.append(1 - new_iou)

    if not losses:
        # if losses is empty, there were no matches
        # we know this case means there were no predictions and no labels, so loss is 0
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    
    losses = torch.stack(losses)

    # check if any element in loss is NaN
    if torch.isnan(losses).any():
        return torch.tensor(1000000.0, dtype=torch.float32, requires_grad=True, device=device)
    else:
        return torch.sum(losses)


# takes in matches dict, calculates giou loss between matched pairs, and returns sum
def giou_loss(matches, no_truths=False, num_preds=None):
    if no_truths and num_preds is not None:
        # worst-case giou loss = 2, so loss should equal 2 * num_preds
        return torch.tensor(2.0 * num_preds, requires_grad=True, device=device, dtype=torch.float32)
    
    losses = []
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.giou(pi, ti)
        losses.append(1 - new_iou)

    if not losses:
        # if losses is empty, there were no matches
        # we know this case means there were no predictions and no labels, so loss is 0
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    
    losses = torch.stack(losses)

    # check if any element in loss is NaN
    if torch.isnan(losses).any():
        return torch.tensor(1000000.0, dtype=torch.float32, requires_grad=True, device=device)
    else:
        return torch.sum(losses)


# takes in matches dict, calculates diou loss between matched pairs, and returns sum
def diou_loss(matches, no_truths=False, num_preds=None):
    if no_truths and num_preds is not None:
        # worst-case diou loss = 2, so loss should equal 2 * num_preds
        return torch.tensor(2.0 * num_preds, requires_grad=True, device=device, dtype=torch.float32)
    
    losses = []
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.diou(pi, ti)
        losses.append(1 - new_iou)

    if not losses:
        # if losses is empty, there were no matches
        # we know this case means there were no predictions and no labels, so loss is 0
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    
    losses = torch.stack(losses)

    # check if any element in loss is NaN
    if torch.isnan(losses).any():
        return torch.tensor(1000000.0, dtype=torch.float32, requires_grad=True, device=device)
    else:
        return torch.sum(losses)


# takes in matches dict, calculates ciou loss between matched pairs, and returns sum
def ciou_loss(matches, no_truths=False, num_preds=None):
    if no_truths and num_preds is not None:
        # worst-case ciou loss = 2, so loss should equal 2 * num_preds
        return torch.tensor(2.0 * num_preds, requires_grad=True, device=device, dtype=torch.float32)
    
    losses = []
    for ti, (pi, iou_score) in matches.items():
        new_iou = u.ciou(pi, ti)
        losses.append(1 - new_iou)

    if not losses:
        # if losses is empty, there were no matches
        # we know this case means there were no predictions and no labels, so loss is 0
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    
    losses = torch.stack(losses)

    # check if any element in loss is NaN
    if torch.isnan(losses).any():
        return torch.tensor(1000000.0, dtype=torch.float32, requires_grad=True, device=device)
    else:
        return torch.sum(losses)

