import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ComboLoss(nn.Module):

    # loss functions and combination weights determined by the config file
    def __init__(self, bbox_loss, cls_loss, bbox_weight, cls_weight):
        super(ComboLoss, self).__init__()
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight
        self.cls_loss = cls_loss
        self.bbox_loss = bbox_loss

    def forward(self, pred_boxes, pred_labels, true_boxes, true_labels):
       combo_loss = self.cls_loss * self.cls_loss(pred_labels, true_labels) + self.bbox_weight * self.bbox_loss(pred_boxes, true_boxes)
       return combo_loss
    
# generalized intersection-over-union loss function based on predicted bbox and true bbox tensors
def giou_loss(pred_boxes, true_boxes):

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
    
    # calculate GIoU loss
    giou_loss = 1 - giou
    
    # should I be returning the mean here?
    return giou_loss.mean()

# distance intersection-over-union loss function based on predicted bbox and true bbox tensors
def diou_loss(pred_boxes, true_boxes):

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

    # calculate DIoU loss
    diou_loss = 1 - diou

    return diou_loss.mean()

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

# normal intersection-over-union loss based on predicted bbox and true bbox tensors
def iou_loss(pred_boxes, true_boxes):
    # convert boxes from [left, top, width, height] to [left, top, right, bottom]
    pred_boxes_x1y1x2y2 = convert_boxes_to_x1y1x2y2(pred_boxes)
    true_boxes_x1y1x2y2 = convert_boxes_to_x1y1x2y2(true_boxes)
    
    # normalize predicted boxes to ensure x2 > x1 and y2 > y1
    pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2 = normalize_boxes(pred_boxes_x1y1x2y2)
    true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2 = true_boxes_x1y1x2y2[:, 0], true_boxes_x1y1x2y2[:, 1], true_boxes_x1y1x2y2[:, 2], true_boxes_x1y1x2y2[:, 3]
    # true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2 = normalize_boxes(true_boxes_x1y1x2y2)
    
    # calculate predicted box area based on normalized predicted and true boxes
    pred_area = calc_box_area(pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2)
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
    # print(f"Pred boxes: {pred_boxes}")
    # print(f"True boxes: {true_boxes}")
    # print(f"Pred boxes (x1, y1, x2, y2): {pred_boxes_x1y1x2y2}")
    # print(f"True boxes (x1, y1, x2, y2): {true_boxes_x1y1x2y2}")
    # print(f"Normalized pred boxes x2 > x1, y2 > y1: {pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2}")
    # print(f"True boxes: {true_boxes_x1, true_boxes_y1, true_boxes_x2, true_boxes_y2}")
    # print(f"Intersection area: {inter_area}")
    # print(f"Union area: {union_area}")
    # print(f"IoU: {iou}")
    iou_loss = 1 - iou

    return iou_loss.mean()

