import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils as u

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
    
    # do I need to define a backward?
    
# generalized intersection-over-union loss function based on predicted bbox and true bbox tensors
def giou_loss(pred_boxes, true_boxes):

    giou = u.giou(pred_boxes, true_boxes)
    
    # calculate GIoU loss
    giou_loss = 1 - giou
    
    # should I be returning the mean here?
    return giou_loss.mean()

# distance intersection-over-union loss function based on predicted bbox and true bbox tensors
def diou_loss(pred_boxes, true_boxes):

    diou = u.diou(pred_boxes, true_boxes)

    # calculate DIoU loss
    diou_loss = 1 - diou

    return diou_loss.mean()

# normal intersection-over-union loss based on predicted bbox and true bbox tensors
def iou_loss(pred_boxes, true_boxes):

    iou = u.iou(pred_boxes, true_boxes)
    
    # calculate IoU loss
    iou_loss = 1 - iou

    return iou_loss.mean()

