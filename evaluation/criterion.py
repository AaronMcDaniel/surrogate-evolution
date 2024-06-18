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
    

# takes in predicted and true bboxes for a single image along with tensor of dim [4] representing the weights for each iou function
# returns tensor of dim [5], which is [iou_mean_loss, giou_mean_loss, diou_mean_loss, ciou_mean_loss, weighted_sum_losses]
def compute_iou_loss(pred_boxes, true_boxes, loss_weights):
    # list of iou functions
    iou_types = ["iou", "giou", "diou", "ciou"]

    # ensure weights tensor is compatible for dotting
    loss_weights = loss_weights.to(torch.float32)

    # initialize tensor for result losses
    result_losses = tensor_zeros = torch.zeros(5, dtype=torch.float32)

    # calculate the mean bbox loss using each of the defined iou functions
    for i, iou_type in enumerate(iou_types):

        # calculate matches with current iou_type 
        # every single prediction made will be matched to a true bbox with no threshold filtering
        matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type)

        # sum loss for all bbox predictions made on the image
        summed_loss = 0.0
        for ti, (pi, iou_score) in matches.items():
            summed_loss += (1 - iou_score)

        # calculate the average loss per prediction made on the image
        mean_loss = summed_loss / len(matches)

        # place loss averaged across predictions in the result losses tensor
        result_losses[i] = mean_loss

    # place weigthed-sum of iou loss functions in the last index of the result tensor
    result_losses[4] = torch.dot(loss_weights, result_losses[:4])
    return result_losses


pred_boxes = torch.tensor([
        [0, 0, 10, 10, 0.9],
        [1, 1, 9, 9, 0.85],
        [2, 2, 8, 8, 0.95]
    ], dtype=torch.float32)

true_boxes = torch.tensor([
    [0, 0, 10, 10],
    [1, 1, 9, 9]
], dtype=torch.float32)

loss_weights = torch.tensor([.25, .25, .25, .25])

print(compute_iou_loss(pred_boxes, true_boxes, loss_weights))