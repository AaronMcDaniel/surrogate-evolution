import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import utils as u
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torchvision.ops import box_iou
import json
from criterion import ComboLoss

# TO-DO:
    # Build loss function
    # Store each epoch's predictions along with ground-truth
    # Compute mAP, FPs, FNs, and IoU on a per-epoch basis
    # Save model weight on a per-epoch basis
    # Manage placement of results in output directories
    # Save predictions with labels to the output director

# parses arguments from sbatch job
if __name__ == "__main__": # makes sure this happens if the script is being run directly
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", type=str), parser.add_argument("config", type=str)
    args = parser.parse_args()

    # passes in the command-line args to begin the evaluation
    evaluation_engine(args.config, args.genome)


def evaluation_engine(cfg, genome):

    # provides iterable batches of data
    train_loader, val_loader = prepare_data(cfg)

    # builds PyTorch model using genome string
    model = build_model(cfg, genome)

    device = torch.device('cuda') if torch.cuda.is_avauilable() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # get relevant parameters from config
    classes = cfg['classes']
    num_classes = cfg['classes'].length()
    img_size = cfg['image_size']
    batch_size = cfg['batch_size']
    num_epochs = cfg['num_epochs']
    det_thresh = cfg['det_thresh']
    conf_thresh = cfg['conf_thresh']
    output_path = cfg['output_path']
    bbox_weight = cfg['bbox_weight']
    cls_weight = cfg['cls_weight']
    bbox_loss = cfg['bbox_loss']
    cls_loss = cfg['cls_loss']

    # get optimizer, loss function, and scheduler according to config
    optimizer = get_optimizer(cfg, params)
    # do we need get_criterion()
    criterion = ComboLoss(bbox_loss, cls_loss, bbox_weight, cls_weight)
    scheduler = get_scheduler()

    # training loop
    for epoch in range(num_epochs):

        all_truths = []

        # training for one epoch
        model.train()
        train_loss = 0.0
        train_preds = []

        # loader yields each batch as a tuple (images, targets)
        for images, targets in tqdm(train_loader):

            # list of image tensors in the batch of shape [channels, height, width]
            images = [img.cuda() for img in images]
            # list of dictionaries where each represents the ground-truth data of one image in the batch
            # each dictionary contains tensor 'boxes' of shape [num_bboxes, 4]
            # each box is represented as [left, top, width, height]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            outputs = model(images)

            true_boxes = torch.cat([target['boxes'] for target in targets], dim=0)
            pred_boxes = torch.cat([output['boxes'] for output in outputs], dim=0)
            train_preds.extend(pred_boxes)
            all_truths.extend(true_boxes)

            # calculate loss, perform backwards step
            loss = criterion(pred_boxes, true_boxes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

        scheduler.step()

        # validation for one epoch
        model.eval()
        val_loss = 0.0
        val_preds = []

        # disables gradient calculations
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = [img.cuda() for img in images]
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                # forward
                outputs = model(images)

                true_boxes = torch.cat([target['boxes'] for target in targets], dim=0)
                pred_boxes = torch.cat([output['boxes'] for output in outputs], dim=0)
                val_preds.extend(pred_boxes)

                # gets the bboxes that actually match from the predictions and truths, as well as false positive and false negative predictions
                matched_preds, matched_truths, fp, fn = u.match_boxes(pred_boxes, true_boxes, det_thresh)


            
# Some functions that may need to be implemented

# def build_model(cfg, genome):

# def get_optimizer(cfg, params):

# def get_scheduler(cfg):

# def compute_metrics():

# def prepare_data():