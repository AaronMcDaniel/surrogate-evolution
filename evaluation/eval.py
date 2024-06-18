import csv
import toml
import sys
sys.path.insert(0, '/gv1/projects/GRIP_Precog_Opt/precog-opt-grip')
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
import criterion as c
import heapq
import pandas as pd
from core.dataset import NewDataset
from torch.utils.data import DataLoader
from codec import Codec
from torch.optim import lr_scheduler

def engine(cfg, genome):

    # retrieves relevant codec attributes from the config and instantiates model
    genome_encoding_strat = cfg['genome_encoding_strat']
    surrogate_encoding_strat = cfg['surrogate_encoding_strat']
    num_classes = cfg['num_classes']
    codec = Codec(genome_encoding_strat, surrogate_encoding_strat, num_classes)
    model_dict = codec.decode_genome(genome)
    # pytorch model: input = 4D tensor -> [batch_size, num_channels, width, height], output = 2D tensor -> []
    model = model_dict['model']

    device = torch.device('cuda') if torch.cuda.is_avauilable() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # TO DO: implement later
    batch_size = dynamic_batch_size(model)

    # TO DO: implement later
    train_loader, val_loader = prepare_data(cfg)

    # get relevant parameters from config/model_dict
    num_epochs = cfg['num_epochs']
    iou_thresh = model_dict['iou_thresh']
    conf_thresh = model_dict['conf_thresh']

    # tensor of dim [4]
    # note: may have to re-normalize
    loss_weights = model_dict['loss_weights']
    # pull priortized iou function from highest weight in vector
    iou_type = pass

    optimizer = get_optimizer(model_dict)
    # TO Do: adjust lr scheduler parameters
    scheduler = model_dict['scheduler'] if 'scheduler' in model_dict else lr_scheduler.CosineAnnealingWarmRestarts(optimizer, )
    
    # initialize loss function
    # criterion = c.ComboLoss(bbox_loss, cls_loss, bbox_weight, cls_weight)

    # dictionary used to map (flight_id, frame_id) -> output dictionary { 'boxes': [box1, box2, ...], 'scores': }
    all_preds = {}

    # create pandas dataframe where each row is different epoch of metrics
    metrics_df = u.create_metrics_df(num_epochs)

    # training loop
    for epoch in range(1, num_epochs + 1):

       train_epoch_loss = train_one_epoch(model, train_loader, loss_weights, criterion, optimizer, scheduler)

        # save model checkpoint with weights
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_epoch_loss.item()
        }, outdir)

        # retrieve metrics for one epoch of validation
        epoch_metrics = val_one_epoch(model, val_loader, iou_thresh, conf_thresh, loss_weights, criterion, all_preds)
        epoch_metrics['epoch'] = epoch

        # log metrics in dataframe
        u.log_epoch_metrics(metrics_df, epoch, epoch_metrics)

# function to train the model for one epoch
def train_one_epoch(model, train_loader, loss_weights, criterion, optimizer, scheduler):
    # set model to training mode
    model.train()

    # initialize accumulators
    train_epoch_loss = 0.0
    num_images = 0

    # loader yields each batch as a tuple (images, targets)
    # iterates through each batch
    for images, targets in tqdm(train_loader):

        # list of image tensors in the batch of shape [channels, height, width]
        images = [img.cuda() for img in images]

        # list of dictionaries where each represents the ground-truth data of one image in the batch
        # each dictionary contains tensor 'boxes' of shape [num_bboxes, 4]
        # each box is represented as [left, top, width, height]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        # list of dictionaries where each represents the predicted data for one image in the batch
        # each dictionary also contains boxes 2D tensor and scores 1D tensor
        outputs = model(images)

        # iterate through the list of outputs to access results on a per-image basis
        for i, output in enumerate(outputs):
            num_images += 1

            # retrive true boxes and predicted boxes + confidence scoress
            true_boxes = targets[i]['bboxes']
            pred_boxes = output['boxes']
            scores = output['scores']

            # concatenate scores on the end of the predicted bounding boxes
            pred_boxes = u.cat_scores(pred_boxes, scores)

            # calculate loss per image and add to batch loss
            train_image_loss = c.compute_iou_loss(pred_boxes, true_boxes, loss_weights)
            train_epoch_loss += train_image_loss

        # backpropagation
        train_epoch_loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    
    # mean epoch loss for number of batches
    train_epoch_loss /= num_images
    scheduler.step()

    # To DO: also return train time with logger
    return train_epoch_loss


# function to validate model for one epoch
def val_one_epoch(model, val_loader, iou_thresh, conf_thresh, loss_weights, criterion, iou_type, all_preds):
    # set model to evaluation mode
    model.eval()

    # initialize accumulators
    val_epoch_loss = 0.0
    num_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_f1_scores = []
    total_precisions = []
    total_recalls = []
    total_avg_pre = []

    # disables gradient calculations
    with torch.no_grad():
        data_iter = tqdm(val_loader)
        # iterates by batch
        for images, targets in data_iter:
            images = [img.cuda() for img in images]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # list of output dictionaries, where each corresponds to the predictions for an img in the batch
            outputs = model(images)

            # iterate through each image's output dictionary
            for i, output in enumerate(outputs):
                num_images += 1

                # gets the true boxes and predicted boxes + scores for the same image
                true_boxes = targets[i]['bboxes']
                pred_boxes = output['boxes']
                scores = output['scores']
                pred_boxes = u.cat_scores(pred_boxes, scores)

                # save mapping of (flight, frame) -> (predictions)
                flight_id = targets[i]['flight_id']
                frame_id = targets[i]['id']

                # save predictied model outputs in all_preds dictionary
                all_preds[(flight_id, frame_id)] = output

                # gets the bboxes that actually match from the predictions and truths, as well as false positive and false negative predictions
                matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)

                # calculate confusion matrix for a single image and add them to accumulators
                tp = len(matches)
                fp = len(fp)
                fn = len(fn)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # calculate f1, precision, and recall for a single image
                pre = u.precision(tp, fp)
                rec = u.recall(tp, fn)
                total_precisions.append(pre)
                total_recalls.append(rec)

                # calculate precision-recall curve and average precision for a single image
                pre_curve, rec_curve = u.precision_recall_curve(pred_boxes, true_boxes, conf_thresh)
                avg_pre = u.AP(pre_curve, rec_curve)
                total_avg_pre.append(avg_pre)

                # default loss calculation is iou but can be configured to another calculation
                val_image_loss = c.compute_iou_loss(pred_boxes, true_boxes, loss_weights)
                # add image loss to the accumulated epoch loss
                val_epoch_loss += val_image_loss

    # mean epoch loss by the number of images seen in the epoch
    val_epoch_loss /= num_images

    # calculate per-epoch metrics
    epoch_pre = sum(total_precisions) / len(total_precisions)
    epoch_rec = sum (total_recalls) / len(total_recalls)
    epoch_f1 = 2 * (epoch_pre * epoch_rec) / (epoch_pre + epoch_rec)
    epoch_avg_pre = sum(total_avg_pre) / len(total_avg_pre)

    # returns dict of metrics from one epoch
    return {
        'epoch_loss': val_epoch_loss,
        'precision': epoch_pre,
        'recall': epoch_rec,
        'f1_score': epoch_f1,
        'average_precision': epoch_avg_pre,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
    }

# returns iterable train and validation dataloaders using custom dataloader defined
# TO DO: implement provided data loader class
def prepare_data(batch_size=64):
    notebook_path = os.path.dirname(os.path.realpath("__file__"))
    local_path = notebook_path + '/data/part1' #this might need to be changed
    s3_path = 's3://airborne-obj-detection-challenge-training/part1/'
    dataset = NewDataset(local_path, s3_path, partial=True, prefix="part1")
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size, shuffle=True)
    return train_loader, val_loader

def dynamic_batch_size(model):
    return

def store_metrics(metrics_df: pd.DataFrame, best_epoch: dict, outdir: str): # stores all metrics for an individual. Expects a dataframe and a dict for the best epoch metrics
    all_metrics_out = f'{outdir}/generation_{gen_num}/{hash}/metrics.csv'
    best_epoch_out = f'{outdir}/generation_{gen_num}/{hash}/best_epoch.csv'
    os.makedirs(os.path.dirname(all_metrics_out), exist_ok=True)
    metrics_df.to_csv(all)
    with open(best_epoch_out, 'w') as fh:
        fh.write(json.dumps(best_epoch))


if __name__ == "__main__": # makes sure this happens if the script is being run directly
    # parses arguments from sbatch job
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    parser.add_argument('-i', '--infile', required=False, default='/gv1/projects/GRIP_Precog_Opt/precog-opt-grip/eval_input.csv')
    parser.add_argument('-o', '--outdir', required=False, default='/gv1/projects/GRIP_Precog_Opt/outputs')
    args = parser.parse_args()
    index = args.index
    infile = args.infile
    outdir = args.outdir

    # load config attributes
    configs = toml.load("conf.toml")
    pipeline_config = configs["pipeline"]
    codec_config = configs["codec"]
    all_config = pipeline_config | codec_config

    # load generated input for current generation
    with open(f'{infile}', 'r') as input_file:
        file = list(csv.reader(input_file))
        line = file[int(index)+1]
        gen_num = line[0]
        hash = line[1]
        genome = line[2]

    # passes in the command-line args to begin the evaluation
    engine(all_config, genome) # note genome is still in string form