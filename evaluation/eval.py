import csv
import toml
import sys

import os
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
sys.path.insert(0, '/gv1/projects/GRIP_Precog_Opt/precog-opt-grip')
from dataset.aot_dataset import AOTDataset
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
from torch.optim import lr_scheduler, optimizer
import inspect
import pickle

def engine(cfg, genome):

    # retrieves codec arguments from the config and decodes the genome into a model_dict
    genome_encoding_strat = cfg['genome_encoding_strat']
    num_classes = cfg['num_classes']
    num_loss_comp = cfg['num_loss_components']
    codec = Codec(genome_encoding_strat, num_classes)

    # dictionary contains model, and other information encoded in the genome (loss, optimizer)
    model_dict = codec.decode_genome(genome, num_loss_comp)

    # pytorch model: input = 4D tensor -> [batch_size, num_channels, width, height],
    model = model_dict['model']

    # move model to device and retrieve trainable parameters
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # note: use dynamic_batch_size function when implemented
    batch_size = cfg['batch_size']

    # TO DO: implement later
    train_loader, val_loader = prepare_data(cfg)

    # get relevant parameters from config/model_dict
    num_epochs = cfg['num_epochs']
    iou_thresh = model_dict['iou_thresh']
    conf_thresh = model_dict['conf_thresh']

    # 1D tensor of dim [4] -> [iou_w, giou_w, diou_w, ciou_w]
    # note: may have to re-normalize, depending on loss function components
    loss_weights = model_dict['loss_weights']
    # pull priortized iou function from highest weight in vector
    # _, max_weight_idx = torch.max(loss_weights, dim=0)
    # iou_type = loss_weights[max_weight_idx]
    iou_type = "ciou"

    # Use model_dict to retrieve optimizer and scheduler
    optimizer = get_optimizer(params, model_dict)
    # pass in num_epochs and batch_size for certain scheduler parameters
    scheduler = get_scheduler(optimizer, model_dict, num_epochs, len(train_loader))

    # all_preds = {epoch_num -> epoch_preds = { (flight_id, frame_id) -> output dictionary { 'boxes': [box1, box2, ...], 'scores': ... }}}
    all_preds = []
    # create pandas dataframe where each row is different epoch of metrics
    metrics_df = create_metrics_df()

    # training loop
    for epoch in range(1, num_epochs + 1):
       
        # epoch_preds = { (flight_id, frame_id) -> output dictionary { 'boxes': [box1, box2, ...], 'scores': ... }}
        epoch_preds = {}

        # train and validate
        train_epoch_loss = train_one_epoch(model, device, train_loader, loss_weights, iou_type, optimizer, scheduler)
        epoch_metrics = val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, epoch_preds)

        # update metrics_df and all_preds with current epoch's data
        epoch_metrics['epoch_num'] = epoch
        epoch_metrics['train_epoch_loss'] = train_epoch_loss
        metrics_df = metrics_df.append(epoch_metrics, ignore_index=True)
        all_preds[epoch - 1] = epoch_preds

        # save metrics_df, best/last epochs, predictions to disc
        save_best_last_epochs(model, metrics_df, epoch)
        store_data(metrics_df, all_preds)

# function to train the model for one epoch
def train_one_epoch(model, device, train_loader, loss_weights, iou_type, optimizer, scheduler):
    # set model to training mode
    model.train()

    # initialize accumulators
    train_epoch_loss = 0.0
    num_preds = 0

    # loader yields each batch as a tuple (images, targets)
    data_iter = tqdm(train_loader)
    for images, targets in data_iter:

        # list of image numpy arrays in the batch of shape [channels, height, width] -> [3, 2048. 2488]
        images = [numpy_to_tensor(img, device) for img in images]

        # list of dictionaries where each represents the ground-truth data of one image in the batch
        # each dictionary contains tensor 'boxes' of shape [num_bboxes, 4]
        # each box is represented as [left, top, width, height]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # list of dictionaries where each represents the predicted data for one image in the batch
        # each dictionary also contains boxes 2D tensor and scores 1D tensor
        outputs = model(images)

        # iterate through the list of outputs to access results on a per-image basis
        for i, output in enumerate(outputs):

            # retrive true boxes and predicted boxes + confidence scoress
            true_boxes = targets[i]['bboxes']
            pred_boxes = output['boxes']
            scores = output['scores']

            # concatenate scores on the end of the predicted bounding boxes
            pred_boxes = u.cat_scores(pred_boxes, scores)
            matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type)
            
            # compute_loss returns a tensor of dim[8]
            # the last loss value in the tensor is the weighted sum of the 7 loss components
            loss_tensor = c.compute_iou_loss(matches, loss_weights, iou_type)
            train_image_loss = loss_tensor[7]
            train_epoch_loss += train_image_loss
            num_preds += len(pred_boxes)

        # backpropagation
        train_epoch_loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    
    # divide total train epoch loss by number of predictions 
    train_epoch_loss /= num_preds
    scheduler.step()

    # returns dict containing the meaned train epoch loss
    return {
        'train_epoch_loss': train_epoch_loss,
    }


# function to validate model for one epoch
def val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, epoch_preds):
    # set model to evaluation mode
    model.eval()

    # initialize accumulators
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0

    # list stores confidences of all predictions
    confidences = []
    # list stores boolean representing whether a prediction is a true positive or false negative
    confusion_status = []

    # disables gradient calculations
    with torch.no_grad():
        data_iter = tqdm(val_loader)
        # iterates by batch
        for images, targets in data_iter:
            # convert image numpy array to tensor and move to device
            images = [numpy_to_tensor(img, device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # list of output dictionaries, where each corresponds to the predictions for an img in the batch
            outputs = model(images)

            # iterate through each image's output dictionary
            for i, output in enumerate(outputs):

                # gets the true boxes and predicted boxes + scores for the same image
                true_boxes = targets[i]['bboxes']
                pred_boxes = output['boxes']
                scores = output['scores']
                pred_boxes = u.cat_scores(pred_boxes, scores)

                # save mapping of (flight, frame) -> (predictions)
                flight_id = targets[i]['flight_id']
                frame_id = targets[i]['id']

                # save predictied model outputs in all_preds dictionary
                epoch_preds[(flight_id, frame_id)] = output

                # gets the bboxes that actually match from the predictions and truths, as well as false positive and false negative predictions
                matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)

                # calculate confusion matrix for a single image and add them to accumulators
                num_tp = len(matches)
                num_fp = len(fp)
                num_fn = len(fn)
                total_tp += num_tp
                total_fp += num_fp
                total_fn += num_fn

                num_preds += len(pred_boxes)
                num_labels += len(true_boxes)

                loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", 'ciou')
                loss_tensor = c.compute_iou_loss(loss_matches, loss_weights, iou_type)

                # weighted sum loss = index 7 of loss tensor
                val_image_loss = loss_tensor[7]
                val_epoch_loss += val_image_loss
                iou_loss += loss_tensor[0]
                giou_loss += loss_tensor[1]
                diou_loss += loss_tensor[2]
                ciou_loss += loss_tensor[3]
                center_loss += loss_tensor[4]
                size_loss += loss_tensor[5]
                obj_loss += loss_tensor[6]

                # update confidences and confusion_status lists
                for _, (true_pos, _) in matches.items():
                    confidences.append(true_pos[4].item())
                    confusion_status.append(True)
                for false_pos in fp:
                    confidences.append(false_pos[4].item())
                    confusion_status.append(False)

    # mean epoch loss by the number of images seen in the epoch
    val_epoch_loss /= num_preds
    iou_loss /= num_preds
    giou_loss /= num_preds
    diou_loss /= num_preds
    ciou_loss/= num_preds
    center_loss/= num_preds
    size_loss /= num_preds
    obj_loss /= num_preds

    # calculate per-epoch metrics
    epoch_f1, epoch_pre, epoch_rec = u.f1_score(total_tp, total_fn, total_fp)

    pre_curve, rec_curve = u.precision_recall_curve(confidences, confusion_status, num_labels)
    epoch_avg_pre = u.AP(pre_curve, rec_curve)

    # returns dict of metrics from one epoch
    return {
        'val_epoch_loss': val_epoch_loss,
        'iou_loss':iou_loss,
        'giou_loss': giou_loss,
        'diou_loss': diou_loss,
        'ciou_loss': ciou_loss,
        'center_loss': center_loss,
        'size_loss': size_loss,
        'obj_loss': obj_loss,
        'precision': epoch_pre,
        'recall': epoch_rec,
        'f1_score': epoch_f1,
        'average_precision': epoch_avg_pre,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
    }

# returns iterable train and validation dataloaders 
def prepare_data(batch_size=64):

    train_dataset = AOTDataset('train')
    val_dataset = AOTDataset('val')

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    return train_loader, val_loader

# TO DO: implement this function
def dynamic_batch_size(model):
    return 64

def create_metrics_df():
    return pd.DataFrame(columns=['epoch_num', 'train_epoch_loss', 'val_epoch_loss', 
                                       'iou_loss', 'giou_loss', 'diou_loss', 'ciou_loss',
                                       'center_loss', 'size_loss', 'obj_loss', 'precision', 
                                       'recall', 'f1_score', 'average_precision', 
                                       'true_positives', 'false_positives', 'false_negatives'])

# converts numpy array to tensor and moves it to the device
def numpy_to_tensor(image, device):
    img = torch.tensor(image, dtype=torch.float32)
    # if image is in [height, width, channels], convert to expected shape of [channels, height, width]
    # tensor.permute changes the order of dimensions in a tensor
    img = img.permute(2, 0, 1)
    return img.to(device)

# saves latest model's weights to disc and checks if current epoch is also the best epoch
def save_best_last_epochs(model, metrics_df, curr_epoch):
    last_epoch_out = f'{outdir}/generation_{gen_num}/{hash}/last_epoch.pth'
    best_epoch_out = f'{outdir}/generation_{gen_num}/{hash}/best_epoch.pth'

    # create directories if they don't exist
    os.makedirs(os.path.dirname(last_epoch_out), exist_ok=True)
    os.makedirs(os.path.dirname(best_epoch_out), exist_ok=True)
    
    # save last epoch weights
    torch.save(model.state_dict(), last_epoch_out)
    
    # retrieve best epoch as epoch with lowest validation loss
    best_epoch = metrics_df['val_epoch_loss'].idxmin()
    if curr_epoch == best_epoch:
        torch.save(model.state_dict(), best_epoch_out)

# saves all model metrics and predictions to disc
def store_data(metrics_df: pd.DataFrame, all_preds: dict):
    metrics_out = f'{outdir}/generation_{gen_num}/{hash}/metrics.csv'
    pickled_preds_out = f'{outdir}/generation_{gen_num}/{hash}/predictions.pkl'
    
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    
    # save metrics dataframe to CSV
    metrics_df.to_csv(metrics_out, index=False)
    
    # pickle the all_preds dictionary
    with open(pickled_preds_out, 'wb') as f:
        # serializes all_preds dictionary and writes to out file
        pickle.dump(all_preds, f)

# returns learning rate scheduler based on configuration defined by the genome
def get_scheduler(optimizer, model_dict, num_epochs, batch_size):

    # set default values for various potential scheduler params
    default_params = {
        'StepLR': {'step_size': 30, 'gamma': 0.1, 'last_epoch': -1},
        'MultiStepLR': {'milestones': [30, 80], 'gamma': 0.1, 'last_epoch': -1},
        'ExponentialLR': {'gamma': 0.1, 'last_epoch': -1},
        'ReduceLROnPlateau': {'mode': 'min', 'factor': 0.1, 'patience': 10, 'threshold': 0.0001, 'cooldown': 0, 'min_lr': 0, 'eps': 1e-08},
        'CosineAnnealingLR': {'T_max': 50, 'eta_min': 0, 'last_epoch': -1},
        'CosineAnnealingWarmRestarts': {'T_0': 10, 'T_mult': 2, 'eta_min': 0, 'last_epoch': -1},
        'OneCycleLR': {'max_lr': 0.1, 'total_steps': None, 'epochs': num_epochs, 'steps_per_epoch': batch_size, 'pct_start': 0.3, 'anneal_strategy': 'cos', 'cycle_momentum': True, 'base_momentum': 0.85, 'max_momentum': 0.95, 'div_factor': 25.0, 'final_div_factor': 1e4, 'three_phase': False, 'last_epoch': -1, 'verbose': False},
        'ConstantLR': {'factor': 1.0, 'total_iters': 5},
        'MultiplicativeLR': {'lr_lambda': lambda epoch: 0.95, 'last_epoch': -1},
        'LambdaLR': {'lr_lambda': lambda epoch: 1, 'last_epoch': -1},
        'LinearLR': {'start_factor': 0.1, 'end_factor': 1.0, 'total_iters': 5, 'last_epoch': -1},
        'PolynomialLR': {'max_lr': 0.1, 'total_steps': 10, 'power': 1.0, 'last_epoch': -1},
        'ChainedScheduler': {'schedulers': [], 'last_epoch': -1},
        'CyclicLR': {'base_lr': 0.001, 'max_lr': 0.1, 'step_size_up': 2000, 'step_size_down': None, 'mode': 'triangular', 'gamma': 1.0, 'scale_fn': None, 'scale_mode': 'cycle', 'cycle_momentum': True, 'base_momentum': 0.8, 'max_momentum': 0.9, 'last_epoch': -1},
        'SequentialLR': {'schedulers': [], 'milestones': []},
    }

    # get scheduler type, StepLR is default
    scheduler_type = model_dict.get('lr_scheduler', 'StepLR')
    # map scheduler types to their respective classes and retrieve the class
    scheduler_class_map = {
        'StepLR': lr_scheduler.StepLR,
        'MultiStepLR': lr_scheduler.MultiStepLR,
        'ExponentialLR': lr_scheduler.ExponentialLR,
        'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
        'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
        'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'OneCycleLR': lr_scheduler.OneCycleLR,
        'ConstantLR': lr_scheduler.ConstantLR,
        'MultiplicativeLR': lr_scheduler.MultiplicativeLR,
        'LambdaLR': lr_scheduler.LambdaLR,
        'LinearLR': lr_scheduler.LinearLR,
        'PolynomialLR': lr_scheduler.PolynomialLR,
        'ChainedScheduler': lr_scheduler.ChainedScheduler,
        'CyclicLR': lr_scheduler.CyclicLR,
        'SequentialLR': lr_scheduler.SequentialLR
    }
    scheduler_class = scheduler_class_map.get(scheduler_type)
    scheduler_defaults = default_params.get(scheduler_type)

    # update default_params with values from model_dict if they exist
    scheduler_params = {k: model_dict.get(f'scheduler_{k}', v) for k, v in scheduler_defaults.items()}

    # get scheduler signature and filter the parameters that are valid for the scheduler
    sig = inspect.signature(scheduler_class)
    valid_params = {k: v for k, v in scheduler_params.items() if k in sig.parameters}
    valid_params['optimizer'] = optimizer

    # instantiate scheduler with correct parameters
    scheduler = scheduler_class(**valid_params)
    return scheduler

# returns optimizer based on configuration defined by the genome
def get_optimizer(params, model_dict):

    # set default values for various potential optimizer parameters
    default_params = {
        'SGD': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0, 'dampening': 0, 'nesterov': False},
        'Adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 0},
        'Adagrad': {'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0, 'eps': 1e-10},
        'Adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False},
        'AdamW': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01, 'amsgrad': False},
        'SparseAdam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8},
        'Adamax': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0},
        'ASGD': {'lr': 0.01, 'lambd': 1e-4, 'alpha': 0.75, 't0': 1e6, 'weight_decay': 0},
        'LBFGS': {'lr': 1.0, 'max_iter': 20, 'max_eval': None, 'tolerance_grad': 1e-5, 'tolerance_change': 1e-9, 'history_size': 100, 'line_search_fn': None},
        'NAdam': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0, 'momentum_decay': 0.004},
        'RAdam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0},
        'RMSprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0, 'momentum': 0, 'centered': False},
        'Rprop': {'lr': 0.01, 'etas': (0.5, 1.2), 'step_sizes': (1e-6, 50)},
    }

    # retrive optimizer type from model_dict and using class mapping to obtain actual optimizer class
    optimizer_type = model_dict.get('optimizer', 'SGD')
    optimizer_class_map = {
        'SGD': optim.SGD,
        'Adadelta': optim.Adadelta,
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'SparseAdam': optim.SparseAdam,
        'Adamax': optim.Adamax,
        'Adagrad': optim.Adagrad,
        'ASGD': optim.ASGD,
        'LBFGS': optim.LBFGS,
        'NAdam': optim.NAdam,
        'RMSprop': optim.RMSprop,
        'Rprop': optim.Rprop
    }
    optimizer_class = optimizer_class_map.get(optimizer_type)
    optimizer_defaults = default_params.get(optimizer_type, {})

    # retrieve any non-default optimizer params from model_dict
    optimizer_params = {k: model_dict.get(f'optimizer_{k}', v) for k, v in optimizer_defaults.items()}

    # filters passed in parameters based on optimizer signature
    sig = inspect.signature(optimizer_class)
    valid_params = {k: v for k, v in optimizer_params.items() if k in sig.parameters}
    valid_params['params'] = params

    # instantiate the optimizer with the valid parameters
    optimizer = optimizer_class(**valid_params)
    return optimizer

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
    input_file = open(f'{infile}', 'r')
    file = list(csv.reader(input_file))
    line = file[int(index)+1]
    gen_num = line[0]
    hash = line[1]
    genome = line[2]
    input_file.close()
    # passes in the command-line args to begin the evaluation
    engine(all_config, genome) # note genome is still in string form