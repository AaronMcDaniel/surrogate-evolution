"""
Script that evaluates a single individual. 
This script is normally sbatched, but can be used to evaluate a single individual if the appropriate command line arguments are provided.
"""


import itertools
import sys
import argparse
import csv
import pickle
import traceback
import toml
import aot_dataset as data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import os
import model_summary as summary
import inspect
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
import utils as u
import criterion as c
from codec import Codec


# wrapper function to log important information
def eval_wrapper(cfg, gen_num, hash, genome, eval: callable):
    os.system('nvidia-smi')
    hostname = os.uname().nodename
    print()
    print("====================")
    print(f'Running on node: {hostname.split(".")[0]}')
    print("--------------------")
    print(f'Gen: {gen_num} Genome hash: {hash}')
    print("====================")
    print()
    summary.tree_genome_summary(genome, cfg['num_loss_components'])
    print()
    try:
        eval(cfg, genome)
    except Exception as e:
        print("==========ERROR==========")
        traceback.print_exc()
        print(e)
        print()
        print('See error log for more information')
        print("--------------------")


# data loader creation
def prepare_data(cfg, train_seed, val_seed, batch_size=5):
    cache_thresh = cfg['cache_thresh']
    max_size = None
    try:
        max_size = cfg['max_size']
    except KeyError:
        pass 
    train_dataset = data.AOTDataset('train', seed=train_seed, string=1, cache_thresh=cache_thresh)
    val_dataset = data.AOTDataset('val', seed=val_seed, string=1, cache_thresh=cache_thresh, max_size=max_size)
    train_sampler = data.AOTSampler(train_dataset, batch_size, train_seed)
    val_sampler = data.AOTSampler(val_dataset, batch_size, val_seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=data.my_collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=data.my_collate, num_workers=4)
    return train_loader, val_loader


# converts images from data loader to pytorch acceptable format
def process_images(images, device):
    # stack images convert to float, and normalize between [0, 1]
    images = torch.stack(images)
    # permute to (B, C, H, W),
    images = images.permute(0, 3, 1, 2).float()
    # normalize between [0, 1]
    images = images / 255.0
    # move to device
    images = images.to(device)
    return images


# converts tagerts from data loader to pytorch acceptable format
def process_targets(targets, device):
    for target in targets:
        num_detections = target['num_detections']
        for k, v in target.items():
            # move to device if tensor
            if isinstance(v, torch.Tensor):
                target[k] = v.to(device)
            if k == 'boxes':
                # slice off empty boxes and convert to [x1, y1, x2, y2]
                target[k] = target[k][:num_detections, :]
                target[k] = u.convert_boxes_to_x1y1x2y2(target[k])
    return targets


# converts model outputs to format for evaluation
def process_preds_truths(target, output):
    flight_id = target['flight_id']
    frame_id = target['frame_id']
    true_boxes = u.norm_box_scale(target['boxes'])
    pred_boxes = u.norm_box_scale(output['boxes'])
    pred_boxes = u.convert_boxes_to_xywh(pred_boxes)
    true_boxes = u.convert_boxes_to_xywh(true_boxes)
    pred_boxes = u.clean_zero_dim(pred_boxes)
    scores = output['scores']
    pred_boxes = u.cat_scores(pred_boxes, scores)
    return pred_boxes, true_boxes, flight_id, frame_id


def create_metrics_df():
    return pd.DataFrame(columns=['epoch_num', 'train_epoch_loss', 'uw_val_epoch_loss', 'val_epoch_loss', 
                                'iou_loss', 'giou_loss', 'diou_loss', 'ciou_loss',
                                'center_loss', 'size_loss', 'obj_loss', 'precision', 
                                'recall', 'f1_score', 'average_precision', 
                                'true_positives', 'false_positives', 'false_negatives'])


# saves latest model's weights to disc and checks if current epoch is also the best epoch
def save_best_last_epochs(model, metrics_df, curr_epoch, criteria):
    last_epoch_out = f'{outdir}/generation_{gen_num}/{hash}/last_epoch.pth'
    best_epoch_out = f'{outdir}/generation_{gen_num}/{hash}/best_epoch.pth'

    os.makedirs(os.path.dirname(last_epoch_out), exist_ok=True)
    
    torch.save(model.state_dict(), last_epoch_out)
    
    # retrieve best epoch as epoch with lowest validation loss
    if criteria[1] == 'max':
        best_epoch = metrics_df[criteria[0]].idxmax() + 1 # NOTE best epoch won't get saved if val loss is nan
    else:
        best_epoch = metrics_df[criteria[0]].idxmin() + 1 # NOTE best epoch won't get saved if val loss is nan
    if curr_epoch == best_epoch:
        torch.save(model.state_dict(), best_epoch_out)


# saves all model metrics and predictions to disc
def store_data(metrics_df: pd.DataFrame, all_preds: dict):
    metrics_out = f'{outdir}/generation_{gen_num}/{hash}/metrics.csv'
    pickled_preds_out = f'{outdir}/generation_{gen_num}/{hash}/predictions.pkl'
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)

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
        'RAdam': optim.RAdam,
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


def step_scheduler(scheduler, loss):
    if type(scheduler) == lr_scheduler.ReduceLROnPlateau:
        scheduler.step(loss)
    else:
        scheduler.step()


# main training and eval loop
def engine(cfg, genome):

    # retrieve config attributes
    genome_encoding_strat = cfg['genome_encoding_strat']
    num_classes = cfg['num_classes']
    num_loss_comp = cfg['num_loss_components']
    batch_size = cfg['batch_size']
    batches_per_epoch = cfg['batches_per_epoch']
    num_epochs = cfg['train_epochs']
    iou_thresh = cfg['iou_thresh']
    conf_thresh = cfg['conf_thresh']
    iou_type = cfg['iou_type']
    best_epoch_criteria = cfg['best_epoch_criteria']
    train_seed = cfg['train_seed']
    val_seed = cfg['val_seed']
    
    # set device and load data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, val_loader = prepare_data(cfg, train_seed, val_seed, batch_size)

    # initialize codec and decode genome
    codec = Codec(num_classes, genome_encoding_strat=genome_encoding_strat)
    model_dict = codec.decode_genome(genome, num_loss_comp)
    loss_weights = model_dict['loss_weights']

    # get model
    model = model_dict['model'].to(device)
    params = model.parameters()

    # get optimizer and scheduler
    optimizer = get_optimizer(params, model_dict)
    scheduler = get_scheduler(optimizer, model_dict, num_epochs, batch_size)
    scaler = GradScaler()

    # all_preds = list of epoch_preds = { (flight_id, frame_id) -> output dict = { 'boxes': [box1, box2, ...], 'scores': ... }}, index is epoch_num - 1
    all_preds = []
    metrics_df = create_metrics_df()

    for epoch in range(1, num_epochs + 1):
       
        # epoch_preds = { (flight_id, frame_id) -> output dictionary { 'boxes': [box1, box2, ...], 'scores': ... }}
        epoch_preds = {}

        train_epoch_loss = train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, loss_weights, iou_type, max_batch=batches_per_epoch)
        epoch_metrics = val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, epoch_preds, max_batch=batches_per_epoch)

        # update metrics_df and all_preds with current epoch's data
        epoch_metrics['epoch_num'] = epoch
        epoch_metrics['train_epoch_loss'] = train_epoch_loss
        epoch_metrics_df = pd.DataFrame([epoch_metrics])
        metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)
        all_preds.append(epoch_preds)

        # save metrics_df, best/last epochs, predictions to disc
        store_data(metrics_df, all_preds)
        save_best_last_epochs(model, metrics_df, epoch, best_epoch_criteria)


def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, loss_weights, iou_type, max_batch=None):
    model.train()
    train_epoch_loss = 0.0

    if max_batch is not None:
        # Slice the dataloader to only include up to max_batch
        train_loader = itertools.islice(train_loader, max_batch)
    data_iter = tqdm(train_loader, desc="Training", total=max_batch)

    for i, (images, targets) in enumerate(data_iter):

        # process images and targets
        images = process_images(images, device)
        targets = process_targets(targets, device)

        # forwards with mixed precision
        with autocast():
            loss_dict, outputs = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # backwards
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        data_iter.set_postfix(loss=losses.item())
        torch.cuda.empty_cache()

        # update accumulating loss
        train_epoch_loss += losses.item()

    train_epoch_loss = train_epoch_loss / max_batch if max_batch is not None else len(train_loader)
    step_scheduler(scheduler, train_epoch_loss)
    return train_epoch_loss


def val_one_epoch(model, device, val_loader, iou_thresh, conf_thresh, loss_weights, iou_type, epoch_preds, max_batch=None):
    confidences, confusion_status = [], []
    val_epoch_loss, iou_loss, giou_loss, diou_loss, ciou_loss, center_loss, size_loss, obj_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_preds, num_labels, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0
    model.eval()
    if max_batch is not None:
        # Slice the dataloader to only include up to max_batch
        val_loader = itertools.islice(val_loader, max_batch)
        data_iter = tqdm(val_loader, desc="Evaluating", total=max_batch)
    else:     
        data_iter = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_iter):
            images = process_images(images, device)
            targets = process_targets(targets, device)
            with autocast():
                outputs = model(images, targets)

            val_batch_loss = torch.zeros(1, dtype=torch.float32, device=device)
            for j, image in enumerate(images):
                pred_boxes, true_boxes, flight_id, frame_id = process_preds_truths(targets[j], outputs[j])
                # NOTE pred boxes are normalized, in xywh format, and have scores, and true boxes are nomalized and in xywh format

                # update epoch_preds
                epoch_preds[(flight_id, frame_id)] = outputs[j]

                matches, fp, fn = u.match_boxes(pred_boxes, true_boxes, iou_thresh, conf_thresh, "val", iou_type)
                num_tp = len(matches)
                num_fp = len(fp)
                num_fn = len(fn)
                total_tp += num_tp
                total_fp += num_fp
                total_fn += num_fn
                num_preds += len(pred_boxes)
                num_labels += len(true_boxes)
                loss_matches = u.match_boxes(pred_boxes, true_boxes, 0.0, 0.0, "train", iou_type)
                loss_tensor = c.compute_weighted_loss(loss_matches, pred_boxes, true_boxes, loss_weights, iou_type)
                
                val_image_loss = loss_tensor[7]
                val_batch_loss += val_image_loss
                val_epoch_loss += val_image_loss
                iou_loss += loss_tensor[0]
                giou_loss += loss_tensor[1]
                diou_loss += loss_tensor[2]
                ciou_loss += loss_tensor[3]
                center_loss += loss_tensor[4]
                size_loss += loss_tensor[5]
                obj_loss += loss_tensor[6]

                curve_matches, curve_fp, _ = u.match_boxes(pred_boxes, true_boxes, iou_thresh, 0.0, 'val', iou_type)

                for _, (true_pos, _) in curve_matches.items():
                    confidences.append(true_pos[4].item())
                    confusion_status.append(True)
                for false_pos in curve_fp:
                    confidences.append(false_pos[4].item())
                    confusion_status.append(False)

            data_iter.set_postfix(loss=val_batch_loss)
            torch.cuda.empty_cache()

    val_epoch_loss /= (num_preds +  1e-9)
    iou_loss /= (num_preds  + 1e-9)
    giou_loss /= (num_preds + 1e-9)
    diou_loss /= (num_preds + 1e-9)
    ciou_loss/= (num_preds + 1e-9)
    center_loss/= (num_preds + 1e-9)
    size_loss /= (num_preds + 1e-9)
    obj_loss /= (num_preds + 1e-9)
    uw_val_epoch_loss = iou_loss + giou_loss + diou_loss + ciou_loss + center_loss + size_loss + obj_loss
    epoch_f1, epoch_pre, epoch_rec = u.f1_score(total_tp, total_fn, total_fp)
    pre_curve, rec_curve = u.precision_recall_curve(confidences, confusion_status, num_labels)
    epoch_avg_pre = u.AP(pre_curve, rec_curve)
    plot_outdir = f'{outdir}/generation_{gen_num}/{hash}'
    u.plot_PR_curve(pre_curve, rec_curve, epoch_avg_pre, plot_outdir)

    epoch_metrics = {
        'uw_val_epoch_loss': uw_val_epoch_loss.item(),
        'val_epoch_loss': val_epoch_loss.item(),
        'iou_loss':iou_loss.item(),
        'giou_loss': giou_loss.item(),
        'diou_loss': diou_loss.item(),
        'ciou_loss': ciou_loss.item(),
        'center_loss': center_loss.item(),
        'size_loss': size_loss.item(),
        'obj_loss': obj_loss.item(),
        'precision': epoch_pre,
        'recall': epoch_rec,
        'f1_score': epoch_f1,
        'average_precision': epoch_avg_pre,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
    }
    return epoch_metrics


if __name__ == '__main__':
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
    configs = toml.load(os.path.join(outdir, "conf.toml"))
    model_config = configs["model"]
    codec_config = configs["codec"]
    data_config = configs["data"]
    pipeline_config = configs["pipeline"]
    all_config = model_config | codec_config | data_config | pipeline_config

    # load generated input for current generation
    input_file = open(f'{infile}', 'r')
    file = list(csv.reader(input_file))
    line = file[int(index)+1]
    gen_num = line[0]
    hash = line[1]
    genome = line[2]
    input_file.close()

    # evaluate
    eval_wrapper(all_config, gen_num, hash, genome, engine)










