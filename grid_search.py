"""
Script to launch grid-search process for surrogates.
"""

import itertools
import argparse
import os
import pickle
import toml
import tqdm
import surrogate_models as sm
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import eval as e
import torch.nn as nn
from torch.optim import lr_scheduler as lr
from sklearn.preprocessing import StandardScaler
import itertools
from functools import partial
import surrogate_dataset as sd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

def engine(cfg, model_str, param_combo, combo_num):

    # pull surrogate train/eval config attributes
    num_epochs = cfg['surrogate_train_epochs']
    batch_size = cfg['surrogate_batch_size']

    # define subset of metrics to train on and prepare data accordingly
    metrics_subset = param_combo['metrics_subset']
    train_loader, val_loader, train_dataset = prepare_data(batch_size, metrics_subset=metrics_subset)
    max_metrics = train_dataset.max_metrics
    min_metrics = train_dataset.min_metrics

    # only use cpu for grid search
    device = torch.device('cpu')
    model, optimizer, scheduler, scaler = build_configuration(model_str=model_str, device=device, param_combo=param_combo)

    # create metrics_df
    metrics_df = create_metrics_df()
    for epoch in range(1, num_epochs + 1):

            # train and validate for one epoch
            train_epoch_loss = train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_metrics=max_metrics, min_metrics=min_metrics)
            epoch_metrics = val_one_epoch(model, device, val_loader, metrics_subset=metrics_subset, max_metrics=max_metrics, min_metrics=min_metrics)

            # update metrics df
            epoch_metrics['param_combo'] = str(param_combo)
            epoch_metrics['epoch_num'] = epoch
            epoch_metrics['train_loss'] = train_epoch_loss
            epoch_metrics_df = pd.DataFrame([epoch_metrics])
            metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)
    
    # store data
    store_data(model_str=model_str, combo_num=combo_num, metrics_df=metrics_df)


# prepare data for grid search
def prepare_data(batch_size, metrics_subset):
    train_df = pd.read_pickle('surrogate_dataset/reg_train_dataset.pkl')
    val_df = pd.read_pickle('surrogate_dataset/reg_val_dataset.pkl')
    train_dataset = sd.SurrogateDataset(train_df, mode='train', metrics_subset=metrics_subset)
    val_dataset = sd.SurrogateDataset(val_df, mode='val', metrics_subset=metrics_subset, metrics_scaler=train_dataset.metrics_scaler, genomes_scaler=train_dataset.genomes_scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, train_dataset       


# build model, optimizer, scheduler, and scaler
def build_configuration(model_str, device, param_combo):
    input_size = 1021

    if model_str == "MLP":
        # build MLP
        hidden_sizes = param_combo['hidden_sizes']
        dropout = param_combo['dropout']
        output_size = len(param_combo['metrics_subset'])
        model = sm.MLP(input_size=input_size, output_size=output_size, dropout=dropout, hidden_sizes=hidden_sizes)
        model = model.to(device)
    
    if model_str == "KAN":
        # build KAN
        hidden_sizes = param_combo['hidden_sizes']
        grid_size = param_combo['grid_size']
        spline_order = param_combo['spline_order']
        output_size = len(param_combo['metrics_subset'])
        model = sm.KAN(input_size, output_size, hidden_sizes, spline_order=spline_order, grid_size=grid_size).to(device)

    # build optimizer
    params = model.parameters()
    lr = param_combo['lr']
    optimizer_func = partial(param_combo['optimizer'])
    optimizer = optimizer_func(params=params, lr=lr)

    # build scheduler
    scheduler_func = param_combo['scheduler']
    if scheduler_func == optim.lr_scheduler.StepLR:
        scheduler = scheduler_func(optimizer=optimizer, step_size=10, gamma=0.1)
    elif scheduler_func == optim.lr_scheduler.MultiStepLR:
        scheduler = scheduler_func(optimizer=optimizer, milestones=[10, 20], gamma=0.1)
    elif scheduler_func == optim.lr_scheduler.CosineAnnealingLR:
        scheduler = scheduler_func(optimizer=optimizer, T_max=10)
    elif scheduler_func == optim.lr_scheduler.ReduceLROnPlateau:
        scheduler = scheduler_func(optimizer=optimizer, mode='min', factor=0.1, patience=5)
    elif scheduler_func == optim.lr_scheduler.CosineAnnealingWarmRestarts:
        scheduler = scheduler_func(optimizer=optimizer, T_0=10, T_mult=2)
    elif scheduler_func == optim.lr_scheduler.ExponentialLR:
        scheduler = scheduler_func(optimizer=optimizer, gamma=0.95)
    else:
        scheduler = scheduler_func(optimizer=optimizer)

    scaler = GradScaler()
    return model, optimizer, scheduler, scaler


# stores metrics csv file 
def store_data(model_str, combo_num, metrics_df):
    metrics_out = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/gs_combos/combo{combo_num}_metrics.csv'
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)


# creates metrics dataframe with appropriate rows
def create_metrics_df():
    return pd.DataFrame(columns=[
        'epoch_num',
        'train_loss',
        'val_loss',
        'mse_uw_val_loss', 
        'mse_iou_loss', 
        'mse_giou_loss', 
        'mse_diou_loss', 
        'mse_ciou_loss', 
        'mse_center_loss', 
        'mse_size_loss', 
        'mse_obj_loss', 
        'mse_precision',
        'mse_recall', 
        'mse_f1_score',
        'mse_average_precision',
        'param_combo'
    ])


def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_metrics, min_metrics):
    model.train()

    # actual surrogate training loss
    surrogate_train_loss = 0.0
    # mean taken for metric regression losses in train
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    data_iter = tqdm(train_loader, desc='Training')
    for i, (genomes, metrics) in enumerate(data_iter):

        # genomes shape: (batch_size, 976)
        genomes = genomes.to(device)
        # metrics shape: (batch_size, 12)
        metrics = metrics.to(device)

        # forwards with mixed precision
        with autocast():
            # outputs shape: (batch_size, 12)
            outputs = model(genomes, update_grid=False)
            clamped_outputs = torch.clamp(outputs, min=(min_metrics.to(device)), max=(max_metrics.to(device)))
            # metric regression loss is meaned, so is scalar tensor value
            loss = criterion(clamped_outputs, metrics)
                   
        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        surrogate_train_loss += loss.item()
        data_iter.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()
    
    # step scheduler
    e.step_scheduler(scheduler, loss)

    # calculate surrogate training loss per batch (NOTE batch loss already meaned by batch size)
    num_batches = len(data_iter)
    surrogate_train_loss /= num_batches

    return surrogate_train_loss


def val_one_epoch(model, device, val_loader, metrics_subset, max_metrics, min_metrics):
    model.eval()

    # actual surrogate validation loss
    surrogate_val_loss = 0.0
    # mse loss matrx for each metric, where rows = num batches and cols = 12 for all predicted metrics
    mse_metrics_per_batch = []
    # no mean taken for losses in validation 
    # criterion = nn.MSELoss(reduction='none')
    criterion = nn.L1Loss()
    
    metric_names = [
        'mse_uw_val_loss', 'mse_iou_loss', 'mse_giou_loss', 'mse_diou_loss', 
        'mse_ciou_loss', 'mse_center_loss', 'mse_size_loss', 'mse_obj_loss', 
        'mse_precision', 'mse_recall', 'mse_f1_score', 'mse_average_precision'
    ]
    selected_metric_names = [metric_names[i] for i in metrics_subset]

    data_iter = tqdm(val_loader, 'Evaluating')
    with torch.no_grad():
        for genomes, metrics in data_iter:
            # genomes shape: (batch_size, 976)
            genomes = genomes.to(device)
            # metrics shape: (batch_size, 12)
            metrics = metrics.to(device)

            # forwards with mixed precision
            with autocast():
                # outputs shape: (batch_size, 12)
                outputs = model(genomes)
                clamped_outputs = torch.clamp(outputs, min=(min_metrics.to(device)), max=(max_metrics.to(device)))
                loss_matrix = criterion(clamped_outputs, metrics)

                # loss tensor shape: (12)
                loss_tensor = torch.mean(loss_matrix, dim=0)
                # loss is meaned, so is scalar tensor value
                loss = torch.mean(loss_tensor)
            
            # update validation loss
            surrogate_val_loss += loss.item()
            # update matrix storing mse loss for each metric for each batch
            mse_metrics_per_batch.append(loss_tensor)
            data_iter.set_postfix(loss=loss)
            torch.cuda.empty_cache()
            
    num_batches = len(data_iter)
    surrogate_val_loss /= num_batches

    # compute the mean of the mse losses for each metric based on num batches
    mse_metrics_per_batch = torch.stack(mse_metrics_per_batch)
    mse_metrics_meaned = mse_metrics_per_batch #.mean(dim=0)

    epoch_metrics = {
        'val_loss': surrogate_val_loss
    }
    print(surrogate_val_loss)
    epoch_metrics.update({
        selected_metric_names[i]: mse_metrics_meaned[i].item() for i in range(len(metrics_subset))
    })

    return epoch_metrics


# uses model string to concatenate grid search resulting metric csvs to one master file
def cat_results(model_str='KAN'):
    search_dir = f'/gv1/projects/GRIP_Precog_Opt/surrogates/KAN/gs_combos'
    master_df = create_metrics_df()

    # change range as necessary for different grid search runs
    for i in range(8640):
        metrics_path = search_dir + f'/combo{i}_metrics.csv'
        try:
            metrics_df = pd.read_csv(metrics_path)
        except:
            continue
        master_df = pd.concat([master_df, metrics_df], ignore_index=True)
    out_path = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/full_gs_master.csv'
    master_df.to_csv(out_path, index=False)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--combo_num', type=int, required=True, default=None)
    parser.add_argument('-cp', '--cfg_path', type=str, required=False, default='/home/tthakur9/precog-opt-grip/conf.toml')

    # argument determines whether old csv files should be overwritten
    parser.add_argument('-o', '--overwrite', type=str, required=False, default='true')
    args = parser.parse_args()
    combo_num = args.combo_num
    cfg_path = args.cfg_path
    overwrite = args.overwrite

    # load config
    all_cfg = toml.load(cfg_path)
    cfg = all_cfg['surrogate']
    model_str = "KAN"

    if model_str == "MLP":
        # define MLP-unique parameter grid
        param_grid = {
                    'dropout': [0.0, 0.2, 0.4, 0.6],
                    'hidden_sizes': [[512, 256], [1024, 512], [2048, 1024, 512]],
                    'optimizer': [optim.SGD, optim.Adam, optim.RMSprop, optim.Adagrad],
                    'lr': [0.0001, 0.001, 0.01, 0.1],
                    'scheduler': [lr.StepLR, lr.MultiStepLR, lr.CosineAnnealingLR, lr.ReduceLROnPlateau],
                    'metrics_subset': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 4, 11], [11], [4], [0]]
        }

    elif model_str == "KAN":
        # define unique KAN param grid
        param_grid = {
                    'hidden_sizes': [[512, 256], [2048, 1024, 512], [2048], []],
                    'optimizer': [optim.SGD, optim.RMSprop, optim.AdamW],
                    'lr': [0.001, 0.01, 0.1],
                    'scheduler': [lr.StepLR, lr.CosineAnnealingWarmRestarts, lr.ReduceLROnPlateau],
                    'metrics_subset': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 4, 11], [11], [4], [0]],
                    'spline_order': [1, 2, 3, 5],
                    'grid_size': [1, 5, 10, 25]
        }

    # use grid's keys & values to create a list of dicts for each combo in search space
    param_names = param_grid.keys()
    param_values = param_grid.values()
    combinations = list(itertools.product(*param_values))
    combinations_dicts = [dict(zip(param_names, combo)) for combo in combinations]
    if combo_num is None or combo_num >= len(combinations_dicts):
        print(f'No more {model_str} parameter combinations to try.')
    else:
        combo = combinations_dicts[combo_num]
        # run a train/eval engine if overwrite is true, else check if the file already exists
        if overwrite == 'true':
            engine(cfg=cfg, model_str=model_str, param_combo=combo, combo_num=combo_num)
        else:
            check_path = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/filtered_gs/gs_combos/combo{combo_num}_metrics.csv'
            if not os.path.exists(check_path):
                engine(cfg=cfg, model_str=model_str, param_combo=combo, combo_num=combo_num)