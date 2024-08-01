"""
Script to launch grid-search process for surrogates.
"""

import inspect
import itertools
import argparse
import os
import pickle
import toml
from tqdm import tqdm
from surrogates import surrogate_models as sm
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
from surrogates import surrogate_dataset as sd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

def engine(cfg, model_str, param_combo, combo_num):

    # pull surrogate train/eval config attributes
    num_epochs = cfg['surrogate_train_epochs']
    batch_size = cfg['surrogate_batch_size']

    # define subset of metrics to train on and prepare data accordingly
    train_loader, val_loader, train_dataset = prepare_data(batch_size)

    # only use cpu for grid search
    device = torch.device('cpu')
    model, optimizer, scheduler, scaler = build_configuration(model_str=model_str, device=device, param_combo=param_combo)

    metrics_df = create_metrics_df()
    for epoch in range(1, num_epochs + 1):

            # train and validate for one epoch
            train_metrics = train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler)
            epoch_metrics = val_one_epoch(model, device, val_loader, scheduler)

            # update metrics df
            epoch_metrics['param_combo'] = str(param_combo)
            epoch_metrics['epoch_num'] = epoch

            for k, v in train_metrics.items():
                epoch_metrics[k] = v
            epoch_metrics_df = pd.DataFrame([epoch_metrics])
            metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)
    
    # store data
    store_data(model_str=model_str, combo_num=combo_num, metrics_df=metrics_df)


# prepare data for grid search
def prepare_data(batch_size):
    train_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/us_surr_cls_train.pkl')
    val_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/us_surr_cls_val.pkl')
    train_dataset = sd.ClassifierSurrogateDataset(train_df, mode='train')
    val_dataset = sd.ClassifierSurrogateDataset(val_df, mode='val', genomes_scaler=train_dataset.genomes_scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, train_dataset       


# build model, optimizer, scheduler, and scaler
def build_configuration(model_str, device, param_combo):

    # build model
    model = param_combo['model']
    sig = inspect.signature(model.__init__)
    filtered_params = {k: v for k, v in param_combo.items() if k in sig.parameters}
    model = model(**filtered_params).to(device)

    # build optimizer
    params = model.parameters()
    lr = param_combo['lr']
    optimizer_func = partial(param_combo['optimizer'])
    optimizer = optimizer_func(params=params, lr=lr)

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
    metrics_out = f'/gv1/projects/GRIP_Precog_Opt/surrogates/classifiers/{model_str}/gs_combos/c{combo_num}_metrics.csv'
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)


def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler):
    model.train()

    # actual surrogate training loss
    surrogate_train_loss = 0.0
    # mean taken for metric regression losses in train
    # criterion = nn.MSELoss()
    y_true = []
    y_pred = []

    criterion = nn.BCEWithLogitsLoss()

    data_iter = tqdm(train_loader, desc='Training')
    for i, (genomes, labels) in enumerate(data_iter):

        # genomes shape: (batch_size, 976)
        genomes = genomes.to(device)
        # metrics shape: (batch_size, 12)
        labels = labels.to(device)

        # forwards with mixed precision
        with autocast():
            outputs = model(genomes)
            loss = criterion(outputs, labels.unsqueeze(1).float())
                   
        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        surrogate_train_loss += loss.item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.sigmoid().cpu().detach().numpy())
        data_iter.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()
    
    # Convert lists to numpy arrays for metric calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Apply threshold to predictions

    # Compute metrics
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)

    # Calculate average training loss
    num_batches = len(data_iter)
    surrogate_train_loss /= num_batches

    epoch_metrics = {
        'train_loss': surrogate_train_loss,
        'train_prec': precision,
        'train_rec': recall,
        'train_acc': accuracy
    }

    return epoch_metrics


def val_one_epoch(model, device, val_loader, scheduler):
    model.eval()

    # actual surrogate validation loss
    surrogate_val_loss = 0.0
    y_true = []
    y_pred = []

    criterion = nn.BCEWithLogitsLoss()

    data_iter = tqdm(val_loader, 'Evaluating')
    data_iter = tqdm(val_loader, 'Evaluating')
    with torch.no_grad():
        for genomes, labels in data_iter:
            genomes = genomes.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision
            with autocast():
                outputs = model(genomes)
                loss = criterion(outputs, labels.unsqueeze(1).float())
            
            # Collect true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.sigmoid().cpu().numpy())

            # Update validation loss
            surrogate_val_loss += loss.item()
            data_iter.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()

    # Step scheduler
    if type(scheduler) is optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(surrogate_val_loss)
    else:
        scheduler.step()

    # Convert lists to numpy arrays for metric calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Apply threshold to predictions

    # Compute metrics
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)

    # Calculate average validation loss
    num_batches = len(data_iter)
    surrogate_val_loss /= num_batches

    epoch_metrics = {
        'val_loss': surrogate_val_loss,
        'val_prec': precision,
        'val_rec': recall,
        'val_acc': accuracy
    }

    return epoch_metrics


# creates metrics dataframe with appropriate rows
def create_metrics_df():
    return pd.DataFrame(columns=[
        'epoch_num', 'val_loss', 'val_prec', 'val_rec', 'val_acc', 'train_loss', 'train_prec', 'train_rec', 'train_acc', 'param_combo'
    ])


# uses model string to concatenate grid search resulting metric csvs to one master file
def cat_results(name, model_str):
    search_dir = f'/gv1/projects/GRIP_Precog_Opt/surrogates/classifiers/{model_str}/gs_combos'
    master_df = create_metrics_df()

    # change range as necessary for different grid search runs
    for i in range(3840):
        metrics_path = search_dir + f'/c{i}_metrics.csv'
        try:
            metrics_df = pd.read_csv(metrics_path)
        except:
            continue
        master_df = pd.concat([master_df, metrics_df], ignore_index=True)
    out_path = f'/gv1/projects/GRIP_Precog_Opt/surrogates/classifiers/{model_str}/{name}_gs.csv'
    master_df.to_csv(out_path, index=False)
    return None


# cat_results('cls_surr', 'KAN')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-cn', '--combo_num', type=int, required=True, default=None)
#     parser.add_argument('-cp', '--cfg_path', type=str, required=False, default='/home/tthakur9/precog-opt-grip/conf.toml')
#     parser.add_argument('-m', '--model', type=str, required=True)

#     # overwrite determines whether old csv files should be overwritten
#     parser.add_argument('-o', '--overwrite', type=str, required=False, default='true')
#     args = parser.parse_args()
#     combo_num = args.combo_num
#     cfg_path = args.cfg_path
#     overwrite = args.overwrite
#     model_str = args.model

#     # load config
#     all_cfg = toml.load(cfg_path)
#     cfg = all_cfg['surrogate']

#     if model_str == "MLP":
#         # define MLP-unique parameter grid
#         param_grid = {
#                     'model': [sm.MLP],
#                     'output_size': [1],
#                     'dropout': [0.0, 0.2, 0.4, 0.6],
#                     'hidden_sizes': [[512, 256], [1024, 512], [2048, 1024, 512]],
#                     'optimizer': [optim.SGD, optim.Adam, optim.RMSprop, optim.Adagrad],
#                     'lr': [0.0001, 0.001, 0.01, 0.1],
#                     'scheduler': [lr.StepLR, lr.MultiStepLR, lr.CosineAnnealingLR, lr.ReduceLROnPlateau],
#         }

#     elif model_str == "KAN":
#         # define unique KAN param grid
#         param_grid = {
#                     'hidden_sizes': [[512, 256], [2048, 1024, 512], [2048], []],
#                     'optimizer': [optim.SGD, optim.RMSprop, optim.AdamW, optim.Adagrad],
#                     'lr': [0.001, 0.01, 0.1, 0.0001],
#                     'scheduler': [lr.StepLR, lr.CosineAnnealingWarmRestarts, lr.ReduceLROnPlateau],
#                     'spline_order': [1, 2, 3, 4, 5],
#                     'grid_size': [1, 5, 10, 25],
#                     'model': [sm.KAN],
#                     'output_size': [1]
#         }

#     # use grid's keys & values to create a list of dicts for each combo in search space
#     param_names = param_grid.keys()
#     param_values = param_grid.values()
#     combinations = list(itertools.product(*param_values))
#     combinations_dicts = [dict(zip(param_names, combo)) for combo in combinations]
#     if combo_num is None or combo_num >= len(combinations_dicts):
#         print(f'No more {model_str} parameter combinations to try.')
#     else:
#         combo = combinations_dicts[combo_num]
#         # run a train/eval engine if overwrite is true, else check if the file already exists
#         if overwrite == 'false':
#             check_path = f'/gv1/projects/GRIP_Precog_Opt/surrogates/classifiers/{model_str}/gs_combos/c{combo_num}_metrics.csv'
#             if not os.path.exists(check_path):
#                 engine(cfg=cfg, model_str=model_str, param_combo=combo, combo_num=combo_num)
#         else:
#             engine(cfg=cfg, model_str=model_str, param_combo=combo, combo_num=combo_num)
