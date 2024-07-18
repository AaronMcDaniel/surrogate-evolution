import argparse
import os
import pickle
import toml
import surrogate_models as sm
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler as lr
from sklearn.preprocessing import StandardScaler
import itertools
import test_surrogate as ts
from functools import partial

def engine(cfg, model_str, param_combo, combo_num):

    # pull surrogate train/eval config attributes
    num_epochs = cfg['surrogate_train_epochs']
    batch_size = cfg['surrogate_batch_size']

    # define subset of metrics to train on and prepare data accordingly
    metrics_subset = param_combo['metrics_subset']
    train_loader, val_loader, max_metrics, min_metrics = ts.prepare_data(batch_size, metrics_subset=metrics_subset)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, optimizer, scheduler, scaler = build_configuration(model_str=model_str, device=device, param_combo=param_combo)

    # create metrics_df
    metrics_df = ts.create_metrics_df()
    for epoch in range(1, num_epochs + 1):

            # train and validate for one epoch
            train_epoch_loss = ts.train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_metrics=max_metrics, min_metrics=min_metrics)
            epoch_metrics = ts.val_one_epoch(model, device, val_loader, metrics_subset=metrics_subset, max_metrics=max_metrics, min_metrics=min_metrics)

            # update metrics df
            epoch_metrics['param_combo'] = str(param_combo)
            epoch_metrics['epoch_num'] = epoch
            epoch_metrics['train_loss'] = train_epoch_loss
            epoch_metrics_df = pd.DataFrame([epoch_metrics])
            metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)
    
    store_data(model_str=model_str, combo_num=combo_num, metrics_df=metrics_df)
            

def build_configuration(model_str, device, param_combo):
    if model_str == "MLP":
        # build model
        input_size = 1021
        hidden_sizes = param_combo['hidden_sizes']
        dropout = param_combo['dropout']
        output_size = len(param_combo['metrics_subset'])
        model = sm.MLP(input_size=input_size, output_size=output_size, dropout=dropout, hidden_sizes=hidden_sizes)
        model.to(device)

        # build optimizer
        params = model.parameters()
        lr = param_combo['lr']
        optimizer_func = partial(param_combo['optimizer'])
        optimizer = optimizer_func(params=params, lr=lr)

        # build scheduler and scaler
        scheduler_func = param_combo['scheduler']
        if scheduler_func == optim.lr_scheduler.StepLR:
            scheduler = scheduler_func(optimizer=optimizer, step_size=10, gamma=0.1)
        elif scheduler_func == optim.lr_scheduler.MultiStepLR:
            scheduler = scheduler_func(optimizer=optimizer, milestones=[10, 20], gamma=0.1)
        elif scheduler_func == optim.lr_scheduler.CosineAnnealingLR:
            scheduler = scheduler_func(optimizer=optimizer, T_max=10)
        elif scheduler_func == optim.lr_scheduler.ReduceLROnPlateau:
            scheduler = scheduler_func(optimizer=optimizer, mode='min', factor=0.1, patience=5)
        else:
            scheduler = scheduler_func(optimizer=optimizer)
        scaler = torch.GradScaler()
        return model, optimizer, scheduler, scaler


def store_data(model_str, combo_num, metrics_df):
    metrics_out = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/gs_combos/combo{combo_num}_metrics.csv'
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)


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


def cat_results(model_str='MLP'):
    search_dir = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/gs_combos'
    master_df = create_metrics_df()
    for i in range(3840):
        metrics_path = search_dir + f'/combo{i}_metrics.csv'
        metrics_df = pd.read_csv(metrics_path)
        master_df = pd.concat([master_df, metrics_df], ignore_index=True)
    out_path = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/gs_master.csv'
    master_df.to_csv(out_path, index=False)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--combo_num', type=int, required=True, default=None)
    parser.add_argument('-cp', '--cfg_path', type=str, required=False, default='/home/tthakur9/precog-opt-grip/conf.toml')
    args = parser.parse_args()
    combo_num = args.combo_num
    cfg_path = args.cfg_path
    all_cfg = toml.load(cfg_path)
    cfg = all_cfg['surrogate']
    models = cfg['models']

    for model_str in models:
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

            # use grid's keys & values to create a list of dicts for each combo in search space
            param_names = param_grid.keys()
            param_values = param_grid.values()
            combinations = list(itertools.product(*param_values))
            combinations_dicts = [dict(zip(param_names, combo)) for combo in combinations]

            # run train/val engine with specific parameter combination
            if combo_num is not None and combo_num < len(combinations_dicts):
                combo = combinations_dicts[combo_num]
                engine(cfg=cfg, model_str=model_str, param_combo=combo, combo_num=combo_num)
            else:
                print(f'No more {model_str} parameter combinations to try.')
                break