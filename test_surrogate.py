import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import toml
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import surrogate_models as sm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import torch.nn as nn
import itertools
import eval as e
from torch.utils.data import DataLoader
import surrogate_dataset as sd
import pickle
#import kan as pykan


def prepare_data(batch_size, metrics_subset):
    train_df = pd.read_pickle('surrogate_dataset/reg_train_dataset.pkl')
    val_df = pd.read_pickle('surrogate_dataset/reg_val_dataset.pkl')
    train_dataset = sd.SurrogateDataset(train_df, mode='train', metrics_subset=metrics_subset)
    val_dataset = sd.SurrogateDataset(val_df, mode='val', metrics_subset=metrics_subset, metrics_scaler=train_dataset.metrics_scaler, genomes_scaler=train_dataset.genomes_scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, train_dataset


def get_model(model_str, metric_subset):
    if model_str == "MLP":
        input_size = 1021
        output_size = len(metrics_subset)
        dropout = 0.0
        hidden_sizes = [512, 256]
        return sm.MLP(input_size=input_size, output_size=output_size, dropout=dropout, hidden_sizes=hidden_sizes)
    if model_str == 'effKAN':
        input_size = 1021
        output_size = 3
        layers_hidden = [512, 256]
        grid_size=25
        spline_order=5
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        #grid_eps=1.0
        grid_range=[-1, 1]
        return sm.KAN(input_size, output_size, layers_hidden, grid_size=grid_size, spline_order=spline_order, scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline, base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
    if model_str == 'pyKAN':
        width=[1021, 2048, 512, 256, 3]
        grid=3
        k=3
        noise_scale=0.1
        scale_base_mu=0.0
        scale_base_sigma=1.0
        base_fun=torch.nn.SiLU()
        symbolic_enabled=True
        bias_trainable=False
        grid_eps=1.0
        grid_range=[-1, 1]
        sp_trainable=True
        sb_trainable=True
        device='cuda'
        seed=42
        return pykan.KAN(width=width, grid=grid, k=k, noise_scale=noise_scale, scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma, base_fun=base_fun, symbolic_enabled=symbolic_enabled, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, device=device, seed=seed)
        
    # TODO implement other surrogate models


def get_optimizer(model_str, params):
    #if model_str == "MLP":
        # NOTE use sparse adam for actual surrogate encoding
        # return optim.RMSprop(params, lr=0.001)
        # baseline is Adam
        # return optim.RMSprop(params, lr=0.01)
    #return optim.Adam(params, 0.0001)
    #if model_str == 'effKAN' or model_str == 'pyKAN':
    return optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    # TODO implement other surrogate optimizers


def get_scheduler(model_str, optimizer, num_epochs, batch_size):
    #if model_str == "MLP":
        # return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
    return lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # return lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        # return lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    #if model_str == 'effKAN' or model_str == 'pyKAN':
    #return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # TODO implement other surrogate schedulers


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
        'mse_average_precision'
    ])


def plot_preds(predictions, truths, metric_names, save_folder='plots'):
    os.makedirs(save_folder, exist_ok=True)

    predictions = np.array(predictions)
    truths = np.array(truths)

    num_metrics = predictions.shape[1]
    
    for i in range(num_metrics):
        plt.figure(figsize=(10, 6))
        
        # Plot predictions histogram in red
        plt.hist(predictions[:, i], bins=100, alpha=0.5, color='red', label='Predictions', edgecolor='black')
        
        # Plot truths histogram in green
        plt.hist(truths[:, i], bins=100, alpha=0.5, color='green', label='Truths', edgecolor='black')
        
        plt.xlabel(metric_names[i])
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {metric_names[i]}')
        plt.legend()
        
        plt.savefig(f'{save_folder}/{metric_names[i]}_histogram.png')
        plt.close()


def store_data(model_str, metrics_df):
    #metrics_out = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/surrogate_metrics.csv'
    metrics_out = f'/home/eharpster3/precog-opt-grip/analysis/kan/{model_str}/surrogate_metrics.csv'
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)


def save_model_weights(model_str, model):
    #weights_out = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/weights.pth'
    weights_out = f'/home/eharpster3/precog-opt-grip/analysis/kan{model_str}/weights.pth'
    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    torch.save(model.state_dict(), weights_out)

    
def engine(cfg, metrics_subset=None):

    models = cfg['models']
    num_epochs = cfg['surrogate_train_epochs']
    batch_size = cfg['surrogate_batch_size']
    if metrics_subset is None:
        metrics_subset = list(range(12))
    train_loader, val_loader, train_dataset = prepare_data(batch_size, metrics_subset=metrics_subset)
    max_metrics = train_dataset.max_metrics
    min_metrics = train_dataset.min_metrics
    genomes_scaler = train_dataset.genomes_scaler
    metrics_scaler = train_dataset.metrics_scaler
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # perform training and validation for each surrogate model
    for model_str in models:
        if model_str == "MLP":
            continue

        # get surrogate model and specific optimizer, scheduler
        model = get_model(model_str, metric_subset=metrics_subset).to(device)
        params = model.parameters()
        optimizer = get_optimizer(model_str, params)
        scheduler = get_scheduler(model_str, optimizer, num_epochs, batch_size)
        scaler = GradScaler()
        metrics_df = create_metrics_df()

        for epoch in range(1, num_epochs + 1):

            # train and validate for one epoch
            train_epoch_loss = train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_metrics, min_metrics, epoch)
            epoch_metrics = val_one_epoch(model, device, val_loader, metrics_subset, max_metrics, min_metrics)

            # update metrics df
            epoch_metrics['epoch_num'] = epoch
            epoch_metrics['train_loss'] = train_epoch_loss
            epoch_metrics_df = pd.DataFrame([epoch_metrics])
            metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)

        store_data(model_str, metrics_df)
        save_model_weights(model_str, model)


def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_metrics, min_metrics, epoch):
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
            #print('before')
            # if epoch == 8 and i == 94:
            #     breakpoint()
            outputs = model(genomes, update_grid=True)
            # if torch.isnan(outputs).any():
            #     break
            #print('after')
            #print(outputs.shape)
            #print(metrics.shape)
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

    # predictions = np.empty((0, len(metrics_subset)))
    # truths = np.empty((0, len(metrics_subset)))

    data_iter = tqdm(val_loader, 'Evaluating')
    with torch.no_grad():
        for genomes, metrics in data_iter:
            # genomes shape: (batch_size, 976)
            genomes = genomes.to(device)
            # metrics shape: (batch_size, 12)
            metrics = metrics.to(device)
            # truths = np.vstack([truths, metrics.cpu().numpy()])

            # forwards with mixed precision
            with autocast():
                # outputs shape: (batch_size, 12)
                outputs = model(genomes)
                # if torch.isnan(outputs).any():
                #     break
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

    
    # if epoch==29:
    #     plot_preds(predictions, truths, ['mse_uw_val_loss', 'mse_ciou_loss', 'mse_average_precision'])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NOTE default config path should change later on
    parser.add_argument('-c', '--config_path', required=False, default='conf_kan.toml')
    args = parser.parse_args()
    config_path = args.config_path
    configs = toml.load(config_path)
    surrogate_config = configs['surrogate']
    metrics_subset = [0, 4, 11]
    engine(surrogate_config, metrics_subset=metrics_subset)

