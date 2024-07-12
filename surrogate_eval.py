import argparse
import os
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


def prepare_data(batch_size):
    train_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/train_dataset.pkl')
    val_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/val_dataset.pkl')
    train_dataset = sd.SurrogateDataset(train_df, mode='train')
    val_dataset = sd.SurrogateDataset(val_df, mode='val', metrics_scaler=train_dataset.metrics_scaler, genomes_scaler=train_dataset.genomes_scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def get_model(model_str):
    if model_str == "MLP":
        # NOTE mlp hyperparameters will be optimized with grid search in the future
        return sm.MLP()
    # TODO implement other surrogate models


def get_optimizer(model_str, params):
    if model_str == "MLP":
        # NOTE use sparse adam for actual surrogate encoding
        # return optim.SparseAdam(params, lr=0.001)
        return optim.Adam(params, lr=0.001)
    # TODO implement other surrogate optimizers


def get_scheduler(model_str, optimizer, num_epochs, batch_size):
    if model_str == "MLP":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
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


def store_data(model_str, metrics_df):
    metrics_out = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/surrogate_metrics.csv'
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    metrics_df.to_csv(metrics_out, index=False)


def save_model_weights(model_str, model):
    weights_out = f'/gv1/projects/GRIP_Precog_Opt/surrogates/{model_str}/weights.pth'
    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    torch.save(model.state_dict(), weights_out)

    
def engine(cfg):

    models = cfg['models']
    num_epochs = cfg['surrogate_train_epochs']
    batch_size = cfg['surrogate_batch_size']
    train_loader, val_loader = prepare_data(batch_size)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # perform training and validation for each surrogate model
    for model_str in models:

        # get surrogate model and specific optimizer, scheduler
        model = get_model(model_str).to(device)
        params = model.parameters()
        optimizer = get_optimizer(model_str, params)
        scheduler = get_scheduler(model_str, optimizer, num_epochs, batch_size)
        scaler = GradScaler()

        # NOTE metrics df for each surrogate must be stored in different directories
        metrics_df = create_metrics_df()
        for epoch in range(1, num_epochs + 1):

            # train and validate for one epoch
            train_epoch_loss = train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler)
            epoch_metrics = val_one_epoch(model, device, val_loader)

            # update metrics df
            epoch_metrics['epoch_num'] = epoch
            epoch_metrics['train_loss'] = train_epoch_loss
            epoch_metrics_df = pd.DataFrame([epoch_metrics])
            metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)

        store_data(model_str, metrics_df)
        save_model_weights(model_str, model)


def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler):
    model.train()

    # actual surrogate training loss
    surrogate_train_loss = 0.0
    # mean taken for metric regression losses in train
    criterion = nn.MSELoss()

    data_iter = tqdm(train_loader, desc='Training')
    for genomes, metrics in data_iter:

        # genomes shape: (batch_size, 976)
        genomes = genomes.to(device)
        # metrics shape: (batch_size, 12)
        metrics = metrics.to(device)

        # forwards with mixed precision
        with autocast():
            # outputs shape: (batch_size, 12)
            outputs = model(genomes)
            # metric regression loss is meaned, so is scalar tensor value
            loss = criterion(outputs, metrics)
        
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


def val_one_epoch(model, device, val_loader):
    model.eval()

    # actual surrogate validation loss
    surrogate_val_loss = 0.0
    # mse loss matrx for each metric, where rows = num batches and cols = 12 for all predicted metrics
    mse_metrics_per_batch = []
    # no mean taken for losses in validation 
    criterion = nn.MSELoss(reduction='none')

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
                # loss_matrix shape: (batch_size, 12)
                loss_matrix = criterion(outputs, metrics)
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

    # calculate surrogate validation loss per batch (NOTE batch loss already meaned by batch size)
    num_batches = len(data_iter)
    surrogate_val_loss /= num_batches

    # compute the mean of the mse losses for each metric based on num batches
    mse_metrics_per_batch = torch.stack(mse_metrics_per_batch)
    mse_metrics_meaned = mse_metrics_per_batch.mean(dim=0)

    epoch_metrics = {
        'val_loss': surrogate_val_loss,
        'mse_uw_val_loss': mse_metrics_meaned[0].item(), 
        'mse_iou_loss': mse_metrics_meaned[1].item(), 
        'mse_giou_loss': mse_metrics_meaned[2].item(), 
        'mse_diou_loss': mse_metrics_meaned[3].item(), 
        'mse_ciou_loss': mse_metrics_meaned[4].item(), 
        'mse_center_loss': mse_metrics_meaned[5].item(), 
        'mse_size_loss': mse_metrics_meaned[6].item(), 
        'mse_obj_loss': mse_metrics_meaned[7].item(), 
        'mse_precision': mse_metrics_meaned[8].item(),
        'mse_recall': mse_metrics_meaned[9].item(), 
        'mse_f1_score': mse_metrics_meaned[10].item(),
        'mse_average_precision': mse_metrics_meaned[11].item()
    }  

    return epoch_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NOTE default config path should change later on
    parser.add_argument('-c', '--config_path', required=False, default='/home/tthakur9/precog-opt-grip/conf.toml')
    args = parser.parse_args()
    config_path = args.config_path
    configs = toml.load(config_path)
    surrogate_config = configs['surrogate']
    engine(surrogate_config)

