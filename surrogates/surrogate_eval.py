"""
Train and Validate operations for the regressor surrogates. Called surrogate_eval for compatibility reasons.
"""


import inspect
import os
import toml
from tqdm import tqdm
from surrogates import surrogate_dataset as sd
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import eval as e
from surrogates import surrogate_models as sm
from torch.cuda.amp import autocast, GradScaler
import numpy as np

def prepare_data(model_dict, batch_size, train_df, val_df):
    train_dataset = sd.SurrogateDataset(train_df, mode='train', metrics_subset=model_dict['metrics_subset'])
    val_dataset = sd.SurrogateDataset(val_df, mode='val', metrics_subset=model_dict['metrics_subset'], metrics_scaler=train_dataset.metrics_scaler, genomes_scaler=train_dataset.genomes_scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, val_loader, train_dataset, val_dataset


# builds model, optimizer and scheduler from a 'model_dict' as defined in the 'Surrogate' class under models (see surrogate.py)
def build_configuration(model_dict, device):
        # build model
        model = model_dict['model']
        output_size = len(model_dict['metrics_subset'])
        sig = inspect.signature(model.__init__)
        filtered_params = {k: v for k, v in model_dict.items() if k in sig.parameters}
        model = model(output_size=output_size, **filtered_params).to(device)

        # build optimizer
        params = model.parameters()
        lr = model_dict['lr']
        optimizer_func = partial(model_dict['optimizer'])
        optimizer = optimizer_func(params=params, lr=lr)

        # build scheduler and scaler
        scheduler_func = model_dict['scheduler']
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
        scaler = GradScaler()
        
        val_subset = model_dict['validation_subset']
        return model, optimizer, scheduler, scaler, val_subset


def create_metrics_df(cfg):
    return pd.DataFrame(columns=[
        'epoch_num',
        'train_loss',
        'val_loss',
    ] + cfg['surrogate_metrics'])


# used to train and evaluate a regressor surrogate
# calling this function will train and validate the model represented by the passed-in model dict
# the model dict includes a metrics_subset and a validation_subset which represent the metrics used to train the model
# and the metrics on which the model makes inferences on respectively.
# returns the genome scaler used (for getting inferences later) and saves best epoch weights by best sum of validation subset losses
def engine(cfg, model_dict, train_df, val_df, weights_dir):
    best_loss_metric = np.inf
    best_epoch = None
    best_epoch_num = None
    best_epoch_metrics = None
    # pull surrogate train/eval config attributes
    num_epochs = cfg['surrogate_train_epochs']
    batch_size = cfg['surrogate_batch_size']
    metric_names = cfg['surrogate_metrics']
    # define subset of metrics to train on and prepare data accordingly
    metrics_subset = model_dict['metrics_subset']
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(model_dict, batch_size, train_df, val_df)
    max_metrics = train_dataset.max_metrics
    min_metrics = train_dataset.min_metrics

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, optimizer, scheduler, scaler, val_subset = build_configuration(model_dict=model_dict, device=device)

    # create metrics_df
    metrics_df = create_metrics_df(cfg)
    for epoch in range(1, num_epochs + 1):
        # train and validate for one epoch
        train_epoch_loss = train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_metrics, min_metrics)
        epoch_metrics = val_one_epoch(cfg, model, device, val_loader, metrics_subset, max_metrics, min_metrics)
        #print(epoch_metrics)
        
        val_losses = []
                
        for val in val_subset:
            val_losses.append(epoch_metrics[metric_names[val]])
            #val_losses.append(loss_tensor[metrics_subset.index(val)])
            #print('CHECK', epoch_metrics, epoch_metrics[metric_names[val]])
        #print(loss_tensor[val_losses[0]])
        loss_metric = sum(val_losses)
        #print(loss_metric)
        #print(loss_metric)
        if loss_metric < best_loss_metric:
            best_loss_metric = loss_metric
            best_epoch = model
            best_epoch_num = epoch
            best_epoch_metrics = epoch_metrics
        # update metrics df
        epoch_metrics['epoch_num'] = epoch
        epoch_metrics['train_loss'] = train_epoch_loss
        epoch_metrics_df = pd.DataFrame([epoch_metrics])
        metrics_df = pd.concat([metrics_df, epoch_metrics_df], ignore_index=True)
        
    if best_epoch is not None:
        torch.save(best_epoch.state_dict(), f'{weights_dir}/{model_dict["name"]}.pth')
        print('        Save epoch #:', best_epoch_num)    

    return metrics_df, best_epoch_metrics, best_epoch_num, train_dataset.genomes_scaler


def get_val_scores(cfg, model_dict, train_df, val_df, weights_dir):
    # pull surrogate train/eval config attributes
    batch_size = cfg['surrogate_batch_size']
    # define subset of metrics to train on and prepare data accordingly
    metrics_subset = model_dict['metrics_subset']
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(model_dict, batch_size, train_df, val_df)
    max_metrics = train_dataset.max_metrics
    min_metrics = train_dataset.min_metrics

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, optimizer, scheduler, scaler, val_subset = build_configuration(model_dict=model_dict, device=device)
    model.load_state_dict(torch.load(f'{weights_dir}/{model_dict["name"]}.pth', map_location=device)) 
    epoch_metrics = val_one_epoch(cfg, model, device, val_loader, metrics_subset, max_metrics, min_metrics)
    return epoch_metrics      


def get_inferences(model_dict, device, inference_df, genome_scaler, weights_dir):
    # get model and load weights
    model, _, _, _, val_subset = build_configuration(model_dict, device)
    model.load_state_dict(torch.load(f'{weights_dir}/{model_dict["name"]}.pth', map_location=device))
    genomes = np.stack(inference_df['genome'].values)

    # scale features with train scaler
    genomes = genome_scaler.transform(genomes)
    genomes = torch.tensor(genomes, dtype=torch.float32, device=device)

    # get inferences
    model.eval()
    with torch.no_grad():
        with autocast():
            inf = model(genomes)
    
    # clamp inferences
    inf = torch.clamp(inf, min=-300, max=300).to(torch.float32)

    # get only the validation subset of inferences
    metrics_subset = model_dict['metrics_subset']
    val_col_indices = [i for i, idx in enumerate(metrics_subset) if idx in val_subset]
    val_inf = inf[:, val_col_indices].cpu().detach().numpy()
    return val_inf
    

def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler, max_metrics, min_metrics):
    model.train()

    # actual surrogate training loss
    surrogate_train_loss = 0.0
    # mean taken for metric regression losses in train
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()

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

    # calculate surrogate training loss per batch (NOTE batch loss already meaned by batch size)
    num_batches = len(data_iter)
    surrogate_train_loss /= num_batches
    
    # step scheduler
    e.step_scheduler(scheduler, surrogate_train_loss)

    return surrogate_train_loss


def val_one_epoch(cfg, model, device, val_loader, metrics_subset, max_metrics, min_metrics):
    model.eval()

    # actual surrogate validation loss
    surrogate_val_loss = 0.0
    # mse loss matrx for each metric, where rows = num batches and cols = 12 for all predicted metrics
    mse_metrics_per_batch = []
    # no mean taken for losses in validation 
    criterion = nn.L1Loss(reduction='none')
    # criterion = nn.MSELoss(reduction='none')
    
    metric_names = cfg['surrogate_metrics']
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
                # loss_matrix shape: (batch_size, 12)
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

    # calculate surrogate validation loss per batch (NOTE batch loss already meaned by batch size)
    num_batches = len(data_iter)
    surrogate_val_loss /= num_batches

    # compute the mean of the mse losses for each metric based on num batches
    mse_metrics_per_batch = torch.stack(mse_metrics_per_batch)
    mse_metrics_meaned = mse_metrics_per_batch.mean(dim=0)

    epoch_metrics = {
        'val_loss': surrogate_val_loss
    }

    epoch_metrics.update({
        selected_metric_names[i]: mse_metrics_meaned[i].item() for i in range(len(metrics_subset))
    })

    return epoch_metrics


def train_all_reg(cfg, reg_train_df, reg_val_df, surrogate, weights_dir='/home/tthakur9/precog-opt-grip/test'):
    for m in surrogate.models:
        weights_path = os.path.join(weights_dir, m['name'] + '.pth')
        if not os.path.exists(weights_path):
            print(f"Training model: {m['name']}")
            # Train the model and save the weights
            engine(cfg, m, reg_train_df, reg_val_df, weights_dir=weights_dir)
        else:
            print(f"Weights for model {m['name']} already exist at {weights_path}. Skipping training.")


# # # TESTING
# config_path = '/home/tthakur9/precog-opt-grip/conf.toml'
# configs = toml.load(config_path)
# cfg = configs['surrogate']
# reg_train_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/us_surr_reg_train.pkl')
# reg_val_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/us_surr_reg_val.pkl')

