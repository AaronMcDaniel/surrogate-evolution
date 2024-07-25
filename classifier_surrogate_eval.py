import inspect
import toml
from tqdm import tqdm
import surrogate_dataset as sd
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import eval as e
import surrogate_models as sm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np


def prepare_data(batch_size, train_df, val_df):
    train_dataset = sd.ClassifierSurrogateDataset(train_df, mode='train')
    val_dataset = sd.ClassifierSurrogateDataset(val_df, mode='val', genomes_scaler=train_dataset.genomes_scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, val_loader, train_dataset, val_dataset


def build_configuration(model_dict, device):
        # build model
        model = model_dict['model']
        sig = inspect.signature(model.__init__)
        filtered_params = {k: v for k, v in model_dict.items() if k in sig.parameters}
        model = model(**filtered_params).to(device)

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
            scheduler = scheduler_func(optimizer=optimizer, mode='min', factor=0.5, patience=5)
        else:
            scheduler = scheduler_func(optimizer=optimizer)
        scaler = GradScaler()
        
        return model, optimizer, scheduler, scaler


def create_metrics_df(cfg):
    return pd.DataFrame(columns=[
        'epoch_num',
        'train_loss',
        'val_loss',
    ] + cfg['surrogate_metrics'])


def engine(cfg, model_dict, train_df, val_df, weights_dir):
    best_loss_metric = np.inf
    best_epoch = None
    best_epoch_num = None
    # pull surrogate train/eval config attributes
    num_epochs = cfg['surrogate_train_epochs']
    batch_size = cfg['surrogate_batch_size']
    # define subset of metrics to train on and prepare data accordingly
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(batch_size, train_df, val_df)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, optimizer, scheduler, scaler = build_configuration(model_dict=model_dict, device=device)
    
    for epoch in range(1, num_epochs + 1):
        # train and validate for one epoch
        train_metrics = train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler)
        val_metrics = val_one_epoch(model, device, val_loader)
        print(f"---- Epoch {epoch} ----")
        print(f"train : loss = {train_metrics['loss']:.4f} | accuracy = {train_metrics['acc']:.4f} | precision = {train_metrics['prec']:.4f} | recall = {train_metrics['rec']:.4f}")
        print(f"val   : loss = {val_metrics['loss']:.4f} | accuracy = {val_metrics['acc']:.4f} | precision = {val_metrics['prec']:.4f} | recall = {val_metrics['rec']:.4f}")
        
        # val_losses = []
                
        # for val in val_subset:
        #     val_losses.append(epoch_metrics[metric_names[val]])
        #     #val_losses.append(loss_tensor[metrics_subset.index(val)])
        #     #print('CHECK', epoch_metrics, epoch_metrics[metric_names[val]])
        # #print(loss_tensor[val_losses[0]])
        # loss_metric = sum(val_losses)
        # #print(loss_metric)
        # #print(loss_metric)
        # if loss_metric < best_loss_metric:
        #     best_loss_metric = loss_metric
        #     best_epoch = model
        #     best_epoch_num = epoch
    

    # torch.save(best_epoch.state_dict(), f'{weights_dir}/{model_dict['name']}.pth')
    # print('        Save epoch #:', best_epoch_num)    

    return best_epoch_num, train_dataset.genomes_scaler


# def get_val_scores(cfg, model_dict, train_df, val_df, weights_dir):
#     # pull surrogate train/eval config attributes
#     batch_size = cfg['surrogate_batch_size']
#     # define subset of metrics to train on and prepare data accordingly
#     metrics_subset = model_dict['metrics_subset']
#     train_loader, val_loader, train_dataset, val_dataset = prepare_data(model_dict, batch_size, train_df, val_df)
#     max_metrics = train_dataset.max_metrics
#     min_metrics = train_dataset.min_metrics

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model, optimizer, scheduler, scaler, val_subset = build_configuration(model_dict=model_dict, device=device)
#     model.load_state_dict(torch.load(f'{weights_dir}/{model_dict['name']}.pth', map_location=device)) 
#     epoch_metrics = val_one_epoch(cfg, model, device, val_loader, metrics_subset, max_metrics, min_metrics)
#     return epoch_metrics               


def train_one_epoch(model, device, train_loader, optimizer, scheduler, scaler):
    model.train()

    # Initialize variables
    surrogate_train_loss = 0.0
    y_true = []
    y_pred = []

    criterion = nn.BCEWithLogitsLoss()

    data_iter = tqdm(train_loader, desc='Training')
    for genomes, labels in data_iter:
        genomes = genomes.to(device)
        labels = labels.to(device)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(genomes)
            loss = criterion(outputs, labels.unsqueeze(1).float())
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        surrogate_train_loss += loss.item()

        # Collect true and predicted labels
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.sigmoid().cpu().detach().numpy())

        data_iter.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()

    # Step scheduler
    scheduler.step()

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
        'loss': surrogate_train_loss,
        'prec': precision,
        'rec': recall,
        'acc': accuracy
    }

    return epoch_metrics

def val_one_epoch(model, device, val_loader):
    model.eval()

    # Initialize variables
    surrogate_val_loss = 0.0
    y_true = []
    y_pred = []

    criterion = nn.BCEWithLogitsLoss()

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
        'loss': surrogate_val_loss,
        'prec': precision,
        'rec': recall,
        'acc': accuracy
    }

    return epoch_metrics



configs = toml.load('conf.toml')
surrogate_config = configs['surrogate']
binary_train_df = pd.read_pickle('surrogate_dataset/train_binary_dataset.pkl')
binary_val_df = pd.read_pickle('surrogate_dataset/val_binary_dataset.pkl')
# # Count the number of 1s and 0s in the 'label' column of the training DataFrame
# train_label_counts = binary_train_df['label'].value_counts()
# print(f"Training DataFrame label counts:\n{train_label_counts}")
# # Count the number of 1s and 0s in the 'label' column of the validation DataFrame
# val_label_counts = binary_val_df['label'].value_counts()
# print(f"Validation DataFrame label counts:\n{val_label_counts}")
model_dict = {
                'name': 'fail_predictor_3000',
                'dropout': 0.2,
                'hidden_sizes': [2048, 1024, 512, 256],
                'optimizer': optim.Adam,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                'model': sm.BinaryClassifier
            }
engine(surrogate_config, model_dict, binary_train_df, binary_val_df, '')