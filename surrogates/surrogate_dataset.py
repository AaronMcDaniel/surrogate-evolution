"""
Functions for building surrogate datasets and dataset classes used for surrogate train/eval.
"""


import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import toml
from codec import Codec
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle


class SurrogateDataset(Dataset):
    # init using df
    def __init__(self, df, mode, metrics_subset=None, metrics_scaler=StandardScaler(), genomes_scaler = StandardScaler()):
        self.df = df
        self.genomes_scaler = genomes_scaler
        self.metrics_scaler = metrics_scaler
        self.mode = mode
        self.genomes = np.stack(df['genome'].values)
        # labels/metrics in the last 12 cols of df
        if metrics_subset is None:
            metrics_subset = list(range(12))
        metrics_subset = [-12 + i for i in metrics_subset]

        self.metrics = df.iloc[:, metrics_subset].values
        self.max_metrics = torch.ones((1, len(metrics_subset))) * 300.0
        self.min_metrics = torch.ones((1, len(metrics_subset))) * -300.0
        
        # standardize genome/metrics data dist if train mode
        if mode == 'train':
            # self.metrics = self.metrics_scaler.fit_transform(self.metrics)
            self.genomes = self.genomes_scaler.fit_transform(self.genomes)
        if mode == 'val':
            # self.metrics = self.metrics_scaler.transform(self.metrics)
            self.genomes = self.genomes_scaler.transform(self.genomes)

        # self.max_metrics = torch.tensor(self.metrics_scaler.transform(self.max_metrics), dtype=torch.float32)
        # self.min_metrics = torch.tensor(self.metrics_scaler.transform(self.min_metrics), dtype=torch.float32)
            
        if np.isnan(self.genomes).any() or np.isnan(self.metrics).any():
            breakpoint()

    # returns num samples in dataset
    def __len__(self):
        return len(self.df)
    
    # retrieve genome, metrics at specific index
    def __getitem__(self, i):
        genome = torch.tensor(self.genomes[i], dtype=torch.float32)
        metrics = torch.tensor(self.metrics[i], dtype=torch.float32)
        return genome, metrics
    
    
class ClassifierSurrogateDataset(Dataset):
    # init using df
    def __init__(self, df, mode, genomes_scaler = StandardScaler()):
        self.df = df
        self.genomes_scaler = genomes_scaler
        self.mode = mode
        self.genomes = np.stack(df['genome'].values)
        self.labels = np.stack(df['label'].values)
        
        # # # standardize genome/metrics data dist if train mode
        if mode == 'train':
            # self.metrics = self.metrics_scaler.fit_transform(self.metrics)
            self.genomes = self.genomes_scaler.fit_transform(self.genomes)
        if mode == 'val':
            # self.metrics = self.metrics_scaler.transform(self.metrics)
            self.genomes = self.genomes_scaler.transform(self.genomes)
            
        if np.isnan(self.genomes).any() or np.isnan(self.labels).any():
            breakpoint()

    # returns num samples in dataset
    def __len__(self):
        return len(self.df)
    
    # retrieve genome, metrics at specific index
    def __getitem__(self, i):
        genome = torch.tensor(self.genomes[i], dtype=torch.float32)
        label = torch.tensor(self.labels[i], dtype=torch.float32)
        return genome, label


# builds classifier and regressor datasets by scraping through a run directory and looking at metric.csv files
# 
# WORKS BUT SHOULD BE PROOFREAD
def build_dataset(
        infile='/gv1/projects/GRIP_Precog_Opt/outputs/out.csv',
        working_dir='/gv1/projects/GRIP_Precog_Opt/outputs', 
        outdir='surrogate_dataset', 
        metrics='uw_val_epoch_loss,iou_loss,giou_loss,diou_loss,ciou_loss,center_loss,size_loss,obj_loss,precision,recall,f1_score,average_precision', 
        exclude=[], include_only=None, val_ratio=0.3, seed=0
    ):
    codec = Codec(num_classes=7)
    metric_headings = metrics.split(',')
    excluded_gens = exclude
    
    local_random = random.Random(seed)
    configs = toml.load(os.path.join(working_dir, "conf.toml"))
    model_config = configs["model"]
    num_epochs = model_config['train_epochs']
    MAX_METRICS = ['precision', 'recall', 'f1_score', 'average_precision']
    genome_max_thresh = 100000

    data = pd.read_csv(infile)
    data = data.to_dict('records')

    all_reg_data = []
    all_cls_data = []

    for line in data:
        genome_hash = line['hash']
        gen = line['gen']
        genome = line['genome']

        if gen in excluded_gens or (include_only is not None and gen not in include_only):
            continue
        
        metrics_path = os.path.join(working_dir, f'generation_{gen}', genome_hash, 'metrics.csv')
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.to_dict('records')

        # initialize datapoint lists
        reg_data = []
        cls_data = []

        for metric_row in metrics:
            # initialize datapoint dicts
            reg_to_add = {}
            cls_to_add = {}

            # if genome is failed, only create classifier datapoint and combined datapoint
            if 'epoch_num' not in metric_row:
                for i in range(num_epochs):
                    cls_to_add = {}
                    # if encoding genome fails, move on to next datapoint
                    try:
                        tensor = codec.encode_surrogate(genome, i + 1)
                    except Exception as e:
                        break
                    # if encoded genome has values greater than threshold, move on to next datapoint
                    if np.any(tensor > genome_max_thresh):
                        break

                    # assign bad metrics and add to combined datapoint dict
                    for heading in metric_headings:
                        if heading in MAX_METRICS:
                            cls_to_add[heading] = -300.0
                        else:
                            cls_to_add[heading] = 300.0

                    # add clipped encoding, hash, and bad label to classifier datapoint dict
                    cls_to_add['genome'] = np.clip(tensor, -1000, 1000)
                    cls_to_add['hash'] = genome_hash
                    cls_to_add['label'] = 1

                    # add combined datapoint to bottom of combined dataframe and lists 
                    cls_data.append(cls_to_add)
                # move on to next genome
                continue

            # if encoding fails, move on to next datapoint
            try:
                tensor = codec.encode_surrogate(genome, metric_row['epoch_num'])
            except Exception as e:
                break
            # if encoded genome has values greater than threshold, move on to next datapoint
            if np.any(tensor > genome_max_thresh):
                break
            
            cls_to_add['hash'] = genome_hash
            cls_to_add['genome'] = np.clip(tensor, -1000, 1000)
            cls_to_add['label'] = 0 
            reg_to_add['hash'] = genome_hash
            reg_to_add['genome'] = np.clip(tensor, -1000, 1000)

            outlier = False
            for heading in metric_headings:
                # if metric is nan, give bad class label, clip value, and identify as outlier
                if math.isnan(metric_row[heading]):
                    outlier = True
                    cls_to_add['label'] = 1
                    if heading in MAX_METRICS:
                        cls_to_add[heading] = -300.0
                    else:
                        cls_to_add[heading] = 300.0
                else:
                    # if metric is not nan but is outside of range, identify as outlier and give bad label
                    if not (-300 <= metric_row[heading] <= 300):
                        outlier = True
                        cls_to_add['label'] = 1
                    reg_to_add[heading] = np.clip(metric_row[heading], -300, 300)
                    cls_to_add[heading] = np.clip(metric_row[heading], -300, 300)
            
            # if there was no outliers, add to regression dataset
            if not outlier:
                reg_data.append(reg_to_add)
            cls_data.append(cls_to_add)
        
        all_reg_data.append(reg_data)
        all_cls_data.append(cls_data)


    # get reg and cls val sizes
    reg_val_size = int(val_ratio * len(all_reg_data))
    cls_val_size = int(val_ratio * len(all_cls_data))

    # shuffle data
    local_random.shuffle(all_reg_data)
    local_random.shuffle(all_cls_data)

    # get train and validation splits
    val_reg_data = all_reg_data[:reg_val_size]
    train_reg_data = all_reg_data[reg_val_size:]
    val_cls_data = all_cls_data[:cls_val_size]
    train_cls_data = all_cls_data[cls_val_size:]

    reg_val_genomes = [epoch for genome in val_reg_data for epoch in genome]
    reg_train_genomes = [epoch for genome in train_reg_data for epoch in genome]
    cls_val_genomes = [epoch for genome in val_cls_data for epoch in genome]
    cls_train_genomes = [epoch for genome in train_cls_data for epoch in genome]

    # shuffle epochs
    local_random.shuffle(reg_train_genomes)
    local_random.shuffle(reg_val_genomes)
    local_random.shuffle(cls_train_genomes)
    local_random.shuffle(cls_val_genomes)

    # create train/val dfs
    reg_train_set = pd.DataFrame(reg_train_genomes)
    reg_val_set = pd.DataFrame(reg_val_genomes)
    cls_train_set = pd.DataFrame(cls_train_genomes)
    cls_val_set = pd.DataFrame(cls_val_genomes)

    # write train/val sets to file
    os.makedirs(outdir, exist_ok=True)
    reg_train_set.to_pickle(os.path.join(outdir, 'reg_train_dataset.pkl'))
    reg_val_set.to_pickle(os.path.join(outdir, 'reg_val_dataset.pkl'))
    cls_train_set.to_pickle(os.path.join(outdir, 'cls_train_dataset.pkl'))
    cls_val_set.to_pickle(os.path.join(outdir, 'cls_val_dataset.pkl'))

    return reg_train_set, reg_val_set, cls_train_set, cls_val_set


def merge_dfs_to_dataset(df1: pd.DataFrame, df2: pd.DataFrame, outdir):
    merged_df = pd.concat([df1, df2])
    merged_df.to_pickle(outdir)


def find_bad_individuals(df, bad_thresh=100000):
    genomes = np.stack(df['genome'].values)
    hashes = df['hash']
    genomes = torch.from_numpy(genomes)
    bad_mask = genomes > bad_thresh
    bad_indices = torch.where(bad_mask.any(dim=1))[0].numpy()
    bad_individuals = hashes[bad_indices]
    return bad_individuals


def merge_csv_to_dataset(
        files_to_merge = [
            '/home/tthakur9/precog-opt-grip/surrogate_dataset/baseline_evolution_complete_dataset.pkl',
            '/home/tthakur9/precog-opt-grip/surrogate_dataset/train_dataset.pkl'
        ],
        output_file='/surrogate_dataset/merged_train_dataset.pkl'
    ):
    dfs = []
    for file in files_to_merge:
        df = pd.read_pickle(file)
        dfs.append(df)
    comb_df =pd.concat(dfs, ignore_index=True)
    comb_df.to_pickle(output_file)


# TESTING SCRIPT
# build_dataset(infile='/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution/out.csv', working_dir='/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution', val_ratio=0.3)
# reg_train_df = pd.read_pickle('surrogate_dataset/reg_train_dataset.pkl')
# reg_val_df = pd.read_pickle('surrogate_dataset/reg_val_dataset.pkl')
# cls_train_df = pd.read_pickle('surrogate_dataset/cls_train_dataset.pkl')
# cls_val_df = pd.read_pickle('surrogate_dataset/cls_val_dataset.pkl')
# breakpoint()
