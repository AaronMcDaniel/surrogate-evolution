import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import toml
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class SurrogateDataset(Dataset):
    # init using df
    def __init__(self, df, mode, scaler=StandardScaler()):
        self.df = df
        self.scaler = scaler
        self.mode = mode

        # features/genomes in the first two cols of df
        # TODO concatenate the epoch number at the front of the genome encoding
        self.genomes = df.iloc[:, :2].values
        # labels/metrics in the last 12 cols of df
        self.metrics = df.iloc[:, -12:].values

        # standardize genome/metrics data dist if train mode
        if mode == 'train':
            self.metrics = scaler.fit_transform(self.metrics)
            # TODO uncomment this line when encodings are finished
            # self.genomes = scaler.fit_transform(self.genomes)
        if mode == 'val':
            self.metrics = scaler.transform(self.metrics)
            # TODO uncomment this line when encodings are finished
            # self.genomes = scaler.transform(self.genoems)

    # returns num samples in dataset
    def __len__(self):
        return len(self.df)
    
    # retrieve genome, metrics at specific index
    def __getitem__(self, i):
        # NOTE this will not work until genomes are encoded
        # genomes = torch.tensor(self.genomes[i], dtype=torch.float32)
        genome = torch.rand(976, dtype=torch.float32)
        metrics = torch.tensor(self.metrics[i], dtype=torch.float32)
        return genome, metrics


def build_dataset(
        infile='/gv1/projects/GRIP_Precog_Opt/outputs/out.csv',
        working_dir='/gv1/projects/GRIP_Precog_Opt/outputs', 
        outdir='surrogate_dataset', 
        metrics='uw_val_epoch_loss,iou_loss,giou_loss,diou_loss,ciou_loss,center_loss,size_loss,obj_loss,precision,recall,f1_score,average_precision', 
        exclude='', val_ratio=0.3, seed=0
    ):
    metric_headings = metrics.split(',')
    excluded_gens = exclude
    if excluded_gens == '':
        excluded_gens = []
    else:
        excluded_gens = [int(gen) for gen in excluded_gens.split(',')]
    
    local_random = random.Random(seed)

    configs = toml.load(os.path.join(working_dir, "conf.toml"))
    model_config = configs["model"]
    num_epochs = model_config['train_epochs']

    MAX_METRICS = ['precision', 'recall', 'f1_score', 'average_precision']
    out_data = pd.DataFrame(columns=['genome', 'epoch_num'] + metric_headings)

    data = pd.read_csv(infile)
    data = data.to_dict('records')

    all_genome_info = []

    for line in data:
        genome_hash = line['hash']
        gen = line['gen']
        genome = line['genome']

        if gen in excluded_gens:
            continue

        metrics_path = os.path.join(working_dir, f'generation_{gen}', genome_hash, 'metrics.csv')
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.to_dict('records')

        genome_info = []

        for metric_row in metrics:
            to_add = {}
            to_add['genome'] = genome

            if 'epoch_num' not in metric_row:
                for i in range(num_epochs):
                    to_add = {}
                    to_add['genome'] = genome
                    to_add['epoch_num'] = i + 1
                    for heading in metric_headings:
                        if heading in MAX_METRICS:
                            to_add[heading] = -1000000.0
                        else:
                            to_add[heading] = 1000000.0
                    out_data.loc[len(out_data)] = to_add
                    genome_info.append(to_add)
                continue

            to_add['epoch_num'] = metric_row['epoch_num']
            for heading in metric_headings:
                if math.isnan(metric_row[heading]):
                    if heading in MAX_METRICS:
                        to_add[heading] = -1000000.0
                    else:
                        to_add[heading] = 1000000.0
                else:
                    to_add[heading] = metric_row[heading]
            out_data.loc[len(out_data)] = to_add
            genome_info.append(to_add)
        
        all_genome_info.append(genome_info)

    val_size = int(val_ratio * len(all_genome_info))
    shuffled_data = all_genome_info[:]
    local_random.shuffle(shuffled_data)
    val_genome_info = shuffled_data[:val_size]
    train_genome_info = shuffled_data[val_size:]
    val_genomes = [epoch for genome in val_genome_info for epoch in genome]
    train_genomes = [epoch for genome in train_genome_info for epoch in genome]
    local_random.shuffle(train_genomes)
    local_random.shuffle(val_genomes)

    train_set = pd.DataFrame(train_genomes)
    val_set = pd.DataFrame(val_genomes)

    complete_output_filename = os.path.join(outdir, 'complete_dataset.csv')
    train_output_filename = os.path.join(outdir, 'train_dataset.csv')
    val_output_filename = os.path.join(outdir, 'val_dataset.csv')
    os.makedirs(os.path.dirname(complete_output_filename), exist_ok=True)
    out_data.to_csv(complete_output_filename, index=False)
    train_set.to_csv(train_output_filename, index=False)
    val_set.to_csv(val_output_filename, index=False)

    return train_set, val_set

# build_dataset(infile='/gv1/projects/GRIP_Precog_Opt/test_evolution/out.csv', working_dir='/gv1/projects/GRIP_Precog_Opt/test_evolution')
