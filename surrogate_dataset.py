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

        # features/genomes in the first two cols of df
        # TODO concatenate the epoch number at the front of the genome encoding
        self.genomes = np.stack(df['genome'].values)
        # labels/metrics in the last 12 cols of df
        if metrics_subset is None:
            metrics_subset = list(range(12))
        metrics_subset = [-12 + i for i in metrics_subset]

        self.metrics = df.iloc[:, metrics_subset].values
        self.max_metrics = torch.ones((1, len(metrics_subset))) * 300.0
        self.min_metrics = torch.ones((1, len(metrics_subset))) * -300.0
        
        # # # standardize genome/metrics data dist if train mode
        if mode == 'train':
            # self.metrics = self.metrics_scaler.fit_transform(self.metrics)
            self.genomes = self.genomes_scaler.fit_transform(self.genomes)
        if mode == 'val':
            # self.metrics = self.metrics_scaler.transform(self.metrics)
            self.genomes = self.genomes_scaler.transform(self.genomes)

        # # self.max_metrics = torch.tensor(self.metrics_scaler.transform(self.max_metrics), dtype=torch.float32)
        # # self.min_metrics = torch.tensor(self.metrics_scaler.transform(self.min_metrics), dtype=torch.float32)
            
        if np.isnan(self.genomes).any() or np.isnan(self.metrics).any():
            breakpoint()

    # returns num samples in dataset
    def __len__(self):
        return len(self.df)
    
    # retrieve genome, metrics at specific index
    def __getitem__(self, i):
        # NOTE this will not work until genomes are encoded
        genome = torch.tensor(self.genomes[i], dtype=torch.float32)
        metrics = torch.tensor(self.metrics[i], dtype=torch.float32)
        return genome, metrics
    
    
class ClassifierSurrogateDataset(Dataset):
    # init using df
    def __init__(self, df, mode, genomes_scaler = StandardScaler()):
        self.df = df
        self.genomes_scaler = genomes_scaler
        self.mode = mode

        # features/genomes in the first two cols of df
        # TODO concatenate the epoch number at the front of the genome encoding
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
        # NOTE this will not work until genomes are encoded
        genome = torch.tensor(self.genomes[i], dtype=torch.float32)
        label = torch.tensor(self.labels[i], dtype=torch.float32)
        return genome, label


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
    out_data = pd.DataFrame(columns=['hash', 'genome', 'epoch_num'] + metric_headings)

    data = pd.read_csv(infile)
    data = data.to_dict('records')

    all_genome_info = []
    all_binary_genome_info = []

    genome_max_thresh = 100000

    for line in data:
        genome_hash = line['hash']
        gen = line['gen']
        genome = line['genome']

        if gen in excluded_gens:
            continue
        
        if include_only is not None and gen not in include_only:
            continue

        metrics_path = os.path.join(working_dir, f'generation_{gen}', genome_hash, 'metrics.csv')
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.to_dict('records')

        genome_info = []
        binary_genome_info = []

        for metric_row in metrics:
            to_add = {}
            binary_to_add = {}

            if 'epoch_num' not in metric_row:
                for i in range(num_epochs):
                    to_add = {}
                    try:
                        tensor = codec.encode_surrogate(genome, i + 1)
                    except:
                        break
                    if np.any(tensor > genome_max_thresh):
                        break
                    to_add['genome'] = np.clip(tensor, -1000, 1000)
                    to_add['hash'] = genome_hash
                    binary_to_add['genome'] = np.clip(tensor, -1000, 1000)
                    binary_to_add['hash'] = genome_hash
                    binary_to_add['label'] = 1
                    for heading in metric_headings:
                        if heading in MAX_METRICS:
                            to_add[heading] = -300.0
                        else:
                            to_add[heading] = 300.0
                    out_data.loc[len(out_data)] = to_add
                    genome_info.append(to_add)
                    binary_genome_info.append(binary_to_add)
                continue

            tensor = codec.encode_surrogate(genome, metric_row['epoch_num'])
            if np.any(tensor > genome_max_thresh):
                break
            to_add['hash'] = genome_hash
            to_add['genome'] = np.clip(tensor, -1000, 1000)
            binary_to_add['hash'] = genome_hash
            binary_to_add['genome'] = np.clip(tensor, -1000, 1000)
            binary_to_add['label'] = 0
            
            for heading in metric_headings:
                if math.isnan(metric_row[heading]):
                    binary_to_add['label'] = 1
                    if heading in MAX_METRICS:
                        to_add[heading] = -300.0
                    else:
                        to_add[heading] = 300.0
                else:
                    to_add[heading] = np.clip(metric_row[heading], -300, 300)
                    if to_add[heading] in [-300, 300]:
                        binary_to_add['label'] = 1
            out_data.loc[len(out_data)] = to_add
            genome_info.append(to_add)
            binary_genome_info.append(binary_to_add)
        
        all_genome_info.append(genome_info)
        all_binary_genome_info.append(binary_genome_info)

    assert len(all_genome_info) == len(all_binary_genome_info)
    val_size = int(val_ratio * len(all_genome_info))
    
    # Create a list of indices and shuffle them
    indices = list(range(len(all_genome_info)))
    local_random.shuffle(indices)

    # Split the indices into train and validation sets
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Create the shuffled and split lists for all_genome_info and all_binary_genome_info
    val_genome_info = [all_genome_info[i] for i in val_indices]
    train_genome_info = [all_genome_info[i] for i in train_indices]
    val_binary_info = [all_binary_genome_info[i] for i in val_indices]
    train_binary_info = [all_binary_genome_info[i] for i in train_indices]

    # Flatten the train and validation sets
    val_genomes = [epoch for genome in val_genome_info for epoch in genome]
    train_genomes = [epoch for genome in train_genome_info for epoch in genome]
    val_binary_genomes = [epoch for genome in val_binary_info for epoch in genome]
    train_binary_genomes = [epoch for genome in train_binary_info for epoch in genome]

    # Pair the elements from the two lists
    train_pairs = list(zip(train_genomes, train_binary_genomes))
    val_pairs = list(zip(val_genomes, val_binary_genomes))

    # Shuffle the pairs
    local_random.shuffle(train_pairs)
    local_random.shuffle(val_pairs)

    # Separate the pairs back into the two lists
    train_genomes, train_binary_genomes = zip(*train_pairs)
    val_genomes, val_binary_genomes = zip(*val_pairs)

    # Convert tuples back to lists
    train_genomes = list(train_genomes)
    train_binary_genomes = list(train_binary_genomes)
    val_genomes = list(val_genomes)
    val_binary_genomes = list(val_binary_genomes)

    # Create dataframes from genome lists 
    train_set = pd.DataFrame(train_genomes)
    val_set = pd.DataFrame(val_genomes)
    binary_train_set = pd.DataFrame(train_binary_genomes)
    binary_val_set = pd.DataFrame(val_binary_genomes)

    # Write stuff to file
    complete_output_filename = os.path.join(outdir, 'complete_dataset.pkl')
    train_output_filename = os.path.join(outdir, 'train_dataset.pkl')
    val_output_filename = os.path.join(outdir, 'val_dataset.pkl')
    train_binary_output_filename = os.path.join(outdir, 'train_binary_dataset.pkl')
    val_binary_output_filename = os.path.join(outdir, 'val_binary_dataset.pkl')
    os.makedirs(os.path.dirname(complete_output_filename), exist_ok=True)
    out_data.to_pickle(complete_output_filename)
    train_set.to_pickle(train_output_filename)
    val_set.to_pickle(val_output_filename)
    binary_train_set.to_pickle(train_binary_output_filename)
    binary_val_set.to_pickle(val_binary_output_filename)

    return train_set, val_set


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


#build_dataset(infile='/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution/out.csv', working_dir='/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution')
# train_df = pd.read_pickle('surrogate_dataset/train_dataset.pkl')
# val_df = pd.read_pickle('surrogate_dataset/val_dataset.pkl')
# binary_train_df = pd.read_pickle('surrogate_dataset/train_binary_dataset.pkl')
# binary_val_df = pd.read_pickle('surrogate_dataset/val_binary_dataset.pkl')
# print(train_df.shape, val_df.shape, binary_train_df.shape, binary_val_df.shape)
# print(train_df.head())
# print(val_df.head())
# print(binary_train_df.head())
# print(binary_val_df.head())
# classifier_train_dataset = ClassifierSurrogateDataset(binary_train_df, mode='train')
# classifier_val_dataset = ClassifierSurrogateDataset(binary_val_df, mode='val', genomes_scaler=classifier_train_dataset.genomes_scaler)