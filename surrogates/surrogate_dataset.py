"""
Functions for building surrogate datasets and dataset classes used for surrogate train/eval.
"""


import argparse
import math
import os
import sys
file_directory = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
repo_dir = os.path.abspath(os.path.join(file_directory, ".."))
sys.path.append(repo_dir)

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
import tqdm
from collections import defaultdict
import time


class SurrogateDataset(Dataset):
    # init using df
    def __init__(self, df, mode, metrics_subset=None, metrics_scaler=StandardScaler(), genomes_scaler = StandardScaler()):
        self.df = df
        self.genomes_scaler = genomes_scaler
        self.metrics_scaler = metrics_scaler
        self.mode = mode
        if metrics_subset is None:
            metrics_subset = list(range(12))
        metrics_subset = [-12 + i for i in metrics_subset]
        self.max_metrics = torch.ones((1, len(metrics_subset))) * 300.0
        self.min_metrics = torch.ones((1, len(metrics_subset))) * -300.0
        
        # standardize genome/metrics data dist if train mode
        if mode == 'train':
            self.genomes = np.stack(df['genome'].values)
            self.metrics = df.iloc[:, metrics_subset].values
            self.genomes = self.genomes_scaler.fit_transform(self.genomes)
        if mode == 'val':
            grouped = df.groupby('hash')
            best_epochs = []
            for _, g in grouped:
                best_epoch_idx = g['uw_val_epoch_loss'].idxmin()
                best_epoch = g.loc[best_epoch_idx]
                best_epochs.append(best_epoch)
            best_epochs_df = pd.DataFrame(best_epochs)
            self.genomes = np.stack(best_epochs_df['genome'].values)
            self.metrics = best_epochs_df.iloc[:, metrics_subset].values
            self.genomes = self.genomes_scaler.transform(self.genomes)
            
        # NOTE commented out for debugging
        # if np.isnan(self.genomes).any() or np.isnan(self.metrics).any():
        #     breakpoint()

    # returns num samples in dataset
    def __len__(self):
        return len(self.genomes)
    
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


class ParentChildSurrogateDataset(Dataset):
    def __init__(self, df, mode, metrics_subset=None, metrics_scaler=StandardScaler(), genomes_scaler=StandardScaler()):
        self.df = df
        self.genomes_scaler = genomes_scaler
        self.metrics_scaler = metrics_scaler
        self.mode = mode
        
        if metrics_subset is None:
            metrics_subset = list(range(12))
        metrics_subset = [-12 + i for i in metrics_subset]
        
        # Extract child genomes
        self.child_genomes = np.stack(df['genome'].values)
        
        # Extract parent genomes
        self.parent1_genomes = np.stack(df['parent1_genome'].values)
        self.parent2_genomes = np.stack(df['parent2_genome'].values)
        
        # Extract parent fitnesses
        self.parent1_fitnesses = np.stack(df['parent1_fitness'].values)
        self.parent2_fitnesses = np.stack(df['parent2_fitness'].values)
        
        # Extract child metrics (target)
        self.child_metrics = df.iloc[:, metrics_subset].values
        
        # Apply scaling
        if mode == 'train':
            # Fit scalers on child genomes
            self.child_genomes = self.genomes_scaler.fit_transform(self.child_genomes)
            # Apply same scaling to parent genomes
            parent_genomes_combined = np.concatenate([self.parent1_genomes, self.parent2_genomes])
            parent_genomes_scaled = self.genomes_scaler.transform(parent_genomes_combined)
            self.parent1_genomes = parent_genomes_scaled[:len(self.parent1_genomes)]
            self.parent2_genomes = parent_genomes_scaled[len(self.parent1_genomes):]
        elif mode == 'val':
            # Apply pre-fitted scaling
            self.child_genomes = self.genomes_scaler.transform(self.child_genomes)
            parent_genomes_combined = np.concatenate([self.parent1_genomes, self.parent2_genomes])
            parent_genomes_scaled = self.genomes_scaler.transform(parent_genomes_combined)
            self.parent1_genomes = parent_genomes_scaled[:len(self.parent1_genomes)]
            self.parent2_genomes = parent_genomes_scaled[len(self.parent1_genomes):]

    def __len__(self):
        return len(self.child_genomes)
    
    def __getitem__(self, i):
        child_genome = torch.tensor(self.child_genomes[i], dtype=torch.float32)
        parent1_genome = torch.tensor(self.parent1_genomes[i], dtype=torch.float32)
        parent2_genome = torch.tensor(self.parent2_genomes[i], dtype=torch.float32)
        parent1_fitness = torch.tensor(self.parent1_fitnesses[i], dtype=torch.float32)
        parent2_fitness = torch.tensor(self.parent2_fitnesses[i], dtype=torch.float32)
        child_metrics = torch.tensor(self.child_metrics[i], dtype=torch.float32)
        
        combined = torch.cat([child_genome, parent1_genome, parent2_genome, 
                          parent1_fitness, parent2_fitness], dim=0)
        return combined, child_metrics


class ParentChildClassifierDataset(Dataset):
    def __init__(self, df, mode, genomes_scaler=StandardScaler()):
        self.df = df
        self.genomes_scaler = genomes_scaler
        self.mode = mode
        
        # Extract data similar to ParentChildSurrogateDataset
        self.child_genomes = np.stack(df['genome'].values)
        self.parent1_genomes = np.stack(df['parent1_genome'].values)
        self.parent2_genomes = np.stack(df['parent2_genome'].values)
        self.parent1_fitnesses = np.stack(df['parent1_fitness'].values)
        self.parent2_fitnesses = np.stack(df['parent2_fitness'].values)
        self.labels = np.stack(df['label'].values)
        
        # Apply scaling similar to above
        if mode == 'train':
            self.child_genomes = self.genomes_scaler.fit_transform(self.child_genomes)
            parent_genomes_combined = np.concatenate([self.parent1_genomes, self.parent2_genomes])
            parent_genomes_scaled = self.genomes_scaler.transform(parent_genomes_combined)
            self.parent1_genomes = parent_genomes_scaled[:len(self.parent1_genomes)]
            self.parent2_genomes = parent_genomes_scaled[len(self.parent1_genomes):]
        elif mode == 'val':
            self.child_genomes = self.genomes_scaler.transform(self.child_genomes)
            parent_genomes_combined = np.concatenate([self.parent1_genomes, self.parent2_genomes])
            parent_genomes_scaled = self.genomes_scaler.transform(parent_genomes_combined)
            self.parent1_genomes = parent_genomes_scaled[:len(self.parent1_genomes)]
            self.parent2_genomes = parent_genomes_scaled[len(self.parent1_genomes):]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        child_genome = torch.tensor(self.child_genomes[i], dtype=torch.float32)
        parent1_genome = torch.tensor(self.parent1_genomes[i], dtype=torch.float32)
        parent2_genome = torch.tensor(self.parent2_genomes[i], dtype=torch.float32)
        parent1_fitness = torch.tensor(self.parent1_fitnesses[i], dtype=torch.float32)
        parent2_fitness = torch.tensor(self.parent2_fitnesses[i], dtype=torch.float32)
        label = torch.tensor(self.labels[i], dtype=torch.float32)
        
        combined = torch.cat([child_genome, parent1_genome, parent2_genome, 
                          parent1_fitness, parent2_fitness], dim=0)
        return combined, label


# builds classifier and regressor datasets by scraping through a run directory and looking at metric.csv files
# 
# WORKS BUT SHOULD BE PROOFREAD
def build_dataset(
        name,
        infile='/gv1/projects/GRIP_Precog_Opt/outputs/out.csv',
        working_dir='/gv1/projects/GRIP_Precog_Opt/outputs', 
        outdir='surrogate_dataset', 
        metrics='uw_val_epoch_loss,iou_loss,giou_loss,diou_loss,ciou_loss,center_loss,size_loss,obj_loss,precision,recall,f1_score,average_precision', 
        exclude=[], include_only=None, val_ratio=0.3, seed=0,
        include_parents=False
    ):

    os.makedirs(outdir, exist_ok=True)

    codec = Codec(num_classes=7)
    metric_headings = metrics.split(',')
    excluded_gens = exclude
    
    local_random = random.Random(seed)
    configs = toml.load(os.path.join(working_dir, "conf.toml"))
    model_config = configs["model"]
    num_epochs = model_config['train_epochs']
    MAX_METRICS = ['precision', 'recall', 'f1_score', 'average_precision']
    genome_max_thresh = 100000

    genealogy = defaultdict(list)  # hash -> list of parent hashes
    genome_to_fitness = {}  # hash -> fitness dict
    
    
    if include_parents:
        # Load genealogy file if it exists
        genealogy_file = os.path.join(working_dir, 'genealogy.csv')
        if os.path.exists(genealogy_file):
            genealogy_df = pd.read_csv(genealogy_file)
            for _, row in genealogy_df.iterrows():
                child_hash = row['child_hash']
                if 'parent1_hash' in row and pd.notna(row['parent1_hash']):
                    genealogy[child_hash].append(row['parent1_hash'])
                if 'parent2_hash' in row and pd.notna(row['parent2_hash']):
                    genealogy[child_hash].append(row['parent2_hash'])
    
    data = pd.read_csv(infile)
    # Keep data as DataFrame for genealogy lookups
    data_df = data.copy()

    # First pass: collect all fitness data
    if include_parents:
        for _, line in data.iterrows():
            genome_hash = line['hash']
            gen = line['gen']
            if gen in excluded_gens or (include_only is not None and len(include_only) > 0 and gen not in include_only):
                continue
            
            # Interact with metrics file for this genome
            metrics_path = os.path.join(working_dir, f'generation_{gen}', genome_hash, 'metrics.csv')
            try:
                # Read metrics file
                metrics_df = pd.read_csv(metrics_path)
                if not metrics_df.empty:
                    # Get best epoch metrics
                    if 'epoch_num' in metrics_df.columns:
                        best_metrics = metrics_df.iloc[-1].to_dict()  # choosing last epoch as best metric
                        genome_to_fitness[genome_hash] = best_metrics
            except FileNotFoundError:
                continue
    
    
    data = data.to_dict('records')
    data = tqdm.tqdm(data)

    all_reg_data = []
    all_cls_data = []
    tensor_shape = None

    # Second pass: build datasets
    for line in data:
        genome_hash = line['hash']
        gen = line['gen']
        genome = line['genome']

        if not tensor_shape:
            tensor_shape = codec.encode_surrogate(genome, 1).shape

        if gen in excluded_gens or (include_only is not None and len(include_only) > 0 and gen not in include_only):
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

            if include_parents:

                # Add parent encodings and fitnesses
                parent_encodings = []
                parent_fitnesses = []
                if genome_hash in genealogy:
                    parents = genealogy[genome_hash]
                    print(f"DEBUG: Found {len(parents)} parents for genome {genome_hash}")
                    print(f"DEBUG: Parents: {parents}")
                    
                    for parent_hash in parents:
                        if parent_hash and parent_hash in genome_to_fitness:
                            # Get parent genome from data
                            parent_row = data_df[data_df['hash'] == parent_hash]
                            print(f"DEBUG: Found parent row for {parent_hash}: {parent_row}")
                            if not parent_row.empty:
                                parent_genome = parent_row.iloc[0]['genome']
                                print(f"DEBUG: Encoding parent genome {parent_hash} for genome {genome_hash}")
                                try:
                                    parent_tensor = codec.encode_surrogate(parent_genome, metric_row.get('epoch_num', 1))
                                    print(f"DEBUG: Encoded parent genome {parent_hash} with tensor shape {parent_tensor.shape}")
                                    parent_encodings.append(np.clip(parent_tensor, -1000, 1000))
                                    
                                    # Get parent fitness
                                    parent_fitness = genome_to_fitness[parent_hash]
                                    parent_fitness_values = [parent_fitness.get(heading, 0) for heading in metric_headings]
                                    parent_fitnesses.append(np.array(parent_fitness_values, dtype=np.float32))
                                except:
                                    continue
                
                # Pad or truncate to exactly 2 parents (for consistency)
                while len(parent_encodings) < 2:
                    parent_encodings.append(np.zeros(tensor_shape))  # Zero padding for missing parents
                    parent_fitnesses.append(np.zeros((len(metric_headings),)))
                
                parent_encodings = parent_encodings[:2]  # Take first 2 parents
                parent_fitnesses = parent_fitnesses[:2]
                
                # Add to datapoint dictionaries
                reg_to_add['parent1_genome'] = parent_encodings[0]
                reg_to_add['parent2_genome'] = parent_encodings[1] 
                reg_to_add['parent1_fitness'] = parent_fitnesses[0]
                reg_to_add['parent2_fitness'] = parent_fitnesses[1]
                
                cls_to_add['parent1_genome'] = parent_encodings[0]
                cls_to_add['parent2_genome'] = parent_encodings[1]
                cls_to_add['parent1_fitness'] = parent_fitnesses[0] 
                cls_to_add['parent2_fitness'] = parent_fitnesses[1]

            # if genome is failed, only create classifier datapoint and combined datapoint
            if 'epoch_num' not in metric_row:
                for i in range(num_epochs):
                    cls_to_add_epoch = {}
                    cls_to_add_epoch.update(cls_to_add)
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
                            cls_to_add_epoch[heading] = -300.0
                        else:
                            cls_to_add_epoch[heading] = 300.0

                    # add clipped encoding, hash, and bad label to classifier datapoint dict
                    cls_to_add_epoch['genome'] = np.clip(tensor, -1000, 1000)
                    cls_to_add_epoch['hash'] = genome_hash
                    cls_to_add_epoch['label'] = 1

                    # add combined datapoint to bottom of combined dataframe and lists
                    cls_data.append(cls_to_add_epoch)
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

    reg_concat_idx = len(reg_train_set)
    cls_concat_idx = len(cls_train_set)

    complete_reg_set = pd.concat([reg_train_set, reg_val_set], axis=0).reset_index(drop=True)
    complete_cls_set = pd.concat([cls_train_set, cls_val_set], axis=0).reset_index(drop=True)

    # print("train_reg_data: ", train_reg_data[:2])
    # print("val_reg_data: ", val_reg_data[:2])
    print("reg_train_set: ", reg_train_set.head())
    print("reg_val_set: ",  reg_val_set.head())
    print("complete_reg_set: ",  complete_reg_set.head())
    print("complete_reg_set cols: ",  complete_reg_set.columns.tolist())

    complete_reg_set, reg_num_train_rem = remove_dupes(complete_reg_set, 'genome', reg_concat_idx)
    complete_cls_set, cls_num_train_rem = remove_dupes(complete_cls_set, 'genome', cls_concat_idx)

    reg_split = reg_concat_idx - reg_num_train_rem
    cls_split = reg_concat_idx - cls_num_train_rem

    reg_train_set = complete_reg_set[:reg_split]
    reg_val_set = complete_reg_set[reg_split:]
    cls_train_set = complete_cls_set[:cls_split]
    cls_val_set = complete_cls_set[cls_split:]

    # write train/val sets to file
    os.makedirs(outdir, exist_ok=True)
    if not reg_train_set.empty:
        reg_train_set.to_pickle(os.path.join(outdir, f'{name}_reg_train.pkl'))
    else:
        print("reg_train_set empty, didnt pickle it")
    if not reg_val_set.empty:
        reg_val_set.to_pickle(os.path.join(outdir, f'{name}_reg_val.pkl'))
    else:
        print("reg_val_set empty, didnt pickle it")
    if not cls_train_set.empty:
        cls_train_set.to_pickle(os.path.join(outdir, f'{name}_cls_train.pkl'))
    else:
        print("cls_train_set empty, didnt pickle it")
    if not cls_val_set.empty:
        cls_val_set.to_pickle(os.path.join(outdir, f'{name}_cls_val.pkl'))
    else:
        print("cls_val_set empty, didnt pickle it")

    return reg_train_set, reg_val_set, cls_train_set, cls_val_set


def merge_dfs_to_dataset(df1: pd.DataFrame, df2: pd.DataFrame, outdir):
    merged_df = pd.concat([df1, df2])
    merged_df.to_pickle(outdir)


def remove_dupes(df, key, concat_idx):
    mask = df[key].apply(lambda x: tuple(x)).duplicated(keep='first')
    num_train_dupes_removed = mask[:concat_idx].sum()
    return df[~mask], num_train_dupes_removed


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
            os.path.join(repo_dir, 'surrogate_dataset/baseline_evolution_complete_dataset.pkl'),
            os.path.join(repo_dir, 'surrogate_dataset/train_dataset.pkl')
        ],
        output_file='/surrogate_dataset/merged_train_dataset.pkl'
    ):
    dfs = []
    for file in files_to_merge:
        df = pd.read_pickle(file)
        dfs.append(df)
    comb_df =pd.concat(dfs, ignore_index=True)
    comb_df.to_pickle(output_file)


def main():


    import argparse
    parser = argparse.ArgumentParser("Generates datasets for training surrogates")
    parser.add_argument('name', type=str, help='The prefix added to generated .pkl dataset files')
    parser.add_argument('--working-dir', type=str, default='/gv1/projects/GRIP_Precog_Opt/outputs', help='The working directory for the evolution to turn into a dataset')
    parser.add_argument('--infile', type=str, default=None, help='The input file, which is an out.csv file from a previous run. Defaults to the out.csv file in the working-dir')
    parser.add_argument('--outdir', type=str, default='surrogate_dataset', help='The directory to save the surrogate dataset to')
    parser.add_argument('--metrics', type=str, default='uw_val_epoch_loss,iou_loss,giou_loss,diou_loss,ciou_loss,center_loss,size_loss,obj_loss,precision,recall,f1_score,average_precision', help='Comma seperated list of metric header names in the out.csv file')
    parser.add_argument('--exclude', type=str, default='', help='Comma seperated generation numbers to exclude from the dataset, corresponding to the gen field in the out.csv file')
    parser.add_argument('--include-only', type=str, default='', help='Comma seperated generation numbers to exclusively include from the dataset, corresponding to the gen field in the out.csv file')
    parser.add_argument('--val-ratio', type=float, default=0.3, help='The proportion of the dataset to allocate to the validation set')
    parser.add_argument('--seed', type=int, default=0, help='The random seed used for generating and shuffling the dataset')
    my_args = parser.parse_args()

    exclude = [int(item) for item in my_args.exclude.split(',') if len(item) > 0]
    include_only = [int(item) for item in my_args.include_only.split(',') if len(item) > 0]
    if my_args.infile is None:
        my_args.infile = os.path.join(my_args.working_dir, 'out.csv')

    build_dataset(my_args.name,
        infile=my_args.infile,
        working_dir=my_args.working_dir, 
        outdir=my_args.outdir, 
        metrics=my_args.metrics, 
        exclude=exclude, include_only=include_only, val_ratio=my_args.val_ratio, seed=my_args.seed,
        include_parents=True
    )

    reg_train_df = pd.read_pickle(os.path.join(my_args.outdir, f'{my_args.name}_reg_train.pkl'))
    reg_val_df = pd.read_pickle(os.path.join(my_args.outdir, f'{my_args.name}_reg_val.pkl'))
    reg_train_ds = ParentChildSurrogateDataset(reg_train_df, 'train', None)
    reg_val_ds = ParentChildSurrogateDataset(reg_val_df, 'val', None, reg_train_ds.metrics_scaler, reg_train_ds.genomes_scaler)
    cls_train_df = pd.read_pickle(os.path.join(my_args.outdir, f'{my_args.name}_cls_train.pkl'))
    cls_val_df = pd.read_pickle(os.path.join(my_args.outdir, f'{my_args.name}_reg_val.pkl'))

if __name__ == "__main__":
    main()

