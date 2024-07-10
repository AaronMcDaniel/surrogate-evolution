'''
File to generate surogate dataset
Steps:
- Load out.csv or equivalent
- Load metrics.csv from run dir
- Process entries to replace NaN values with large numbers
- Create new dataset
- Split data into train/val or similar
'''

import argparse
import math
import os
import random

import pandas as pd
import toml
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', required=False, default='/gv1/projects/GRIP_Precog_Opt/outputs/out.csv')
parser.add_argument('-w', '--working', required=False, default='/gv1/projects/GRIP_Precog_Opt/outputs')
parser.add_argument('-o', '--outdir', required=False, default='surrogate_dataset')
parser.add_argument('-m', '--metrics', required=False, default='uw_val_epoch_loss,iou_loss,giou_loss,diou_loss,ciou_loss,center_loss,size_loss,obj_loss,precision,recall,f1_score,average_precision')
parser.add_argument('-x', '--exclude', required=False, default='')
parser.add_argument('-v', '--valratio', required=False, default=0.3)
parser.add_argument('-s', '--seed', required=False, default=0)
args = parser.parse_args()
infile = args.infile
working_dir = args.working
outdir = args.outdir
metric_headings = args.metrics
metric_headings = metric_headings.split(',')
excluded_gens = args.exclude
if excluded_gens == '':
    excluded_gens = []
else:
    excluded_gens = [int(gen) for gen in excluded_gens.split(',')]
val_ratio = args.valratio
seed = args.seed

local_random = random.Random(seed)


configs = toml.load(os.path.join(working_dir, "conf.toml"))
model_config = configs["model"]
num_epochs = model_config['train_epochs']


MAX_METRICS = ['precision', 'recall', 'f1_score', 'average_precision']
out_data = pd.DataFrame(columns=['genome', 'epoch_num']+metric_headings)

data = pd.read_csv(infile)
data = data.to_dict('records')

all_genome_info = []

for line in data:
    genome_hash = line['hash']
    gen = line['gen']
    genome = line['genome']

    if gen in excluded_gens:
        continue

    metrics = pd.read_csv(os.path.join(working_dir, f'generation_{gen}', genome_hash, 'metrics.csv'))
    metrics = metrics.to_dict('records')

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
                        to_add[heading] = torch.tensor(-1000000, dtype=torch.float32)
                    else:
                        to_add[heading] = torch.tensor(1000000.0, dtype=torch.float32)
                out_data.loc[len(out_data)] = to_add
                genome_info.append(to_add)
            continue

        to_add['epoch_num'] = metric_row['epoch_num']
        for heading in metric_headings:
            if math.isnan(metric_row[heading]):
                if heading in MAX_METRICS:
                    to_add[heading] = torch.tensor(-1000000, dtype=torch.float32)
                else:
                    to_add[heading] = torch.tensor(1000000.0, dtype=torch.float32)
            else:
                to_add[heading] = torch.tensor(metric_row[heading], dtype=torch.float32)
        out_data.loc[len(out_data)] = to_add
        genome_info.append(to_add)
    
    all_genome_info.append(genome_info)


val_size = int(val_ratio*len(all_genome_info))
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
