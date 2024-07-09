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

import pandas as pd
import toml


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', required=False, default='/gv1/projects/GRIP_Precog_Opt/outputs/out.csv')
parser.add_argument('-w', '--working', required=False, default='/gv1/projects/GRIP_Precog_Opt/outputs')
parser.add_argument('-o', '--outdir', required=False, default='surrogate_dataset')
parser.add_argument('-m', '--metrics', required=False, default='uw_val_epoch_loss,iou_loss,giou_loss,diou_loss,ciou_loss,center_loss,size_loss,obj_loss,precision,recall,f1_score,average_precision')
parser.add_argument('-x', '--exclude', required=False, default='')
args = parser.parse_args()
infile = args.infile
working_dir = args.working
outdir = args.outdir
metric_headings = args.metrics
metric_headings = metric_headings.split(',')
excluded_gens = args.exclude
excluded_gens = [int(gen) for gen in excluded_gens.split(',')]


configs = toml.load(os.path.join(working_dir, "conf.toml"))
model_config = configs["model"]
num_epochs = model_config['train_epochs']


MAX_METRICS = ['precision', 'recall', 'f1_score', 'average_precision']
out_data = pd.DataFrame(columns=['genome', 'epoch_num']+metric_headings)

data = pd.read_csv(infile)
data = data.to_dict('records')


for line in data:
    genome_hash = line['hash']
    gen = line['gen']
    genome = line['genome']

    if gen in excluded_gens:
        continue

    metrics = pd.read_csv(os.path.join(working_dir, f'generation_{gen}', genome_hash, 'metrics.csv'))
    metrics = metrics.to_dict('records')

    for metric_row in metrics:
        to_add = {}
        to_add['genome'] = genome

        if 'epoch_num' not in metric_row:
            for i in range(num_epochs):
                to_add['epoch_num'] = i + 1
                for heading in metric_headings:
                    if heading in MAX_METRICS:
                        to_add[heading] = -1000000
                    else:
                        to_add[heading] = 1000000
                out_data.loc[len(out_data)] = to_add
            continue

        to_add['epoch_num'] = metric_row['epoch_num']
        for heading in metric_headings:
            if math.isnan(metric_row[heading]):
                if heading in MAX_METRICS:
                    to_add[heading] = -1000000
                else:
                    to_add[heading] = 1000000
            else:
                to_add[heading] = metric_row[heading]
        out_data.loc[len(out_data)] = to_add

output_filename = os.path.join(outdir, 'complete_dataset.csv')
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
out_data.to_csv(output_filename, index=False)
        