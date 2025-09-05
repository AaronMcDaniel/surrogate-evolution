"""
Plots the average and median of main metrics (uw_val_epoch_loss, ciou_loss, average_precision) per generation from out.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
from pareto_utils import find_pareto_indices



parser = argparse.ArgumentParser()
parser.add_argument('username', type=str)
parser.add_argument('--hall_of_fame', action='store_true', help='If set, only log metrics for the Pareto front (hall of fame) per generation')
parser.add_argument('--cross_gen', action='store_true', help='If set, hall of fame is computed from all populations up to and including the current generation')
parser.add_argument('--include_min_max', action='store_true', help='If set, include min/max values in the plot')
args = parser.parse_args()


csv_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/out.csv'
USER = args.username
output_dir = f'/home/hice1/{USER}/scratch/surrogate-evolution/analysis/graphs/metricsPlotting/'

hall_of_fame = args.hall_of_fame
cross_generation_pareto_front = args.cross_gen
include_min_max = args.include_min_max

# Main metrics to track
metrics = ['uw_val_epoch_loss', 'ciou_loss', 'average_precision']

# Directions: True = minimize, False = maximize (as in pareto_front.py)
directions = [True, True, False]
directions_dict = {'uw_val_epoch_loss': True, 'ciou_loss': True, 'average_precision': False}

# Read data
df = pd.read_csv(csv_path)

# Drop rows with missing values in main metrics
df = df.dropna(subset=metrics + ['gen'])

generations = sorted(df['gen'].unique())

# Prepare data for plotting
extreme_dict = {m: [] for m in metrics}
med_dict = {m: [] for m in metrics}




# Efficient cross-generation Pareto front caching (like pareto_front.py)
cached_front = None
for idx, gen in enumerate(generations):
    if hall_of_fame:
        if cross_generation_pareto_front:
            df_current_gen = df[df['gen'] == gen]
            if idx == 0 or cached_front is None:
                # First generation, just use current gen
                df_combined = df_current_gen.copy()
            else:
                # Combine previous front with current gen
                df_combined = pd.concat([cached_front, df_current_gen], ignore_index=True)
            front, _ = find_pareto_indices(df_combined, metrics, directions)
            cached_front = front.copy()
            df_to_use = front
        else:
            # Use only this generation
            df_hof = df[df['gen'] == gen]
            front, _ = find_pareto_indices(df_hof, metrics, directions)
            df_to_use = front
    else:
        df_to_use = df[df['gen'] == gen]
    for m in metrics:
        if include_min_max:
            extreme_dict[m].append(df_to_use[m].min() if directions_dict[m] else df_to_use[m].max())
        med_dict[m].append(df_to_use[m].median())

# Plot average and median for each metric
plt.figure(figsize=(12, 8))
for i, m in enumerate(metrics):
    plt.subplot(3, 1, i+1)
    if include_min_max:
        plt.plot(generations, extreme_dict[m], label=f'Min/Max {m}', marker='o', color='tab:blue')
    plt.plot(generations, med_dict[m], label=f'Median {m}', marker='s', color='tab:orange')
    plt.title(f'{m} per Generation')
    plt.xlabel('Generation')
    plt.ylabel(m)
    plt.legend()
    plt.grid(True)
plt.tight_layout()

if hall_of_fame:
    if cross_generation_pareto_front:
        plot_path = os.path.join(output_dir, 'population_metrics_cross_generation_hall_of_fame_per_generation.jpg')
    else:
        plot_path = os.path.join(output_dir, 'population_metrics_hall_of_fame_per_generation.jpg')
else:
    plot_path = os.path.join(output_dir, 'population_metrics_per_generation.jpg')
plt.savefig(plot_path)
plt.close()
print(f'Plot saved to {plot_path}')
