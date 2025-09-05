"""
Plots the metrics for individuals with best combined ranking in ciou_loss and average_precision.
Finds the Pareto-dominant individual with minimal (rank_ciou + rank_avg_precision) and also shows
the individuals that are one rank better in ciou_loss and average_precision respectively.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
from pareto_utils import find_pareto_indices



parser = argparse.ArgumentParser()
parser.add_argument('username', type=str)
parser.add_argument('--cross_gen', action='store_true', help='If set, hall of fame is computed from all populations up to and including the current generation')
args = parser.parse_args()


csv_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/out.csv'
USER = args.username
output_dir = f'/home/hice1/{USER}/scratch/surrogate-evolution/analysis/graphs/metricsPlotting/'

cross_generation_pareto_front = args.cross_gen

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

# For each generation, find the best individuals for each metric
best_individuals_per_gen = {}

# Efficient cross-generation Pareto front caching (like pareto_front.py)
cached_front = None
for idx, gen in enumerate(generations):
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
    
    # Find best individuals for each metric in this generation
    best_individuals = {}
    for metric in metrics:
        if directions_dict[metric]:  # minimize
            best_idx = df_to_use[metric].idxmin()
        else:  # maximize
            best_idx = df_to_use[metric].idxmax()
        best_individuals[metric] = df_to_use.loc[best_idx]
    
    best_individuals_per_gen[gen] = best_individuals

# Prepare data for plotting - track each metric for each of the 3 best individuals
plot_data = {}
for metric in metrics:
    plot_data[metric] = {}
    for best_metric in metrics:
        plot_data[metric][f'best_{best_metric}'] = []

# Collect data for plotting
for gen in generations:
    best_individuals = best_individuals_per_gen[gen]
    for metric in metrics:
        for best_metric in metrics:
            value = best_individuals[best_metric][metric]
            plot_data[metric][f'best_{best_metric}'].append(value)

# Plot metrics for best individuals
plt.figure(figsize=(15, 12))

colors = ['tab:red', 'tab:green', 'tab:blue']
metric_labels = ['Best UW Val Loss', 'Best CIOU Loss', 'Best Avg Precision']

for i, metric in enumerate(metrics):
    plt.subplot(3, 1, i+1)
    
    for j, best_metric in enumerate(metrics):
        label = f'{metric_labels[j]} Individual'
        linestyle = '--' if j != i else '-'  # Solid line for the metric's own best, dashed for others
        linewidth = 2 if j == i else 1
        plt.plot(generations, plot_data[metric][f'best_{best_metric}'], 
                label=label, marker='o', color=colors[j], 
                linestyle=linestyle, linewidth=linewidth)
    
    plt.title(f'{metric} for Best Individuals per Generation')
    plt.xlabel('Generation')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)

plt.tight_layout()

# Save plot
if cross_generation_pareto_front:
    plot_path = os.path.join(output_dir, 'best_individuals_metrics_cross_generation.jpg')
else:
    plot_path = os.path.join(output_dir, 'best_individuals_metrics.jpg')

plt.savefig(plot_path)
plt.close()
print(f'Plot saved to {plot_path}')
