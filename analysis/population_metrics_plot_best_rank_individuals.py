"""
Plots the metrics for the top 3 individuals based on combined ranking in ciou_loss and average_precision.
Finds the top 3 individuals with minimal (rank_ciou + rank_avg_precision) and plots all metrics for them.
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

# For each generation, find individuals based on combined ranking in ciou_loss and average_precision
selected_individuals_per_gen = {}

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

    # Rank individuals within the population based on ciou_loss and average_precision
    # For ciou_loss: lower is better (rank 1 = best)
    # For average_precision: higher is better (rank 1 = best)
    df_to_use = df_to_use.copy()
    df_to_use['rank_ciou'] = df_to_use['ciou_loss'].rank(method='min', ascending=True)
    df_to_use['rank_avg_precision'] = df_to_use['average_precision'].rank(method='min', ascending=False)
    df_to_use['combined_rank'] = df_to_use['rank_ciou'] + df_to_use['rank_avg_precision']
    
    # Find the top 3 individuals with best combined ranks
    df_sorted = df_to_use.sort_values('combined_rank')
    
    # Get top 3 (or fewer if less than 3 individuals available)
    top_1 = df_sorted.iloc[0] if len(df_sorted) >= 1 else None
    top_2 = df_sorted.iloc[1] if len(df_sorted) >= 2 else top_1
    top_3 = df_sorted.iloc[2] if len(df_sorted) >= 3 else top_2
    
    selected_individuals_per_gen[gen] = {
        'best_combined': top_1,
        'second_best': top_2,
        'third_best': top_3
    }

# Prepare data for plotting - track each metric for the 3 selected individuals
plot_data = {}
for metric in metrics:
    plot_data[metric] = {
        'best_combined': [],
        'second_best': [],
        'third_best': []
    }

# Collect data for plotting
for gen in generations:
    selected_individuals = selected_individuals_per_gen[gen]
    for metric in metrics:
        plot_data[metric]['best_combined'].append(selected_individuals['best_combined'][metric])
        plot_data[metric]['second_best'].append(selected_individuals['second_best'][metric])
        plot_data[metric]['third_best'].append(selected_individuals['third_best'][metric])

# Plot metrics for selected individuals
plt.figure(figsize=(15, 12))

colors = ['tab:purple', 'tab:red', 'tab:blue']
individual_labels = ['1st Best Combined Rank', '2nd Best Combined Rank', '3rd Best Combined Rank']

for i, metric in enumerate(metrics):
    plt.subplot(3, 1, i+1)
    
    individual_types = ['best_combined', 'second_best', 'third_best']
    for j, individual_type in enumerate(individual_types):
        label = individual_labels[j]
        linestyle = '-' if j == 0 else '--'  # Solid line for best combined, dashed for others
        linewidth = 2 if j == 0 else 1
        plt.plot(generations, plot_data[metric][individual_type], 
                label=label, marker='o', color=colors[j], 
                linestyle=linestyle, linewidth=linewidth)
    
    plt.title(f'{metric} for Selected Individuals per Generation')
    plt.xlabel('Generation')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)

plt.tight_layout()

# Save plot
if cross_generation_pareto_front:
    plot_path = os.path.join(output_dir, 'combined_rank_individuals_metrics_cross_generation.jpg')
else:
    plot_path = os.path.join(output_dir, 'combined_rank_individuals_metrics.jpg')

plt.savefig(plot_path)
plt.close()
print(f'Plot saved to {plot_path}')
