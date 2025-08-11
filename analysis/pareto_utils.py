import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV, Hypervolume
from matplotlib.ticker import MaxNLocator
import toml
import argparse
import time

def stepify_pareto_points_2d(x, y, metric_directions):
    """
    Returns the pareto front points, including the steps, from the x/y points of a 2D pareto front
    Takes in the optimization directions for the x and y parameters as a list of booleans (True=Minimized, False=Maximized)
    """

    # Handle empty arrays
    if len(x) == 0 or len(y) == 0:
        return [], []

    is_x_minimized, is_y_minimized = metric_directions
    # sort for pareto steps
    y_argsort = np.argsort(y)  
    # the last Y value should be the worst
    if is_y_minimized == False:
        y_argsort = y_argsort[::-1]
    x = x[y_argsort]
    y = y[y_argsort]
    x_argsort = np.argsort(x)
    # the last X value should be the worst
    if is_x_minimized == False:
        x_argsort = x_argsort[::-1]
    x = x[x_argsort]
    y = y[x_argsort]
    
    last_x, last_y = x[0], y[0]
    x_steps = [last_x]
    y_steps = [last_y]
    # step direction is based on the optimization direction of each axis
    for i, x_val in enumerate(x):
        y_val = y[i]
        if last_x != x_val:
            # add the stair step
            y_steps.append(last_y)
            x_steps.append(x_val)
            # add the point
        y_steps.append(y_val)
        x_steps.append(x_val)
        last_x, last_y = x_val, y_val
    return x_steps, y_steps


def find_pareto_indices(df, objectives, directions):
    """
    Efficient Pareto front calculation using vectorized operations
    """
    if df.empty:
        return df.copy(), []
    
    # Convert to numpy array for faster operations
    values = df[objectives].values
    n_points = len(values)
    
    # Initialize dominated array
    is_dominated = np.zeros(n_points, dtype=bool)
    
    # For each point, check if it's dominated by any other point
    for i in range(n_points):
        if is_dominated[i]:
            continue
            
        # Compare point i with all other points
        for j in range(n_points):
            if i == j or is_dominated[j]:
                continue
                
            # Check if point j dominates point i
            dominates_count = 0
            for k, direction in enumerate(directions):
                if direction:  # minimize
                    if values[j, k] < values[i, k]:
                        dominates_count += 1
                    elif values[j, k] > values[i, k]:
                        break
                else:  # maximize
                    if values[j, k] > values[i, k]:
                        dominates_count += 1
                    elif values[j, k] < values[i, k]:
                        break
            
            # If point j dominates point i in all objectives
            if dominates_count == len(objectives):
                is_dominated[i] = True
                break
    
    # Get non-dominated points
    pareto_mask = ~is_dominated
    front = df.iloc[pareto_mask].copy()
    dominated_indices = df.index[is_dominated].tolist()
    
    return front, dominated_indices


def gen_plot(all_fronts, benchmarks, gen, objectives, directions, bounds, bounds_margin, best_epoch, best_epoch_direction, USER):
    metric_a = objectives[0]
    metric_b = objectives[1]
    metric_c = objectives[2]

    #PLOT 1
    plt.subplot(2, 1, 1)
    xrange = bounds[1] - bounds[0]
    yrange = bounds[3] - bounds[2]
    plt.xlim(bounds[0] -  (bounds_margin * xrange), bounds[1] + (bounds_margin * xrange))
    plt.ylim(bounds[2] - (bounds_margin * yrange), bounds[3] + (bounds_margin * yrange))
    plt.title("Pareto Front Generation " + str(gen))
    plt.xlabel(metric_a)
    plt.ylabel(metric_b)
    
    for one_front in all_fronts:
        front = one_front['front']
        front_top = one_front['front_top']
        colors = one_front['colors']
        marker = one_front['marker']
        name = one_front['name']
        x_steps, y_steps = stepify_pareto_points_2d(front_top[metric_a].to_numpy(), front_top[metric_b].to_numpy(), [directions[0], directions[1]])
        if one_front['reached_max']:
            color1 = one_front['colors'][2]
            color2 = one_front['colors'][3]
        else:
            color1 = one_front['colors'][0]
            color2 = one_front['colors'][1]
        plt.scatter(front[metric_a], front[metric_b], color=color1, marker=marker, label=name[0] + ': Overall Pareto Optimal')
        plt.scatter(front_top[metric_a], front_top[metric_b], color=color2, marker=marker, label=name[0] + ': Recalculated Pareto Optimal')
        plt.plot(x_steps, y_steps, color=color2, label='_' + name[0] + ': Pareto Frontier')
    
    for benchmark in benchmarks:
        benchmark_df = benchmark['df']
        plt.scatter(benchmark_df[metric_a], benchmark_df[metric_b], color=benchmark['color'], marker=benchmark['marker'], label=benchmark['name'])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #PLOT 2
    plt.subplot(2, 1, 2)
    xrange = bounds[5] - bounds[4]
    yrange = bounds[3] - bounds[2]
    plt.xlim(bounds[4] -  (bounds_margin * xrange), bounds[5] + (bounds_margin * xrange))
    plt.ylim(bounds[2] - (bounds_margin * yrange), bounds[3] + (bounds_margin * yrange))
    plt.title("Pareto Front Generation " + str(gen))
    plt.xlabel(metric_c)
    plt.ylabel(metric_b)
    
    for one_front in all_fronts:
        front = one_front['front']
        front_bottom = one_front['front_bottom']
        colors = one_front['colors']
        marker = one_front['marker']
        name = one_front['name']
        x_steps, y_steps = stepify_pareto_points_2d(front_bottom[metric_c].to_numpy(), front_bottom[metric_b].to_numpy(), [directions[2], directions[1]])
        if one_front['reached_max']:
            color1 = one_front['colors'][2]
            color2 = one_front['colors'][3]
        else:
            color1 = one_front['colors'][0]
            color2 = one_front['colors'][1]
        plt.scatter(front[metric_c], front[metric_b], color=color1, marker=marker, label=name[0] + ': Overall Pareto Optimal')
        plt.scatter(front_bottom[metric_c], front_bottom[metric_b], color=color2, marker=marker, label=name[0] + ': Recalculated Pareto Optimal')
        plt.plot(x_steps, y_steps, color=color2, label='_' + name[0] + ': Pareto Frontier')


    for benchmark in benchmarks:
        benchmark_df = benchmark['df']
        plt.scatter(benchmark_df[metric_c], benchmark_df[metric_b], color=benchmark['color'], marker=benchmark['marker'], label=benchmark['name'])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'/home/hice1/{USER}/scratch/surrogate-evolution/analysis/graphs/paretoTestingBaseline/pareto_gen' + str(gen) + '.jpg')
    plt.close()
    print('plot ' + str(gen) + ' done', flush=True)
