"""
Generates hypervolume and pareto front plots given out.csv's from evolutions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV, Hypervolume
from matplotlib.ticker import MaxNLocator
import toml

def stepify_pareto_points_2d(x, y, metric_directions):
    """
    Returns the pareto front points, including the steps, from the x/y points of a 2D pareto front
    Takes in the optimization directions for the x and y parameters as a list of booleans (True=Minimized, False=Maximized)
    """

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
    dominated = []
    front = df.copy()
    for i, point_a in front.iterrows():
        for j, point_b in front.iterrows():
            if (not point_a.equals(point_b)) and dominates(point_a, point_b, objectives, directions):
                front = front.drop(index=j)
                dominated.append(j)
    return front, dominated

def dominates(point_a, point_b, objectives, directions):
    count = 0
    for i, objective in enumerate(objectives):
        if (point_a[objective] < point_b[objective]) == directions[i]:
            count += 1

    if count == len(objectives):
        return True
    else:
        return False

def gen_plot(all_fronts, benchmarks, gen, objectives, directions, bounds, bounds_margin, best_epoch, best_epoch_direction):
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
    plt.savefig('/home/hice1/hweston3/scratch/surrogate-evolution/analysis/graphs/pareto/pareto_gen' + str(gen) + '.jpg')
    plt.close()
    print('plot ' + str(gen) + ' done')

def generate_fronts(df, objectives, directions, name, gen, colors, marker, reached_max):
    df_current = df[(df['gen'] <= gen)]
    
    front, dominated = find_pareto_indices(df_current, objectives, directions)
    front_top, dominated_top = find_pareto_indices(df_current, objectives[:2], directions[:2])
    front_bottom, dominated_bottom = find_pareto_indices(df_current, objectives[1:], directions[1:])
    
    all_fronts = {}
    all_fronts['name'] = name
    all_fronts['colors'] = colors
    all_fronts['marker'] = marker
    all_fronts['front'] = front
    all_fronts['df'] = df_current
    all_fronts['dominated'] = dominated
    all_fronts['front_top'] = front_top
    all_fronts['dominated_top'] = dominated_top
    all_fronts['front_bottom'] = front_bottom
    all_fronts['dominated_bottom'] = dominated_bottom
    all_fronts['reached_max'] = reached_max
    
    return all_fronts


if __name__ == "__main__":
    # grab the objectives and best epoch criteria from the config and transform them to how I was previously representing that data (True = want to minimize, False = maximize)
    # configs = toml.load('conf.toml')
    configs = toml.load('/storage/ice-shared/vip-vvk/data/AOT/baseline_evo_working/conf.toml')
    pipeline_config = configs["pipeline"]
    cfg_objectives = pipeline_config['objectives']
    cfg_best_epoch = pipeline_config['best_epoch_criteria']
    objectives = list(cfg_objectives.keys())
    directions = [val == -1 for val in list(cfg_objectives.values())]
    best_epoch = cfg_best_epoch[0]
    best_epoch_direction = cfg_best_epoch[1] == 'min'
    
    # HERE IS WHERE YOU ADD FRONTS
    # need to create a pandas dataframe then add an entry to the dataframes list with all the needed info
    baseline_path = '/storage/ice-shared/vip-vvk/data/AOT/baseline_evo_working/out.csv'
    df_baseline = pd.read_csv(baseline_path)
    surrogate_path = '/storage/ice-shared/vip-vvk/data/AOT/tfs_se_v2/out.csv'
    df_surrogate = pd.read_csv(surrogate_path)
    # every dataframe needs an actual pandas dataframe, a name to display on legends, 4 colors (overall pareto optimal, pareto optimal for 2 objectives, and their past max gen alternatives), and the marker to use on graphs
    dataframes = [
        {'df': df_baseline, 'name': 'Baseline', 'colors': ['xkcd:lightblue', 'xkcd:blue', 'xkcd:grey', 'xkcd:charcoal'], 'marker': 'o'}, 
        {'df': df_surrogate, 'name': 'Surrogate', 'colors': ['xkcd:orange', 'xkcd:dark orange', 'xkcd:grey', 'xkcd:charcoal'], 'marker': '^'}
            ]

    min_gens = []
    max_gens = []
    # here are the limits for setting the effective max in the dataset for graphing purposes
    bounds_limits = [-np.inf, 60, -np.inf, 2.1, -1000000, np.inf]
    bounds_margin = 0.1
    bounds = {'min_objective_1': [], 'max_objective_1': [], 'min_objective_2': [], 'max_objective_2': [], 'min_objective_3': [], 'max_objective_3': []}
    #bounds = {'min_uw_val_epoch_loss': [], 'max_uw_val_epoch_loss': [], 'min_ciou_loss': [], 'max_ciou_loss': [], 'min_average_precision': [], 'max_average_precision': []}
    for dataframe in dataframes:
        df = dataframe['df']
        df = df[['gen', 'hash', 'genome', objectives[0], objectives[1], objectives[2]]]
        df = df.dropna()
        df = df.drop_duplicates(subset=[objectives[0], objectives[1], objectives[2]])
        min_gens.append(df['gen'].min())
        max_gens.append(df['gen'].max())
        
        bounds['min_objective_1'].append(df[df[objectives[0]] > bounds_limits[0]][objectives[0]].min())
        bounds['max_objective_1'].append(df[df[objectives[0]] < bounds_limits[1]][objectives[0]].max())
        bounds['min_objective_2'].append(df[df[objectives[1]] > bounds_limits[2]][objectives[1]].min())
        bounds['max_objective_2'].append(df[df[objectives[1]] < bounds_limits[3]][objectives[1]].max())
        bounds['min_objective_3'].append(df[df[objectives[2]] > bounds_limits[4]][objectives[2]].min())
        bounds['max_objective_3'].append(df[df[objectives[2]] < bounds_limits[5]][objectives[2]].max())
        
        dataframe['df'] = df



    
    
    # HERE IS WHERE YOU ADD BENCHMARKS THAT HAVE NO FRONTS
    # same process as adding a front except it only takes in one color instead of a list of colors
    # reduced_path = 'dmytro_metrics/combined_metric.csv'
    # full_path = 'dmytro_metrics/metrics.csv'
    # df_simple = pd.read_csv(reduced_path)
    # df_complex = pd.read_csv(full_path)
    
    benchmarks = []
    # benchmarks = [
    #     {'df': df_simple, 'name': 'Reduced Dmytro',  'color': 'xkcd:purple', 'marker': 'X'}, 
    #     {'df': df_complex,'name': 'Full Dmytro', 'color': 'xkcd:green', 'marker': 'X'}
    # ]

    for benchmark in benchmarks:
        df = benchmark['df']
        #Get rid of points that will never be plotted
        benchmark_df = df
        dfs = []
        # grab the row with the best value for that objective
        for i, objective in enumerate(objectives):
            if directions[i]:
                df_objective = benchmark_df.loc[benchmark_df[objective].idxmin()]
            else:
                df_objective = benchmark_df.loc[benchmark_df[objective].idxmax()]
            dfs.append(df_objective)
        # if we did not already, grab the row that would be selected by best_epoch criteria
        if best_epoch not in objectives:
            if best_epoch_direction:
                df_objective = benchmark_df.loc[benchmark_df[best_epoch].idxmin()]
            else:
                df_objective = benchmark_df.loc[benchmark_df[best_epoch].idxmax()]
            dfs.append(df_objective)
        # recombine all the rows we grabbed into a pandas dataframe
        benchmark_df = pd.concat(dfs, axis=1).transpose()
        df = benchmark_df.reset_index(drop=True)
        benchmark['df'] = df
        # grab the max and mins for each objective in the df
        bounds['min_objective_1'].append(df[df[objectives[0]] > bounds_limits[0]][objectives[0]].min())
        bounds['max_objective_1'].append(df[df[objectives[0]] < bounds_limits[1]][objectives[0]].max())
        bounds['min_objective_2'].append(df[df[objectives[1]] > bounds_limits[2]][objectives[1]].min())
        bounds['max_objective_2'].append(df[df[objectives[1]] < bounds_limits[3]][objectives[1]].max())
        bounds['min_objective_3'].append(df[df[objectives[2]] > bounds_limits[4]][objectives[2]].min())
        bounds['max_objective_3'].append(df[df[objectives[2]] < bounds_limits[5]][objectives[2]].max())

    # find the max and mins for each objective for all data that will be plotted
    bounds = [min(bounds['min_objective_1']), max(bounds['max_objective_1']), min(bounds['min_objective_2']), max(bounds['max_objective_2']), min(bounds['min_objective_3']), max(bounds['max_objective_3'])]
    print(bounds)
 
    gens = []
    min_gen = min(min_gens)
    max_gen = max(max_gens)
    all_hvs = {}

    for gen in range(min_gen, max_gen + 1): 
        all_fronts = []
        
        for dataframe in dataframes:
            if gen <= dataframe['df']['gen'].max():
                reached_max = False
            else:
                reached_max = True    
            all_fronts.append(generate_fronts(dataframe['df'], objectives, directions, dataframe['name'], gen, dataframe['colors'], dataframe['marker'], reached_max))

        gen_plot(all_fronts, benchmarks, gen, objectives, directions, bounds, bounds_margin, best_epoch, best_epoch_direction)
        
        print('GEN:', gen)
        for one_front in all_fronts:
            if not one_front['reached_max']:
                df_current = one_front['df']
                front = one_front['front']
                front_top = one_front['front_top']
                front_bottom = one_front['front_bottom']
                name = one_front['name']
                print('length of ' + name.lower() + ' fronts:', len(df_current), len(front), len(front_top), len(front_bottom))   

                hv_front = front[[objectives[0], objectives[1], objectives[2]]].to_numpy() 
                hv_front[:, 2] = hv_front[:, 2] * -1
                hv_max = np.array([1000000, 2, 0]) #hv_front.max(axis=0)
                hv_min = np.array([0, 0, -1]) #hv_front.min(axis=0)
                hv_front = (hv_front - hv_min) / (hv_max - hv_min)
                ref_point = np.array([1, 1, 1])
                hv = Hypervolume(ref_point=ref_point)
                if name not in list(all_hvs.keys()):
                    all_hvs[name] = []
                all_hvs[name].append(hv(hv_front))
                print(name + ' hypervolume:', hv(hv_front))

        print()

        gens.append(gen)
    # print('GENS', len(gens))
    # print(gens)
    # print()
    # print('HYPERVOLUMES', len(all_hvs))
    # print(all_hvs)
    for dataframe in dataframes:
        name = dataframe['name']
        plt.plot(range(1, len(all_hvs[name]) + 1), all_hvs[name], marker=dataframe['marker'], color=dataframe['colors'][1], label=name)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.title("Pareto Front Hypervolumes Per Generation")
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.tight_layout()
    plt.savefig('/home/hice1/hweston3/scratch/surrogate-evolution/analysis/graphs/pareto/pareto_hypervolume.jpg')
    plt.close()