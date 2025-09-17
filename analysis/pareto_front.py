"""
Generates hypervolume and pareto front plots given out.csv's from evolutions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV, Hypervolume
from matplotlib.ticker import MaxNLocator
import toml
import argparse
from pareto_utils import find_pareto_indices, gen_plot

parser = argparse.ArgumentParser()
parser.add_argument('username', type=str)
parser.add_argument('--two_objectives', action='store_true', help='If set, only two objectives are used for pareto optimality')
parser.add_argument('--cross_gen', action='store_true', help='If set, hall of fame is computed from all populations up to and including the current generation')

args = parser.parse_args()
USER = args.username
CROSS_GENERATION_PARETO_FRONT = args.cross_gen
TWO_OBJECTIVES = args.two_objectives


def generate_fronts(df, objectives, directions, name, gen, colors, marker, reached_max, cached_fronts=None):
    if CROSS_GENERATION_PARETO_FRONT and cached_fronts is not None and gen > 1:
        # Use cached front from previous generation and combine with current generation
        prev_front = cached_fronts.get('front', pd.DataFrame())
        prev_front_top = cached_fronts.get('front_top', pd.DataFrame())
        prev_front_bottom = cached_fronts.get('front_bottom', pd.DataFrame())
        
        # Get current generation data
        df_current_gen = df[(df['gen'] == gen)]
        
        # Combine previous front with current generation for recalculation
        if not prev_front.empty and not df_current_gen.empty:
            df_combined_front = pd.concat([prev_front, df_current_gen], ignore_index=True)
            df_combined_front_top = pd.concat([prev_front_top, df_current_gen], ignore_index=True)
            df_combined_front_bottom = pd.concat([prev_front_bottom, df_current_gen], ignore_index=True)
        elif not df_current_gen.empty:
            df_combined_front = df_current_gen.copy()
            df_combined_front_top = df_current_gen.copy()
            df_combined_front_bottom = df_current_gen.copy()
        else:
            df_combined_front = prev_front.copy()
            df_combined_front_top = prev_front_top.copy()
            df_combined_front_bottom = prev_front_bottom.copy()
        
        # Calculate new fronts from combined data (much smaller than all generations)
        front, dominated = find_pareto_indices(df_combined_front, objectives, directions)
        front_top, dominated_top = find_pareto_indices(df_combined_front_top, objectives[:2], directions[:2])
        front_bottom, dominated_bottom = find_pareto_indices(df_combined_front_bottom, objectives[1:], directions[1:])
        
        # For plotting, we need all data up to current generation
        df_current = df[(df['gen'] <= gen)]
    elif CROSS_GENERATION_PARETO_FRONT:
        # First generation or no cache - plot the pareto front considering all generations up through the current generation
        df_current = df[(df['gen'] <= gen)]
        front, dominated = find_pareto_indices(df_current, objectives, directions)
        front_top, dominated_top = find_pareto_indices(df_current, objectives[:2], directions[:2])
        front_bottom, dominated_bottom = find_pareto_indices(df_current, objectives[1:], directions[1:])
    else:
        # plot the pareto front only considering the current generation population
        df_current = df[(df['gen'] == gen)]
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
    configs = toml.load('/home/hice1/psomu3/scratch/surrogate-evolution/conf_gens_vae.toml')
    pipeline_config = configs["pipeline"]
    cfg_objectives = pipeline_config['objectives']
    cfg_best_epoch = pipeline_config['best_epoch_criteria']
    objectives = list(cfg_objectives.keys())
    directions = [val == -1 for val in list(cfg_objectives.values())]
    best_epoch = cfg_best_epoch[0]
    best_epoch_direction = cfg_best_epoch[1] == 'min'
    
    # HERE IS WHERE YOU ADD FRONTS
    # need to create a pandas dataframe then add an entry to the dataframes list with all the needed info
    baseline_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_baseline_30/out.csv'
    df_baseline = pd.read_csv(baseline_path)
    surrogate_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/out.csv'
    df_surrogate = pd.read_csv(surrogate_path)
    ssi_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_baseline_two_30/out.csv'
    df_ssi = pd.read_csv(ssi_path)
    fourth_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_two_30/out.csv'
    df_four = pd.read_csv(fourth_path)
    # every dataframe needs an actual pandas dataframe, a name to display on legends, 4 colors (overall pareto optimal, pareto optimal for 2 objectives, and their past max gen alternatives), and the marker to use on graphs
    dataframes = [
        {'df': df_baseline, 'name': 'Base', 'colors': ['xkcd:cerulean', 'xkcd:azure', 'xkcd:slate grey', 'xkcd:sky blue'], 'marker': 'o'}, 
        {'df': df_surrogate, 'name': 'SSI_vae', 'colors': ['xkcd:gold', 'xkcd:amber', 'xkcd:dark grey', 'xkcd:charcoal'], 'marker': '^'},
        {'df': df_ssi, 'name': 'Base_two', 'colors': ['xkcd:lime green', 'xkcd:forest green', 'xkcd:grey', 'xkcd:slate'], 'marker': 's'},
        {'df': df_four, 'name': 'SSI_vae_two', 'colors': ['xkcd:red', 'xkcd:red', 'xkcd:red', 'xkcd:red'], 'marker': 'o'},
    ]

    min_gens = []
    max_gens = []
    # here are the limits for setting the effective max in the dataset for graphing purposes
    bounds_limits = [-np.inf, 60, -np.inf, 2.1, -1000000, np.inf]
    bounds_margin = 0.1
    if TWO_OBJECTIVES:
        bounds = {'min_objective_1': [], 'max_objective_1': [], 'min_objective_2': [], 'max_objective_2': []}
    else:
        bounds = {'min_objective_1': [], 'max_objective_1': [], 'min_objective_2': [], 'max_objective_2': [], 'min_objective_3': [], 'max_objective_3': []}

    for dataframe in dataframes:
        df = dataframe['df']
        
        if TWO_OBJECTIVES and len(objectives) > 2:
            objectives = objectives[1:]
            directions = directions[1:]
            bounds_limits = bounds_limits[2:]
        if TWO_OBJECTIVES:
            df = df[['gen', 'hash', 'genome', objectives[0], objectives[1]]]
        else:
            df = df[['gen', 'hash', 'genome', objectives[0], objectives[1], objectives[2]]]
        df = df.dropna()
        if TWO_OBJECTIVES:
            df = df.drop_duplicates(subset=[objectives[0], objectives[1]])
        else:
            df = df.drop_duplicates(subset=[objectives[0], objectives[1], objectives[2]])
        min_gens.append(df['gen'].min())
        max_gens.append(df['gen'].max())
        
        bounds['min_objective_1'].append(df[df[objectives[0]] > bounds_limits[0]][objectives[0]].min())
        bounds['max_objective_1'].append(df[df[objectives[0]] < bounds_limits[1]][objectives[0]].max())
        bounds['min_objective_2'].append(df[df[objectives[1]] > bounds_limits[2]][objectives[1]].min())
        bounds['max_objective_2'].append(df[df[objectives[1]] < bounds_limits[3]][objectives[1]].max())
        if not TWO_OBJECTIVES:
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
        if not TWO_OBJECTIVES:
            bounds['min_objective_3'].append(df[df[objectives[2]] > bounds_limits[4]][objectives[2]].min())
            bounds['max_objective_3'].append(df[df[objectives[2]] < bounds_limits[5]][objectives[2]].max())

    # find the max and mins for each objective for all data that will be plotted
    if TWO_OBJECTIVES:
        bounds = [min(bounds['min_objective_1']), max(bounds['max_objective_1']), min(bounds['min_objective_2']), max(bounds['max_objective_2'])]
    else:
        bounds = [min(bounds['min_objective_1']), max(bounds['max_objective_1']), min(bounds['min_objective_2']), max(bounds['max_objective_2']), min(bounds['min_objective_3']), max(bounds['max_objective_3'])]
    print(bounds)
 
    gens = []
    min_gen = min(min_gens)
    max_gen = max(max_gens)
    all_hvs = {}
    
    # Cache for storing fronts from previous generation (for CROSS_GENERATION_PARETO_FRONT optimization)
    cached_fronts_by_dataframe = {}

    for gen in range(min_gen, max_gen + 1): 
        all_fronts = []
        
        for i, dataframe in enumerate(dataframes):
            if gen <= dataframe['df']['gen'].max():
                reached_max = False
            else:
                reached_max = True
            
            # Get cached fronts from previous generation for this dataframe
            dataframe_name = dataframe['name']
            cached_fronts = cached_fronts_by_dataframe.get(dataframe_name, None) if CROSS_GENERATION_PARETO_FRONT else None
            
            front_result = generate_fronts(dataframe['df'], objectives, directions, dataframe['name'], gen, dataframe['colors'], dataframe['marker'], reached_max, cached_fronts)
            all_fronts.append(front_result)
            
            # Cache the fronts for next generation (only if CROSS_GENERATION_PARETO_FRONT is enabled and we haven't reached max)
            if CROSS_GENERATION_PARETO_FRONT and not reached_max:
                cached_fronts_by_dataframe[dataframe_name] = {
                    'front': front_result['front'].copy(),
                    'front_top': front_result['front_top'].copy(),
                    'front_bottom': front_result['front_bottom'].copy()
                }

        gen_plot(all_fronts, benchmarks, gen, objectives, directions, bounds, bounds_margin, best_epoch, best_epoch_direction, USER, TWO_OBJECTIVES)
        
        print('GEN:', gen, flush=True)
        for one_front in all_fronts:
            if not one_front['reached_max']:
                df_current = one_front['df']
                front = one_front['front']
                front_top = one_front['front_top']
                front_bottom = one_front['front_bottom']
                name = one_front['name']
                print('length of ' + name.lower() + ' fronts:', len(df_current), len(front), len(front_top), len(front_bottom))   

                if TWO_OBJECTIVES:
                    hv_front = front[[objectives[0], objectives[1]]].to_numpy() 
                    hv_front[:, 1] = hv_front[:, 1] * -1
                    hv_max = np.array([2, 0]) #hv_front.max(axis=0)
                    hv_min = np.array([0, -1]) #hv_front.min(axis=0)
                    hv_front = (hv_front - hv_min) / (hv_max - hv_min)
                    ref_point = np.array([1, 1])
                else:
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
    plt.savefig(f'/home/hice1/{USER}/scratch/surrogate-evolution/analysis/graphs/paretoTestingBaseline/pareto_hypervolume.jpg')
    plt.close()