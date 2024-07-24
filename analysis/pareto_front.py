import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from deap.benchmarks.tools import hypervolume
from pymoo.indicators.hv import HV, Hypervolume

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
            # if is_x_minimized:
            # else:
            #     y_steps.append(y_val)
            #     x_steps.append(last_x)
            # # add the point
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
            #if check_unique(point_a, point_b, objectives) and dominates(point_a, point_b, objectives, directions):
                front = front.drop(index=j)
                dominated.append(j)
    return front, dominated

# def find_pareto_top(df, objectives, directions):


# def find_pareto_bottom(df, objectives, directions):


def dominates(point_a, point_b, objectives, directions):
    count = 0
    for i, objective in enumerate(objectives):
        if (point_a[objective] < point_b[objective]) == directions[i]:
            count += 1

    if count == len(objectives):
        return True
    else:
        return False
    # if (point_a[objectives[0]] < point_b[objectives[0]]) and (point_a[objectives[1]] < point_b[objectives[1]]) and (point_a[objectives[2]] > point_b[objectives[2]]):
    #     return True
    # else:
    #     return False

# def recalc_doms(front, objectives, directions):
#     dominated = []
#     new_front = front.copy()
#     for i, point_a in new_front.iterrows():
#         for j, point_b in new_front.iterrows():
#             if (not point_a.equals(point_b)) and dominates(point_a, point_b, objectives, directions):
#             # if check_unique(point_a, point_b, objectives) and dominates(point_a, point_b, objectives, directions):
#                 front = front.drop(index=j)
#                 dominated.append(j)
#     return new_front, dominated

# def check_unique(point_a, point_b, objectives):
#     if (not point_a.equals(point_b)):
#         count = 0
#         for objective in objectives:
#             if point_a[objective] != point_b[objective]:
#                 count += 1
#         if count == len(objectives):
#             return True
#     return False

def gen_plot(df_baseline_current, gen, path, objectives, directions, baseline_front, baseline_front_top, baseline_front_bottom, surrogate_path, surrogate_front, surrogate_front_top, surrogate_front_bottom, bounds, df_simple, df_complex):
    metric_a = objectives[0]
    metric_b = objectives[1]
    metric_c = objectives[2]

    #plt.figure(figsize=(5, 8))
    #PLOT 1
    plt.subplot(2, 1, 1)
    plt.xlim(0, 60)
    #plt.xlim(1, 1.1 * bounds[1])
    #plt.xscale('log')
    #plt.xlim(0.8 * 1.989222, 1.2 * 8568.095703)
    plt.ylim(0.9 * bounds[2], 2.1) #1.1 * bounds[3])
    #plt.scatter(df_surrogate_current[metric_a], df_surrogate_current[metric_b], color='xkcd:orange')
    #plt.scatter(df_surrogate_current[metric_a], df_surrogate_current[metric_b], color='xkcd:orange')
    plt.title("Pareto Front Generation " + str(gen))
    plt.xlabel(metric_a)
    plt.ylabel(metric_b)
    
    plt.scatter(baseline_front[metric_a], baseline_front[metric_b], color='xkcd:lightblue', marker='o')
    
    #front1, dominated1 = find_pareto_indices(front, objectives[:2], directions[:2])
    baseline_x_steps, baseline_y_steps = stepify_pareto_points_2d(baseline_front_top[metric_a].to_numpy(), baseline_front_top[metric_b].to_numpy(), [directions[0], directions[1]])
    surrogate_x_steps, surrogate_y_steps = stepify_pareto_points_2d(surrogate_front_top[metric_a].to_numpy(), surrogate_front_top[metric_b].to_numpy(), [directions[0], directions[1]])
    
    plt.scatter(baseline_front_top[metric_a], baseline_front_top[metric_b], color='xkcd:blue', marker='o')
    plt.plot(baseline_x_steps, baseline_y_steps, color='xkcd:blue')
    
    #UNCOMMENT WHEN WE HAVE SURROGATE
    # plt.scatter(surrogate_front[metric_a], surrogate_front[metric_b], color='xkcd:orange', marker='^')
    # plt.scatter(surrogate_front_top[metric_a], surrogate_front_top[metric_b], color='xkcd:red', marker='^')
    # plt.plot(surrogate_x_steps, surrogate_y_steps, color='xkcd:red')
    
    # plt.scatter(df_metric_a[metric_a], df_metric_a[metric_b], color='xkcd:pink', marker='X')
    # plt.scatter(df_metric_b[metric_a], df_metric_b[metric_b], color='xkcd:yellow', marker='X')
    # plt.scatter(df_metric_c[metric_a], df_metric_c[metric_b], color='xkcd:grey', marker='X')
    plt.scatter(df_simple[metric_a], df_simple[metric_b], color='xkcd:purple', marker='X')
    plt.scatter(df_complex[metric_a], df_complex[metric_b], color='xkcd:green', marker='X')
    #UNCOMMENT WHEN WE HAVE SURROGATE
    #plt.legend(['B: Overall Pareto Optimal', 'B: Recalculated Pareto Optimal', 'B: Pareto Frontier', 'S: Overall Pareto Optimal', 'S: Recalculated Pareto Optimal', 'S: Pareto Frontier', 'Reduced Dmytro', 'Full Dmytro'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(['Overall Pareto Optimal', 'Recalculated Pareto Optimal', 'Pareto Frontier', 'Reduced Dmytro', 'Full Dmytro'], loc='center left', bbox_to_anchor=(1, 0.5))

    #PLOT 2
    plt.subplot(2, 1, 2)
    plt.xlim(bounds[4] - 0.01, 1.8 * bounds[5])
    #plt.xscale('symlog')
    plt.ylim(0.9 * bounds[2], 2.1) #1.1 * bounds[3])
    #plt.scatter(df_baseline_current[metric_c], df_baseline_current[metric_b], color='xkcd:orange')
    plt.title("Pareto Front Generation " + str(gen))
    plt.xlabel(metric_c)
    plt.ylabel(metric_b)
    
    plt.scatter(baseline_front[metric_c], baseline_front[metric_b], color='xkcd:lightblue', marker='o')
    
    #front2, dominated2 = find_pareto_indices(front, objectives[1:], directions[1:])
    baseline_x_steps, baseline_y_steps = stepify_pareto_points_2d(baseline_front_bottom[metric_c].to_numpy(), baseline_front_bottom[metric_b].to_numpy(), [directions[2], directions[1]])
    surrogate_x_steps, surrogate_y_steps = stepify_pareto_points_2d(surrogate_front_bottom[metric_c].to_numpy(), surrogate_front_bottom[metric_b].to_numpy(), [directions[2], directions[1]])
    
    plt.scatter(baseline_front_bottom[metric_c], baseline_front_bottom[metric_b], color='xkcd:blue', marker='o')
    plt.plot(baseline_x_steps, baseline_y_steps, color='xkcd:blue')

    # plt.scatter(surrogate_front[metric_c], surrogate_front[metric_b], color='xkcd:orange', marker='^')
    # plt.scatter(surrogate_front_bottom[metric_c], surrogate_front_bottom[metric_b], color='xkcd:red', marker='^')
    # plt.plot(surrogate_x_steps, surrogate_y_steps, color='xkcd:red')
    
    #plt.legend(['Normal Individual', 'Overall Pareto Optimal', 'Pareto Frontier', 'Recalculated Pareto Optimal'], loc='center left', bbox_to_anchor=(1, 0.5))
    
    # plt.scatter(df_metric_a[metric_c], df_metric_a[metric_b], color='xkcd:pink', marker='X')
    # plt.scatter(df_metric_b[metric_c], df_metric_b[metric_b], color='xkcd:yellow', marker='X')
    # plt.scatter(df_metric_c[metric_c], df_metric_c[metric_b], color='xkcd:grey', marker='X')
    plt.scatter(df_simple[metric_c], df_simple[metric_b], color='xkcd:purple', marker='X')
    plt.scatter(df_complex[metric_c], df_complex[metric_b], color='xkcd:green', marker='X')
    #plt.legend(['B: Overall Pareto Optimal', 'B: Recalculated Pareto Optimal', 'B: Pareto Frontier', 'S: Overall Pareto Optimal', 'S: Recalculated Pareto Optimal', 'S: Pareto Frontier', 'Reduced Dmytro', 'Full Dmytro'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(['Overall Pareto Optimal', 'Recalculated Pareto Optimal', 'Pareto Frontier', 'Reduced Dmytro', 'Full Dmytro'], loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    #plt.savefig('/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/graphs/pareto/pareto_gen' + str(gen) + '.jpg')
    plt.savefig('graphs/pareto/pareto_gen' + str(gen) + '.jpg')
    plt.close()
    print('plot ' + str(gen) + ' done')

# def calc_hypervolume(front, ref):
#     hv = hypervolume(front, ref)
#     return hv

if __name__ == "__main__":
    baseline_path = '/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution/out.csv'
    surrogate_path = '/gv1/projects/GRIP_Precog_Opt/baseline_evolution/out.csv'
    simple_path = '/home/eharpster3/precog-opt-grip/dmytro_metrics/combined_metric.csv'
    complex_path = '/home/eharpster3/precog-opt-grip/dmytro_metrics/complex/metrics.csv'

    objectives = ['uw_val_epoch_loss', 'ciou_loss', 'average_precision']
    #True if minimized, False if maximized
    directions = [True, True, False]
    df_baseline = pd.read_csv(baseline_path)
    df_baseline = df_baseline[['gen', 'hash', 'genome', 'uw_val_epoch_loss', 'ciou_loss', 'average_precision']]

    df_surrogate = pd.read_csv(surrogate_path)
    df_surrogate = df_surrogate[['gen', 'hash', 'genome', 'uw_val_epoch_loss', 'ciou_loss', 'average_precision']]

    df_simple = pd.read_csv(simple_path)
    df_complex = pd.read_csv(complex_path)

    df_metric_a = df_simple.loc[df_simple['uw_val_epoch_loss'].idxmin()]
    df_metric_b = df_simple.loc[df_simple['ciou_loss'].idxmin()]
    df_metric_c = df_simple.loc[df_simple['average_precision'].idxmax()]
    df_simple = df_simple.loc[df_simple['val_epoch_loss'].idxmin()]
    df_simple = pd.concat([df_simple, df_metric_a, df_metric_b, df_metric_c], axis=1).transpose()
    df_simple = df_simple.reset_index(drop=True)
    df_simple = df_simple[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
    df_complex = df_complex[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
    # df_metric_a = df_metric_a[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
    # df_metric_b = df_metric_b[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
    # df_metric_c = df_metric_c[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
    #df = df.drop(df['uw_val_epoch_loss'].max())
    #df = df[(df['uw_val_epoch_loss'] != 1000000) & (df['ciou_loss'] != 1000000) & (df['average_precision'] != -1000000)]
    df_baseline = df_baseline.dropna()
    df_surrogate = df_surrogate.dropna()

    df_baseline = df_baseline.drop_duplicates(subset=['uw_val_epoch_loss', 'ciou_loss', 'average_precision'])
    df_surrogate = df_surrogate.drop_duplicates(subset=['uw_val_epoch_loss', 'ciou_loss', 'average_precision'])
    #print(df[df['uw_val_epoch_loss'] == df['uw_val_epoch_loss'].max()])
    bounds = [df_baseline['uw_val_epoch_loss'].min(), df_baseline[df_baseline['uw_val_epoch_loss'] <= 200]['uw_val_epoch_loss'].max(), df_baseline['ciou_loss'].min(), df_baseline[df_baseline['ciou_loss'] < 1000000]['ciou_loss'].max(), df_baseline[df_baseline['average_precision'] > -1000000]['average_precision'].min(), df_baseline['average_precision'].max()]
    print(bounds)
    #need a list of all individuals, list of pareto front individuals, then 2 lists of pareto per graph
    baseline_hvs = []
    surrogate_hvs = []
    gens = []
    print('newgen')
    #print(df[df['ciou_loss'] == df['ciou_loss'].max()])
    min_gen = min(df_baseline['gen'].min(), df_surrogate['gen'].min())
    max_gen = max(df_baseline['gen'].max(), df_surrogate['gen'].max())
    for gen in range(min_gen, max_gen + 1): #also 6, 12, and beyond, basically skip 7-11
        #if gen >= 7 and gen <= 11:
        #    continue
        df_baseline_current = df_baseline[(df_baseline['gen'] <= gen)] #& (~df.gen.isin([7, 8, 9, 10, 11]))]
        df_surrogate_current = df_surrogate[(df_surrogate['gen'] <= gen)]
        #print(objectives, directions)
        baseline_front, baseline_dominated = find_pareto_indices(df_baseline_current, objectives, directions)
        baseline_front_top, baseline_dominated_top = find_pareto_indices(df_baseline_current, objectives[:2], directions[:2])
        baseline_front_bottom, baseline_dominated_bottom = find_pareto_indices(df_baseline_current, objectives[1:], directions[1:])

        surrogate_front, surrogate_dominated = find_pareto_indices(df_surrogate_current, objectives, directions)
        surrogate_front_top, surrogate_dominated_top = find_pareto_indices(df_surrogate_current, objectives[:2], directions[:2])
        surrogate_front_bottom, surrogate_dominated_bottom = find_pareto_indices(df_surrogate_current, objectives[1:], directions[1:])

        gen_plot(df_baseline_current, gen, baseline_path, objectives, directions, baseline_front, baseline_front_top, baseline_front_bottom, surrogate_path, surrogate_front, surrogate_front_top, surrogate_front_bottom, bounds, df_simple, df_complex)
        print('GEN:', gen)
        print('length of baseline fronts:', len(df_baseline_current), len(baseline_front), len(baseline_front_top), len(baseline_front_bottom))
        print('length of surrogate fronts:', len(df_surrogate_current), len(surrogate_front), len(surrogate_front_top), len(surrogate_front_bottom))
        #front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
        #rows_set = set(map(tuple, baseline_front.values))
        print(baseline_front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']].drop_duplicates().shape)
        print(surrogate_front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']].drop_duplicates().shape)
        baseline_hv_front = baseline_front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']].to_numpy()
        surrogate_hv_front = surrogate_front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']].to_numpy()
        
        baseline_hv_front[:, 2] = baseline_hv_front[:, 2] * -1
        surrogate_hv_front[:, 2] = surrogate_hv_front[:, 2] * -1
        #[1000000, 2, 1]
        #[0, 0, 0]
        hv_max = np.array([1000000, 2, 0]) #hv_front.max(axis=0)
        hv_min = np.array([0, 0, -1]) #hv_front.min(axis=0)
        baseline_hv_front = (baseline_hv_front - hv_min) / (hv_max - hv_min)
        surrogate_hv_front = (surrogate_hv_front - hv_min) / (hv_max - hv_min)
        #print(hv_front)
        #ref_point = np.array([np.max(hv_front[:, 0]) + 1, np.max(hv_front[:, 1]) + 1, np.max(hv_front[:, 2]) + 1])
        #ref_point = np.array([-10,-10,-10])
        ref_point = np.array([1, 1, 1])
        #print('refpoint', ref_point)
        baseline_hv = Hypervolume(ref_point=ref_point)
        baseline_hvs.append(baseline_hv(baseline_hv_front))
        surrogate_hv = Hypervolume(ref_point=ref_point)
        surrogate_hvs.append(surrogate_hv(surrogate_hv_front))
        gens.append(gen)
        print('baseline hypervolume:', baseline_hv(baseline_hv_front))
        print('surrogate hypervolume:', surrogate_hv(surrogate_hv_front))
        print()
    
    print('baseline front')
    print(baseline_front[['gen', 'hash']])
    print('surrogate front')
    print(surrogate_front[['gen', 'hash']])
    plt.plot(baseline_hvs, marker='o')
    #plt.plot(surrogate_hvs, marker='^')
    #plt.legend(['Baseline', 'Surrogate'])
    #plt.plot(hvs, marker='o')
    plt.xticks(range(len(gens)), gens)
    plt.title("Pareto Front Hypervolumes Per Generation")
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.tight_layout()
    #plt.savefig('/home/eharpster3/precog-opt-grip/analysis/graphs/pareto/pareto_hypervolume.jpg')
    plt.savefig('graphs/pareto/pareto_hypervolume.jpg')
    plt.close()
        
    

