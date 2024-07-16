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

def gen_plot(df, gen, path, objectives, directions, front, front_top, front_bottom, bounds):
    metric_a = objectives[0]
    metric_b = objectives[1]
    metric_c = objectives[2]

    #plt.figure(figsize=(5, 8))
    #PLOT 1
    plt.subplot(2, 1, 1)
    plt.xlim(1, 1.1 * bounds[1])
    plt.xscale('log')
    #plt.xlim(0.8 * 1.989222, 1.2 * 8568.095703)
    plt.ylim(0.9 * bounds[2], 2.1) #1.1 * bounds[3])
    plt.scatter(df[metric_a], df[metric_b], color='xkcd:orange')
    plt.title("Pareto Front Generation " + str(gen))
    plt.xlabel(metric_a)
    plt.ylabel(metric_b)
    
    plt.scatter(front[metric_a], front[metric_b], color='xkcd:lightblue', marker='o')
    #front1, dominated1 = find_pareto_indices(front, objectives[:2], directions[:2])
    x_steps, y_steps = stepify_pareto_points_2d(front_top[metric_a].to_numpy(), front_top[metric_b].to_numpy(), [directions[0], directions[1]])
    plt.plot(x_steps, y_steps, color='xkcd:blue')
    plt.scatter(front_top[metric_a], front_top[metric_b], color='xkcd:blue', marker='o')
    plt.legend(['Normal Individual', 'Overall Pareto Optimal', 'Pareto Frontier', 'Recalculated Pareto Optimal'], loc='center left', bbox_to_anchor=(1, 0.5))

    #PLOT 2
    plt.subplot(2, 1, 2)
    plt.xlim(bounds[4] - 0.01, 1.1 * bounds[5])
    #plt.xscale('symlog')
    plt.ylim(0.9 * bounds[2], 2.1) #1.1 * bounds[3])
    plt.scatter(df[metric_c], df[metric_b], color='xkcd:orange')
    plt.title("Pareto Front Generation " + str(gen))
    plt.xlabel(metric_c)
    plt.ylabel(metric_b)
    
    plt.scatter(front[metric_c], front[metric_b], color='xkcd:lightblue', marker='o')
    #front2, dominated2 = find_pareto_indices(front, objectives[1:], directions[1:])
    x_steps, y_steps = stepify_pareto_points_2d(front_bottom[metric_c].to_numpy(), front_bottom[metric_b].to_numpy(), [directions[2], directions[1]])
    plt.plot(x_steps, y_steps, color='xkcd:blue')
    plt.scatter(front_bottom[metric_c], front_bottom[metric_b], color='xkcd:blue', marker='o')
    plt.legend(['Normal Individual', 'Overall Pareto Optimal', 'Pareto Frontier', 'Recalculated Pareto Optimal'], loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig('/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/graphs/pareto/pareto_gen' + str(gen) + '.jpg')
    plt.close()
    print('plot ' + str(gen) + ' done')

# def calc_hypervolume(front, ref):
#     hv = hypervolume(front, ref)
#     return hv

if __name__ == "__main__":
    path = '/gv1/projects/GRIP_Precog_Opt/baseline_evolution/out.csv'
    objectives = ['uw_val_epoch_loss', 'ciou_loss', 'average_precision']
    #True if minimized, False if maximized
    directions = [True, True, False]
    df = pd.read_csv(path)
    df = df[['gen', 'hash', 'genome', 'uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
    #df = df.drop(df['uw_val_epoch_loss'].max())
    #df = df[(df['uw_val_epoch_loss'] != 1000000) & (df['ciou_loss'] != 1000000) & (df['average_precision'] != -1000000)]
    df = df.dropna()
    #print(df[df['uw_val_epoch_loss'] == df['uw_val_epoch_loss'].max()])
    bounds = [df['uw_val_epoch_loss'].min(), df[df['uw_val_epoch_loss'] <= 200]['uw_val_epoch_loss'].max(), df['ciou_loss'].min(), df[df['ciou_loss'] < 1000000]['ciou_loss'].max(), df[df['average_precision'] > -1000000]['average_precision'].min(), df['average_precision'].max()]
    print(bounds)
    #need a list of all individuals, list of pareto front individuals, then 2 lists of pareto per graph
    hvs = []
    gens = []
    print('newgen')
    #print(df[df['ciou_loss'] == df['ciou_loss'].max()])
    for gen in range(df['gen'].min(), df['gen'].max() + 1): #also 6, 12, and beyond, basically skip 7-11
        #if gen >= 7 and gen <= 11:
        #    continue
        df_current = df[(df['gen'] <= gen)] #& (~df.gen.isin([7, 8, 9, 10, 11]))]
        #print(objectives, directions)
        front, dominated = find_pareto_indices(df_current, objectives, directions)
        #print(len(front), len(dominated))
        front_top, dominated_top = find_pareto_indices(df_current, objectives[:2], directions[:2])
        front_bottom, dominated_bottom = find_pareto_indices(df_current, objectives[1:], directions[1:])
        gen_plot(df_current, gen, path, objectives, directions, front, front_top, front_bottom, bounds)
        print('GEN:', gen)
        print('length of fronts:', len(df_current), len(front), len(front_top), len(front_bottom))
        #front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']]
        rows_set = set(map(tuple, front.values))
        print(front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']].drop_duplicates().shape)
        hv_front = front[['uw_val_epoch_loss', 'ciou_loss', 'average_precision']].to_numpy()
        
        hv_front[:, 2] = hv_front[:, 2] * -1
        #[1000000, 2, 1]
        #[0, 0, 0]
        hv_max = np.array([1000000, 2, 0]) #hv_front.max(axis=0)
        hv_min = np.array([0, 0, -1]) #hv_front.min(axis=0)
        hv_front = (hv_front - hv_min) / (hv_max - hv_min)
        #print(hv_front)
        #ref_point = np.array([np.max(hv_front[:, 0]) + 1, np.max(hv_front[:, 1]) + 1, np.max(hv_front[:, 2]) + 1])
        #ref_point = np.array([-10,-10,-10])
        ref_point = np.array([1, 1, 1])
        #print('refpoint', ref_point)
        hv = Hypervolume(ref_point=ref_point)
        hvs.append(hv(hv_front))
        gens.append(gen)
        print('hypervolume:', hv(hv_front))
        print()
    
    print(front[['gen', 'hash']])
    plt.plot(hvs, marker='o')
    #plt.plot(hvs, marker='o')
    plt.xticks(range(len(gens)), gens)
    plt.title("Pareto Front Hypervolumes Per Generation")
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.tight_layout()
    plt.savefig('/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/graphs/pareto/pareto_hypervolume.jpg')
    plt.close()
        
    

