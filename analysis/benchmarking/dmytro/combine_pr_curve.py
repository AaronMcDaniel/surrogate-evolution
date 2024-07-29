import pickle
import os
import matplotlib.pyplot as plt

def plot_PR_curve(dmytro_precisions, dmytro_recalls, dmytro_ap, our_precisions, our_recalls, our_ap, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,'pr_curve.png')
    plt.figure(figsize=(8, 6))
    plt.plot(dmytro_recalls, dmytro_precisions, marker='o', label=f'Benchmark PR curve (AP = {dmytro_ap:.2f})')
    plt.plot(our_recalls, our_precisions, marker='o', label=f'Baseline PR curve (AP = {our_ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

dmytro_path = '/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/6/pr_curve_metrics.pkl'
our_path = 'dmytro_metrics/compare/pr_curve_metrics_comparison.pkl'


#path = '/home/eharpster3/precog-opt-grip/dmytro_metrics/pr_curve_metrics_comparison.pkl'
#path = '/home/eharpster3/precog-opt-grip/targets.pkl'
with open(dmytro_path, 'rb') as f:
    dmytro_pr_curve_metrics = pickle.load(f)
with open(our_path, 'rb') as f:
    our_pr_curve_metrics = pickle.load(f)

dmytro_pre_curve = dmytro_pr_curve_metrics['pre_curve']
dmytro_rec_curve = dmytro_pr_curve_metrics['rec_curve']
dmytro_epoch_avg_pre = dmytro_pr_curve_metrics['epoch_avg_pre']

our_pre_curve = our_pr_curve_metrics['pre_curve']
our_rec_curve = our_pr_curve_metrics['rec_curve']
our_epoch_avg_pre = our_pr_curve_metrics['epoch_avg_pre']

plot_PR_curve(dmytro_pre_curve, dmytro_rec_curve, dmytro_epoch_avg_pre, our_pre_curve, our_rec_curve, our_epoch_avg_pre, save_dir='/home/eharpster3/precog-opt-grip/dmytro_metrics/compare')