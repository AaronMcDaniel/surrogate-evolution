import pandas as pd

one = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/1/metrics.csv')
two = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/2/metrics.csv')
three = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/3/metrics.csv')
four = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/4/metrics.csv')
five = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/5/metrics.csv')
six = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/6/metrics.csv')
seven = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/7/metrics.csv')
eight = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/8/metrics.csv')
nine = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/9/metrics.csv')
ten = pd.read_csv('/home/eharpster3/precog-opt-grip/dmytro_metrics/epochs/10/metrics.csv')
dfs = [one, two, three, four, five, six, seven, eight, nine, ten]

combined_df = pd.concat(dfs, axis=0, ignore_index=True)
combined_df.to_csv('combined_metric.csv', index=False)
# Print or use the combined DataFrame
