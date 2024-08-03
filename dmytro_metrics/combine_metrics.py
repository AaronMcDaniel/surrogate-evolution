import pandas as pd
import os
file_directory = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
repo_dir = os.path.abspath(os.path.join(file_directory, "../../.."))
metrics_folder = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))

one = pd.read_csv(os.path.join(metrics_folder, '1/metrics.csv'))
two = pd.read_csv(os.path.join(metrics_folder, '2/metrics.csv'))
three = pd.read_csv(os.path.join(metrics_folder, '3/metrics.csv'))
four = pd.read_csv(os.path.join(metrics_folder, '4/metrics.csv'))
five = pd.read_csv(os.path.join(metrics_folder, '5/metrics.csv'))
six = pd.read_csv(os.path.join(metrics_folder, '6/metrics.csv'))
seven = pd.read_csv(os.path.join(metrics_folder, '7/metrics.csv'))
eight = pd.read_csv(os.path.join(metrics_folder, '8/metrics.csv'))
nine = pd.read_csv(os.path.join(metrics_folder, '9/metrics.csv'))
ten = pd.read_csv(os.path.join(metrics_folder, '10/metrics.csv'))
dfs = [one, two, three, four, five, six, seven, eight, nine, ten]

combined_df = pd.concat(dfs, axis=0, ignore_index=True)
combined_df.to_csv('combined_metric.csv', index=False)
# Print or use the combined DataFrame
