import os, sys
import glob
import pandas as pd
import numpy as np
import argparse
import tqdm
import toml

file_directory = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
repo_dir = os.path.abspath(os.path.join(file_directory, ".."))
sys.path.append(repo_dir)

def main(input_folder, output_file, cfg_path, is_classifier):
    # combine all CSVs and save
    print("Looking for CSV files...")
    all_csvs = glob.glob(os.path.join(input_folder, "**/*_metrics.csv"), recursive=True)
    print(f"Found {len(all_csvs)} csv files, loading now...")
    all_dfs = []
    all_csv_iterator = tqdm.tqdm(all_csvs)
    for csv_file in all_csv_iterator:
        surr_df = pd.read_csv(csv_file)
        all_dfs.append(surr_df)

    final_df = pd.concat(all_dfs, axis=0)
    final_df.index = np.arange(len(final_df))
    final_df.to_csv(output_file)
    print(f"Saved CSV to: {output_file}")

    kan_mask = np.array(["spline_order" in conf for conf in final_df["param_combo"]])
    mlp_mask = np.array(["spline_order" not in conf for conf in final_df["param_combo"]])

    if is_classifier:
        objectives = {"val_acc": 1}
    else:
        configs = toml.load(cfg_path)
        objectives = configs["pipeline"]["objectives"]
        objectives = {"mse_" + key.replace("_epoch", ""): value for key,value in objectives.items()}

    column_names = [col for col in objectives]

    column_sets = [[col] for col in column_names] + [column_names]
    for column_set in column_sets:
        sum_col = None
        for column in column_set:
            weighted_vals = final_df[column] * -1 if "mse" not in column.lower() else final_df[column]
            if sum_col is None:
                sum_col = weighted_vals
            else:
                sum_col += weighted_vals
        
        best_names = ["KAN", "MLP"]
        best_indices = []
        sum_col_arr = np.array(sum_col)

        argsort_indices = np.argsort(sum_col_arr)
        sorted_kan_mask = kan_mask[argsort_indices]
        sorted_mlp_mask = mlp_mask[argsort_indices]
        # # best overall
        # best_indices.append(argsort_indices[0])
        # best KAN
        best_indices.append(argsort_indices[sorted_kan_mask][0])
        # best MLP
        best_indices.append(argsort_indices[sorted_mlp_mask][0])

        print(f"\n\nTop surrogates by {column_set}:")
        for index, name in zip(best_indices, best_names):
            config = final_df.iloc[index]["param_combo"]
            print(f"Best {name} Individual: {config}\nHas Scores:\n{final_df.iloc[index]}")
        print(sum_col.describe())

        

if __name__ == "__main__":
    regressor_dir = '/gv1/projects/GRIP_Precog_Opt/surrogates/regressors'
    classifier_dir = '/gv1/projects/GRIP_Precog_Opt/surrogates/classifiers'

    parser = argparse.ArgumentParser(f'Aggrigates all grid search results into one CSV and shows the best individuals from it')
    parser.add_argument('--input-folder', type=str, default=regressor_dir,  help=f'The folder containing surrogate outputs from a agrid search to summarize.')
    parser.add_argument('--output-file', type=str, default=None, help='Location to save the file to, defaults to a file in the input folder')
    parser.add_argument('--cfg-path', type=str, required=False, default=os.path.join(repo_dir, 'conf.toml'), help='The config path defining the objectives')
    parser.add_argument('-c', '--is-classifier', default=False, action="store_true", help='Flag indicating that classifiers are being evaluated')
    my_args = parser.parse_args()

    if my_args.output_file is None:
        my_args.output_file = os.path.join(my_args.input_folder, "summary_surrogates.csv")

    main(my_args.input_folder, my_args.output_file, my_args.cfg_path, my_args.is_classifier)

# get best individuals based on holy trinity, both individual metrics and all 3
