"""
Script to launch a re-evaluation process of a run.
"""


import re
import argparse
import os
import subprocess
import time

import pandas as pd
import toml

# main re-eval loop
def engine(outdir, excluded_gens):

    # get generation folders
    pattern = re.compile(r'generation_\d+$')
    generation_folders = [os.path.join(outdir, item) for item in os.listdir(outdir)
                          if os.path.isdir(os.path.join(outdir, item)) and pattern.match(item)]
    generation_folders.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))

    # loop through each generation
    for folder in generation_folders:
        gen_num = int(os.path.basename(folder).split('_')[-1])
        if gen_num in excluded_gens:
            continue

        print(f'Re-evaluating Generation {gen_num}...')

        genome_folders = [os.path.join(folder, subfolder) for subfolder in os.listdir(folder)
                      if os.path.isdir(os.path.join(folder, subfolder))]
        
        # loop through each genome in the generation
        for genome_folder in genome_folders:
            predictions_path = os.path.join(genome_folder, 'predictions.pkl')
            if not os.path.isfile(predictions_path):
                continue # don't do anything if individual actually failed
            
            print(f'Evaluating individual {os.path.basename(genome_folder)}...')

            # dispatch stuff here (give outdir, and genome_folder)
            os.popen(f"sbatch re_evaluation/re_eval.job -o {outdir} -g {genome_folder}" )
        
        # wait for job to finish here
        while True:
            time.sleep(5)
            p = subprocess.Popen(['squeue', '-n', 'precog_re_eval'], stdout=subprocess.PIPE)
            text = p.stdout.read().decode('utf-8')
            jobs = text.split('\n')[1:-1]
            if len(jobs) == 0:
                break

        print('Updating holy grail...')

        # load existing holy grail
        holy_grail = pd.read_csv(os.path.join(outdir, 'out.csv'))
        for genome_folder in genome_folders:
            genome_hash = os.path.basename(genome_folder)
            metrics_df = pd.read_csv(os.path.join(genome_folder, 'metrics.csv'))
            # get the recalulated best epoch to replace with in the holy grail
            if (best_epoch_criteria[1].lower() == 'min'):
                best_epoch = metrics_df[metrics_df[best_epoch_criteria[0]] == metrics_df[best_epoch_criteria[0]].min()]
            else :
                best_epoch = metrics_df[metrics_df[best_epoch_criteria[0]] == metrics_df[best_epoch_criteria[0]].max()]

            # replace old entry in out.csv
            columns_to_replace = [col for col in best_epoch.columns if col != 'epoch_num' and col in holy_grail.columns]
            holy_grail.loc[holy_grail['hash'] == genome_hash, columns_to_replace] = best_epoch[columns_to_replace].values[0]

        holy_grail.to_csv(os.path.join(outdir, 'out.csv'), index=False)
                
        print('========================================')


if __name__ == '__main__':
    # parses arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', required=False, default='/gv1/projects/GRIP_Precog_Opt/outputs')
    parser.add_argument('-x', '--exclude', required=False, default = '')
    args = parser.parse_args()
    outdir = args.outdir
    excluded_gens = args.exclude
    if excluded_gens == '':
        excluded_gens = []
    else:
        excluded_gens = [int(gen) for gen in excluded_gens.split(',')]
    
    configs = toml.load(os.path.join(outdir, "conf.toml"))
    best_epoch_criteria = configs['pipeline']['best_epoch_criteria']

    # evaluate
    engine(outdir, excluded_gens)










