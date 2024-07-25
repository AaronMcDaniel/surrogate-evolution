'''
Script to calculate surrogate trustworthiness over generation off of an existing run
'''

import os
import pickle
import pandas as pd
import toml
from surrogate import Surrogate
from surrogate_dataset import SurrogateDataset, build_dataset
from surrogate_eval import engine

NUM_GENERATIONS = 20
RUN_FOLDER = '/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution'
CONFIG_DIR = 'conf.toml'
DATASETS_FOLDER = 'test/trust_gens_datasets' # temp directory to store datasets
SUB_SURROGATES = [4, 5, 6]
TRUSTS_DIR = 'test/trusts_over_gens.pkl'

configs = toml.load(CONFIG_DIR)
surrogate_config = configs["surrogate"]

surrogate = Surrogate(CONFIG_DIR)
seen_gens = []
trusts = []

for i in range(1, NUM_GENERATIONS+1):
    trusts.append(surrogate.trust)
    print(f'----------Generation {i} stats----------')
    print(f'Surrogate trustworthiness: {surrogate.trust}')
    if i == 1:
        print('No data to train/val on yet')
    elif i == 2:
        print('Trained on a portion of generation 1. Validated on a portion of generation 1')
    else:
        print(f'Trained on generations upto and including {i-2}. Validated on portion of generations upto and including generation {i-2} and generation {i-1}')
    
    print('------------------------------------')
    if i == NUM_GENERATIONS:
        break
    # generate dataset based on seen gens
    print('Building train and val datasets...')
    if i == 1:
        build_dataset(f'{RUN_FOLDER}/out.csv', RUN_FOLDER, DATASETS_FOLDER, val_ratio=0.2, include_only=[1])
        train_df = pd.read_pickle(f'{DATASETS_FOLDER}/train_dataset.pkl')
        val_df = pd.read_pickle(f'{DATASETS_FOLDER}/val_dataset.pkl')
    else:
        build_dataset(f'{RUN_FOLDER}/out.csv', RUN_FOLDER, DATASETS_FOLDER, val_ratio=0.2, include_only=seen_gens)
        train_df = pd.read_pickle(f'{DATASETS_FOLDER}/train_dataset.pkl')
        prev_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/val_dataset.pkl')
        build_dataset(f'{RUN_FOLDER}/out.csv', RUN_FOLDER, DATASETS_FOLDER, val_ratio=1, include_only=[i])
        curr_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/val_dataset.pkl')
        val_df = pd.concat([prev_val_df, curr_val_df])
    
    calc_pool = surrogate.get_individuals_from_file(f'{RUN_FOLDER}/out.csv', hashes=val_df['hash'].to_list())
    seen_gens.append(i)
    
    print('Training surrogate ensenble...')
    model_dicts = [surrogate.models[i] for i in SUB_SURROGATES]
    for model_dict in model_dicts:
        print(f'    Training {model_dict['name']}...')
        metrics, best_epoch_num, genome_scaler = engine(surrogate_config, model_dict, train_df, val_df)    
    
    print('Getting trust score...')
    surrogate.trust = surrogate.calc_ensemble_trust(SUB_SURROGATES, genome_scaler, calc_pool)
    
    # pickle the trusts
    with open(TRUSTS_DIR, 'wb') as file:
        pickle.dump(trusts, file) 
    
    print("====================================")

        
    