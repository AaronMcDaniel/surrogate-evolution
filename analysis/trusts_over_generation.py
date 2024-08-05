"""
OUTDATED
--------
Script to calculate surrogate trustworthiness over generations off of an existing run
"""

import os
import pickle
import numpy as np
import pandas as pd
import toml
import argparse
from surrogates.surrogate import Surrogate
from surrogates.surrogate_dataset import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Surrogate Evolution Script')
    parser.add_argument('-n', '--num_generations', type=int, required=True, help='Number of generations to run')
    parser.add_argument('-r', '--run_folder', type=str, required=True, help='Directory for run files')
    parser.add_argument('-c', '--config_dir', type=str, default='conf.toml', help='Path to configuration file')
    parser.add_argument('-d', '--datasets_folder', type=str, default='test/trust_gens_datasets', help='Temporary directory to store datasets')
    parser.add_argument('-t', '--trusts_dir', type=str, default='test/trusts_over_gens.pkl', help='Directory to store trust pickles')
    parser.add_argument('-s', '--subsurrogates_dir', type=str, default='test/subsurrogates_over_gens.pkl', help='Directory to store subsurrogate pickles')
    parser.add_argument('-w', '--weights_dir', type=str, default='test/trust_over_gens_weights', help='Directory to store weight files')
    return parser.parse_args()

def get_reg_indices(scores, surrogate, objectives, surrogate_metrics):
    best_models = {f'mse_{objective}': {'model': '', 'score': np.inf} for objective in objectives}

    name_to_dict = {d['name']: d for d in surrogate.models}
    indices = [surrogate_metrics.index(f'mse_{objective}') for objective in objectives]
    print(indices)

    for reg_key, reg_val in scores['regressors'].items():
        for idx, objective in zip(indices, best_models.keys()):
            if idx in name_to_dict[reg_key]['validation_subset']:
                if reg_val[objective] < best_models[objective]['score']:
                    best_models[objective]['model'] = reg_key
                    best_models[objective]['score'] = reg_val[objective]

    condensed = [list(name_to_dict.keys()).index(model['model']) for model in best_models.values()]
    
    return condensed

def main():
    args = parse_args()

    NUM_GENERATIONS = args.num_generations
    RUN_FOLDER = args.run_folder
    CONFIG_DIR = args.config_dir
    DATASETS_FOLDER = args.datasets_folder
    TRUSTS_DIR = args.trusts_dir
    SUBSURROGATES_DIR = args.subsurrogates_dir
    WEIGHTS_DIR = args.weights_dir

    configs = toml.load(CONFIG_DIR)
    surrogate_config = configs["surrogate"]
    pipeline_config = configs["pipeline"]

    surrogate = Surrogate(CONFIG_DIR, WEIGHTS_DIR)
    trusts = []
    chosen_models = []

    objectives = pipeline_config['objectives']
    surrogate_metrics = surrogate_config['surrogate_metrics']

    for i in range(1, NUM_GENERATIONS+1):
        print(f'---------- Generation {i} ----------')
        print('Building surrogate train and val datasets...')
        seen_gens = list(range(1, i))
        name = 'surr_evolution'
        
        if i == 1:
            trusts.append((surrogate.cls_trust, surrogate.reg_trust))
            chosen_models.append(None)
            with open(TRUSTS_DIR, 'wb') as file:
                pickle.dump(trusts, file)
            with open(SUBSURROGATES_DIR, 'wb') as file:
                pickle.dump(chosen_models, file)
            print('Nothing to do')  
            print("====================================")
            continue

        if i == 2:
            build_dataset(name, os.path.join(RUN_FOLDER, 'out.csv'), RUN_FOLDER, DATASETS_FOLDER, val_ratio=0.2, include_only=[1])
            reg_train_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_reg_train.pkl')
            reg_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_reg_val.pkl')
            cls_train_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_cls_train.pkl')
            cls_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_cls_val.pkl')
        elif i < 7:
            build_dataset(name, os.path.join(RUN_FOLDER, 'out.csv'), RUN_FOLDER, DATASETS_FOLDER, val_ratio=0, include_only=[1])
            reg_train_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_reg_train.pkl')
            cls_train_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_cls_train.pkl')
            build_dataset(name, os.path.join(RUN_FOLDER, 'out.csv'), RUN_FOLDER, DATASETS_FOLDER, val_ratio=1, include_only=seen_gens[1:])
            reg_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_reg_val.pkl')
            cls_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_cls_val.pkl')
        else:
            build_dataset(name, os.path.join(RUN_FOLDER, 'out.csv'), RUN_FOLDER, DATASETS_FOLDER, val_ratio=0, include_only=seen_gens[:-5])
            reg_train_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_reg_train.pkl')
            cls_train_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_cls_train.pkl')
            build_dataset(name, os.path.join(RUN_FOLDER, 'out.csv'), RUN_FOLDER, DATASETS_FOLDER, val_ratio=1, include_only=seen_gens[-5:])
            reg_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_reg_val.pkl')
            cls_val_df = pd.read_pickle(f'{DATASETS_FOLDER}/{name}_cls_val.pkl')

        print('Training surrogate ensemble...')
        train_reg = len(reg_val_df) >= surrogate_config['surrogate_batch_size'] * surrogate_config['min_batch_in_val_data']
        if not train_reg:
            print('----Warning: not enough valid data for regressors... skipping surrogate preparation----')
        
        scores, cls_genome_scaler, reg_genome_scaler = surrogate.train(cls_train_df, cls_val_df, reg_train_df, reg_val_df, train_reg=train_reg)
        
        print('Calculating trust and selecting best sub-surrogates...')
        sub_surrogates = []
        cls_trust = 0
        max_cls_model = ''
        for key, val in scores['classifiers'].items():
            if val['acc'] > cls_trust:
                cls_trust = val['acc']
                max_cls_model = key
        cls_to_dict = {d['name']: d for d in surrogate.classifier_models}
        max_cls_model_idx = list(cls_to_dict.keys()).index(max_cls_model)
        sub_surrogates.append(max_cls_model_idx)
        
        if train_reg:
            if surrogate_config['sub_surrogate_sel_strat'] == 'trust':
                result = surrogate.optimize_trust(cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df)
                reg_trust = result[0]
                sub_surrogates += result[1]
            else:
                reg_indices = get_reg_indices(scores, surrogate, objectives, surrogate_metrics)
                sub_surrogates.extend(reg_indices)
                cls_trust, reg_trust = surrogate.calc_trust(sub_surrogates, cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df)
                
            surrogate.cls_trust, surrogate.reg_trust = cls_trust, reg_trust
        else:
            sub_surrogates += [0] * len(objectives)
        
        trusts.append((surrogate.cls_trust, surrogate.reg_trust))
        print(f'classifier trust: {surrogate.cls_trust}, regressor trust: {surrogate.reg_trust}')
        
        with open(TRUSTS_DIR, 'wb') as file:
            pickle.dump(trusts, file)
        with open(SUBSURROGATES_DIR, 'wb') as file:
            pickle.dump(chosen_models, file)  
        
        print("====================================")

if __name__ == "__main__":
    main()
    