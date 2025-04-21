#!/usr/bin/env python
import os
import sys
import pickle
import pandas as pd
import toml
from surrogates.surrogate import Surrogate
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train surrogate models on GPU')
    parser.add_argument('gen_num', type=int, help='Generation number')
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    
    # Load data
    train_data_dir = os.path.join(args.output_dir, 'surrogate_train_data')
    cls_train_df = pd.read_pickle(os.path.join(train_data_dir, f'cls_train.pkl'))
    cls_val_df = pd.read_pickle(os.path.join(train_data_dir, f'cls_val.pkl'))
    reg_train_df = pd.read_pickle(os.path.join(train_data_dir, f'reg_train.pkl'))
    reg_val_df = pd.read_pickle(os.path.join(train_data_dir, f'reg_val.pkl'))
    
    with open(os.path.join(train_data_dir, f'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    # Create and train surrogate
    config_path = os.path.join(args.output_dir, 'conf.toml')
    surrogate_weights_dir = os.path.join(args.output_dir, 'surrogate_weights')
    surrogate = Surrogate(config_path, surrogate_weights_dir)
    
    print(f"Training surrogate for generation {args.gen_num}")
    print(f"Dataset sizes - cls_train: {len(cls_train_df)}, cls_val: {len(cls_val_df)}, " 
          f"reg_train: {len(reg_train_df)}, reg_val: {len(reg_val_df)}")
    
    # Train surrogate and get results
    scores, cls_genome_scaler, reg_genome_scaler = surrogate.train(
        cls_train_df, 
        cls_val_df, 
        reg_train_df, 
        reg_val_df, 
        train_reg=config['train_reg']
    )
    
    # Save results
    results = {
        'scores': scores,
        'cls_genome_scaler': cls_genome_scaler,
        'reg_genome_scaler': reg_genome_scaler
    }
    
    results_path = os.path.join(train_data_dir, f'surrogate_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Training completed. Results saved to {results_path}")

if __name__ == "__main__":
    main()