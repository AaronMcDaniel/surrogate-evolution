import pickle
from sklearn.model_selection import GridSearchCV
import surrogate_models as sm
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler as lr
from skorch.callbacks import LRScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from skorch import NeuralNetRegressor
from sklearn.compose import TransformedTargetRegressor
from hypopt import GridSearch
import itertools

def fine_tune(model_str='MLP', n_jobs=-1, cv=5):
    df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset_baseline/complete_dataset.pkl')
    genomes = np.stack(df['genome'].values).astype(np.float32)
    metrics = df.iloc[:, -12:].values.astype(np.float32)
    genomes_scaler = StandardScaler()
    metrics_scaler = StandardScaler()
    genomes = genomes_scaler.fit_transform(genomes)
    metrics = metrics_scaler.fit_transform(metrics)

    if model_str == 'MLP':

        model = NeuralNetRegressor(
            module=sm.MLP,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            lr=0.001,
            max_epochs=30,
            batch_size=16,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        param_grid = {
            'module__dropout': [0.0, 0.2, 0.4, 0.6],
            'module__hidden_sizes': [[512, 256, 12], [1024, 512, 12], [2048, 1024, 512, 12]],
            'optimizer': [optim.SGD, optim.Adam, optim.RMSprop, optim.Adagrad],
            'optimizer__lr': [0.0001, 0.001, 0.01, 0.1],
            'callbacks': [
                [LRScheduler(policy=lr.StepLR, step_size=10, gamma=0.1)],
                [LRScheduler(policy=lr.MultiStepLR, milestones=[10, 20], gamma=0.1)],
                [LRScheduler(policy=lr.CosineAnnealingLR, T_max=10)],
                [LRScheduler(policy=lr.ReduceLROnPlateau, mode='min', factor=0.1, patience=5)]
            ]
        }
        # param_names = param_grid.keys()
        # param_values = param_grid.values()
        # combinations = list(itertools.product(*param_values))
        # combinations_dicts = [dict(zip(param_names, combination)) for combination in combinations]
        # for combo in combinations_dicts:
        #     print(combo)
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs, cv=cv)
        result = grid.fit(genomes, metrics)

    with open(f'{model_str}_all_combos.pkl', 'wb') as f:
        pickle.dump(result.best_estimator_, f)
        pickle.dump(result.cv_results_, f)

    with open(f'{model_str}_best_combo.txt', 'w') as f:
        f.write(f'Best Parameters: {result.best_params_}\n')
        f.write(f'Best Score: {result.best_score_}')

    return None

fine_tune()