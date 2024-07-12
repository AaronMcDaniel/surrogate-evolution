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

def fine_tune(model_str='MLP', n_jobs=-1, cv=5):
    train_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/train_dataset.pkl')
    X_t = np.stack(train_df.iloc[:, 0].values).astype(np.float32)
    Y_t = train_df.iloc[:, -12:].values.astype(np.float32)

    val_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/val_dataset.pkl')
    X_v = np.stack(val_df.iloc[:, 0].values).astype(np.float32)
    Y_v = val_df.iloc[:, -12:].values.astype(np.float32)

    if model_str == 'MLP':
        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()
        X_t = X_scaler.fit_transform(X_t)
        Y_t = Y_scaler.fit_transform(Y_t)
        X_v = X_scaler.transform(X_v)
        Y_v = Y_scaler.transform(Y_v)

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
            'max_epochs': [10, 20],
            'hidden_sizes': [[1024, 512, 12], [2048, 1024, 512, 12]]
        }

        opt = GridSearch(model = model, param_grid = param_grid)
        opt.fit(X_t, Y_t, X_v, Y_v)
        print(opt.score)

        # pipeline = make_pipeline(
        #     StandardScaler(),
        #     TransformedTargetRegressor(
        #         regressor=model,
        #         transformer=StandardScaler()
        #     )
        # )

        # param_grid = {
        #     'max_epochs': [10, 20, 30],
        #     'batch_size': [8, 16, 32, 64],
        #     'module__dropout': [0.0, 0.2, 0.4, 0.6],
        #     'module__hidden_sizes': [[512, 256, 12], [1024, 512, 12], [2048, 1024, 512, 12]],
        #     'optimizer': [optim.SGD, optim.Adam, optim.RMSprop, optim.Adagrad],
        #     'optimizer__lr': [0.0001, 0.001, 0.01, 0.1],
        #     'callbacks': [
        #         [LRScheduler(policy=lr.StepLR, step_size=10, gamma=0.1)],
        #         [LRScheduler(policy=lr.MultiStepLR, milestones=[10, 20], gamma=0.1)],
        #         [LRScheduler(policy=lr.CosineAnnealingLR, T_max=10)],
        #         [LRScheduler(policy=lr.ReduceLROnPlateau, mode='min', factor=0.1, patience=5)]
        #     ]
        # }

        # grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=n_jobs, cv=cv)
        # result = grid.fit(X, Y)

    result_dict = {
        'best_params': result.best_params_,
        'best_score': result.best_score_,
        'best_model': result.best_estimator_
    }
    with open(f'{model_str}_fine_tuned.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    return result_dict

print(fine_tune())