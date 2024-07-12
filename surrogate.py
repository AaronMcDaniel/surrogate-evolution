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
from skorch import NeuralNetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, x_scaler, y_scaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def fit(self, X, y):
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        self.model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        X_scaled = self.x_scaler.transform(X)
        y_scaled_pred = self.model.predict(X_scaled)
        return self.y_scaler.inverse_transform(y_scaled_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)  # Use negative MSE for GridSearchCV because it maximizes

def fine_tune(model_str='MLP', n_jobs=-1, cv=5):
    # Load dataset
    df = pd.read_pickle('surrogate_dataset/complete_dataset.pkl')
    X = np.stack(df.iloc[:, 0].values).astype(np.float32)
    Y = df.iloc[:, -12:].values.astype(np.float32)

    # Check for NaN values in the dataset
    if np.isnan(X).any() or np.isnan(Y).any():
        raise ValueError("Input data contains NaN values.")

    if model_str == 'MLP':
        # Define the model
        model = NeuralNetRegressor(
            module=sm.MLP,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            lr=0.001,
            max_epochs=30,
            batch_size=16,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        # Define the custom regressor with scalers
        custom_regressor = CustomRegressor(
            model=model,
            x_scaler=StandardScaler(),
            y_scaler=StandardScaler()
        )

        # Define the parameter grid for grid search
        param_grid = {
            'model__max_epochs': [10, 20, 30],
            'model__batch_size': [8, 16, 32, 64],
            'model__module__dropout': [0.0, 0.2, 0.4, 0.6],
            'model__module__hidden_sizes': [[512, 256, 12], [1024, 512, 12], [2048, 1024, 512, 12]],
            'model__optimizer': [optim.SGD, optim.Adam, optim.RMSprop, optim.Adagrad],
            'model__optimizer__lr': [0.0001, 0.001, 0.01, 0.1],
            'model__callbacks': [
                [LRScheduler(policy=lr.StepLR, step_size=10, gamma=0.1)],
                [LRScheduler(policy=lr.MultiStepLR, milestones=[10, 20], gamma=0.1)],
                [LRScheduler(policy=lr.CosineAnnealingLR, T_max=10)],
                [LRScheduler(policy=lr.ReduceLROnPlateau, mode='min', factor=0.1, patience=5)]
            ]
        }

        # Create the GridSearchCV object with negative MSE scoring
        grid = GridSearchCV(estimator=custom_regressor, param_grid=param_grid, n_jobs=n_jobs, cv=cv, scoring='neg_mean_squared_error')
        result = grid.fit(X, Y)

    # Save the best model and parameters
    result_dict = {
        'best_params': result.best_params_,
        'best_score': -result.best_score_,  # Convert back to positive MSE
        'best_model': result.best_estimator_
    }
    with open(f'{model_str}_fine_tuned.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    return result_dict

print(fine_tune())
