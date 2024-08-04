"""
Surrogate class and related surrogate functions. This class is used in the pipeline.
"""


import copy
import hashlib
import inspect
import itertools

from sklearn.metrics import accuracy_score
from codec import Codec
from deap import creator, gp, base, tools
import numpy as np
import pandas as pd
from primitive_tree import CustomPrimitiveTree
import primitives
from surrogates import surrogate_models as sm
import toml
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from surrogates import surrogate_dataset as sd
from surrogates import classifier_surrogate_eval as cse
from surrogates import surrogate_eval as se
from surrogates import surrogate_eval as rse
import random
import os

file_directory = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
repo_dir = os.path.abspath(os.path.join(file_directory, ".."))


def ensure_deap_classes(objectives, codec_config):
    # Check if the 'FitnessMulti' class exists, if not, create it
    if not hasattr(creator, 'FitnessMulti'):
        creator.create("FitnessMulti", base.Fitness, weights=tuple(objectives.values()))

    # TODO: add other cases for encoding strategy
        genome_type = gp.PrimitiveTree # default
        match codec_config["genome_encoding_strat"].lower():
            case "tree":
                genome_type = gp.PrimitiveTree

    # Check if the 'Individual' class exists, if not, create it
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", genome_type, fitness=creator.FitnessMulti)
    
    creator.create("TrustIndividual", str, fitness=creator.FitnessMulti)


class Surrogate():
    def __init__(self, config_dir, weights_dir): # this config is the overall config, not just the surrogate specific one
        configs = toml.load(config_dir)
        surrogate_config = configs["surrogate"]
        self.surrogate_config = surrogate_config
        pipeline_config = configs["pipeline"]
        codec_config = configs["codec"]
        model_config = configs["model"]
        self.models = [ # these are the regressor models but are simply called 'models' for compatibility reasons with the pipeline
            {
                'name': 'mlp_best_overall',
                'dropout': 0.2,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.Adam,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.StepLR,
                'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'validation_subset': [0, 4, 11],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_uwvl',
                'dropout': 0.2,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.Adam,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.StepLR,
                'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'validation_subset': [0],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_cioul',
                'dropout': 0.4,
                'hidden_sizes': [1024, 512],
                'optimizer': optim.Adam,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.MultiStepLR,
                'metrics_subset': [4],
                'validation_subset': [4],
                'model': sm.MLP
            },
            {
              'name': 'mlp_best_ap',
              'dropout': 0.0,
              'hidden_sizes': [512, 256],
              'optimizer': optim.Adam,
              'lr': 0.1,
              'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
              'metrics_subset': [11],
              'validation_subset': [11],
              'model': sm.MLP
            },
            {
                'name': 'mlp_best_uwvl_2',
                'dropout': 0.0,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.SGD,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'metrics_subset': [0],
                'validation_subset': [0],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_cioul_2',
                'dropout': 0.2,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.Adam,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                'metrics_subset': [4],
                'validation_subset': [4],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_ap_2',
                'dropout': 0.2,
                'hidden_sizes': [1024, 512],
                'optimizer': optim.RMSprop,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.MultiStepLR,
                'metrics_subset': [11],
                'validation_subset': [11],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_overall_2',
                'dropout': 0.4,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.RMSprop,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.StepLR,
                'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'validation_subset': [0, 4, 11],
                'model': sm.MLP
            },   
            {
              'name': 'kan_best_uwvl',
              'model': sm.KAN,
              'hidden_sizes': [512, 256],
              'optimizer': torch.optim.AdamW,
              'lr': 0.01,
              'scheduler': torch.optim.lr_scheduler.StepLR,
              'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              'validation_subset': [0],
              'grid_size': 25,
              'spline_order': 5
            },
            {
                'name': 'mlp_best_uwvl_2',
                'dropout': 0.0,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.SGD,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'metrics_subset': [0],
                'validation_subset': [0],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_cioul_2',
                'dropout': 0.2,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.Adam,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                'metrics_subset': [4],
                'validation_subset': [4],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_ap_2',
                'dropout': 0.2,
                'hidden_sizes': [1024, 512],
                'optimizer': optim.RMSprop,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.MultiStepLR,
                'metrics_subset': [11],
                'validation_subset': [11],
                'model': sm.MLP
            },
            {
                'name': 'mlp_best_overall_2',
                'dropout': 0.4,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.RMSprop,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.StepLR,
                'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'validation_subset': [0, 4, 11],
                'model': sm.MLP
            },
            {
              'name': 'kan_best_cioul',
              'model': sm.KAN,
              'hidden_sizes': [2048, 1024, 512],
              'optimizer': torch.optim.AdamW,
              'lr': 0.001,
              'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
              'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              'validation_subset': [4],
              'grid_size': 10,
              'spline_order': 1
            },
            {
              'name': 'kan_best_ap',
              'hidden_sizes': [2048, 1024, 512],
              'optimizer': optim.AdamW,
              'lr': 0.001,
              'scheduler': optim.lr_scheduler.StepLR,
              'metrics_subset': [11],
              'validation_subset': [11],
              'model': sm.KAN,
              'spline_order': 1,
              'grid_size': 25,
              'model': sm.KAN
            },
            {
                'name': 'kan_best_overall',
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.AdamW,
                'lr': 0.001,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'validation_subset': [0, 4, 11],
                'spline_order': 1,
                'grid_size': 25,
                'model': sm.KAN
            },
            {
                'name': 'kan_best_uwvl_2',
                'hidden_sizes': [512, 256],
                'optimizer': optim.AdamW,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts,
                'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'validation_subset': [0],
                'spline_order': 2,
                'grid_size': 1,
                'model': sm.KAN
            },
            {
                'name': 'kan_best_cioul_2',
                'hidden_sizes': [2048],
                'optimizer': optim.AdamW,
                'lr': 0.001,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'metrics_subset': [4],
                'validation_subset': [4],
                'spline_order': 2,
                'model': sm.KAN,
                'grid_size': 5
            },
            {
                'name': 'kan_best_ap_2',
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.AdamW,
                'lr': 0.001,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'metrics_subset': [11],
                'validation_subset': [11],
                'spline_order': 1,
                'model': sm.KAN,
                'grid_size': 5
            }
        ]
        self.classifier_models = [
            {
                'name': 'fail_predictor_3000',
                'dropout': 0.2,
                'hidden_sizes': [1024, 512, 256, 128],
                'optimizer': optim.Adam,
                'lr': 0.001,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'model': sm.MLP,
                'output_size': 1
            },
            {
                'name': 'fail_predictor_6000',
                'output_size': 1,
                'model': sm.MLP,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.Adagrad,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'lr': 0.01,
                'dropout': 0.2,
            },
            {
                'name': 'fail_predictor_turbo',
                'hidden_sizes': [512, 256],
                'optimizer': optim.RMSprop,
                'lr': 0.001,
                'spline_order': 2,
                'grid_size': 25,
                'model': sm.KAN,
                'output_size': 1,
                'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts,
                'scale_noise': 0.5
            }
        ]
        self.trust_calc_strategy = surrogate_config["trust_calc_strategy"]
        self.trust_calc_ratio = surrogate_config["trust_calc_ratio"]
        self.objectives = pipeline_config["objectives"]
        self.genome_epochs = model_config["train_epochs"]
        self.weights_dir = weights_dir
        self.num_epochs = surrogate_config['surrogate_train_epochs']
        self.batch_size = surrogate_config['surrogate_batch_size']
        
        self.pset = primitives.pset
        self.reg_trust = 0
        self.cls_trust = 0
        self.codec = Codec(num_classes=model_config["num_classes"], genome_encoding_strat=codec_config["genome_encoding_strat"], surrogate_encoding_strat=codec_config["surrogate_encoding_strat"])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.METRICS = surrogate_config["surrogate_metrics"]
        self.opt_directions = surrogate_config["opt_directions"]
        
        ensure_deap_classes(self.objectives, codec_config)
        self.toolbox = base.Toolbox()
    
    
    # This function converts string representations of genomes from a file like out.csv into deap individuals
    # with fitness that can be used to either train on or calculate trust with.
    # The generations parameter tells us what generations to get individuals from. Will use all individuals in a file if unspecified
    # CLIPS TARGET VALUES: values outside [-300, 300] will be clipped so metrics don't have abnormally large values for outliers
    def get_individuals_from_file(self, filepath, generations=None, hashes=None):
        # Read the CSV file into a DataFrame
        genomes_df = pd.read_csv(filepath)

        # Raise an exception if generations is provided but 'gen' column is missing
        if generations is not None and 'gen' not in genomes_df.columns:
            raise ValueError("The DataFrame does not contain a 'gen' column.")

        # Filter the DataFrame based on generations if provided
        if generations is not None:
            genomes_df = genomes_df[genomes_df['gen'].isin(generations)]
            
        # Filter the DataFrame based on hashes if provided
        if hashes is not None:
            if 'hash' not in genomes_df.columns:
                raise ValueError("The DataFrame does not contain a 'hash' column.")
            genomes_df = genomes_df[genomes_df['hash'].isin(hashes)]

        # Convert the DataFrame to a list of DEAP individuals
        genomes = genomes_df['genome'].values  
        fitness_values = genomes_df[list(self.objectives.keys())].values
        fitness_values = np.clip(fitness_values, -300, 300)
        
        # Check for NaN values and replace them
        for i, key in enumerate(self.objectives.keys()):
            if self.objectives[key] < 0:
                fitness_values[:, i] = np.where(np.isnan(fitness_values[:, i]), 300, fitness_values[:, i])
            elif self.objectives[key] > 0:
                fitness_values[:, i] = np.where(np.isnan(fitness_values[:, i]), -300, fitness_values[:, i])
            
        valid_rows = ~((fitness_values == 300) | (fitness_values == -300)).any(axis=1)
        fitness_values = fitness_values[valid_rows]
        genomes = genomes[valid_rows]

        # Convert genome string to deap individual
        individuals = [
            creator.Individual(CustomPrimitiveTree.from_string(genome, self.pset))
            for genome in genomes
        ]

        # Set fitnesses
        for individual, fitness in zip(individuals, fitness_values):
            individual.fitness.values = tuple(fitness)

        return individuals
    
    
    def __get_hash(self, s):
        layer_list = self.codec.get_layer_list(s)
        return hashlib.shake_256(str(layer_list).encode()).hexdigest(5)
    
    '''
    Section below is for two-stage surrogate implementation and is WOP.
    '''
    
    # trains all the classifiers and regressors and stores their individual weights and metrics
    def train(self, classifier_train_df, classifier_val_df, regressor_train_df, regressor_val_df, train_reg=True):
        scores = {
            'classifiers': {},
            'regressors': {}
        }
        cls_genome_scaler = None
        reg_genome_scaler = None
        # loop through the classifier models
        for classifier_dict in self.classifier_models:
            metrics, gs = cse.engine(self.surrogate_config, classifier_dict, classifier_train_df, classifier_val_df, self.weights_dir)
            if cls_genome_scaler is None: cls_genome_scaler = gs
            scores['classifiers'][classifier_dict['name']] = metrics
        
        # loop through regressor models
        if train_reg:
            for regressor_dict in self.models:
                metrics, best_epoch_metrics, best_epoch_num, gs = rse.engine(self.surrogate_config, regressor_dict, regressor_train_df, regressor_val_df, self.weights_dir)
                if reg_genome_scaler is None: reg_genome_scaler = gs
                scores['regressors'][regressor_dict['name']] = best_epoch_metrics
            
        return scores, cls_genome_scaler, reg_genome_scaler
    
    
    # inference models is a list of models where the first entry is the classifier model index to use and the
    # rest are the indices of the sub-surrogate regressor models 
    def get_inferences(self, inference_models, inference_df, cls_genome_scaler, reg_genome_scaler):
        # inference with cls model
        cls_model = inference_models[0]
        cls_dict = self.classifier_models[cls_model]
        cls_infs = cse.get_inferences(cls_dict, self.device, inference_df, cls_genome_scaler, self.weights_dir) # list of inferences. status of 1 means failed 0 means not

        # make df with successful individuals
        success_indices = [i for i, status in enumerate(cls_infs) if status == 0]
        reg_inf_df = inference_df.iloc[success_indices]

        # inference with reg models
        reg_infs = self.get_reg_inferences(inference_models[1:], reg_inf_df, reg_genome_scaler)

        return cls_infs, reg_infs
    

    def get_reg_inferences(self, model_idxs, inf_df, genome_scaler):
        # only inference with unique models
        unique_reg_models = list(set(model_idxs))
        reg_dicts = [self.models[x] for x in unique_reg_models]

        # create returned dataframe and populate hash column
        reg_infs = pd.DataFrame(columns=['hash'] + list(self.objectives.keys()))
        reg_infs['hash'] = inf_df['hash']

        # dynamically create column mapping using metrics list indices
        col_mapping = {}
        for i, metric in enumerate(self.METRICS):
            if metric == 'mse_uw_val_loss':
                metric = 'mse_uw_val_epoch_loss'
            name = metric.replace('mse_', '')
            if name in list(self.objectives.keys()):
                col_mapping[i] = name

        # get regression inferences
        for reg_dict in reg_dicts:
            val_subset = reg_dict['validation_subset']
            inf = se.get_inferences(reg_dict, self.device, inf_df, genome_scaler, self.weights_dir)

            # use val_subset to map inferences to correct df cols
            for i, col_name in col_mapping.items():
                if i in val_subset:
                    reg_infs[col_name] = inf[:, val_subset.index(i)]

        return reg_infs
    

    # takes in a list of deap individuals and assigns them inferred fitnesses
    # used for downselecting 
    def set_fitnesses(self, inference_models, cls_genome_scaler, reg_genome_scaler, deap_list):
        deap_list = copy.deepcopy(deap_list)
        invalid_deap = []
        valid_deap = []
        remaining_deap = []
        # step 1: encode the genomes to create an inference df
        inference_df = pd.DataFrame(columns=['hash', 'genome']) # this df will hold encoded genomes
        for genome in deap_list:
            try:
                encoded_genome = self.codec.encode_surrogate(str(genome), self.genome_epochs) # we're going to infer for the last epoch
                to_add = {'hash': self.__get_hash(str(genome)), 'genome': encoded_genome}
                inference_df.loc[len(inference_df)] = to_add
                remaining_deap.append(genome)
            except:
                invalid_deap.append(genome)
        # step 2: get inferences on these genomes
        failed, inferred_df = self.get_inferences(inference_models, inference_df, cls_genome_scaler, reg_genome_scaler)
        # at this stage, the inferred_df contains the set of individuals predicted as valid by the classifier

        # step 3 & 4: split individuals into those predicted to fail and those predicted to be valid 
        # and assign metrics based on inferred_df and assign bad metrics if invalid
        bad_fitnesses = tuple([300 if x < 0 else -300 for x in self.objectives.values()])
        for i, v in enumerate(failed):
            individual = remaining_deap[i]
            if v == 1:
                individual.fitness.values = bad_fitnesses
                invalid_deap.append(individual)
            else:
                h = self.__get_hash(str(individual))
                row = inferred_df[inferred_df['hash'] == h]
                if row.empty:
                    raise ValueError(f"Hash value {h} not found in the dataframe.")
                fitness = tuple([float(row[obj].values[0]) for obj in self.objectives.keys()])
                individual.fitness.values = fitness
                valid_deap.append(individual)
        
        return invalid_deap, valid_deap
    
    
    def calc_trust(self, inference_models, cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df):
        reg_val_df = reg_val_df.drop_duplicates(subset='hash') # get rid of duplicate individuals for trust calc
        cls_val_df = cls_val_df.drop_duplicates(subset='hash')
        # step 1: get classifier accuracy
        cls_dict = self.classifier_models[inference_models[0]]
        cls_inferences = cse.get_inferences(cls_dict, self.device, cls_val_df, cls_genome_scaler, self.weights_dir)
        truths = cls_val_df['label'].to_list()
        accuracy = accuracy_score(np.array(truths), np.array(cls_inferences))
        cls_trust = accuracy
        # step 2: get reg inferences
        reg_inferences_df = self.get_reg_inferences(inference_models[1:], reg_val_df, reg_genome_scaler)
        # step 3: assign fitness
        objective_keys = list(self.objectives.keys())
        true_individuals = []
        for row in reg_val_df.to_dict('records'):
            individual = creator.TrustIndividual(row['hash'])
            individual.fitness.values = tuple([row[x] for x in objective_keys])
            true_individuals.append(individual)
        inferred_individuals = []
        for row in reg_inferences_df.to_dict('records'):
            individual = creator.TrustIndividual(row['hash'])
            individual.fitness.values = tuple([row[x] for x in objective_keys])
            inferred_individuals.append(individual)
        
        # step 4: select down using trust_calc_strategy
        match self.trust_calc_strategy.lower():
            case 'spea2':
                self.toolbox.register("select", tools.selSPEA2, k = int(len(true_individuals)*self.trust_calc_ratio))
        
        selected = [str(g) for g in self.toolbox.select(true_individuals)]
        surrogate_selected = [str(g) for g in self.toolbox.select(inferred_individuals)]
        
        # step 5: check intersection of selected individuals and return
        selected = set(selected)
        surrogate_selected = set(surrogate_selected)
        intersection = selected.intersection(surrogate_selected)
        reg_trust = len(intersection)/len(selected)
        return cls_trust, reg_trust
        

    def optimize_trust(self, cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df):
        objectives_indices = []
        for i, metric in enumerate(self.METRICS):
            if metric == 'mse_uw_val_loss':
                metric = 'mse_uw_val_epoch_loss'
            name = metric.replace('mse_', '')
            if name in list(self.objectives.keys()):
                objectives_indices.append(i)

        grid = {}
        for i, objective in enumerate(objectives_indices):
            for j, m in enumerate(self.models):
                if objective in m['validation_subset']:
                    if i not in grid:
                        grid[i] = []
                    grid[i].append(j)

        compatible_models = grid.values()
        combos = [list(c) for c in list(itertools.product(*compatible_models))]
        max_trust = [float('-inf'), None]
        for c in combos:
            _, trust = self.calc_trust([0] + c, cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df)
            if trust > max_trust[0]:
                max_trust[0] = trust
                max_trust[1] = c
        return max_trust
    
    
def main():
    surrogate = Surrogate('conf.toml', os.path.join(repo_dir, 'test/weights/surrogate_weights'))
    reg_train_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_reg_train.pkl'))
    reg_val_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_reg_val.pkl'))
    cls_train_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_cls_train.pkl'))
    cls_val_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_cls_val.pkl'))
    # inference_models = [0, 5, 6, 7]
    cls_train_dataset = sd.ClassifierSurrogateDataset(cls_train_df, mode='train')
    reg_train_dataset = sd.SurrogateDataset(reg_train_df, mode='train')
    cls_genome_scaler = cls_train_dataset.genomes_scaler
    reg_genome_scaler = reg_train_dataset.genomes_scaler
    print(surrogate.optimize_trust(cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df))
    # print(surrogate.set_fitnesses(inference_models, cls_genome_scaler, reg_genome_scaler, individuals))

    print(surrogate.calc_ensemble_trust([4, 5, 6], reg_genome_scaler, individuals))
    print(surrogate.calc_ensemble_trust([1, 2, 3], genome_scaler, individuals))
    print(surrogate.calc_trust(-2, genome_scaler, individuals))

    surrogate = Surrogate('conf.toml', os.path.join(repo_dir, 'test/weights/surrogate_weights'))
    cls_train_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_cls_train.pkl'))
    cls_val_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_cls_val.pkl'))
    reg_train_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_reg_train.pkl'))
    reg_val_df = pd.read_pickle(os.path.join(repo_dir, 'surrogate_dataset/us_surr_reg_val.pkl'))
    cls_train_dataset = sd.ClassifierSurrogateDataset(cls_train_df, mode='train')
    reg_train_dataset = sd.SurrogateDataset(reg_train_df, mode='train', metrics_subset=[0, 4, 11])
    cls_genome_scaler = cls_train_dataset.genomes_scaler
    reg_genome_scaler = reg_train_dataset.genomes_scaler
    scores, cls_genome_scaler, reg_genome_scaler = surrogate.train(cls_train_df, cls_val_df, reg_train_df, reg_val_df)
    print(scores)
    
    
    print(surrogate.calc_trust([0, 5, 6, 7], cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df))
    print(surrogate.set_fitnesses([0, 5, 6, 7], cls_genome_scaler, reg_genome_scaler, individuals))
    print(surrogate.calc_ensemble_trust([5, 6, 7], reg_genome_scaler, individuals))
    print(surrogate.calc_trust(-2, genome_scaler, individuals))

if __name__ == "__main__":
    main()
