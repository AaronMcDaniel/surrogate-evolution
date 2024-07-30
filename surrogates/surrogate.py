"""
Surrogate class and related surrogate functions. This class is used in the pipeline.
"""


import copy
import hashlib
import inspect
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
                'name': 'best_overall',
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
                'name': 'best_mse_uw_val_loss',
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
                'name': 'best_mse_ciou_loss',
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
                'name': 'best_mse_average_precision',
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
                'name': 'kan_test',
                'layers_hidden': [1021, 2048, 512, 256, 12],
                'grid_size': 5,
                'spline_order': 3,
                'scale_noise': 0.1,
                'scale_base': 1.0,
                'scale_spline': 1.0,
                'base_activation': torch.nn.SiLU,
                'grid_eps': 0.02,
                'grid_range': [-1, 1],
                'optimizer': optim.Adam,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                'validation_subset': [0, 4, 11],
                'model': sm.KAN
            },      
            {'name': 'kan_best_uwvl', 
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
            {'name': 'kan_best_cioul', 
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
                'grid_size': 25
            }
        ]
        self.classifier_models = [
            {
                'name': 'fail_predictor_3000',
                'dropout': 0.3,
                'hidden_sizes': [2048],
                'optimizer': optim.AdamW,
                'lr': 0.0005,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'model': sm.BinaryClassifier
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
        self.trust = 0
        self.cls_trust = 0
        self.codec = Codec(num_classes=model_config["num_classes"], genome_encoding_strat=codec_config["genome_encoding_strat"], surrogate_encoding_strat=codec_config["surrogate_encoding_strat"])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.METRICS = surrogate_config["surrogate_metrics"]
        self.opt_directions = surrogate_config["opt_directions"]
        
        ensure_deap_classes(self.objectives, codec_config)
        self.toolbox = base.Toolbox()
        
    
    # takes in a list of deap individuals as the inference pool and then predicts fitnesses for them using the surrogate blend
    # described by model_idxs. The fitnesses are then assigned to the deap individuals so they can be selected using a deap
    # selection algorithm like SPEA2
    def set_inferred_fitness(self, model_idxs, genome_scaler, inference_pool):        
        unique_model_idxs = list(set(model_idxs))
        unique_inferences = []
        
        for model_idx in unique_model_idxs:
            # get inferences on copy of calc_pool and assign fitness to copy
            print(f'    Getting inferences using {self.models[model_idx]['name']}...')
            inferences = self.get_surrogate_inferences(model_idx, genome_scaler, inference_pool)
            unique_inferences.append(inferences)
            
            #fitness_idx = model_idxs.index(model_idx)
        #print('START')
        #print('info', unique_model_idxs, unique_inferences[0][0], unique_inferences[1][0])
        print('    Constructing fitnesses...')
        constructed_inferences = []
        #print(len(unique_inferences[0]), len(unique_inferences[1]), len(surrogate_pool))
        for idx in range(len(inference_pool)):
            fitnesses = []
            for i, model_idx in enumerate(model_idxs):
                unique_idx = unique_model_idxs.index(model_idx)
                if len(unique_inferences[unique_idx][idx]) == 1:
                    i = 0
                #print(idx, unique_idx, i, unique_inferences[unique_idx][idx][i])
                fitnesses.append(unique_inferences[unique_idx][idx][i])
            #print('fitnesses', fitnesses)
            constructed_inferences.append(tuple(fitnesses))
        #print('constructed inferences', constructed_inferences)
            #for inference in unique_inferences[unique_idx]:

        for i, individual in enumerate(inference_pool):
            individual.fitness.values = constructed_inferences[i]
    
    
    # Calculates trust from a blend of sub_surrogates represented by model_idxs
    # The calc_pool is a list of deap individuals with calculated fitnesses. The model infers the metrics and 
    # we see the intersection in selections
    def calc_ensemble_trust(self, model_idxs, genome_scaler, calc_pool, rand = False):        
        # create copy of calc_pool
        surrogate_pool = copy.deepcopy(calc_pool)
        
        if not rand:
            self.set_inferred_fitness(model_idxs, genome_scaler, surrogate_pool)
                    
        # '''TESTING'''
        # for i in range(len(calc_pool)):
        #     print(calc_pool[i].fitness.values, surrogate_pool[i].fitness.values)
        
        # run trust-calc strategy to select trust_calc_ratio-based number of individuals for both calc_pool and its copy
        # TODO: add other cases of trust_calc_strategy
        print('    Calculating trust...')
        match self.trust_calc_strategy.lower():
            case 'spea2':
                self.toolbox.register("select", tools.selSPEA2, k = int(len(calc_pool)*self.trust_calc_ratio))
        
        selected = [self.__get_hash(str(g)) for g in self.toolbox.select(calc_pool)]
        if not rand:
            surrogate_selected = [self.__get_hash(str(g)) for g in self.toolbox.select(surrogate_pool)]
        else:
            surrogate_selected = [self.__get_hash(str(g)) for g in random.sample(surrogate_pool, int(len(calc_pool)*self.trust_calc_ratio))]
        
        # check intersection of selected individuals and return
        selected = set(selected)
        surrogate_selected = set(surrogate_selected)
        intersection = selected.intersection(surrogate_selected)
        trust = len(intersection)/len(selected)   #len(selected.union(surrogate_selected)) # for iou-based trust calculation
        return trust


    # OUTDATED
    # calculate trust for a single model (will still work, but only if the validation subset of the model used contains all optimization objectives)
    def calc_trust(self, model_idx, genome_scaler, calc_pool):        
        # create copy of calc_pool
        surrogate_pool = copy.deepcopy(calc_pool)
        
        # get inferences on copy of calc_pool and assign fitness to copy
        inferences = self.get_surrogate_inferences(model_idx, genome_scaler, surrogate_pool)
        for i, individual in enumerate(surrogate_pool):
            individual.fitness.values = inferences[i]
        
        # '''TESTING'''
        # for i in range(len(calc_pool)):
        #     print(calc_pool[i].fitness.values, surrogate_pool[i].fitness.values)
        
        # run trust-calc strategy to select trust_calc_ratio-based number of individuals for both calc_pool and its copy
        # TODO: add other cases of trust_calc_strategy
        match self.trust_calc_strategy.lower():
            case 'spea2':
                self.toolbox.register("select", tools.selSPEA2, k = int(len(calc_pool)*self.trust_calc_ratio))
        
        selected = [self.__get_hash(str(g)) for g in self.toolbox.select(calc_pool)]
        surrogate_selected = [self.__get_hash(str(g)) for g in self.toolbox.select(surrogate_pool)]
        
        # check intersection of selected individuals and return
        selected = set(selected)
        surrogate_selected = set(surrogate_selected)
        intersection = selected.intersection(surrogate_selected)
        trust = len(intersection)/len(selected)
        return trust
    
    
    # Get surrogate inferences on a list of deap individuals using a single model
    # CLIPS ENCODED GENOME VALUES: any encoded genome array values are forced be in the range [-1000, 1000]
    # so as to not cause issues when scaling the features for training/inference
    def get_surrogate_inferences(self, model_idx, genome_scaler: StandardScaler, inference_pool):
        encoded_genomes = []
    
        # Encode genomes
        for genome in inference_pool:
            for i in range(self.genome_epochs):
                try:
                    encoded_genome = self.codec.encode_surrogate(str(genome), i+1)
                    encoded_genomes.append(np.clip(encoded_genome, -1000, 1000)) 
                except:
                    encoded_genomes.append(np.full(1021, np.nan))
        # Get model dictionary and initialize the model
        model_dict = self.models[model_idx]
        model = model_dict['model']
        model_name = model_dict['name']
        metrics_subset = model_dict['metrics_subset']
        val_subset = model_dict['validation_subset']
        val_names = [self.METRICS[i] for i in val_subset]
        output_size = len(model_dict['metrics_subset'])
        sig = inspect.signature(model.__init__)
        filtered_params = {k: v for k, v in model_dict.items() if k in sig.parameters}
        model = model(output_size=output_size, **filtered_params)
        model = model.to(self.device)
        
        # Load model weights
        model.load_state_dict(torch.load(f'{self.weights_dir}/{model_name}.pth', map_location=self.device))
        model.eval()
        
        all_inferences = []
        
        # Transform encoded genomes
        encoded_genomes = np.array(encoded_genomes)
        encoded_genomes = genome_scaler.transform(encoded_genomes)
        
        for genome in encoded_genomes:
            if np.isnan(genome).any():
                inference_dict = {}
                for i in range(len(metrics_subset)):
                    inference_dict[self.METRICS[metrics_subset[i]]] = 300 * (1 if self.opt_directions[metrics_subset[i]] == 'min' else -1)
                all_inferences.append(inference_dict)
                continue
            genome = torch.tensor(genome, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                inference = model(genome)
                inference = torch.clamp(inference, min=(torch.ones_like(inference) * -300.0), max=(torch.ones_like(inference) * 300.0))
                # here we have all values in the metrics subset inferred on
                inference = inference.squeeze()
                inference = inference.tolist()
                if type(inference) is float:
                    inference = [inference]
              
                #print('before', inference)
                inference = tuple(inference)
                #print('after', inference)
                inference_dict = {}
                for i, val in enumerate(inference):
                    inference_dict[self.METRICS[metrics_subset[i]]] = val
                all_inferences.append(inference_dict)

        # Select the best inference for each genome and format to tuple
        def get_overall_loss(x):
            loss = 0
            for i in val_subset:
                loss += x[self.METRICS[i]] * (1 if self.opt_directions[i] == 'min' else -1)
            return loss 
        final_inferences = []
        for i in range(0, len(all_inferences), self.genome_epochs):
            epoch_inferences = all_inferences[i:i + self.genome_epochs]
            best_inference = min(epoch_inferences, key=lambda x: get_overall_loss(x))
            best_inference = tuple([best_inference[key] for key in val_names])
            final_inferences.append(best_inference)

        return final_inferences
    
    
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
    def train(self, classifier_train_df, classifier_val_df, regressor_train_df, regressor_val_df):
        scores = {
            'classifiers': {},
            'regressors': {}
        }
        genome_scaler = None
        print('Training classifier')
        # loop through the classifier models
        for classifier_dict in self.classifier_models:
            metrics, gs = cse.engine(self.surrogate_config, classifier_dict, classifier_train_df, classifier_val_df, self.weights_dir)
            if genome_scaler is None: genome_scaler = gs
            scores['classifiers'][classifier_dict['name']] = metrics
        
        print('training regressor')
        # loop through regressor models
        for regressor_dict in self.models:
            metrics, best_epoch_metrics, best_epoch_num, gs = rse.engine(self.surrogate_config, regressor_dict, regressor_train_df, regressor_val_df, self.weights_dir)
            if genome_scaler is None: genome_scaler = gs
            scores['regressors'][regressor_dict['name']] = best_epoch_metrics
            
        return scores, genome_scaler
    
    
    # inference models is a list of models where the first entry is the classifier model index to use and the
    # rest are the indices of the sub-surrogate regressor models 
    def get_inferences(self, inference_models, inference_df, cls_genome_scaler, reg_genome_scaler):
        # get classifier model and unique regression model dicts
        cls_model = inference_models[0]
        unique_reg_models = list(set(inference_models[1:]))
        cls_dict = self.classifier_models[cls_model]
        reg_dicts = [self.models[x] for x in unique_reg_models]

        # get classifier inferences
        cls_infs = cse.get_inferences(cls_dict, self.device, inference_df, cls_genome_scaler, self.weights_dir) # list of inferences. status of 1 means failed 0 means not

        # make df with successful individuals
        success_indices = [i for i, status in enumerate(cls_infs) if status == 0]
        reg_inf_df = inference_df.iloc[success_indices]
        
        # create returned dataframe and populate hash column
        reg_infs = pd.DataFrame(columns=['hash'] + list(self.objectives.keys()))
        reg_infs['hash'] = reg_inf_df['hash']

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
            inf = se.get_inferences(reg_dict, self.device, reg_inf_df, reg_genome_scaler, self.weights_dir)

            # use val_subset to map inferences to correct df cols
            for i, col_name in col_mapping.items():
                if i in val_subset:
                    reg_infs[col_name] = inf[:, val_subset.index(i)]

        return cls_infs, reg_infs
    

# TESTING SCRIPT
# surrogate = Surrogate('/home/tthakur9/precog-opt-grip/conf.toml', '/home/tthakur9/precog-opt-grip/test')
# # individuals = surrogate.get_individuals_from_file("/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution/out.csv", generations=[21, 22, 23, 24])
# reg_train_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/reg_train_dataset.pkl')
# reg_val_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/reg_val_dataset.pkl')
# cls_train_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/cls_train_dataset.pkl')
# cls_val_df = pd.read_pickle('/home/tthakur9/precog-opt-grip/surrogate_dataset/cls_val_dataset.pkl')
# inference_models = [0, 5, 6, 7]
# cls_train_dataset = sd.ClassifierSurrogateDataset(cls_train_df, mode='train')
# reg_train_dataset = sd.SurrogateDataset(reg_train_df, mode='train')
# cls_genome_scaler = cls_train_dataset.genomes_scaler
# reg_genome_scaler = reg_train_dataset.genomes_scaler
# print(surrogate.get_inferences(inference_models, cls_val_df, cls_genome_scaler, reg_genome_scaler))

# print(surrogate.calc_ensemble_trust([1, 2, 3], genome_scaler, individuals))
# print(surrogate.calc_trust(-2, genome_scaler, individuals))