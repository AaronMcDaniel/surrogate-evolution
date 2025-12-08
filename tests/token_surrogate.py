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
from surrogates import transformer
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
import json
import argparse
from functools import partial


file_directory = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
# repo_dir = os.path.abspath(os.path.join(file_directory, ".."))
repo_dir = "/storage/ice-shared/vip-vvk/data/AOT/"


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
    def __init__(self, config_dir, weights_dir, cls_steps=None, reg_steps=None): # this config is the overall config, not just the surrogate specific one
        configs = toml.load(config_dir)
        surrogate_config = configs["surrogate"]
        self.surrogate_config = surrogate_config
        pipeline_config = configs["pipeline"]
        codec_config = configs["codec"]
        model_config = configs["model"]
        self.codec = Codec(num_classes=model_config["num_classes"], genome_encoding_strat=codec_config["genome_encoding_strat"], surrogate_encoding_strat=codec_config["surrogate_encoding_strat"])
        self.codec._build_discrete_vocab()
        self.vocab_size = len(self.codec.vocab)
        self.num_epochs = surrogate_config['surrogate_train_epochs']
        self.cls_steps = cls_steps
        self.reg_steps = reg_steps
        self.models = [ # these are the regressor models but are simply called 'models' for compatibility reasons with the pipeline
            # {
            #     'name': 'test_conv_overall',
            #     'dropout': 0.1,
            #     'optimizer': partial(optim.AdamW, weight_decay=1e-2),
            #     'lr': 0.0001,
            #     'scheduler': optim.lr_scheduler.CosineAnnealingLR,
            #     'model': sm.SurrogateCNN,
            #     'vocab_size': self.vocab_size,
            #     'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            #     'validation_subset': [0, 4, 11],
            # },
            # {
            #     'name': 'test_transformer_overall',
            #     'model': sm.SurrogateTransformer,
            #     'optimizer': partial(optim.AdamW, weight_decay=1e-2),
            #     'scheduler': partial(optim.lr_scheduler.OneCycleLR, epochs=self.num_epochs, steps_per_epoch=self.reg_steps),
            #     'lr': 0.0003,
            #     'dropout': 0.1,
            #     'vocab_size': self.vocab_size,
            #     'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            #     'validation_subset': [0, 4, 11],
            # },
            {
                'name': 'test_transformer_cioul',
                'model': sm.SurrogateTransformer,
                'optimizer': partial(optim.AdamW, weight_decay=1e-3),
                # 'scheduler': partial(optim.lr_scheduler.OneCycleLR, epochs=self.num_epochs, steps_per_epoch=self.reg_steps),
                'scheduler': None,
                # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'lr': 0.00005,
                'dropout': 0.001,
                'vocab_size': self.vocab_size,
                'metrics_subset': [4],
                'validation_subset': [4],
            },
            {
                'name': 'test_transformer_ap',
                'model': sm.SurrogateTransformer,
                'optimizer': partial(optim.AdamW, weight_decay=1e-3),
                # 'scheduler': partial(optim.lr_scheduler.OneCycleLR, epochs=self.num_epochs, steps_per_epoch=self.reg_steps),
                'scheduler': None,
                # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'lr': 0.00005,
                'dropout': 0.001,
                'vocab_size': self.vocab_size,
                'metrics_subset': [11],
                'validation_subset': [11],
            },
            {
                'name': 'test_transformer_dual',
                'model': sm.SurrogateTransformer,
                'optimizer': partial(optim.AdamW, weight_decay=1e-3),
                # 'scheduler': partial(optim.lr_scheduler.OneCycleLR, epochs=self.num_epochs, steps_per_epoch=self.reg_steps),
                'scheduler': None,
                # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'lr': 0.00005,
                'dropout': 0.001,
                'vocab_size': self.vocab_size,
                'metrics_subset': [4, 11],
                'validation_subset': [4, 11],
            },
            # {
            #     'name': 'test_transformer_cioul_0.083',
            #     'model': sm.SurrogateTransformer,
            #     'optimizer': partial(optim.AdamW, weight_decay=1e-2),
            #     # 'scheduler': partial(optim.lr_scheduler.OneCycleLR, epochs=self.num_epochs, steps_per_epoch=self.reg_steps),
            #     'scheduler': None,
            #     'lr': 0.0006,
            #     'dropout': 0.1,
            #     'vocab_size': self.vocab_size,
            #     'metrics_subset': [4],
            #     'validation_subset': [4],
            # },
            # {
            #     'name': 'test_transformer_ap',
            #     'model': sm.SurrogateTransformer,
            #     'optimizer': partial(optim.AdamW, weight_decay=1e-2),
            #     'scheduler': partial(optim.lr_scheduler.OneCycleLR, epochs=self.num_epochs, steps_per_epoch=self.reg_steps),
            #     'lr': 0.0003,
            #     'dropout': 0.1,
            #     'vocab_size': self.vocab_size,
            #     'metrics_subset': [11],
            #     'validation_subset': [11],
            # }
        ]
        self.classifier_models = [
            {
                'name': 'test_transformer_classifier',
                'output_size': 1,
                'model': sm.SurrogateTransformer,
                'optimizer': partial(optim.AdamW, weight_decay=3e-1),
                # 'scheduler': partial(optim.lr_scheduler.OneCycleLR, epochs=self.num_epochs, steps_per_epoch=self.reg_steps),
                'scheduler': None,
                # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'lr': 0.00005,
                'dropout': 0.4,
                'vocab_size': self.vocab_size,
            },
        ]
        self.inference_models = None
        self.trust_calc_strategy = surrogate_config["trust_calc_strategy"]
        self.trust_calc_ratio = surrogate_config["trust_calc_ratio"]
        self.objectives = pipeline_config["objectives"]
        self.genome_epochs = model_config["train_epochs"]
        self.weights_dir = weights_dir
        self.batch_size = surrogate_config['surrogate_batch_size']
        
        self.pset = primitives.pset
        self.reg_trust = 0
        self.cls_trust = 0
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
    def train(self, classifier_train_df, classifier_val_df, regressor_train_df, regressor_val_df, train_reg=True, reg_lambda=0.001, train_cls=True):
        scores = {
            'classifiers': {},
            'regressors': {}
        }
        cls_genome_scaler = None
        reg_genome_scaler = None

        if train_cls:
            # loop through the classifier models
            for classifier_dict in self.classifier_models:
                metrics, gs = cse.engine(self.surrogate_config, classifier_dict, classifier_train_df, classifier_val_df, self.weights_dir, token_mode=True)
                if cls_genome_scaler is None: cls_genome_scaler = gs
                scores['classifiers'][classifier_dict['name']] = metrics
        
        # loop through regressor models
        if train_reg:
            for regressor_dict in self.models:
                metrics, best_epoch_metrics, best_epoch_num, gs = rse.engine(self.surrogate_config, regressor_dict, regressor_train_df, regressor_val_df, self.weights_dir, reg_lambda, token_mode=True)
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
    
    
    def predict(
        self, 
        z_latent: torch.Tensor,
        genome_scaler = None
    ) -> torch.Tensor:
        """
        Differentiable prediction method for inverse design optimization.
        
        This method takes latent architecture vectors and returns predicted fitness
        values using the trained surrogate ensemble. It maintains the computation
        graph for backpropagation through the generator.
        
        Args:
            z_latent: Latent architecture vectors, shape [B, z_dim], torch.Tensor
            genome_scaler: Scaler for genome features (if None, assumes z_latent is pre-scaled)
            
        Returns:
            predicted_fitness: Tensor of shape [B, num_objectives]
        """
        import inspect
        
        # Ensure input is on correct device
        z_latent = z_latent.to(self.device)
        batch_size = z_latent.shape[0]
        
        # Use default inference models if not provided
        # This would be the last trained/selected sub-surrogates
        if self.inference_models is None:
            # Default: use first classifier and all regressors for all objectives
            # You may want to set this based on your pipeline's sub_surrogates
            self.inference_models = [0] + list(range(len(self.models)))
        
        cls_model_idx = self.inference_models[0]
        reg_model_idxs = self.inference_models[1:]
        
        # Step 1: Scale features if scaler is provided
        if genome_scaler is not None:
            # Apply scaling - need to convert to numpy, scale, then back to torch
            # This breaks differentiability, so we'll skip scaling if not provided
            # For inverse design, we assume z_latent is already in the right scale
            z_scaled = z_latent
        else:
            z_scaled = z_latent
        
        # Step 2: Classifier inference (optional - for now we'll skip and assume all valid)
        # In the full pipeline, classifier predicts pass/fail
        # For inverse design, we'll skip this and go straight to regression
        # If you want to include it, you'd need a differentiable classifier forward pass
        
        # Step 3: Regressor inference - DIFFERENTIABLE
        # We need to run inference for each unique regressor model
        unique_reg_models = list(set(reg_model_idxs))
        
        # Create mapping from metric index to objective name
        col_mapping = {}
        for i, metric in enumerate(self.METRICS):
            if metric == 'mse_uw_val_loss':
                metric = 'mse_uw_val_epoch_loss'
            name = metric.replace('mse_', '')
            if name in list(self.objectives.keys()):
                col_mapping[i] = name
        
        # Initialize output tensor
        num_objectives = len(self.objectives)
        predictions = torch.zeros(batch_size, num_objectives, device=self.device)
        
        # Run each unique regressor
        for model_idx in unique_reg_models:
            model_dict = self.models[model_idx]
            
            # Build model architecture
            model_class = model_dict['model']
            output_size = len(model_dict['metrics_subset'])
            sig = inspect.signature(model_class.__init__)
            filtered_params = {k: v for k, v in model_dict.items() if k in sig.parameters}
            model = model_class(output_size=output_size, **filtered_params).to(self.device)
            
            # Load trained weights
            weights_path = f'{self.weights_dir}/{model_dict["name"]}.pth'
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.eval()
            
            # Freeze model parameters - ensure no gradients accumulate in surrogate
            for param in model.parameters():
                param.requires_grad = False
            
            # Forward pass (differentiable w.r.t. inputs only, not model params)
            with torch.set_grad_enabled(True):
                model_output = model(z_scaled)  # [B, output_size]
            
            # Clamp predictions to prevent extreme values
            model_output = torch.clamp(model_output, min=-300, max=300)
            
            # Map outputs to objective columns
            metrics_subset = model_dict['metrics_subset']
            val_subset = model_dict['validation_subset']
            
            # Get indices in model output that correspond to validation subset
            val_col_indices = [i for i, idx in enumerate(metrics_subset) if idx in val_subset]
            
            # Map to objective positions
            for i, col_idx in enumerate(val_col_indices):
                metric_idx = val_subset[i]
                if metric_idx in col_mapping:
                    obj_name = col_mapping[metric_idx]
                    obj_position = list(self.objectives.keys()).index(obj_name)
                    predictions[:, obj_position] = model_output[:, col_idx]
        
        return predictions


    def predict_soft(self, soft_tokens, model_indices=None):
        """
        Differentiable prediction path.
        Args:
            soft_tokens: [B, L, Vocab] Probability distribution from Generator
            model_indices: List of regressor indices to use (default: first 2)
        """
        if model_indices is None:
            # Default to using the models relevant for optimization (e.g., CIoU and AP)
            # Assuming these are models 0 and 1 in self.models based on your config
            model_indices = [0, 1]

        batch_size = soft_tokens.shape[0]
        num_objectives = len(self.objectives)
        predictions = torch.zeros(batch_size, num_objectives, device=self.device)
        
        # Iterate over selected regressors
        for model_idx in model_indices:
            model_dict = self.models[model_idx]
            
            # Re-instantiate the model structure (weights are loaded in self.train or manually)
            # NOTE: In your main script, ensure these models are loaded and stored in self.loaded_models
            # To avoid reloading weights every step, we assume self.loaded_models exists or we hack it here:
            
            # --- Fast Hacking for Optimization Loop ---
            # We assume the external loop passes the ACTUAL loaded pytorch model objects, 
            # OR we assume they are stored in self.models[i]['loaded_model']
            if 'loaded_model' not in model_dict:
                raise ValueError("Models must be loaded into memory before optimization loop.")
            
            surrogate_model = model_dict['loaded_model']
            
            # --- THE SOFT HANDOFF ---
            # 1. Get Surrogate's Embedding Matrix [Vocab, Emb_Dim]
            w_emb = surrogate_model.embedding.weight 
            
            # 2. Linear Combination: [B, L, Vocab] @ [Vocab, Emb_Dim] -> [B, L, Emb_Dim]
            inputs_embeds = torch.matmul(soft_tokens, w_emb)
            
            # 3. Forward pass through Transformer
            outputs = surrogate_model(x=None, inputs_embeds=inputs_embeds)
            
            # 4. Map outputs to predictions tensor (same logic as get_inferences)
            metrics_subset = model_dict['metrics_subset'] # e.g., [4] for CIoU
            
            # Map specific output heads to the global objective vector
            # (Simplified for the 2-objective case)
            if 4 in metrics_subset: # CIoU
                idx_in_output = metrics_subset.index(4)
                predictions[:, 0] = outputs[:, idx_in_output] # Assumes CIoU is 0th objective
            if 11 in metrics_subset: # AP
                idx_in_output = metrics_subset.index(11)
                predictions[:, 1] = outputs[:, idx_in_output] # Assumes AP is 1st objective

        return predictions
    
    
def main():
    # surrogate = Surrogate('conf.toml', os.path.join(repo_dir, 'psomu3/test/weights/surrogate_weights'))
    # # inference_models = [0, 5, 6, 7]
    # cls_train_dataset = sd.ClassifierSurrogateDataset(cls_train_df, mode='train')
    # reg_train_dataset = sd.SurrogateDataset(reg_train_df, mode='train')
    # cls_genome_scaler = cls_train_dataset.genomes_scaler
    # reg_genome_scaler = reg_train_dataset.genomes_scaler

    # use argparse to get mode
    parser = argparse.ArgumentParser(description="Surrogate training script")
    parser.add_argument('--mode', type=str, default="mix_dataset", help="Mode for training")
    args = parser.parse_args()
    FORCE_DATASET_RETOKENIZE = False

    mode = args.mode
    print("MODE:", mode, flush=True)

    scores_record = {}

    USER_ENV_VAR = os.getenv('USER', 'psomu3')
    testing_dir = f"{USER_ENV_VAR}/codestral/surrogate_training"
    # dataset_dir = "/storage/ice-shared/vip-vvk/data/AOT/surrogate_dataset"
    dataset_dir = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset"
    scores_file = os.path.join("/storage/ice-shared/vip-vvk/data/AOT/", testing_dir, f"scores_{mode}.txt")
    # cls_train_df = pd.read_pickle(os.path.join(dataset_dir, f'mix_dataset_cls_train.pkl'))
    # cls_val_df = pd.read_pickle(os.path.join(dataset_dir, f'mix_dataset_cls_val.pkl'))
    cls_train_df = pd.read_pickle(os.path.join(dataset_dir, f'{mode}_cls_train.pkl'))
    cls_val_df = pd.read_pickle(os.path.join(dataset_dir, f'{mode}_cls_val.pkl'))
    reg_train_df = pd.read_pickle(os.path.join(dataset_dir, f'{mode}_reg_train.pkl'))
    reg_val_df = pd.read_pickle(os.path.join(dataset_dir, f'{mode}_reg_val.pkl'))


    if FORCE_DATASET_RETOKENIZE:
        codec = Codec(num_classes=1)
        codec._build_discrete_vocab()
        pad_token_id = codec.vocab["<PAD>"]
        max_seq_len = 350

        def process_genome_string(genome_str):
            try:
                # Encode
                tokens = codec.encode_discrete(genome_str)
            except Exception as e:
                # Fallback for errors
                # Assuming pad_token_id is 0 based on context, change if needed
                tokens = np.array([0] * max_seq_len) 

            # Pad / Truncate
            seq_len = len(tokens)
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                padded_tokens = np.pad(tokens, (0, pad_len), 'constant', constant_values=0) # 0 is pad_token_id
            else:
                padded_tokens = tokens[:max_seq_len]
                
            return padded_tokens

        print("Tokenizing")
        dfs = [cls_train_df, cls_val_df, reg_train_df, reg_val_df]
        for i, df in enumerate(dfs):
            print(f"Processing dataframe {i+1}/{len(dfs)}")
            
            # Apply the function to the entire column at once
            # This creates a Series of numpy arrays and assigns it to the new column
            df['genome'] = df['str_genome'].apply(process_genome_string)

            # write to pkl file
            df.to_pickle(os.path.join(dataset_dir, f'{mode}_small_token_{["cls_train", "cls_val", "reg_train", "reg_val"][i]}.pkl'))

        print("Tokenizing done")
        print("Preview:")
        print(cls_train_df.iloc[0]['genome'])
    

    # cls_train_df = pd.read_pickle("/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/temp_surrogate_datasets/surr_evolution_cls_train.pkl")
    # cls_val_df = pd.read_pickle("/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/temp_surrogate_datasets/surr_evolution_cls_val.pkl")
    # reg_train_df = pd.read_pickle("/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/temp_surrogate_datasets/surr_evolution_reg_train.pkl")
    # reg_val_df = pd.read_pickle("/storage/ice-shared/vip-vvk/data/AOT/psomu3/full_vae_30/temp_surrogate_datasets/surr_evolution_reg_val.pkl")
    if not os.path.exists(os.path.join(repo_dir, testing_dir)):
        os.mkdir(os.path.join(repo_dir, testing_dir))
    if not os.path.exists(os.path.join(repo_dir, testing_dir, 'surrogate_weights')):
        os.mkdir(os.path.join(repo_dir, testing_dir, 'surrogate_weights'))
    for i in range(1):
        surrogate = Surrogate('conf.toml', os.path.join(repo_dir, os.path.join(testing_dir, 'surrogate_weights')), cls_steps=len(cls_train_df), reg_steps=len(reg_train_df))
        scores, cls_genome_scaler, reg_genome_scaler = surrogate.train(cls_train_df, cls_val_df, reg_train_df, reg_val_df, reg_lambda=0, train_cls=False)
        print("SAVING SCORES TO", scores_file)
        with open(scores_file, 'a') as f:
            json.dump(scores, f)
            f.write('\n')
        
    
    

if __name__ == "__main__":
    main()
