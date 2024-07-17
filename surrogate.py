import copy
import hashlib
import inspect
from codec import Codec
from deap import creator, gp, base, tools
import numpy as np
import pandas as pd
from primitive_tree import CustomPrimitiveTree
import primitives
import surrogate_models as sm
import toml
import torch
import torch.optim as optim


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
    def __init__(self, config_dir):
        # Begin by loading config attributes
        configs = toml.load(config_dir)
        surrogate_config = configs["surrogate"]
        pipeline_config = configs["pipeline"]
        codec_config = configs["codec"]
        model_config = configs["model"]
        self.models = [
            {
                'name': 'best_overall',
                'dropout': 0.0,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.RMSprop,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'metrics_subset': [0, 4, 11], 
                'validation_subset': [0, 4, 11],
                'model': sm.MLP
            },
            {
                'name': 'best_mse_uw_val_loss',
                'dropout': 0.6,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.Adam,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                'metrics_subset': [0, 4, 11],
                'validation_subset': [0],
                'model': sm.MLP
            },
            {
                'name': 'best_mse_ciou_loss',
                'dropout': 0.6,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.Adam,
                'lr': 0.1,
                'scheduler': optim.lr_scheduler.StepLR,
                'metrics_subset': [0, 4, 11],
                'validation_subset': [4],
                'model': sm.MLP
            },
            {
                'name': 'best_mse_average_precision',
                'dropout': 0.6,
                'hidden_sizes': [2048, 1024, 512],
                'optimizer': optim.RMSprop,
                'lr': 0.01,
                'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                'metrics_subset': [11],
                'validation_subset': [11],
                'model': sm.MLP
            }            
        ]
        self.trust_calc_strategy = surrogate_config["trust_calc_strategy"]
        self.trust_calc_ratio = surrogate_config["trust_calc_ratio"]
        self.objectives = pipeline_config["objectives"]
        self.genome_epochs = model_config["train_epochs"]
        self.best_epoch_criteria = pipeline_config["best_epoch_criteria"]
        
        self.pset = primitives.pset
        self.trust = 0
        self.codec = Codec(num_classes=model_config["num_classes"], genome_encoding_strat=codec_config["genome_encoding_strat"], surrogate_encoding_strat=codec_config["surrogate_encoding_strat"])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.METRICS = surrogate_config["surrogate_metrics"]
        
        ensure_deap_classes(self.objectives, codec_config)
        self.toolbox = base.Toolbox()
        
    
    # The calc_pool is a list of deap individuals with calculated fitnesses. The model infers the metrics and 
    # we see the intersection in selections
    def calc_trust(self, model_idx, calc_pool):        
        # create copy of calc_pool
        surrogate_pool = copy.deepcopy(calc_pool)
        
        # get inferences on copy of calc_pool and assign fitness to copy
        inferences = self.get_surrogate_inferences(model_idx, surrogate_pool, list(self.objectives.keys()))
        for i, individual in enumerate(surrogate_pool):
            individual.fitness.values = inferences[i]
        
        '''TESTING'''
        for i in range(len(calc_pool)):
            print(calc_pool[i].fitness.values, surrogate_pool[i].fitness.values)
        
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
        print(intersection)
        trust = len(intersection)/(len(selected)+len(surrogate_selected))
        return trust
    
    
    # Get surrogate inferences on a list of deap individuals
    def get_surrogate_inferences(self, model_idx, inference_pool, objectives):
        encoded_genomes = []
        for genome in inference_pool:
            for i in range(self.genome_epochs):
                encoded_genomes.append(self.codec.encode_surrogate(str(genome), i+1)) # error handling needs to be done in case encoding breaks (punish)
        
        model_dict = self.models[model_idx]
        # build model
        model = model_dict['model']
        output_size = len(model_dict['metrics_subset'])
        sig = inspect.signature(model.__init__)
        filtered_params = {k: v for k, v in model_dict.items() if k in sig.parameters}
        model = model(output_size=output_size, **filtered_params).to(self.device)
        model.load_state_dict(torch.load('test/weights/weights.pth', map_location=self.device)) # weights dir is hardcoded rn
        model.eval()
        all_inferences = []
        for genome in encoded_genomes:
            genome = torch.tensor(genome, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                inference = model(genome)
                all_inferences.append(tuple(inference.squeeze().tolist()))

        # Select the best inference for each genome
        final_inferences = []
        criteria_metric = self.best_epoch_criteria[0]
        criteria_type = self.best_epoch_criteria[1]
        criteria_index = objectives.index(criteria_metric)

        for i in range(0, len(all_inferences), self.genome_epochs):
            epoch_inferences = all_inferences[i:i + self.genome_epochs]

            if criteria_type == 'min':
                best_inference = min(epoch_inferences, key=lambda x: x[criteria_index])
            elif criteria_type == 'max':
                best_inference = max(epoch_inferences, key=lambda x: x[criteria_index])
            else:
                raise ValueError(f"Invalid criteria type: {criteria_type}")

            final_inferences.append(best_inference)

        return final_inferences
    
    
    # This function converts string representations of genomes from a file like out.csv into deap individuals
    # with fitness that can be used to either train on or calculate trust with
    # parameter generations tells us what generations to get individuals from. Will use all individuals in a file if unspecified
    def get_individuals_from_file(self, filepath, generations=None):
        # Read the CSV file into a DataFrame
        genomes_df = pd.read_csv(filepath)

        # Raise an exception if generations is provided but 'gen' column is missing
        if generations is not None and 'gen' not in genomes_df.columns:
            raise ValueError("The DataFrame does not contain a 'gen' column.")

        # Filter the DataFrame based on generations if provided
        if generations is not None:
            genomes_df = genomes_df[genomes_df['gen'].isin(generations)]

        # Convert the DataFrame to a list of DEAP individuals
        genomes = genomes_df['genome'].values  
        fitness_values = genomes_df[list(self.objectives.keys())].values
        
        # Check for NaN values and replace them
        for i, key in enumerate(self.objectives.keys()):
            if self.objectives[key] < 0:
                fitness_values[:, i] = np.where(np.isnan(fitness_values[:, i]), 1000000, fitness_values[:, i])
            elif self.objectives[key] > 0:
                fitness_values[:, i] = np.where(np.isnan(fitness_values[:, i]), -1000000, fitness_values[:, i])

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
        return hashlib.shake_256(s.encode()).hexdigest(5)


surrogate = Surrogate('conf.toml')
individuals = surrogate.get_individuals_from_file("/gv1/projects/GRIP_Precog_Opt/baseline_evolution/out.csv")
print(surrogate.calc_trust(0, individuals))
     