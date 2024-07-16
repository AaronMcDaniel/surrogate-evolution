from deap import creator, gp, base, tools
import numpy as np
import pandas as pd
from pipeline import CustomPrimitiveTree
import primitives
import toml


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
        self.models = surrogate_config["models"]
        self.trust_calc_strategy = surrogate_config["trust_calc_strategy"]
        self.trust_calc_ratio = surrogate_config["trust_calc_ratio"]
        self.objectives = pipeline_config["objectives"]
        
        self.pset = primitives.pset
        self.trusts = [0 for _ in self.models]
        
        ensure_deap_classes(self.objectives, codec_config)
        
    
    # The calc_pool is a list of deap individuals with calculated fitnesses. The model infers the metrics and 
    # we see the intersection in selections
    def calc_trust(self, model_name, calc_pool):
        if model_name.lower() not in self.models:
            raise ValueError(f'{model_name} provided is not in list of surrogate models')
        
        # create copy of calc_pool
        
        # convert deap individuals to string and call codec.encode_surrogate
        
        # get inferences on copy of calc_pool and assign fitness to copy
        
        # run trust-calc strategy to select trust_calc_ratio-based number of individuals for both calc_pool and its copy
        
        # check intersection of selected individuals and return
        pass
    
    
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
    
    
    def get_best_model(self):
        return self.models[max(enumerate(self.trusts), key=lambda x: x[1])[0]]
     