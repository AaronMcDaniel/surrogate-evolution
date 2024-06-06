import toml
import sys, os
import itertools

import pandas as pd

from deap import gp
import primitives

class Pipeline:
    def __init__(self) -> None:
        # Begin by loading config attributes
        try:
            pipeline_config = toml.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf.toml"))["pipeline"]
        except Exception:
            raise Exception('Could not find configuration toml file with "pipeline" table')
        self.holy_grail = pd.DataFrame(columns=['gen', 'hash', 'genome', 'metrics']) # schema needs to be figured out
        self.surrogate_data = pd.DataFrame(columns=['gen', 'model', 'metrics']) # schema needs to be figured out
        self.current_population = [] # list of genomes associated with their metrics and hash
        self.population_size = pipeline_config['population_size']
        self.unsustainable_population_size = pipeline_config['unsustainable_population_size']
        self.num_parents = pipeline_config['num_parents']
        self.crossovers = pipeline_config['crossovers']
        self.mutations = pipeline_config['mutations']
        self.percent_elite = pipeline_config['percent_elite']
        self.surrogate = None # Surrogate class to be defined
        self.surrogate_enabled = pipeline_config['surrogate_enabled']
        self.objectives = pipeline_config['objectives']
        self.train_epochs = pipeline_config['train_epochs']
        self.selection_method_trusted = pipeline_config['selection_method_trusted']
        self.selection_method_untrusted = pipeline_config['selection_method_untrusted']
        self.selection_method_parents = pipeline_config['selection_method_parents']
        self.selection_method_elite_pool = pipeline_config['selection_method_elite_pool']
        self.elite_pool = [] # list of hashes representing genomes in the elite pool
        self.max_elite_pool = pipeline_config['max_elite_pool']
        self.hall_of_fame = [] # list of hashes representing genomes in the HOF
        self.train_pool_source = pipeline_config['train_pool_source']
        self.trust_pool_source = pipeline_config['trust_pool_source']
        self.gen_count = 0

        # Add primitives to primitive set
        pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(primitives.Tensor3D, 1), primitives.FinalTensor)
        
        self.pset = pset

    
    def init_pop(self, seed_file = None):
        pass

    def evaluate_gen(self):
        pass

    def select_parents(self): # current population is an attribute so no need for input args
        pass

    def overpopulate(self):
        pass

    def downselect(self, unsustainable_pop):
        pass

    def cross(self, parents: list):
        pass

    def mutate(self, mutant):
        pass

    def log_info(self):
        pass