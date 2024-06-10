import csv
import hashlib
import toml
import sys, os

import pandas as pd
from deap import creator, gp, base, tools

import primitives

class Pipeline:
    def __init__(self) -> None:
        # Begin by loading config attributes
        try:
            configs = toml.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf.toml"))
            pipeline_config = configs["pipeline"]
            codec_config = configs["codec"]
        except Exception:
            raise Exception('Could not find configuration toml file with "pipeline" and "codec" table')
        self.population_size = pipeline_config['population_size']
        self.unsustainable_population_size = pipeline_config['unsustainable_population_size']
        self.num_parents = pipeline_config['num_parents']
        self.crossovers = pipeline_config['crossovers']
        self.mutations = pipeline_config['mutations']
        self.percent_elite = pipeline_config['percent_elite']
        self.surrogate_enabled = pipeline_config['surrogate_enabled']
        self.objectives = pipeline_config['objectives']
        self.train_epochs = pipeline_config['train_epochs']
        self.selection_method_trusted = pipeline_config['selection_method_trusted']
        self.selection_method_untrusted = pipeline_config['selection_method_untrusted']
        self.selection_method_parents = pipeline_config['selection_method_parents']
        self.selection_method_elite_pool = pipeline_config['selection_method_elite_pool']
        self.max_elite_pool = pipeline_config['max_elite_pool']
        self.train_pool_source = pipeline_config['train_pool_source']
        self.trust_pool_source = pipeline_config['trust_pool_source']

        # Useful attributes
        self.holy_grail = pd.DataFrame(columns=['gen', 'hash', 'genome', 'metrics']) # schema needs to be figured out
        self.surrogate_data = pd.DataFrame(columns=['gen', 'model', 'metrics']) # schema needs to be figured out
        self.current_population = [] # list of genomes associated with their metrics and hash
        self.selected_parents = [] # list of hashes of parents selected to be mated
        self.elite_pool = [] # list of hashes representing genomes in the elite pool
        self.surrogate = None # Surrogate class to be defined
        self.hall_of_fame = [] # list of hashes representing genomes in the HOF
        self.pset = primitives.pset # primitive set
        self.gen_count = 1

        # Setting up pipeline
        creator.create("FitnessMulti", base.Fitness, weights=tuple(self.objectives.values()))
        
        # TODO: add other cases for encoding strategy
        genome_type = gp.PrimitiveTree
        match codec_config["genome_encoding_strat"].lower():
            case "tree":
                genome_type = gp.PrimitiveTree
                
        creator.create("Individual", genome_type, fitness=creator.FitnessMulti)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=3, max_=10)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # TODO: add other cases of selection_method_parents
        match self.selection_method_parents.lower():
            case 'nsga2':
                self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
    

    def init_pop(self, seed_file = None):
        # TODO: add seed file support
        self.current_population = []
        pop = self.toolbox.population(n=self.population_size)
        for genome in pop:
            genome_string = str(genome)
            self.current_population.append({'genome': genome_string, 'hash': hashlib.shake_256(genome_string.encode()).hexdigest(5), 'metrics': None})


    def evaluate_gen(self):
        # build input file
        lines = []
        for genome in self.current_population:
                lines.append({'hash': genome['hash'], 'genome': genome['genome']}) # eval script can get num_epochs, genome_encoding_strat and other parameters from config file
        input_data = pd.DataFrame(lines)
        input_data.to_csv('eval_input.csv', index=False)

        # create bash script for job file
        create_job_file()

        # TODO: dispatch job and wait for completion
        # os.popen(f"sbatch --array 0-{self.population_size - 1} eval_gen.job" )
        
        # read eval_gen output file

        # update current population metrics and add to dataframe(s) 

        # set fitness of current population

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


# TODO: Fix this function
def create_job_file():
    lines = []
    lines.append("#!/bin/bash\n")
    lines.append("#SBATCH --job-name={}\n".format('???'))
    lines.append("#SBATCH --output={}\n".format('???')) # a path to shared folder 
    lines.append("#SBATCH --error={}\n".format('???'))
    lines.append("#SBATCH --exclude={}\n".format(0))  # ice nodes to exclude?
    lines.append("#SBATCH --time={}\n".format(100000)) # some maximum time
    lines.append("#SBATCH --mem=1000M\n") # unsure how much mem is required
    lines.append("#SBATCH -p testflight\n") # unsure what this is
    # will need more lines as required

    # with open('eval_gen.job', 'w') as fh:
    #     fh.writelines(lines)
