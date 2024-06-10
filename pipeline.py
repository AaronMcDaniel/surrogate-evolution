import copy
import hashlib
import json
import random
import re
import toml
import os

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

        # Other useful attributes
        self.holy_grail = pd.DataFrame(columns=['gen', 'hash', 'genome', 'metrics']) # all data regarding individuals not stored by eval script; metrics are from best epoch since all other metric data is already stored by eval_script
        self.surrogate_data = pd.DataFrame(columns=['gen', 'model', 'metrics']) # all data regarding surrogates; metrics are per epoch
        self.current_population = {} # dict of genomes associated with their metrics with hash as key
        self.current_deap_pop = [] # list of deap individuals representing the current population; no other info
        self.elite_pool = [] # list of deap individuals in the elite pool
        self.hall_of_fame = tools.HallOfFame(10000) # hall of fame object
        self.surrogate = None # Surrogate class to be defined
        self.pset = primitives.pset # primitive set
        self.gen_count = 1

        # Setting up pipeline
        creator.create("FitnessMulti", base.Fitness, weights=tuple(self.objectives.values()))
        
        # TODO: add other cases for encoding strategy
        genome_type = gp.PrimitiveTree # default
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
    
        # TODO: add other cases of selection_method_elite_pool
        match self.selection_method_elite_pool.lower():
            case 'spea2':
                self.toolbox.register("select_elitists", tools.selSPEA2, k = self.max_elite_pool)
        
        for crossover in self.crossovers.keys():
            init, *temp = crossover.split('_')
            res = ''.join([init.lower(), *map(str.title, temp)])
            self.toolbox.register(crossover, eval(f'gp.cx{str.upper(res[0])+res[1:]}'))

        for mutation in self.mutations.keys():
            init, *temp = mutation.split('_')
            res = ''.join([init.lower(), *map(str.title, temp)])
            self.toolbox.register(mutation, eval(f'gp.mut{str.upper(res[0])+res[1:]}'))


    def init_pop(self, seed_file = None):
        # TODO: add seed file support
        self.current_population = {}
        pop = self.toolbox.population(n=self.population_size)
        self.current_deap_pop = pop
        for genome in pop:
            genome_string = str(genome)
            hash = hashlib.shake_256(genome_string.encode()).hexdigest(5)
            self.current_population[hash] = {'genome': genome_string, 'metrics': None}


    def evaluate_gen(self):
        # build input file for eval_script
        lines = []
        for hash, genome in self.current_population.items():
                lines.append({'gen': self.gen_count, 'hash': hash, 'genome': genome['genome']}) # eval script can get num_epochs, genome_encoding_strat and other parameters from config file
        input_data = pd.DataFrame(lines)
        input_data.to_csv('eval_input.csv', index=False)

        # create bash script for job file
        create_job_file()

        # TODO: dispatch job
        # os.popen(f"sbatch --array 0-{self.population_size - 1} eval_gen.job" )
        # wait for job to finish

        # read eval_gen output file
        for hash, genome in self.current_population.items():
            with open(f'/gv1/projects/GRIP_Precog_Opt/outputs/generation_{self.gen_count}/{hash}/best_epoch.json', 'r') as metrics_f:
            #with open(f'test/sample_out.json', 'r') as metrics_f:
                data = json.load(metrics_f)
                # update current population metrics
                self.current_population[hash]['metrics'] = data
                # update holy grail dataframe
                self.holy_grail.loc[len(self.holy_grail.index)] = [self.gen_count, hash, genome['genome'], data]
                # set fitness of current population
                for g in self.current_deap_pop:
                    if str(g) == genome['genome']:
                        g.fitness.values = tuple([data[key] for key in self.objectives.keys()])


    def select_parents(self): # current population is an attribute so no need for input args
        selected_parents = self.toolbox.select_parents(self.current_deap_pop)
        return selected_parents


    def update_elite_pool(self): # updates the elite pool and returns selected elites from the current population
        self.elite_pool = self.toolbox.select_elitists(self.current_deap_pop + self.elite_pool)
        return self.elite_pool
    

    def update_hof(self): # updates HOF object
        self.hall_of_fame.update(self.current_deap_pop)


    def overpopulate(self, mating_pool): # mating pool is selected_parents + elite pool, looks like some individuals can be broken
        new_pop = {}
        while len(new_pop) < self.unsustainable_population_size:
            copies = copy.deepcopy(mating_pool)
            parent_pairs = []
            while len(parent_pairs) < len(mating_pool)/2:
                parent1 = copies.pop(random.randrange(0, len(copies)))
                parent2 = copies.pop(random.randrange(0, len(copies)))
                parent_pairs.append([parent1, parent2])
            
            for pair in parent_pairs:
                try: # try catch due to some issues when crossovering
                    offsprings = self.cross(pair)
                    mutants = []
                    for offspring in offsprings:
                        mutants += self.mutate(offspring)
                    for genome in offsprings + mutants:
                        new_pop[hashlib.shake_256(str(genome).encode()).hexdigest(5)] = genome
                        if (len(new_pop) == self.unsustainable_population_size):
                            return new_pop
                except:
                    continue
        return new_pop


    def downselect(self, unsustainable_pop):
        if (self.surrogate_enabled):
            # model = self.surrogate.getBestModel()
            # predictions = self.surrogate.getPredictions(model, unsustainable_pop)
            # ...
            pass # surrogate code will eventually go here
        else :
            # self.current_population = 
            pass # chose randomly


    def cross(self, parents):
        offspring = []
        for crossover, probability in self.crossovers.items():
            additional_param = ''
            if (crossover == 'one_point_leaf_biased'):
                additional_param = ',0.5'
            if random.random() < probability:
                offspring += list(eval(f'self.toolbox.{crossover}(parents[0], parents[1]{additional_param})'))
        return offspring


    def mutate(self, mutant):
        mutants = []
        for mutation, probability in self.mutations.items():
            if random.random() < probability:
                mutants += list(eval(f'self.toolbox.{mutation}')(mutant, self.pset))
        return mutants


    def log_info(self):
        pass


    def step_gen(self):
        self.gen_count += 1


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
