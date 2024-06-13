import copy
import hashlib
import json
import random
import subprocess
import time
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
        self.holy_grail = pd.DataFrame(columns=['gen', 'hash', 'genome', 'metrics']) # data regarding every evaluated individual; metrics are from best epoch since all other metric data is already stored by eval_script
        self.surrogate_data = pd.DataFrame(columns=['gen', 'model', 'metrics']) # all data regarding surrogates; metrics are per epoch
        self.current_population = {} # dict of genomes associated with their metrics with hash as key
        self.current_deap_pop = [] # list of deap individuals representing the current population; no other info
        self.elite_pool = [] # list of deap individuals in the elite pool
        self.hall_of_fame = tools.ParetoFront() # hall of fame as a ParetoFront object
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
            res = ''.join([init.lower(), *map(str.title, temp)]) # convert from readable snake_case config to camelCase function name 
            self.toolbox.register(crossover, eval(f'gp.cx{str.upper(res[0])+res[1:]}'))

        for mutation in self.mutations.keys():
            init, *temp = mutation.split('_')
            res = ''.join([init.lower(), *map(str.title, temp)]) # convert from readable snake_case config to camelCase function name 
            self.toolbox.register(mutation, eval(f'gp.mut{str.upper(res[0])+res[1:]}'))


    def init_pop(self, seed_file = None):
        print('Initializing population...')
        # TODO: add seed file support
        self.current_population = {}
        pop = self.toolbox.population(n=self.population_size)
        self.current_deap_pop = pop
        for genome in pop:
            genome_string = str(genome)
            hash = self.__get_hash(genome_string)
            self.current_population[hash] = {'genome': genome_string, 'metrics': None}
        print('Done!')


    def evaluate_gen(self):
        print(f'Evaluating generation {self.gen_count}...')
        # build input file for eval_script
        lines = []
        for hash, genome in self.current_population.items():
                lines.append({'gen': self.gen_count, 'hash': hash, 'genome': genome['genome']}) # eval script can get num_epochs, genome_encoding_strat and other parameters from config file
        input_data = pd.DataFrame(lines)
        input_data.to_csv('eval_input.csv', index=False)

        # create bash script for job file
        create_job_file(self.population_size)

        # dispatch job
        print('    Dispatching jobs...')
        os.popen(f"sbatch {JOB_NAME}.job" )
        print('    Waiting for jobs...')
        # wait for job to finish
        while True:
            time.sleep(5)
            p = subprocess.Popen(['squeue', '-n', JOB_NAME], stdout=subprocess.PIPE)
            text = p.stdout.read().decode('utf-8')
            jobs = text.split('\n')[1:-1]
            if len(jobs) == 0:
                break
        print('    Done!')

        # read eval_gen output file
        for hash, genome in self.current_population.items():
            with open(f'{OUTPUTS_FOLDER}/generation_{self.gen_count}/{hash}/best_epoch.json', 'r') as metrics_f:
                data = json.load(metrics_f)
                # update current population metrics
                self.current_population[hash]['metrics'] = data
                # update holy grail dataframe
                self.holy_grail.loc[len(self.holy_grail.index)] = [self.gen_count, hash, genome['genome'], data]
                # set fitness of current population
                for g in self.current_deap_pop:
                    if str(g) == genome['genome']:
                        g.fitness.values = tuple([data[key] for key in self.objectives.keys()])
        print(f'Generation {self.gen_count} evaluation done!')


    def select_parents(self): # current population is an attribute so no need for input args
        print('Selecting parents...')
        selected_parents = self.toolbox.select_parents(self.current_deap_pop)
        print('Done!')
        return selected_parents


    def update_elite_pool(self): # updates the elite pool and returns selected elites from the current population
        print('Updating elite pool...')
        self.elite_pool = self.toolbox.select_elitists(self.current_deap_pop + self.elite_pool)
        print('Done!')
        return self.elite_pool
    

    def update_hof(self): # updates ParetaFront object
        print('Updating Hall of Fame...')
        self.hall_of_fame.update(self.current_deap_pop)
        print('Done!')


    def overpopulate(self, mating_pool): # mating pool is selected_parents + elite pool
        print('Overpopulating...')
        new_pop = {}
        # repeat till target overpopulation size is met
        while len(new_pop) < self.unsustainable_population_size:
            copies = copy.deepcopy(mating_pool)
            # make pairs of parents randomly
            parent_pairs = []
            while len(parent_pairs) < len(mating_pool)/2:
                parent1 = copies.pop(random.randrange(0, len(copies)))
                parent2 = copies.pop(random.randrange(0, len(copies)))
                parent_pairs.append([parent1, parent2])
            # mate pairs
            for pair in parent_pairs:
                try: # try except due to errors with particular genomes when crossovering
                    offsprings = self.cross(pair)
                    mutants = []
                    # mutate on the offsprings
                    for offspring in offsprings:
                        offspring = copy.deepcopy(offspring)
                        mutants += self.mutate(offspring)
                    for genome in offsprings + mutants:
                        hash = self.__get_hash(str(genome))
                        if (hash in self.holy_grail.get('hash').to_list()): # avoid genomes that have been in the population before
                            continue
                        new_pop[hash] = genome
                        if (len(new_pop) == self.unsustainable_population_size):
                            print('Done!')
                            return new_pop
                except:
                    continue
        print('Done!')
        return new_pop


    def downselect(self, unsustainable_pop):
        print('Downselecting...')
        if (self.surrogate_enabled):
            # model = self.surrogate.getBestModel()
            # predictions = self.surrogate.getPredictions(model, unsustainable_pop)
            # ...
            pass # surrogate code will eventually go here
        else :
            if (self.selection_method_untrusted.lower() == 'random'): # choose randomly
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.population_size)
                new_deap_pop = []
                new_pop = {}
                for hash in new_hashes:
                    new_deap_pop.append(unsustainable_pop[hash])
                    new_pop[hash] = {'genome': str(unsustainable_pop[hash]), 'metrics': None}
                self.current_population = new_pop
                self.current_deap_pop = new_deap_pop
        print('Done!')


    def cross(self, parents):
        parents = copy.deepcopy(parents)
        offspring = []
        for crossover, probability in self.crossovers.items():
            additional_param = ''
            if (crossover == 'one_point_leaf_biased'):
                additional_param = ',0.5'
            if random.random() < probability:
                offspring += list(eval(f'self.toolbox.{crossover}(parents[0], parents[1]{additional_param})'))
        return offspring


    def mutate(self, mutant):
        mutant = copy.deepcopy(mutant)
        mutants = []
        for mutation, probability in self.mutations.items():
            if random.random() < probability:
                mutants += list(eval(f'self.toolbox.{mutation}')(mutant, self.pset))
        return mutants


    def log_info(self):
        print('Logging data...')
        # store holy grail
        holy_grail_expanded = self.holy_grail.join(pd.json_normalize(self.holy_grail['metrics'])).drop('metrics', axis='columns')
        holy_grail_expanded.to_csv(f'{OUTPUTS_FOLDER}/out.csv', index=False)
        # get all entries from holy grail that share the same hashes as the elite pool members
        elites_df = holy_grail_expanded[holy_grail_expanded['hash'].isin([self.__get_hash(str(genome)) for genome in self.elite_pool])]
        elites_df.to_csv(f'{OUTPUTS_FOLDER}/elites.csv', index=False)
        # get all entries from holy grail that share the same hashes as the hall of fame members
        hof_df = holy_grail_expanded[holy_grail_expanded['hash'].isin([self.__get_hash(str(genome)) for genome in self.hall_of_fame.items])]
        hof_df.to_csv(f'{OUTPUTS_FOLDER}/hall_of_fame.csv', index=False)
        print('Done!')


    def step_gen(self):
        self.gen_count += 1


    def __get_hash(self, s):
        return hashlib.shake_256(s.encode()).hexdigest(5)
    
    def clear_outputs(self):
        print('Clearing outputs and logs...')
        os.system(f'rm -rf {OUTPUTS_FOLDER}/*')
        os.system(f'rm -rf {LOGS_FOLDER}/*')
        print('Done!')

# job file params
JOB_NAME = 'eval'
NODES = 1
CORES = 8
MEM = '16GB'
JOB_TIME = '1-00:00'
SCRIPT = 'test/dummy_eval.py'
OUTPUTS_FOLDER = "/gv1/projects/GRIP_Precog_Opt/outputs"
LOGS_FOLDER = "/gv1/projects/GRIP_Precog_Opt/logs"

# TODO: Fix this function
def create_job_file(num_jobs):
    batch_script = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes={NODES}
#SBATCH -G 1
#SBATCH --cpus-per-task={CORES}
#SBATCH --mem={MEM}
#SBATCH --time={JOB_TIME}
#SBATCH --output={LOGS_FOLDER}/evaluation.%A.%a.log
#SBATCH --error={LOGS_FOLDER}/evaluation_error.%A.%a.log
#SBATCH --array=0-{num_jobs-1}

module load anaconda3/2023.07
module load cuda/12.1.1

# Activate conda environment
conda activate myenv

# Execute the Python script with SLURM_ARRAY_TASK_ID as argument. Script also has optional args -i and -o to specify input file and output directory respectively
python {SCRIPT} $((SLURM_ARRAY_TASK_ID))

conda deactivate
"""
    with open(f'{JOB_NAME}.job', 'w') as fh:
        fh.write(batch_script)
