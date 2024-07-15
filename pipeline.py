from collections import deque
import copy
import csv
import hashlib
import pickle
import random
import re
import shutil
import subprocess
import time
import toml
import os

import pandas as pd
from deap import creator, gp, base, tools

import primitives

# job file params
JOB_NAME = 'precog_eval'
NODES = 1
CORES = 8
MEM = '32GB'
JOB_TIME = '04:00:00'
SCRIPT = 'eval.py'
ENV_NAME = 'myenv'
EXCEPTED_NODES = ['ice109', 'ice111', 'ice161', 'ice113', 'ice116', 'ice114', 'ice170', 'ice149', 'ice158', 'ice177', 'ice178', 'ice120']
GPUS = ["TeslaV100-PCIE-32GB", "TeslaV100S-PCIE-32GB", "NVIDIAA100-SXM4-80GB", "NVIDIAA10080GBPCIe", "TeslaP100-SXM2-16GB", "TeslaK80"]
# ALLOWED_NODES = ['ice108', 'ice107', 'ice110', 'ice143', 'ice144', 'ice145', 'ice151', 'ice162', 'ice163', 'ice164', 'ice165', 'ice175', 'ice176', 'ice179', 'ice183', 'ice185', 'ice191', 'ice192', 'ice193']


class Pipeline:
    def __init__(self, output_dir, config_dir, force_wipe = False, clean = False) -> None:
        self.output_dir = output_dir
        self.force_wipe = force_wipe
        self.clean = clean
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.attempt_resume = False

        # Check if output location already exists
        if os.path.exists(self.output_dir):
            if not self.force_wipe:
                self.attempt_resume = True
            else:
                self.clear_outputs()
                os.makedirs(self.logs_dir)
                shutil.copy(config_dir, output_dir)
        else:
            os.makedirs(self.output_dir)
            os.makedirs(self.logs_dir)
            shutil.copy(config_dir, output_dir)

        # Begin by loading config attributes
        configs = toml.load(config_dir)
        pipeline_config = configs["pipeline"]
        codec_config = configs["codec"]

        self.initial_population_size = pipeline_config['initial_population_size']
        self.population_size = pipeline_config['population_size']
        self.unsustainable_population_size = pipeline_config['unsustainable_population_size']
        self.num_parents = pipeline_config['num_parents']
        self.crossovers = pipeline_config['crossovers']
        self.mutations = pipeline_config['mutations']
        self.surrogate_enabled = pipeline_config['surrogate_enabled']
        self.objectives = pipeline_config['objectives']
        self.selection_method_trusted = pipeline_config['selection_method_trusted']
        self.selection_method_untrusted = pipeline_config['selection_method_untrusted']
        self.selection_method_parents = pipeline_config['selection_method_parents']
        self.selection_method_elite_pool = pipeline_config['selection_method_elite_pool']
        self.max_elite_pool = pipeline_config['max_elite_pool']
        self.train_pool_source = pipeline_config['train_pool_source']
        self.trust_pool_source = pipeline_config['trust_pool_source']
        self.best_epoch_criteria = pipeline_config['best_epoch_criteria']

        # Other useful attributes
        self.holy_grail = pd.DataFrame(columns=['gen', 'hash', 'genome', 'metrics']) # data regarding every evaluated individual; metrics are from best epoch since all other metric data is already stored by eval_script
        self.surrogate_data = pd.DataFrame(columns=['gen', 'model', 'metrics']) # all data regarding surrogates; metrics are per epoch
        self.current_population = {} # dict of genomes associated with their metrics with hash as key
        self.current_deap_pop = [] # list of deap individuals representing the current population; no other info
        self.elite_pool = [] # list of deap individuals in the elite pool
        self.elite_pool_history = {} # dict keeping track of elite pool through generations
        self.hall_of_fame = tools.ParetoFront() # hall of fame as a ParetoFront object
        self.hof_history = {} # dict keeping track of hall of fame through generations
        self.surrogate = None # Surrogate class to be defined
        self.pset = primitives.pset # primitive set
        self.gen_count = 1
        self.num_genome_fails = 0
        self.total_evaluated_individuals = 0

        # Setting up pipeline
        creator.create("FitnessMulti", base.Fitness, weights=tuple(self.objectives.values()))
        
        # TODO: add other cases for encoding strategy
        genome_type = gp.PrimitiveTree # default
        match codec_config["genome_encoding_strat"].lower():
            case "tree":
                genome_type = gp.PrimitiveTree
                
        creator.create("Individual", genome_type, fitness=creator.FitnessMulti)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=4, max_=8)
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


    def initialize(self, seed_file = None):
        if self.attempt_resume:
            print('Attempting to resume...')
            self.holy_grail = pd.read_csv(os.path.join(self.output_dir, 'out.csv'))
            metric_columns = [col for col in self.holy_grail.columns if col not in ['gen', 'hash', 'genome']]
            self.holy_grail['metrics'] = self.holy_grail[metric_columns].apply(lambda row: row.to_dict(), axis=1)
            self.holy_grail = self.holy_grail.drop(columns=metric_columns)
            print('Found holy grail (out.csv)!')
            self.gen_count = self.holy_grail['gen'].iloc[-1]
            current_pop = self.holy_grail[self.holy_grail['gen'] == self.gen_count].to_dict('records')
            self.current_deap_pop = []
            self.current_population = {}
            for genome in current_pop:
                metrics = genome.copy()
                del metrics['gen']
                del metrics['hash']
                del metrics['genome']
                self.current_population[genome['hash']] = {'genome': genome['genome'], 'metrics': metrics}
            checkpoint_path = os.path.join(self.output_dir, 'checkpoint')
            with open(os.path.join(checkpoint_path,'latest_pop.pkl'), 'rb') as f:
                self.current_deap_pop = pickle.load(f)
            print('Found latest_pop.pkl!')
            with open(os.path.join(checkpoint_path,'elites.pkl'), 'rb') as f:
                self.elite = pickle.load(f)
            print('Found elites.pkl!')
            with open(os.path.join(checkpoint_path,'hof.pkl'), 'rb') as f:
                self.hall_of_fame = pickle.load(f)
            print('Found hof.pkl!')
        else:
            self.init_pop(seed_file)
            

    def init_pop(self, seed_file = None):
        print('Initializing population...')
        seeded_individuals = []
        if seed_file is not None:
            shutil.copy(seed_file, self.output_dir)
            with open(seed_file, 'r') as seed_file:
                genomes = seed_file.readlines()
                genomes = [x[:-1] if '\n' in x else x for x in genomes]
                for genome in genomes:
                    individual = creator.Individual(CustomPrimitiveTree.from_string(genome, self.pset))
                    seeded_individuals.append(individual)
        self.current_population = {}
        pop = self.toolbox.population(n=self.initial_population_size-len(seeded_individuals))
        self.current_deap_pop = pop + seeded_individuals
        for genome in self.current_deap_pop:
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
        eval_input_path = os.path.join(self.output_dir, 'eval_inputs', f'eval_input_gen{self.gen_count}.csv')
        os.makedirs(os.path.dirname(eval_input_path), exist_ok=True)
        input_data.to_csv(eval_input_path, index=False)

        # create bash script for job file
        self.create_job_file(len(self.current_population), self.gen_count)

        # create this gen's log diretory
        os.makedirs(os.path.join(self.logs_dir, f'generation_{self.gen_count}'))

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

        fails = 0
        # read eval_gen output file
        for hash, genome in self.current_population.items():
            try:
                dataframe = pd.read_csv(f'{self.output_dir}/generation_{self.gen_count}/{hash}/metrics.csv')
                if (self.best_epoch_criteria[1].lower() == 'max'):
                    data = dataframe.nlargest(1, self.best_epoch_criteria[0]).squeeze().to_dict()
                else:
                    data = dataframe.nsmallest(1, self.best_epoch_criteria[0]).squeeze().to_dict()
                del data['epoch_num']
                # update current population metrics
                self.current_population[hash]['metrics'] = data
                # update holy grail dataframe
                self.holy_grail.loc[len(self.holy_grail.index)] = [self.gen_count, hash, genome['genome'], data]
                # set fitness of current population
                for g in self.current_deap_pop:
                    if str(g) == genome['genome']:
                        g.fitness.values = tuple([data[key] for key in self.objectives.keys()])
            except FileNotFoundError:
                # in the case the file isn't found we generate a file with bad metrics and then use that
                print(f'    Couldn\'t find individual {hash} evaluation... Assuming genome failure and assigning bad metrics')
                self.num_genome_fails += 1
                fails += 1
                test_metrics = {}
                for objective, weight in self.objectives.items():
                    test_metrics[objective] = 1000000 if weight < 0 else -1000000
                output_filename = f'{self.output_dir}/generation_{self.gen_count}/{hash}/metrics.csv'
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                with open(output_filename, 'w') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(list(test_metrics.keys()))
                    writer.writerow(list(test_metrics.values()))

                data = test_metrics
                # update current population metrics
                self.current_population[hash]['metrics'] = data
                # update holy grail dataframe
                self.holy_grail.loc[len(self.holy_grail.index)] = [self.gen_count, hash, genome['genome'], data]
                # set fitness of current population
                for g in self.current_deap_pop:
                    if str(g) == genome['genome']:
                        g.fitness.values = tuple([data[key] for key in self.objectives.keys()])
        
        self.total_evaluated_individuals += len(self.current_population)
        print(f'Generation {self.gen_count} evaluation done! Genome failures: {fails}')


    def select_parents(self, selection_pool):
        print('Selecting parents...')
        selected_parents = self.toolbox.select_parents(selection_pool)
        print('Done!')
        return selected_parents


    def update_elite_pool(self): # updates the elite pool and returns selected elites from the current population
        print('Updating elite pool...')
        self.elite_pool = self.toolbox.select_elitists(self.current_deap_pop + self.elite_pool)
        self.elite_pool_history[self.gen_count] = self.elite_pool
        print('Done!')
        return self.elite_pool
    

    def update_hof(self): # updates ParetoFront object
        print('Updating Hall of Fame...')
        self.hall_of_fame.update(self.current_deap_pop)
        self.hof_history[self.gen_count] = self.hall_of_fame
        print('Done!')
        if self.clean:
            print('Cleaning up non-pareto-optimal indivuduals...')
            hof_hashes = [self.__get_hash(str(x)) for x in self.hall_of_fame]
            to_remove = self.holy_grail[~self.holy_grail['hash'].isin(hof_hashes)]
            to_remove = to_remove[['gen', 'hash']].values.tolist()
            for gen, hash in to_remove:
                os.popen(f'rm {self.output_dir}/generation_{gen}/{hash}/best_epoch.pth')
                os.popen(f'rm {self.output_dir}/generation_{gen}/{hash}/last_epoch.pth')
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
                if (len(copies) >= 2):
                    parent1 = copies.pop(random.randrange(0, len(copies)))
                    parent2 = copies.pop(random.randrange(0, len(copies)))
                    parent_pairs.append([parent1, parent2])
                else:
                    break
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
        # store current pop, elite pool, and hof as pickle files
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint') 
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open(os.path.join(checkpoint_path,'latest_pop.pkl'), 'wb') as f:
            pickle.dump(self.current_deap_pop, f)
        with open(os.path.join(checkpoint_path,'elites.pkl'), 'wb') as f:
            pickle.dump(self.elite_pool, f)
        with open(os.path.join(checkpoint_path,'hof.pkl'), 'wb') as f:
            pickle.dump(self.hall_of_fame, f)
        # store elite pool and hof history as pickle files
        with open(os.path.join(self.output_dir,'elites_history.pkl'), 'wb') as f:
            pickle.dump(self.elite_pool_history, f)
        with open(os.path.join(self.output_dir,'hof_history.pkl'), 'wb') as f:
            pickle.dump(self.hof_history, f)
        # store holy grail
        holy_grail_expanded = self.holy_grail.join(pd.json_normalize(self.holy_grail['metrics'])).drop('metrics', axis='columns')
        holy_grail_expanded.to_csv(f'{self.output_dir}/out.csv', index=False)
        # get all entries from holy grail that share the same hashes as the elite pool members
        elites_df = holy_grail_expanded[holy_grail_expanded['hash'].isin([self.__get_hash(str(genome)) for genome in self.elite_pool])]
        elites_df.to_csv(f'{self.output_dir}/elites.csv', index=False)
        # get all entries from holy grail that share the same hashes as the hall of fame members
        hof_df = holy_grail_expanded[holy_grail_expanded['hash'].isin([self.__get_hash(str(genome)) for genome in self.hall_of_fame.items])]
        hof_df.to_csv(f'{self.output_dir}/hall_of_fame.csv', index=False)
        print('Done!')


    def step_gen(self):
        self.gen_count += 1
        self.attempt_resume = False


    def __get_hash(self, s):
        return hashlib.shake_256(s.encode()).hexdigest(5)
    

    def clear_outputs(self):
        print('Clearing old outputs and logs...')
        os.system(f'rm -rf {self.output_dir}/*')
        print('Done!')


    def create_job_file(self, num_jobs, gen_num):
        batch_script = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes={NODES}
#SBATCH -G 1
#SBATCH -x {','.join(EXCEPTED_NODES)}
#SBATCH --cpus-per-task={CORES}
#SBATCH --mem={MEM}
#SBATCH --time={JOB_TIME}
#SBATCH --output={self.logs_dir}/generation_{gen_num}/evaluation.%A.%a.log
#SBATCH --error={self.logs_dir}/generation_{gen_num}/evaluation_error.%A.%a.log
#SBATCH --array=0-{num_jobs-1}
#SBATCH --constraint="{'|'.join(GPUS)}"

module load anaconda3/2023.07
module load cuda/12.1.1

# Execute the Python script with SLURM_ARRAY_TASK_ID as argument. Script also has optional args -i and -o to specify input file and output directory respectively
conda run -n {ENV_NAME} --no-capture-output python -u {SCRIPT} $((SLURM_ARRAY_TASK_ID)) -i {self.output_dir}/eval_inputs/eval_input_gen{gen_num}.csv -o {self.output_dir}
"""
        with open(f'{JOB_NAME}.job', 'w') as fh:
            fh.write(batch_script)
    

class CustomPrimitiveTree(gp.PrimitiveTree):
    @classmethod
    def from_string(cls, string, pset):
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                expr.append(primitive)
                if isinstance(primitive, gp.Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                expr.append(gp.Terminal(token, False, type_))
        return cls(expr)
        
