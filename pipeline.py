"""
Evolutionary pipeline. Deals with evolutionary processes like selection, mating, overpopulation.
Makes calls to surrogate appropriately to train/validate and use for downselection.
"""


import copy
import csv
import hashlib
import pickle
import random
import shutil
import subprocess
import time
import numpy as np
import toml
import os
import re

import pandas as pd
from deap import creator, gp, base, tools

import primitives
from codec import Codec
from surrogates.surrogate import Surrogate
from primitive_tree import CustomPrimitiveTree
from surrogates.surrogate_eval import engine, get_val_scores
from surrogates.surrogate_dataset import build_dataset
import numpy as np
import re

# job file params
JOB_NAME = 't_evo'
NODES = 1
CORES = 8
MEM = '16GB'
JOB_TIME = '08:00:00'
SCRIPT = 'eval.py'
ENV_NAME = 'nas'
GPUS = ["V100-16GB", "V100-32GB", "L40S", "A100-40GB", "H100"]


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


class Pipeline:
    def __init__(self, output_dir, config_dir, force_wipe = False, clean = False) -> None:
        self.output_dir = output_dir
        self.force_wipe = force_wipe
        self.clean = clean
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.attempt_resume = False
        
        # init file names
        self.trust_file = os.path.join(self.output_dir, 'surrogate_trusts.csv')
        self.cls_surrogate_data_file = os.path.join(self.output_dir, 'cls_surrogate_data.csv')
        self.reg_surrogate_data_file = os.path.join(self.output_dir, 'reg_surrogate_data.csv')
        self.selected_surrogate_data_file = os.path.join(self.output_dir, 'selected_surrogate_data.csv')
        self.checkpoint_path = os.path.join(self.output_dir, 'checkpoint')
        self.holy_grail_file = os.path.join(self.output_dir, 'out.csv')
        self.elites_file = os.path.join(self.output_dir,'elites.csv')
        self.hall_of_fame_file = os.path.join(self.output_dir, 'hall_of_fame.csv')
        self.surrogate_data_file = os.path.join(self.output_dir, 'surrogate_data.csv')
        self.surrogate_data_extra_file = os.path.join(self.output_dir, 'surrogate_data_extra.csv')

        self.latest_pop_file = os.path.join(self.checkpoint_path,'latest_pop.pkl')
        self.elites_checkpoint_file = os.path.join(self.checkpoint_path,'elites.pkl')
        self.hof_file = os.path.join(self.checkpoint_path,'hof.pkl')
        self.reg_genome_scaler_file = os.path.join(self.checkpoint_path,'reg_genome_scaler.pkl')
        self.cls_genome_scaler_file = os.path.join(self.checkpoint_path,'cls_genome_scaler.pkl')
        self.sub_surrogate_selection_file = os.path.join(self.checkpoint_path,'sub_surrogate_selection.pkl')
        self.elites_history_file = os.path.join(self.output_dir,'elites_history.pkl')
        self.hof_history_file = os.path.join(self.output_dir,'hof_history.pkl')

        # Check if output location already exists
        if os.path.exists(self.output_dir):
            if not self.force_wipe:
                self.attempt_resume = True
            else:
                self.clear_outputs()
                os.makedirs(self.logs_dir) #, exist_ok=True)
                shutil.copy(config_dir, output_dir)
        else:
            os.makedirs(self.output_dir)
            os.makedirs(self.logs_dir)
            shutil.copy(config_dir, output_dir)

        # Begin by loading config attributes
        configs = toml.load(config_dir)
        pipeline_config = configs["pipeline"]
        codec_config = configs["codec"]
        surrogate_config = configs["surrogate"]
        self.surrogate_config = surrogate_config

        self.initial_population_size = pipeline_config['initial_population_size']
        self.population_size = pipeline_config['population_size']
        self.unsustainable_population_size = pipeline_config['unsustainable_population_size']
        self.num_parents = pipeline_config['num_parents']
        self.crossovers = pipeline_config['crossovers']
        self.mutations = pipeline_config['mutations']
        self.surrogate_enabled = pipeline_config['surrogate_enabled']
        self.surrogate_temp_dataset_path = os.path.join(self.output_dir, 'temp_surrogate_datasets')
        self.surrogate_weights_dir = os.path.join(output_dir, 'surrogate_weights')
        self.surrogate_metrics = surrogate_config['surrogate_metrics']
        self.sub_surrogate_sel_strat = surrogate_config['sub_surrogate_sel_strat']
        if 'pretrained_dir' in surrogate_config and 'pretrained' in surrogate_config:
            self.surrogate_pretrained = surrogate_config['pretrained']
            self.surrogate_pretrained_dir = surrogate_config['pretrained_dir']
        else:
            self.surrogate_pretrained = False
            self.surrogate_pretrained_dir = None
        self.objectives = pipeline_config['objectives']
        self.selection_method_trusted = pipeline_config['selection_method_trusted']
        self.selection_method_untrusted = pipeline_config['selection_method_untrusted']
        self.selection_method_parents = pipeline_config['selection_method_parents']
        self.selection_method_elite_pool = pipeline_config['selection_method_elite_pool']
        self.max_elite_pool = pipeline_config['max_elite_pool']
        self.train_pool_source = pipeline_config['train_pool_source']
        self.trust_pool_source = pipeline_config['trust_pool_source']
        self.best_epoch_criteria = pipeline_config['best_epoch_criteria']
        self.num_gens_ssi = pipeline_config['num_gens_ssi']

        # Other useful attributes
        self.holy_grail = pd.DataFrame(columns=['gen', 'hash', 'genome', 'metrics']) # data regarding every evaluated individual; metrics are from best epoch since all other metric data is already stored by eval_script
        self.all_cls_surrogate_data = pd.DataFrame() # data regarding all classifier surrogates
        self.all_reg_surrogate_data = pd.DataFrame() # data regarding all regressor surrogates
        self.selected_surrogate_data = pd.DataFrame() # chosen surrogates per generation with their trusts.
        self.cls_surrogate_pretrained_data = None
        self.reg_surrogate_pretrained_data = None
        self.current_population = {} # dict of genomes associated with their metrics with hash as key
        self.current_deap_pop = [] # list of deap individuals representing the current population; no other info
        self.elite_pool = [] # list of deap individuals in the elite pool
        self.elite_pool_history = {} # dict keeping track of elite pool through generations
        self.hall_of_fame = tools.ParetoFront() # hall of fame as a ParetoFront object
        self.hof_history = {} # dict keeping track of hall of fame through generations
        self.codec = Codec(0, genome_encoding_strat=codec_config['genome_encoding_strat']) # only used for getting hash, so initialization values don't matter
        self.surrogate = Surrogate(config_dir, self.surrogate_weights_dir) # Surrogate class to be defined
        self.reg_genome_scaler = None # scaler used to transform genomes on regression training and inference
        self.cls_genome_scaler = None # scaler used to transform genomes on classification training and inference
        self.sub_surrogates = [0] * (len(self.objectives) + 1) # list of sub-surrogate indices to use
        self.surrogate_trusts = {"gen":[], "cls_trust":[], "reg_trust":[]} # list to keep track of surrogate trust over the generations (SHOULD BE SAVED)
        self.pset = primitives.pset # primitive set
        self.gen_count = 1
        self.num_genome_fails = 0
        self.total_evaluated_individuals = 0

        assert self.sub_surrogate_sel_strat in ['mse', 'trust']

        # Setting up pipeline
        ensure_deap_classes(self.objectives, codec_config)
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
            self.holy_grail = pd.read_csv(self.holy_grail_file)
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
            with open(self.latest_pop_file, 'rb') as f:
                self.current_deap_pop = pickle.load(f)
            print('Found latest_pop.pkl!')
            with open(self.elites_checkpoint_file, 'rb') as f:
                self.elite = pickle.load(f)
            print('Found elites.pkl!')
            with open(self.hof_file, 'rb') as f:
                self.hall_of_fame = pickle.load(f)
            print('Found hof.pkl!')
            if self.surrogate_enabled:
                with open(self.reg_genome_scaler_file, 'rb') as f:
                    self.reg_genome_scaler = pickle.load(f)
                print('Found reg_genome_scaler.pkl!')
                with open(self.cls_genome_scaler_file, 'rb') as f:
                    self.cls_genome_scaler = pickle.load(f)
                print('Found cls_genome_scaler.pkl!')
                with open(self.sub_surrogate_selection_file, 'rb') as f:
                    self.sub_surrogates = pickle.load(f)
                print('Found sub_surrogate_selection.pkl!')
                try:
                    self.all_cls_surrogate_data = pd.read_csv(self.cls_surrogate_data_file)
                    self.all_reg_surrogate_data = pd.read_csv(self.reg_surrogate_data_file)
                    self.selected_surrogate_data = pd.read_csv(self.selected_surrogate_data_file)
                except pd.errors.EmptyDataError:
                    pass 
                if os.path.exists(self.trust_file):
                    df = pd.read_csv(self.trust_file)
                    self.surrogate_trusts = {key: list(df[key]) for key in df if "unnamed" not in key.lower()}
                    print('Found surrogate_trusts.csv!')
                else:
                    self.surrogate_trusts["gen"] = list(range(1, self.gen_count+1))
                    self.surrogate_trusts["cls_trust"] = [0] * self.gen_count
                    self.surrogate_trusts["reg_trust"] = [0] * self.gen_count
                    print('Found no surrogate trust, assuming they are all 0')
        else:
            os.makedirs(self.surrogate_weights_dir)
            self.init_pop(seed_file)

        # if surrogate should be "pretrained", load existing data
        if self.surrogate_pretrained:
            self.cls_surrogate_pretrained_data = pd.read_pickle(f'{self.surrogate_pretrained_dir}/surr_evolution_gen9_cls_train.pkl')
            self.reg_surrogate_pretrained_data = pd.read_pickle(f'{self.surrogate_pretrained_dir}/surr_evolution_gen9_reg_train.pkl')
        else:
            self.cls_surrogate_pretrained_data = pd.DataFrame()
            self.reg_surrogate_pretrained_data = pd.DataFrame()
            

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
        generation_string = f'generation_{self.gen_count}'
        log_folder = os.path.join(self.logs_dir, generation_string)
        os.system(f'rm -rf {log_folder}')
        os.makedirs(log_folder)

        # dispatch job
        print('    Dispatching jobs...')
        sbatch_result = os.popen(f"sbatch {JOB_NAME}_{self.gen_count}.job" ).read()
        
        #parse sbatch_result for job id:
        match = re.search(r'Submitted batch job (\d+)', sbatch_result)

        if match:
            job_id = match.group(1)
            # print(f"Job ID: {job_id}")
        else:
            print("Failed to submit job or parse job id" )
            print(sbatch_result)
        
        if self.surrogate_enabled:
            print('    Preparing surrogate...')
            all_subsurrogate_metrics = self.prepare_surrogate()
                        
        print('    Waiting for jobs...')
        while True:
            time.sleep(120)
            p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE)
            text = p.stdout.read().decode('utf-8')
            jobs = text.split('\n')[1:-1]
            if len(jobs) == 0:
                # print("SQUEUE COMMAND OUTPUT:")
                # print(text)
                break
        print('    Done!')

        fails = 0
        # read eval_gen output file
        for hash, genome in self.current_population.items():
            try:
                path = f'{self.output_dir}/generation_{self.gen_count}/{hash}/metrics.csv'
                #print(path)
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
            except FileNotFoundError as e:
                # import pdb; pdb.set_trace()
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


    def add_metrics_to_dfs(self, output_dict):
        classifier_columns = ['gen', 'model'] + list(next(iter(output_dict['classifiers'].values())).keys())
        regressor_columns = ['gen', 'model'] + list(next(iter(output_dict['regressors'].values())).keys())
        classifiers_df = pd.DataFrame(columns=classifier_columns)
        regressors_df = pd.DataFrame(columns=regressor_columns)
        for model_name, metrics in output_dict['classifiers'].items():
            row = {'gen': self.gen_count, 'model': model_name, **metrics}
            classifiers_df.loc[len(classifiers_df)] = row
        for model_name, metrics in output_dict['regressors'].items():
            row = {'gen': self.gen_count, 'model': model_name, **metrics}
            regressors_df.loc[len(regressors_df)] = row
        
        selected_surr_dict = {}
        selected_surr_dict['gen'] = self.gen_count
        selected_surr_dict['cls_model'] = list(output_dict['classifiers'].keys())[self.sub_surrogates[0]]
        reg_dicts = list(output_dict['regressors'].keys())
        for i, model_idx in enumerate(self.sub_surrogates[1:]):
            selected_surr_dict[list(self.objectives.keys())[i]+'_reg_model'] = reg_dicts[model_idx]
        selected_surr_dict['cls_trust'] = self.surrogate.cls_trust
        selected_surr_dict['reg_trust'] = self.surrogate.reg_trust        
        
        self.selected_surrogate_data = pd.concat([self.selected_surrogate_data, pd.DataFrame([selected_surr_dict])], ignore_index=True)
        self.all_cls_surrogate_data = pd.concat([self.all_cls_surrogate_data, classifiers_df], ignore_index=True)
        self.all_reg_surrogate_data = pd.concat([self.all_reg_surrogate_data, regressors_df], ignore_index=True)
        
        print(self.all_cls_surrogate_data)
        print(self.all_reg_surrogate_data)
        print(self.selected_surrogate_data)



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
    
    
    # trains the surrogate (all sub-surrogates) and gets eval scores which are used to calculate a trustworthiness
    # surrogate weights are stored to be used for inference when downselecting
    def prepare_surrogate(self):
        seen_gens = list(range(1, self.gen_count))
        if self.gen_count == 1:
            return None
        print('    Building surrogate train and val datasets...')
        # implement growing sliding window till gen 7 (then use prev 5 gens as val and everything before that as train)
        name = 'surr_evolution'
        if self.gen_count == 2: # use train val split from gen 1 at gen 2
            build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=0.2, include_only=[1])
            reg_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_train.pkl')
            reg_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
            # reg_subset_val_df = reg_val_df
            cls_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_train.pkl')
            cls_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
            # cls_subset_val_df = cls_val_df
        elif self.gen_count < 7: # grows here
            build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=0, include_only=[1])
            reg_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_train.pkl')
            cls_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_train.pkl')
            build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[1:])
            reg_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
            cls_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
            # build_dataset(name, os.path.join(self.output_dir, 'out.csv'), self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[-1:])
            # reg_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
            # cls_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
        else: # slides here
            build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=0, include_only=seen_gens[:-5])
            reg_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_train.pkl')
            cls_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_train.pkl')
            build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[-5:])
            reg_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
            cls_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
            # build_dataset(name, os.path.join(self.output_dir, 'out.csv'), self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[-1:])
            # reg_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
            # cls_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
            

        # concatenate online datasets wit pretrained datasets
        reg_train_df = pd.concat([reg_train_df, self.reg_surrogate_pretrained_data], axis=0)
        cls_train_df = pd.concat([cls_train_df, self.cls_surrogate_pretrained_data], axis=0)
        
        print('++++++++++++++++++++++++')
        print('reg train size:', reg_train_df.shape)
        print('reg val size:', reg_val_df.shape)
        print('cls train size:', cls_train_df.shape)
        print('cls val size:', cls_val_df.shape)
        print('++++++++++++++++++++++++')
        
        # check if there's enough data to train regressors
        train_reg = True
        if len(reg_val_df) < self.surrogate_config['surrogate_batch_size']*self.surrogate_config['min_batch_in_val_data']:
            train_reg = False
            print('----Warning: not enough valid data for regressors... skipping surrogate preparation----')
        #first call train function and receive the scores, then find the best model for each objective plus cls, then calculate their trust
        print('    Training surrogate ensemble...')
        if len(reg_val_df) < self.surrogate_config['surrogate_batch_size']*self.surrogate_config['min_batch_in_val_data']:
            print('    ----Warning: not enough valid data for regressors... skipping surrogate preparation----')
            return None
        # print(f'     Regression validation data shape: {reg_val_df.shape}      {reg_val_df.head()}')
        scores, cls_genome_scaler, reg_genome_scaler = self.surrogate.train(cls_train_df, cls_val_df, reg_train_df, reg_val_df, train_reg=train_reg)
            
        print('    Selecting best sub-surrogates...')
        sub_surrogates = []
        # finding best classifier
        cls_trust = 0
        max_cls_model = ''
        for key, val in scores['classifiers'].items():
            if val['acc'] > cls_trust:
                cls_trust = val['acc']
                max_cls_model = key
        cls_to_dict = {d['name']: d for d in self.surrogate.classifier_models}
        max_cls_model_idx = list(cls_to_dict.keys()).index(max_cls_model)
        sub_surrogates.append(max_cls_model_idx)
        # print(f'    Selected {max_cls_model} as classifier')
        
        if train_reg:
            if self.sub_surrogate_sel_strat == 'trust':
                # finding best regressor
                result = self.surrogate.optimize_trust(cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df)
                reg_trust = result[0]
                sub_surrogates += result[1]
            else:
                #get my model indices, tack on cls in front, pass to calc trust to get my own trusts, pass below
                reg_indices = self.get_reg_indices(scores)
                sub_surrogates.extend(reg_indices)
                cls_trust, reg_trust = self.surrogate.calc_trust(sub_surrogates, cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df)
        else:
            sub_surrogates.extend(self.sub_surrogates[1:])
    
            
        self.cls_genome_scaler = cls_genome_scaler
        self.reg_genome_scaler = reg_genome_scaler
        self.surrogate.cls_trust, self.surrogate.reg_trust = cls_trust, reg_trust
        self.sub_surrogates = sub_surrogates

        # log trusts
        # NOTE this should work but may need to log trusts as well
        if scores is not None:
                self.add_metrics_to_dfs(scores)
        print('    Done!')
        return scores
    

    def get_reg_indices(self, scores):
        best_models = {}
        for objective, direction in self.objectives.items():
            objective = 'mse_' + objective
            best_models[objective.replace('epoch_', '')] = {'model': '', 'score': np.inf}

        name_to_dict = {d['name']: d for d in self.surrogate.models}
        indices = []
        for objective in self.objectives.keys():
            loss_objective = 'mse_' + objective
            loss_objective = loss_objective.replace('epoch_', '')

            indices.append(self.surrogate_metrics.index(loss_objective))
        print(indices)

        for reg_key, reg_val in scores['regressors'].items():
            for idx, objective in zip(indices, best_models.keys()):
                if idx in name_to_dict[reg_key]['validation_subset']:
                    if reg_val[objective] < best_models[objective]['score']:
                        best_models[objective]['model'] = reg_key
                        best_models[objective]['score'] = reg_val[objective]

        condensed = []
        for name, model in best_models.items():
            condensed.append(list(name_to_dict.keys()).index(model['model']))
        
        return condensed
    

    def downselect(self, unsustainable_pop):
        print('Downselecting...')
        if self.surrogate_enabled and self.gen_count != 1:
            unsustainable_pop_copy = copy.deepcopy(unsustainable_pop)

            # get surrogate inferred fitnesses using classification and regression
            invalid_deap, valid_deap = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(unsustainable_pop.values()))

            # define sizes for stages of selection
            # selection where we trust surrogate classification AND regression values
            num_tw_select = int(self.surrogate.reg_trust * self.surrogate.cls_trust * self.population_size)
            # selection where we trust surrogate classification AND NOT regression
            num_utw_select = int((1 - self.surrogate.reg_trust) * self.surrogate.cls_trust * self.population_size)
            # selection where we don't trust any of the surrogate's predictions
            num_rand_other_select = (self.population_size - num_tw_select - num_utw_select) 
            
            # create downselect function for trustwrothy surrogate ratio
            if self.selection_method_trusted == 'NSGA2':
                self.toolbox.register("downselect", tools.selNSGA2, k = num_tw_select)
            # TODO add other if statements for other selections strategies

            if len(valid_deap) < (self.population_size * self.surrogate.cls_trust):
                downselect_pool = invalid_deap + valid_deap
            else:
                downselect_pool = valid_deap
            
            # downselect using surrogate
            downselected = self.toolbox.downselect(downselect_pool)
            
            # create new population dict
            new_pop = {}
            new_deap_pop = []

            # add trustworthy downselected individuals to new population
            # remove added individuals from valid_deap and unsustainable population
            for individual in downselected:
                hash = self.__get_hash(str(individual))
                new_pop[hash] = {'genome': str(unsustainable_pop_copy[hash]), 'metrics': None}
                new_deap_pop.append(individual)
                # delete surrogate selected individuals to avoid duplicates being selected by other technique
                del unsustainable_pop_copy[hash]
                downselect_pool.remove(individual)

            # randomly select from valid deap individuals    
            if (self.selection_method_untrusted.lower() == 'random'): # choose randomly
                to_utw_select = random.sample((valid_deap), num_utw_select)
                for individual in to_utw_select:
                    hash = self.__get_hash(str(individual))
                    new_pop[hash] = {'genome': str(unsustainable_pop_copy[hash]), 'metrics': None}
                    new_deap_pop.append(individual)
                    del unsustainable_pop_copy[hash]
                    downselect_pool.remove(individual)

                other_rand_hashes = random.sample(list(unsustainable_pop_copy.keys()), num_rand_other_select)
                for hash in other_rand_hashes:
                    new_pop[hash] = {'genome': str(unsustainable_pop_copy[hash]), 'metrics': None}
                    new_deap_pop.append(unsustainable_pop_copy[hash])
           
            # set new current pop and current deap pop
            self.current_population = new_pop
            self.current_deap_pop = new_deap_pop
            
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

    
    def simulated_surrogate_injection(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        curr_deap_pop = list(curr_pop.values())
        for i in range(self.num_gens_ssi):
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, curr_deap_pop)
            parents = self.select_parents(valid) 
            unsustainable_pop = self.overpopulate(parents)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.population_size)
                new_pop = {}
                new_deap_pop = []
                for hash in new_hashes:
                    new_deap_pop.append(unsustainable_pop[hash])
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop
                curr_deap_pop = new_deap_pop
        return curr_pop, new_deap_pop


    def log_info(self):
        print('Logging data...')
        # store current pop, elite pool, and hof as pickle files
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        with open(self.latest_pop_file, 'wb') as f:
            pickle.dump(self.current_deap_pop, f)
        with open(self.elites_checkpoint_file, 'wb') as f:
            pickle.dump(self.elite_pool, f)
        with open(self.hof_file, 'wb') as f:
            pickle.dump(self.hall_of_fame, f)
        if self.surrogate_enabled:
            # store current genome scalers to be used for downselect
            with open(self.cls_genome_scaler_file, 'wb') as f:
                pickle.dump(self.cls_genome_scaler, f)
            with open(self.reg_genome_scaler_file, 'wb') as f:
                pickle.dump(self.reg_genome_scaler, f)
            # store current selected sub-surrogates
            with open(self.sub_surrogate_selection_file, 'wb') as f:
                pickle.dump(self.sub_surrogates, f)
        # store elite pool and hof history as pickle files
        with open(self.elites_history_file, 'wb') as f:
            pickle.dump(self.elite_pool_history, f)
        with open(self.hof_history_file, 'wb') as f:
            pickle.dump(self.hof_history, f)
        holy_grail_expanded = self.holy_grail.join(pd.json_normalize(self.holy_grail['metrics'])).drop('metrics', axis='columns')
        holy_grail_expanded.to_csv(self.holy_grail_file, index=False)
        # get all entries from holy grail that share the same hashes as the elite pool members
        elites_df = holy_grail_expanded[holy_grail_expanded['hash'].isin([self.__get_hash(str(genome)) for genome in self.elite_pool])]
        elites_df.to_csv(self.elites_file, index=False)
        # get all entries from holy grail that share the same hashes as the hall of fame members
        hof_df = holy_grail_expanded[holy_grail_expanded['hash'].isin([self.__get_hash(str(genome)) for genome in self.hall_of_fame.items])]
        hof_df.to_csv(self.hall_of_fame_file, index=False)
        if self.surrogate_enabled:
            # write surrogate information to file
            self.all_cls_surrogate_data.to_csv(self.cls_surrogate_data_file, index=False)
            self.all_reg_surrogate_data.to_csv(self.reg_surrogate_data_file, index=False)
            self.selected_surrogate_data.to_csv(self.selected_surrogate_data_file, index=False)                
        print('Done!')


    def step_gen(self):
        self.gen_count += 1
        self.attempt_resume = False


    def __get_hash(self, s):
        layer_list = self.codec.get_layer_list(s)
        return hashlib.shake_256(str(layer_list).encode()).hexdigest(5)
    

    def clear_outputs(self):
        print('Clearing old outputs and logs...')
        os.system(f'rm -rf {self.output_dir}/*')
        print('Done!')

# alternate solution to waiting for generations to finish is have --job-name={JOB_NAME}_{gen_num} and check for same job-name in wait loop
    def create_job_file(self, num_jobs, gen_num):
        batch_script = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}_{gen_num}
#SBATCH --nodes={NODES}
#SBATCH -G 1
#SBATCH --cpus-per-task={CORES}
#SBATCH --mem={MEM}
#SBATCH --time={JOB_TIME}
#SBATCH --output={self.logs_dir}/generation_{gen_num}/evaluation.%A.%a.log
#SBATCH --error={self.logs_dir}/generation_{gen_num}/evaluation_error.%A.%a.log
#SBATCH --array=0-{num_jobs-1}
#SBATCH --constraint={'|'.join(GPUS)}

module load anaconda3/2023.03
module load anaconda3/2023.03
module load cuda/12.1.1
mkdir -p {self.logs_dir}/generation_{gen_num}

# Execute the Python script with SLURM_ARRAY_TASK_ID as argument. Script also has optional args -i and -o to specify input file and output directory respectively
conda run -n {ENV_NAME} --no-capture-output python -u {SCRIPT} $SLURM_ARRAY_TASK_ID -i {self.output_dir}/eval_inputs/eval_input_gen{gen_num}.csv -o {self.output_dir}
"""
        with open(f'{JOB_NAME}_{gen_num}.job', 'w') as fh:
            fh.write(batch_script)