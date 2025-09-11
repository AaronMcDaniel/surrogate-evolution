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

import pandas as pd
from deap import creator, gp, base, tools

import primitives
from codec import Codec
from surrogates.surrogate import Surrogate
from primitive_tree import CustomPrimitiveTree, type_fixed_mut
from surrogates.surrogate_eval import engine, get_val_scores
from surrogates.surrogate_dataset import build_dataset
import numpy as np
import re
from custom_selection import dbea_selection, lexicase_selection
import surrogates.surrogate_dataset as sd
from torch.utils.data import DataLoader


# job file params
JOB_NAME = 's_evo'
NODES = 1
CORES = 8
MEM = '24GB'
JOB_TIME = '08:00:00'
SCRIPT = 'eval.py'
ENV_NAME = 'nas'
GPUS = ["V100-16GB", "V100-32GB", "L40S", "A100-40GB", "H100", "A40", "H200"]
# ALLOWED_NODES = ['ice108', 'ice107', 'ice110', 'ice143', 'ice144', 'ice145', 'ice151', 'ice162', 'ice163', 'ice164', 'ice165', 'ice175', 'ice176', 'ice179', 'ice183', 'ice185', 'ice191', 'ice192', 'ice193']


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
        global JOB_NAME
        JOB_NAME = f'{JOB_NAME}_{os.path.basename(output_dir)}'
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
        self.ssi_holy_grail_file = os.path.join(self.output_dir, 'outssi.csv')
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
                os.makedirs(self.logs_dir)
                shutil.copy(config_dir, os.path.join(output_dir, "conf.toml"))
        else:
            os.makedirs(self.output_dir)
            os.makedirs(self.logs_dir)
            shutil.copy(config_dir, os.path.join(output_dir, "conf.toml"))

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
        if 'preprocess' in surrogate_config:
            self.preprocess = surrogate_config['preprocess']
            self.preprocess_batch_size = surrogate_config['preprocess_batch_size']
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
        self.ssi_unsus_pop_size = pipeline_config['ssi_unsustainable_population_size']
        self.ssi_sus_pop_size = pipeline_config['ssi_sustainable_population_size']
        self.num_parents_ssi = pipeline_config['num_parents_ssi']
        if 'ssi_population_percentage' in pipeline_config:
            self.ssi_population_percentage = pipeline_config['ssi_population_percentage']
        else:
            self.ssi_population_percentage = 1
        self.surrogate_in_final_downselect = pipeline_config['surrogate_in_final_downselect']

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
            if mutation == "type_fixed_mut":
                self.toolbox.register("type_fixed_mut", eval("type_fixed_mut"))
            else:
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
                self.elite_pool = pickle.load(f)
            print('Found elites.pkl!')
            with open(self.hof_file, 'rb') as f:
                self.hall_of_fame = pickle.load(f)
            print('Found hof.pkl!')
            if self.surrogate_enabled:
                if os.path.exists(self.reg_genome_scaler_file):
                    with open(self.reg_genome_scaler_file, 'rb') as f:
                        self.reg_genome_scaler = pickle.load(f)
                    print('Found reg_genome_scaler.pkl!')
                if os.path.exists(self.cls_genome_scaler_file):
                    with open(self.cls_genome_scaler_file, 'rb') as f:
                        self.cls_genome_scaler = pickle.load(f)
                    print('Found cls_genome_scaler.pkl!')
                if os.path.exists(self.sub_surrogate_selection_file):
                    with open(self.sub_surrogate_selection_file, 'rb') as f:
                        self.sub_surrogates = pickle.load(f)
                    print('Found sub_surrogate_selection.pkl!')
                try:
                    if os.path.exists(self.cls_surrogate_data_file):
                        self.all_cls_surrogate_data = pd.read_csv(self.cls_surrogate_data_file)
                    if os.path.exists(self.reg_surrogate_data_file):
                        self.all_reg_surrogate_data = pd.read_csv(self.reg_surrogate_data_file)
                    if os.path.exists(self.selected_surrogate_data_file):
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

        print("Output dir:", self.output_dir, flush=True)

        # if surrogate should be "pretrained", load existing data
        if self.surrogate_pretrained:
            self.cls_surrogate_pretrained_data = pd.read_pickle(f'{self.surrogate_pretrained_dir}/pretrain_cls_train.pkl')
            self.reg_surrogate_pretrained_data = pd.read_pickle(f'{self.surrogate_pretrained_dir}/pretrain_reg_train.pkl')
        else:
            self.cls_surrogate_pretrained_data = pd.DataFrame()
            self.reg_surrogate_pretrained_data = pd.DataFrame()
            

    def init_pop(self, seed_file = None):
        print('Initializing population...')
        seeded_individuals = []
        if seed_file is not None and seed_file and os.path.exists(seed_file):
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
        # os.popen(f"sbatch {JOB_NAME}_{self.gen_count}.job" )
        sbatch_result = os.popen(f"sbatch {JOB_NAME}_{self.gen_count}.job" ).read()
        
        #parse sbatch_result for job id:
        match = re.search(r'Submitted batch job (\d+)', sbatch_result)
        if match:
            job_id = match.group(1)
            print(f"Job ID: {job_id}")
        else:
            print("Failed to submit job or parse job id" )
            print(sbatch_result)
        
        if self.surrogate_enabled:
            print('    Preparing surrogate...')
            try:
                all_subsurrogate_metrics = self.prepare_surrogate()
            except Exception as e:
                print("Error occurred while preparing surrogate. Likely due to training data being empty.")
                all_subsurrogate_metrics = None

        print('    Waiting for jobs...')
        # wait for job to finish
        while True:
            time.sleep(300)
            # p = subprocess.Popen(['squeue', '-n', JOB_NAME], stdout=subprocess.PIPE)
            p = subprocess.Popen(['squeue', '-n', f'{JOB_NAME}_{self.gen_count}'], stdout=subprocess.PIPE)
            text = p.stdout.read().decode('utf-8')
            jobs = text.split('\n')[1:-1]
            if len(jobs) == 0:
                print(f'JOBS FINISHED{text}')
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


    def overpopulate(self, mating_pool, ssi=False, custom_pop_size=None): # mating pool is selected_parents + elite pool
        print('Overpopulating...')
        new_pop = {}
        # repeat till target overpopulation size is met
        unsus_pop_size = self.unsustainable_population_size if not ssi else self.ssi_unsus_pop_size
        if custom_pop_size is not None:
            unsus_pop_size = custom_pop_size
        while len(new_pop) < unsus_pop_size:
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
                        if (len(new_pop) == unsus_pop_size):
                            print('Done!')
                            return new_pop
                except:
                    continue
        print('Done!')
        return new_pop

    def serialize_surrogate_training_inputs(self, cls_train_df, cls_val_df, reg_train_df, reg_val_df, train_reg):
        """Serialize surrogate training inputs to disk for batch processing."""
        train_data_dir = os.path.join(self.output_dir, 'surrogate_train_data')
        os.makedirs(train_data_dir, exist_ok=True)
        
        # Save DataFrames to disk
        cls_train_df.to_pickle(os.path.join(train_data_dir, f'cls_train.pkl'))
        cls_val_df.to_pickle(os.path.join(train_data_dir, f'cls_val.pkl'))
        reg_train_df.to_pickle(os.path.join(train_data_dir, f'reg_train.pkl'))
        reg_val_df.to_pickle(os.path.join(train_data_dir, f'reg_val.pkl'))
        
        # Save training configuration
        config = {
            'gen': self.gen_count,
            'train_reg': train_reg,
            'preprocess': self.preprocess,
            'preprocess_batch_size': self.preprocess_batch_size
        }
        with open(os.path.join(train_data_dir, f'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        return train_data_dir

    def wait_for_surrogate_training_job(self, job_id):
        """Wait for the surrogate training job to complete and return its results."""
        print('    Waiting for surrogate training job to complete...')
        print(time.time(), flush=True)
        while True:
            time.sleep(30)  # Check status every 30 seconds
            print(f"Checking squeue -j {job_id}", flush=True)
            p = subprocess.Popen(['squeue', '-j', job_id], stdout=subprocess.PIPE)
            text = p.stdout.read().decode('utf-8')
            print("squeue response:\n", text, flush=True)
            jobs = text.split('\n')[1:-1]
            if len(jobs) == 0:  # Only header line remains
                print('    Training job completed!')
                break
        print(time.time(), flush=True)
        
        # Load and return results
        results_path = os.path.join(self.output_dir, 'surrogate_train_data', f'surrogate_results.pkl')
        
        # Wait a bit for filesystem to catch up
        max_wait = 60  # Maximum wait time in seconds
        wait_time = 0
        while not os.path.exists(results_path) and wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
        
        if not os.path.exists(results_path):
            print('    Error: Training results file not found')
            return None, None, None
        
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        return results['scores'], results['cls_genome_scaler'], results['reg_genome_scaler']
    
    
    # trains the surrogate (all sub-surrogates) and gets eval scores which are used to calculate a trustworthiness
    # surrogate weights are stored to be used for inference when downselecting
    def prepare_surrogate(self, rebuild=True):
        seen_gens = list(range(1, self.gen_count))
        if self.gen_count == 1:
            return None
        # if self.surrogate_enabled:
        print('    Building surrogate train and val datasets...')
        # implement growing sliding window till gen 7 (then use prev 5 gens as val and everything before that as train)
        name = 'surr_evolution'

        try:
            if self.gen_count == 2: # use train val split from gen 1 at gen 2
                if rebuild:
                    build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=0.2, include_only=[1])
                reg_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_train.pkl')
                reg_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
                # reg_subset_val_df = reg_val_df
                cls_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_train.pkl')
                cls_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
                # cls_subset_val_df = cls_val_df
            elif self.gen_count < 7: # grows here
                if rebuild:
                    build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=0, include_only=[1])
                reg_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_train.pkl')
                cls_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_train.pkl')
                if rebuild:
                    build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[1:])
                reg_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
                cls_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
                # build_dataset(name, os.path.join(self.output_dir, 'out.csv'), self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[-1:])
                # reg_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
                # cls_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
            else: # slides here
                if rebuild:
                    build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=0, include_only=seen_gens[:-5])
                reg_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_train.pkl')
                cls_train_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_train.pkl')
                if rebuild:
                    build_dataset(name, self.holy_grail_file, self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[-5:])
                reg_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
                cls_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
                # build_dataset(name, os.path.join(self.output_dir, 'out.csv'), self.output_dir, self.surrogate_temp_dataset_path, val_ratio=1, include_only=seen_gens[-1:])
                # reg_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_reg_val.pkl')
                # cls_subset_val_df = pd.read_pickle(f'{self.surrogate_temp_dataset_path}/{name}_cls_val.pkl')
        except FileNotFoundError as e:
            print('    ----Warning: no surrogate training data found. failed to prepare surrogate----')
            return None
            

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
        
        
        train_data_dir = self.serialize_surrogate_training_inputs(      # Serialize training inputs
            cls_train_df, cls_val_df, reg_train_df, reg_val_df, train_reg
        )
        
        job_file = self.create_surrogate_train_job(self.gen_count)      # Create job file
        
        print('    Submitting surrogate training job to GPU node...')   # Submit job
        result = os.popen(f"sbatch {job_file}").read()
        match = re.search(r'Submitted batch job (\d+)', result)
        
        if match:
            job_id = match.group(1)
            print(f"    Surrogate training job submitted with ID: {job_id}")
            scores, cls_genome_scaler, reg_genome_scaler = self.wait_for_surrogate_training_job(job_id)
        else:
            print("    Failed to submit surrogate training job")
            return None
        
        # Reload train/val dataframes if they were preprocessed
        if self.preprocess:
            cls_train_df = pd.read_pickle(os.path.join(train_data_dir, 'cls_train.pkl'))
            cls_val_df = pd.read_pickle(os.path.join(train_data_dir, 'cls_val.pkl'))
            reg_train_df = pd.read_pickle(os.path.join(train_data_dir, 'reg_train.pkl'))
            reg_val_df = pd.read_pickle(os.path.join(train_data_dir, 'reg_val.pkl'))

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
    

    def downselect(self, unsustainable_pop, fully_trust_surrogate=False, for_ssi=False, custom_population_size=None):
        population_size = self.population_size
        if custom_population_size:
            population_size = custom_population_size
        print('Downselecting...')
        if self.surrogate_enabled and (self.surrogate_in_final_downselect or for_ssi) and self.gen_count != 1 and os.listdir(self.surrogate_weights_dir):
            unsustainable_pop_copy = copy.deepcopy(unsustainable_pop)

            # get surrogate inferred fitnesses using classification and regression
            invalid_deap, valid_deap = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(unsustainable_pop.values()))

            # define sizes for stages of selection
            # selection where we trust surrogate classification AND regression values
            num_tw_select = int(self.surrogate.reg_trust * self.surrogate.cls_trust * population_size)
            # selection where we trust surrogate classification AND NOT regression
            num_utw_select = int((1 - self.surrogate.reg_trust) * self.surrogate.cls_trust * population_size)
            # selection where we don't trust any of the surrogate's predictions
            num_rand_other_select = (population_size - num_tw_select - num_utw_select) 
            
            # create downselect function for trustwrothy surrogate ratio
            if self.selection_method_trusted == 'NSGA2':
                if fully_trust_surrogate:
                    self.toolbox.register("downselect", tools.selNSGA2, k = population_size)
                else:
                    self.toolbox.register("downselect", tools.selNSGA2, k = num_tw_select)
            # TODO add other if statements for other selections strategies

            if len(valid_deap) < (population_size * self.surrogate.cls_trust):
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

            if not fully_trust_surrogate:
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
                new_hashes = None
                if len(unsustainable_pop) == self.population_size:
                    print("Entered preservation case", flush=True)
                    new_hashes = list(unsustainable_pop.keys())
                else:
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
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi):
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                new_pop = {}
                for hash in new_hashes:
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop

    def simulated_surrogate_injection_ablation(self, curr_pop, override_fitnesses: bool,
                        downselect_incoming_population: bool, normal_unsustainable_population_size: bool,
                        mix_elites: bool, old_downselect: bool, partitioned_population: bool):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection')
        self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents_ssi)
        self.toolbox.register("select_elitists", tools.selSPEA2, k = 10)
        elite_list = []
        
        for i in range(self.num_gens_ssi):
            print("Len of cur pop", len(curr_pop), flush=True)
            valid = None
            if i > 0 or override_fitnesses:
                _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            else:
                valid = list(curr_pop.values())  
            self.save_ssi_metrics(i, valid)
            
            parents = None
            if len(valid) != self.num_parents or downselect_incoming_population:
                parents = self.select_parents(valid) 
            else:
                parents = valid

            
            if mix_elites:
                elite_list = self.toolbox.select_elitists(valid + elite_list)
                parents += elite_list

            unsustainable_pop = self.overpopulate(parents, ssi=(not normal_unsustainable_population_size))
            if i == self.num_gens_ssi - 1:
                population_size_to_use = self.population_size
                if partitioned_population:
                    population_size_to_use = int(self.population_size*self.ssi_population_percentage)

                if old_downselect:
                    self.downselect(valid, for_ssi=True, custom_population_size=population_size_to_use)
                    downselected = self.current_deap_pop
                    curr_pop = self.current_population
                else:
                    downselected = tools.selNSGA2(valid, population_size_to_use)
                    curr_pop = {self.__get_hash(str(x)):x for x in downselected}
                self.save_ssi_metrics(i+1, downselected)
            else:
                curr_pop = unsustainable_pop
                print("Len of cur pop", len(unsustainable_pop), flush=True)

            print(f'{i + 1} Generations of SSI Completed')
        self.toolbox.register("select_elitists", tools.selSPEA2, k = self.max_elite_pool)
        self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
        return curr_pop
    
    
    def simulated_surrogate_injection_new(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection')
        self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents_ssi)
        
        for i in range(self.num_gens_ssi):
            print("Len of cur pop", len(curr_pop), flush=True)
            valid = None
            if i > 0:
                _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            else:
                valid = list(curr_pop.values())  
            self.save_ssi_metrics(i, valid)
            
            parents = None
            if len(valid) != self.num_parents:
                parents = self.select_parents(valid) 
            else:
                parents = valid

            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                downselected = tools.selNSGA2(valid, int(self.population_size*self.ssi_population_percentage))
                self.save_ssi_metrics(i+1, downselected)
                curr_pop = {self.__get_hash(str(x)):x for x in downselected}
            else:
                curr_pop = unsustainable_pop
                print("Len of cur pop", len(unsustainable_pop), flush=True)

            print(f'{i + 1} Generations of SSI Completed')
        self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
        return curr_pop

    
    def save_ssi_metrics(self, ssi_gen, valid_individuals):
        """Save surrogate-predicted metrics to outssi.csv."""
        print(f'Saving SSI metrics for generation {self.gen_count}.{ssi_gen}...')
        
        # Create a DataFrame for the SSI metrics
        ssi_metrics_df = pd.DataFrame(columns=['gen', 'hash', 'genome', 'metrics'])
        
        for individual in valid_individuals:
            # Only process individuals that have fitness values
            if individual.fitness is None or individual.fitness.values is None:
                continue
                
            hash_val = self.__get_hash(str(individual))
            genome_str = str(individual)
            metrics = {key: individual.fitness.values[i] for i, key in enumerate(self.objectives.keys())}
            
            # Format generation as main_gen.ssi_gen (e.g., 1.3)
            gen_str = f"{self.gen_count}.{ssi_gen}"
            
            # Add to DataFrame
            ssi_metrics_df.loc[len(ssi_metrics_df.index)] = [gen_str, hash_val, genome_str, metrics]
        
        # If file doesn't exist, create it; otherwise append to it
        if not os.path.exists(self.ssi_holy_grail_file):
            ssi_metrics_expanded = ssi_metrics_df.join(pd.json_normalize(ssi_metrics_df['metrics'])).drop('metrics', axis='columns')
            ssi_metrics_expanded.to_csv(self.ssi_holy_grail_file, index=False)
        else:
            # Read existing file, append new data, and save
            existing_df = pd.read_csv(self.ssi_holy_grail_file)
            
            # Convert metrics column in existing_df back to dictionary format if it exists
            if 'metrics' not in existing_df.columns:
                metrics_columns = [col for col in existing_df.columns if col not in ['gen', 'hash', 'genome']]
                existing_df['metrics'] = existing_df[metrics_columns].apply(lambda row: row.to_dict(), axis=1)
                existing_df = existing_df[['gen', 'hash', 'genome', 'metrics']]
            
            # Concatenate existing and new data
            combined_df = pd.concat([existing_df, ssi_metrics_df], ignore_index=True)
            
            # Expand the metrics column and save
            combined_expanded = combined_df.join(pd.json_normalize(combined_df['metrics'])).drop('metrics', axis='columns')
            combined_expanded.to_csv(self.ssi_holy_grail_file, index=False)
        
        print('Done!')

    def log_ssi_info(self, ssi_gen, curr_deap_pop):
        """Log population snapshot during SSI iterations (unused currently)"""
        print(f'Logging SSI generation {ssi_gen} data...')
        
        # Create SSI checkpoint directory
        ssi_checkpoint_path = os.path.join(self.checkpoint_path, f'ssi_gen_{self.gen_count}_{ssi_gen}')
        os.makedirs(ssi_checkpoint_path, exist_ok=True)
        
        # Save current SSI population
        with open(os.path.join(ssi_checkpoint_path, 'pop.pkl'), 'wb') as f:
            pickle.dump(curr_deap_pop, f)
            
        # Create a mini hall of fame and elite pool just for this SSI iteration
        ssi_hof = tools.ParetoFront()
        ssi_hof.update(curr_deap_pop)
        with open(os.path.join(ssi_checkpoint_path, 'hof.pkl'), 'wb') as f:
            pickle.dump(ssi_hof, f)
            
        # Save elites from current SSI population 
        ssi_elites = self.toolbox.select_elitists(curr_deap_pop)
        with open(os.path.join(ssi_checkpoint_path, 'elites.pkl'), 'wb') as f:
            pickle.dump(ssi_elites, f)
        
        self.save_ssi_metrics(ssi_gen, curr_deap_pop)

        print('Done!')

    def simulated_surrogate_injection_spea(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi):
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = tools.selSPEA2(valid, self.num_parents) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                new_pop = {}
                for hash in new_hashes:
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop

    def simulated_surrogate_injection_no_downselect(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi):
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                new_pop = {}
                for hash in new_hashes:
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop


    def simulated_surrogate_injection_tsdea(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection')
        self.num_gens_ssi = 10
        for i in range(self.num_gens_ssi):
            valid = None
            if i > 0:
                _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            else:
                valid = list(curr_pop.values())
            parents = tools.selSPEA2(valid, self.num_parents) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                original_crowding_dist = tools.emo.assignCrowdingDist
                tools.emo.assignCrowdingDist = self.euclidean_crowding_distance_assignment
                curr_pop = tools.selNSGA2(list(unsustainable_pop.values()), self.population_size)
                curr_pop = {self.__get_hash(str(x)):x for x in curr_pop}
                tools.emo.assignCrowdingDist = original_crowding_dist
            else:
                curr_pop = unsustainable_pop
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop


    def simulated_surrogate_injection_tsdea_short(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection')
        self.num_gens_ssi = 3
        for i in range(self.num_gens_ssi):
            valid = None
            if i > 0:
                _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            else:
                valid = list(curr_pop.values())
            parents = tools.selSPEA2(valid, self.num_parents) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                original_crowding_dist = tools.emo.assignCrowdingDist
                tools.emo.assignCrowdingDist = self.euclidean_crowding_distance_assignment
                curr_pop = tools.selNSGA2(list(unsustainable_pop.values()), self.population_size)
                curr_pop = {self.__get_hash(str(x)):x for x in curr_pop}
                tools.emo.assignCrowdingDist = original_crowding_dist
            else:
                curr_pop = unsustainable_pop
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop


    def simulated_surrogate_injection_tsdea_elitism(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        elite_list = tools.selSPEA2(list(curr_pop.values()), 5)
        print('Beginning Simulated Surrogate Injection')
        self.num_gens_ssi = 10
        for i in range(self.num_gens_ssi):
            valid = None
            if i > 0:
                _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            else:
                valid = list(curr_pop.values())
            parents = tools.selSPEA2(valid, self.num_parents-5) + elite_list
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                original_crowding_dist = tools.emo.assignCrowdingDist
                tools.emo.assignCrowdingDist = self.euclidean_crowding_distance_assignment
                curr_pop = tools.selNSGA2(list(unsustainable_pop.values()), self.population_size)
                curr_pop = {self.__get_hash(str(x)):x for x in curr_pop}
                tools.emo.assignCrowdingDist = original_crowding_dist
            else:
                curr_pop = unsustainable_pop
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop
    

    def select_with_dual_population(self, valid_individuals, diversity_sel, k):
        # Non-dominated sorting for main population (exploitation)
        self.toolbox.register("non_dominated_selection", tools.selNSGA2, k=int(k*0.7))
        main_pop = self.toolbox.non_dominated_selection(valid_individuals)
        
        # Diversity-based selection for vice population (exploration)
        remaining = [ind for ind in valid_individuals if ind not in main_pop]
        original_crowding_dist = tools.emo.assignCrowdingDist
        leftover = k-int(k*0.7)
        if diversity_sel == "nsga_euclidean":
            tools.emo.assignCrowdingDist = self.euclidean_crowding_distance_assignment
            self.toolbox.register("diversity_selection", tools.selNSGA2, k=leftover)
        elif diversity_sel == "dbea":
            self.toolbox.register("diversity_selection", dbea_selection, k=leftover, n_objectives=3)
        elif diversity_sel == "lexicase":
            self.toolbox.register("diversity_selection", lexicase_selection, n_objectives=3, k=leftover)
        vice_pop = self.toolbox.diversity_selection(remaining)
        if diversity_sel == "nsga_euclidean":
            tools.emo.assignCrowdingDist = original_crowding_dist
        
        return main_pop + vice_pop
    

    def euclidean_crowding_distance_assignment(self, individuals):
        """
        A replacement for DEAP's assignCrowdingDist function that uses
        Euclidean distance instead of the traditional crowding distance.
        
        This function modifies individuals' fitness.crowding_dist in-place.
        """
        if len(individuals) <= 1:
            for ind in individuals:
                ind.fitness.crowding_dist = float("inf")
            return
        
        # Calculate pairwise Euclidean distances in objective space
        for i, ind in enumerate(individuals):
            objectives = np.array(ind.fitness.values)
            
            # Calculate distances to all other individuals
            distances = []
            for j, other in enumerate(individuals):
                if i != j:
                    other_objectives = np.array(other.fitness.values)
                    dist = np.sqrt(np.sum((objectives - other_objectives) ** 2))
                    distances.append(dist)
            
            # Assign average distance as the crowding distance
            if distances:
                ind.fitness.crowding_dist = np.mean(distances)
            else:
                ind.fitness.crowding_dist = 0.0


    def simulated_surrogate_injection_nsga_euclidean_p(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        elite_list = tools.selSPEA2(list(curr_pop.values()), 5)
        self.toolbox.register("select_parents", self.select_with_dual_population, diversity_sel='nsga_euclidean', k = self.num_parents)
        print('Beginning Simulated Surrogate Injection')
        self.num_gens_ssi = 33
        for i in range(self.num_gens_ssi+1):
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            if i == self.num_gens_ssi - 1:
                curr_pop = {self.__get_hash(str(x)):x for x in parents}
            else:
                curr_pop = self.overpopulate(parents+elite_list, ssi=True)
            print(f'{i + 1} Generations of SSI Completed')
        self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
        return curr_pop


    def simulated_surrogate_injection_dbea_p(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        elite_list = tools.selSPEA2(list(curr_pop.values()), 5)
        self.toolbox.register("select_parents", self.select_with_dual_population, diversity_sel='dbea', k = self.num_parents)
        print('Beginning Simulated Surrogate Injection')
        self.num_gens_ssi = 33
        for i in range(self.num_gens_ssi+1):
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            if i == self.num_gens_ssi - 1:
                curr_pop = {self.__get_hash(str(x)):x for x in parents}
            else:
                curr_pop = self.overpopulate(parents+elite_list, ssi=True)
            print(f'{i + 1} Generations of SSI Completed')
        self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
        return curr_pop


    def simulated_surrogate_injection_lexicase_p(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        elite_list = tools.selSPEA2(list(curr_pop.values()), 5)
        self.toolbox.register("select_parents", self.select_with_dual_population, diversity_sel='lexicase', k = self.num_parents)
        print('Beginning Simulated Surrogate Injection')
        self.num_gens_ssi = 33
        for i in range(self.num_gens_ssi+1):
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            if i == self.num_gens_ssi - 1:
                curr_pop = {self.__get_hash(str(x)):x for x in parents}
            else:
                curr_pop = self.overpopulate(parents+elite_list, ssi=True)
            print(f'{i + 1} Generations of SSI Completed')
        self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
        return curr_pop

    def simulated_surrogate_injection_final_parents(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi+1):
            # if i!=0:
            #     print(type(list(curr_pop.values())[0]), flush=True)
            #     print(list(curr_pop.values())[0], flush=True)
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            if i == self.num_gens_ssi - 1:
                curr_pop = {self.__get_hash(str(x)):x for x in parents}
            else:
                unsustainable_pop = self.overpopulate(parents, ssi=True)
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                new_pop = {}
                for hash in new_hashes:
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop
                print(f'{i + 1} Generations of SSI Completed')
        return curr_pop

    def simulated_surrogate_injection_final_parents_elitism_static(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        self.toolbox.register("select_elitists", tools.selSPEA2, k = 10)
        elite_list = self.toolbox.select_elitists(list(curr_pop.values()))
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi+1):
            # if i!=0:
            #     print(type(list(curr_pop.values())[0]), flush=True)
            #     print(list(curr_pop.values())[0], flush=True)
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            if i == self.num_gens_ssi - 1:
                curr_pop = {self.__get_hash(str(x)):x for x in parents}
                parents = self.select_parents(valid) 
            else:
                self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents_ssi)
                parents = self.select_parents(valid) 
                self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
                parents = parents + random.sample(elite_list, 5)            
                unsustainable_pop = self.overpopulate(parents, ssi=True)
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                new_pop = {}
                for hash in new_hashes:
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop
                print(f'{i + 1} Generations of SSI Completed')
        self.toolbox.register("select_elitists", tools.selSPEA2, k = self.max_elite_pool)
        return curr_pop

    def simulated_surrogate_injection_elitism(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        elite_list = []
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi):
            # if i!=0:
            #     print(type(list(curr_pop.values())[0]), flush=True)
            #     print(list(curr_pop.values())[0], flush=True)

            # Reintroduce elites at each loop of SSI to prevent diversity loss
            elite_list = self.toolbox.select_elitists(list(curr_pop.values()) + elite_list)
            for e in elite_list:
                curr_pop[self.__get_hash(str(e))] = e
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                new_pop = {}
                for hash in new_hashes:
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop

            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop

    def simulated_surrogate_injection_elitism_static(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        self.toolbox.register("select_elitists", tools.selSPEA2, k = 5)
        elite_list = self.toolbox.select_elitists(list(curr_pop.values()))
        self.toolbox.register("select_elitists", tools.selSPEA2, k = self.max_elite_pool)
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi):
            # if i!=0:
            #     print(type(list(curr_pop.values())[0]), flush=True)
            #     print(list(curr_pop.values())[0], flush=True)

            # Reintroduce elites at each loop of SSI to prevent diversity loss
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents_ssi)
            parents = self.select_parents(valid) 
            self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)
            parents = parents + elite_list
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                new_pop = {}
                for hash in new_hashes:
                    new_pop[hash] = unsustainable_pop[hash]
                curr_pop = new_pop

            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop

    def simulated_surrogate_injection_elitism_no_downselect(self, curr_pop):
        curr_pop = copy.deepcopy(curr_pop)
        elite_list = []
        print('Beginning Simulated Surrogate Injection')
        for i in range(self.num_gens_ssi):
            # if i!=0:
            #     print(type(list(curr_pop.values())[0]), flush=True)
            #     print(list(curr_pop.values())[0], flush=True)

            # Reintroduce elites at each loop of SSI to prevent diversity loss
            elite_list = self.toolbox.select_elitists(list(curr_pop.values()) + elite_list)
            for e in elite_list:
                curr_pop[self.__get_hash(str(e))] = e
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            
            curr_pop = unsustainable_pop

            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop

    def simulated_surrogate_injection_parents(self, curr_pop, pure_nsga=False):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection with NSGA Downselection')
        for i in range(self.num_gens_ssi):
            # if i!=0:
            #     print(list(curr_pop.values())[0], flush=True)
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents_ssi)
            parents = self.select_parents(valid) 
            self.toolbox.register("select_parents", tools.selNSGA2, k = self.num_parents)

            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                self.downselect(unsustainable_pop, fully_trust_surrogate=pure_nsga, for_ssi=True)
                # new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                # new_pop = {}
                # for hash in new_hashes:
                #     new_pop[hash] = unsustainable_pop[hash]
                curr_pop = {self.__get_hash(str(x)):x for x in self.current_deap_pop}
                print(len(curr_pop.keys()))
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop



    def simulated_surrogate_injection_nsga(self, curr_pop, pure_nsga=False):
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection with NSGA Downselection')
        for i in range(self.num_gens_ssi):
            # if i!=0:
            #     print(list(curr_pop.values())[0], flush=True)
            _, valid = self.surrogate.set_fitnesses(self.sub_surrogates, self.cls_genome_scaler, self.reg_genome_scaler, list(curr_pop.values()))
            parents = self.select_parents(valid) 
            unsustainable_pop = self.overpopulate(parents, ssi=True)
            if i == self.num_gens_ssi - 1:
                curr_pop = unsustainable_pop
            else:
                self.downselect(unsustainable_pop, fully_trust_surrogate=pure_nsga, for_ssi=True)
                # new_hashes = random.sample(list(unsustainable_pop.keys()), self.ssi_sus_pop_size)
                # new_pop = {}
                # for hash in new_hashes:
                #     new_pop[hash] = unsustainable_pop[hash]
                curr_pop = {self.__get_hash(str(x)):x for x in self.current_deap_pop}
                print(len(curr_pop.keys()))
            print(f'{i + 1} Generations of SSI Completed')
        return curr_pop

    def simulated_surrogate_injection_pure_nsga(self, curr_pop):
        return self.simulated_surrogate_injection_nsga(curr_pop, pure_nsga=True)
    
    def simulated_surrogate_injection_islands(self, curr_pop):
        """
        A complete implementation of surrogate-assisted evolutionary algorithm using an island model.
        This approach evolves separate sub-populations (islands) that periodically exchange individuals.
        """
        curr_pop = copy.deepcopy(curr_pop)
        print('Beginning Simulated Surrogate Injection with Island Model')
        
        # Configuration parameters - these could be moved to class initialization
        num_islands = 3
        migration_interval = 2  # How often to perform migration (every n generations)
        migration_rate = 0.2    # Fraction of population to migrate
        
        # Split initial population into islands
        islands = self.split_into_islands(curr_pop, num_islands)
        island_size = len(curr_pop) // num_islands  # Approximate size per island
        
        for i in range(self.num_gens_ssi):
            print(f'Starting generation {i+1} of SSI with Island Model')
            new_islands = []
            
            # Process each island separately
            for island_idx, island in enumerate(islands):
                print(f'Processing island {island_idx+1}/{num_islands}')
                
                # Set fitness values using surrogate model
                _, valid = self.surrogate.set_fitnesses(
                    self.sub_surrogates,
                    self.cls_genome_scaler, 
                    self.reg_genome_scaler, 
                    list(island.values())
                )
                
                # Select parents within this island
                parents = self.select_parents(valid)
                
                # Generate offspring
                island_offspring = self.overpopulate(parents, ssi=True)
                
                # Select new population for this island
                if i == self.num_gens_ssi - 1:
                    # For the final generation, keep all offspring
                    new_island = island_offspring
                else:
                    # Otherwise, maintain the island size
                    new_hashes = random.sample(
                        list(island_offspring.keys()), 
                        min(island_size, len(island_offspring))
                    )
                    new_island = {hash: island_offspring[hash] for hash in new_hashes}
                
                new_islands.append(new_island)
            
            # Perform migration between islands at specified intervals
            if i > 0 and i % migration_interval == 0 and i < self.num_gens_ssi - 1:
                print(f'Performing migration at generation {i+1}')
                new_islands = self.migrate_between_islands(new_islands, migration_rate)
                
                # Optional: Log island statistics for monitoring
                # stats = self.get_island_statistics(new_islands)
                # for stat in stats:
                #     print(f"Island {stat['island_index']+1}: "
                #         f"Avg fitness: {stat['avg_fitness']:.4f}, "
                #         f"Best: {stat['best_fitness']:.4f}")
            
            islands = new_islands
            print(f'{i + 1} Generations of SSI with Island Model Completed')
        
        # For the final output, merge all islands
        final_pop = self.merge_islands(islands)
        
        # If the merged population is larger than desired, sample it down
        if len(final_pop) > self.ssi_sus_pop_size:
            selected_hashes = random.sample(list(final_pop.keys()), self.ssi_sus_pop_size)
            final_pop = {hash: final_pop[hash] for hash in selected_hashes}
        
        print(f'Completed Island Model SSI with final population size: {len(final_pop)}')
        return final_pop

    def split_into_islands(self, population, num_islands):
        """
        Split the population into roughly equal-sized islands.
        
        Args:
            population (dict): Dictionary of genome hashes to genomes
            num_islands (int): Number of islands to create
        
        Returns:
            list: List of dictionaries, each representing an island population
        """
        all_hashes = list(population.keys())
        random.shuffle(all_hashes)  # Randomize population before splitting
        
        # Calculate base size and remainder for uneven splits
        base_size = len(all_hashes) // num_islands
        remainder = len(all_hashes) % num_islands
        
        islands = []
        start_idx = 0
        
        for i in range(num_islands):
            # Add one extra individual to some islands if population size isn't perfectly divisible
            island_size = base_size + (1 if i < remainder else 0)
            island_hashes = all_hashes[start_idx:start_idx + island_size]
            
            # Create island population dictionary
            island_pop = {hash: population[hash] for hash in island_hashes}
            islands.append(island_pop)
            
            start_idx += island_size
        
        return islands

    def migrate_between_islands(self, islands, migration_rate=0.1):
        """
        Perform migration between islands using a ring topology.
        
        Args:
            islands (list): List of dictionaries representing island populations
            migration_rate (float): Fraction of population to migrate (default: 0.1)
        
        Returns:
            list: Updated list of island populations after migration
        """
        new_islands = []
        num_islands = len(islands)
        
        for i in range(num_islands):
            current_island = islands[i]
            next_island_idx = (i + 1) % num_islands  # Ring topology
            
            # Calculate number of individuals to migrate
            num_migrants = max(1, int(len(current_island) * migration_rate))
            
            # Select random migrants from current island
            migrant_hashes = random.sample(list(current_island.keys()), num_migrants)
            
            # Remove migrants from current island
            remaining_pop = {hash: current_island[hash] 
                            for hash in current_island 
                            if hash not in migrant_hashes}
            
            # Add migrants from previous island
            prev_island_idx = (i - 1) % num_islands
            prev_island_migrants = random.sample(
                list(islands[prev_island_idx].keys()),
                num_migrants
            )
            
            for hash in prev_island_migrants:
                remaining_pop[hash] = islands[prev_island_idx][hash]
            
            new_islands.append(remaining_pop)
        
        return new_islands

    def merge_islands(self, islands):
        """
        Merge all islands back into a single population.
        
        Args:
            islands (list): List of dictionaries representing island populations
        
        Returns:
            dict: Merged population
        """
        merged_population = {}
        
        # Simple merge of all island populations
        for island in islands:
            merged_population.update(island)
        
        return merged_population

    def get_island_statistics(self, islands):
        """
        Calculate statistics for each island to monitor diversity.
        
        Args:
            islands (list): List of dictionaries representing island populations
        
        Returns:
            list: List of dictionaries containing statistics for each island
        """
        stats = []
        
        for i, island in enumerate(islands):
            island_genomes = list(island.values())
            
            # Calculate fitness statistics
            fitness_values = [genome.fitness for genome in island_genomes]
            
            island_stats = {
                'island_index': i,
                'population_size': len(island),
                'avg_fitness': np.mean(fitness_values),
                'fitness_std': np.std(fitness_values),
                'best_fitness': max(fitness_values),
                'worst_fitness': min(fitness_values)
            }
            
            stats.append(island_stats)
        
        return stats


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
        # store holy grail
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
            df = pd.DataFrame(self.surrogate_trusts)      
            df.to_csv(self.trust_file, index=False)                
        print('Done!')


    def step_gen(self):
        self.gen_count += 1
        self.attempt_resume = False


    def __get_hash(self, s):
        layer_list = self.codec.get_layer_list(s)
        return hashlib.shake_256(str(layer_list).encode()).hexdigest(5)
    
    def get_hash_public(self, s):
        return self.__get_hash(s)
    

    def clear_outputs(self):
        print('Clearing old outputs and logs...')
        os.system(f'rm -rf {self.output_dir}/*')
        print('Done!')


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
#SBATCH --constraint="{'|'.join(GPUS)}"

module load anaconda3/2023.03
module load cuda/12.1.1
mkdir -p {self.logs_dir}/generation_{gen_num}

# Execute the Python script with SLURM_ARRAY_TASK_ID as argument. Script also has optional args -i and -o to specify input file and output directory respectively
conda run -n {ENV_NAME} --no-capture-output python -u {SCRIPT} $SLURM_ARRAY_TASK_ID -i {self.output_dir}/eval_inputs/eval_input_gen{gen_num}.csv -o {self.output_dir}
"""
        with open(f'{JOB_NAME}_{gen_num}.job', 'w') as fh:
            fh.write(batch_script)


    def create_surrogate_train_job(self, gen_num):
        """Create SBATCH job file for surrogate training using permanent script."""
        log_dir = os.path.join(self.logs_dir, f'generation_{gen_num}', 'surrogate')
        os.makedirs(log_dir, exist_ok=True)
        
        job_file_path = os.path.join(self.output_dir, f'surrogate_train_{gen_num}.job')
        
        # Get path to permanent surrogate_trainer.py file
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_trainer.py')
        
        batch_script = f"""#!/bin/bash
#SBATCH --job-name=surr_train_{gen_num}
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output={log_dir}/surrogate_train.%j.log
#SBATCH --error={log_dir}/surrogate_train_error.%j.log
#SBATCH --constraint="{'|'.join(GPUS)}"

module load anaconda3/2023.03
module load cuda/12.1.1

conda run -n nas python -u {script_path} {gen_num} {self.output_dir}
"""
    
        with open(job_file_path, 'w') as f:
            f.write(batch_script)
        
        return job_file_path


