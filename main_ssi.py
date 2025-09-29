"""
Launches an evolution.
"""

import toml
import argparse
from pipeline import Pipeline
import numpy as np
import random
import time
from deap import tools
import copy
import hashlib
import math
import torch
import pandas

pandas.set_option('display.max_colwidth', None)
pandas.set_option('display.max_columns', None)

parser = argparse.ArgumentParser()
    
parser.add_argument('-o', '--outputs', type=str, required=True, help='The output directory')
parser.add_argument('-f', '--force', action='store_true', help='Force overwrite if output directory exists. Will attempt to resume run without this flag')
parser.add_argument('-n', '--num_generations', type=int, required=True, help='The number of generations to run the evolution for')
parser.add_argument('-r', '--remove', action='store_true', help='Cleans output directory of non-pareto-optimal individual weights')
parser.add_argument('-conf', '--configuration', type=str, required=True, help='The path to the configuration file')
parser.add_argument('-s', '--seed_file', type=str, required=False, help='The path to seeding .txt file')
parser.add_argument('-i', '--initialization_seed', type=int, required=False, help='The initialization seed', default=93)
parser.add_argument('-x', '--experimentation', action='store_true', help='Enters experimentation mode')

# 0-indexed in this order: override_fitnesses, downselect_incoming_population, normal_unsustainable_population_size, mix_elites, old_downselect, partitioned_population
parser.add_argument('-a', '--ablation', type=int, required=False, default=-1, help='Index of ablation flag to enable (0–5) in simulated_surrogate_injection_ablation(). If None, all flags are False.')
parser.add_argument('-rp', '--remove_partition', action='store_true', required=False, default=False, help='Flag to choose running with or without partition. Part of ablation.')
parser.add_argument('-sg', '--ssi_loops',type=int, required=False, default=-1, help='Number of generations to run SSI for.')

args = parser.parse_args()

output_dir = args.outputs
config_dir = args.configuration
force_flag = args.force
num_gen = args.num_generations
clean = args.remove
seed_file = args.seed_file
ablation_index = args.ablation
rp = args.remove_partition
ssi_loops = args.ssi_loops

# Map ablation index to keyword arguments
ablation_kwargs = {
    'override_fitnesses': ablation_index == 0,
    'downselect_incoming_population': ablation_index == 1,
    'normal_unsustainable_population_size': ablation_index == 2,
    'mix_elites': ablation_index == 3,
    'random_downselect': ablation_index == 4,
    'keep_same_population_size': ablation_index == 5,
    'num_ssi_loops': ssi_loops
}

print(f"Ablation args: {ablation_kwargs}")
print(f'Remove partitioned population ablation: {rp}')
SEED = args.initialization_seed
print("INITIALIZATION SEED:", SEED, flush=True)
if SEED in [None, 0, '', 'None']:
    SEED = 93
experimentation = args.experimentation

configs = toml.load(config_dir)
pipeline_config = configs["pipeline"]
ssi = pipeline_config['ssi']
ssi_start_gen = pipeline_config['ssi_start_gen']
ssi_freq = pipeline_config['ssi_freq']

num_evals = 0


def print_random_state_fingerprint(random_state, np_random_state):
    print('RANDOM STATE INFO', flush=True)
    py_state_bytes = str(random_state[1]).encode()
    py_hash = hashlib.md5(py_state_bytes).hexdigest()
    
    np_state_bytes = np_random_state[1].tobytes()
    np_hash = hashlib.md5(np_state_bytes).hexdigest()
    print("RANDOM HASH", py_hash)
    print("NP RANDOM HASH", np_hash)


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
setSeed(SEED)

REMOVE_PARTITIONED_POPULATION_ABLATION = rp


def perform_ssi(elites, do_ablation):
    selection_pool = copy.deepcopy(elites + GaPipeline.current_deap_pop)
    selection_pool = {GaPipeline.get_hash_public(str(x)):x for x in selection_pool}
    if not experimentation and not do_ablation:
        unsustainable_pop = GaPipeline.simulated_surrogate_injection_new(selection_pool)
    elif do_ablation:
        print("IN ABLATION MODE", flush=True)
        unsustainable_pop = GaPipeline.simulated_surrogate_injection_ablation(selection_pool, **ablation_kwargs)
    else:
        print("IN EXPERIMENTATION MODE", flush=True)
        unsustainable_pop = GaPipeline.simulated_surrogate_injection_stepwise_balanced(selection_pool, fill_interval=1, start_generation=3)
    if not REMOVE_PARTITIONED_POPULATION_ABLATION:
        remove_hashes = set()
        for hash in selection_pool:
            if hash in unsustainable_pop:
                remove_hashes.add(hash)
        for hash in remove_hashes:
            del selection_pool[hash]
        k = math.ceil(GaPipeline.population_size*(1-GaPipeline.ssi_population_percentage))
        retained_pop = tools.selNSGA2(list(selection_pool.values()), k=k)
        retained_pop = GaPipeline.overpopulate(retained_pop, custom_pop_size=k)
        unsustainable_pop.update(retained_pop)
    return unsustainable_pop


GaPipeline = Pipeline(output_dir, config_dir, force_flag, clean)
GaPipeline.initialize(seed_file)
while GaPipeline.gen_count <= num_gen:
    random.seed(int(SEED*(GaPipeline.gen_count+1)))
    np.random.seed(int(SEED*(GaPipeline.gen_count+1)))
    torch.manual_seed(int(SEED*(GaPipeline.gen_count+1)))
    
    print(f'---------- Generation {GaPipeline.gen_count} ----------')
    if not GaPipeline.attempt_resume:
        GaPipeline.evaluate_gen()
        num_evals += 1
    else:
        # just train the surrogate, don't evaluate generation on resume
        all_subsurrogate_metrics = GaPipeline.prepare_surrogate()

    setSeed(int(SEED*(GaPipeline.gen_count+1)))

    if not GaPipeline.attempt_resume:
        elites = GaPipeline.update_elite_pool() # elites are selected from existing elite pool and current pop
    else :
        elites = GaPipeline.elite_pool
    if not GaPipeline.attempt_resume:
        GaPipeline.update_hof()
        GaPipeline.log_info()

    print_random_state_fingerprint(random.getstate(), np.random.get_state())

    unsustainable_pop = None
    if all_subsurrogate_metrics is None:
        print('NO SURROGATE METRICS; SSI WILL NOT RUN IF ATTEMPTED THIS GENERATION', flush=True)
    if ssi and (GaPipeline.gen_count >= ssi_start_gen) and ((GaPipeline.gen_count - ssi_start_gen) % ssi_freq == 0) and all_subsurrogate_metrics is not None:
        # returns pop dict
        retry_counter = 0
        max_retries = 5
        while retry_counter < max_retries:
            try:
                unsustainable_pop = perform_ssi(elites, ablation_index != -1)
                break  # If successful, exit the loop
            except Exception as e:
                retry_counter += 1
                setSeed(int(SEED*(GaPipeline.gen_count+1)) + retry_counter)
                print(f"SSI attempt {retry_counter} failed with error: {e}. Retrying...")
        setSeed(int(SEED*(GaPipeline.gen_count+1)))
        
        if unsustainable_pop is None:
            print("All SSI attempts failed.", flush=True)
    else:
        selected_parents = GaPipeline.select_parents(elites + GaPipeline.current_deap_pop) 
        unsustainable_pop = GaPipeline.overpopulate(selected_parents)
    # takes in pop dict
    GaPipeline.downselect(unsustainable_pop)
    

    GaPipeline.step_gen()

print('====================')
print(f'total genome fails: {GaPipeline.num_genome_fails}/{GaPipeline.total_evaluated_individuals}')
