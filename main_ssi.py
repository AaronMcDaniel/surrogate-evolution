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

parser = argparse.ArgumentParser()
    
parser.add_argument('-o', '--outputs', type=str, required=True, help='The output directory')
parser.add_argument('-f', '--force', action='store_true', help='Force overwrite if output directory exists. Will attempt to resume run without this flag')
parser.add_argument('-n', '--num_generations', type=int, required=True, help='The number of generations to run the evolution for')
parser.add_argument('-r', '--remove', action='store_true', help='Cleans output directory of non-pareto-optimal individual weights')
parser.add_argument('-conf', '--configuration', type=str, required=True, help='The path to the configuration file')
parser.add_argument('-s', '--seed_file', type=str, required=False, help='The path to seeding .txt file')

# 0-indexed in this order: override_fitnesses, downselect_incoming_population, normal_unsustainable_population_size, mix_elites, old_downselect, partitioned_population
parser.add_argument('-a', '--ablation', type=int, required=False, default=-1, help='Index of ablation flag to enable (0–5) in simulated_surrogate_injection_ablation(). If None, all flags are False.')
parser.add_argument('-rp', '--remove_partition', action='store_true', required=False, default=False, help='Flag to choose running with or without partition. Part of ablation.')
args = parser.parse_args()

output_dir = args.outputs
config_dir = args.configuration
force_flag = args.force
num_gen = args.num_generations
clean = args.remove
seed_file = args.seed_file
ablation_index = args.ablation
rp = args.remove_partition

print(ablation_index)
ablation_args = [i == ablation_index for i in range(5)]

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

SEED = 93
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

REMOVE_PARTITIONED_POPULATION_ABLATION = rp

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

    random.seed(int(SEED*(GaPipeline.gen_count+1)))
    np.random.seed(int(SEED*(GaPipeline.gen_count+1)))
    torch.manual_seed(int(SEED*(GaPipeline.gen_count+1)))
    
    if not GaPipeline.attempt_resume:
        elites = GaPipeline.update_elite_pool() # elites are selected from existing elite pool and current pop
    else :
        elites = GaPipeline.elite_pool
    if not GaPipeline.attempt_resume:
        GaPipeline.update_hof()
        GaPipeline.log_info()

    print_random_state_fingerprint(random.getstate(), np.random.get_state())

    unsustainable_pop = None
    if ssi and (GaPipeline.gen_count >= ssi_start_gen) and ((GaPipeline.gen_count - ssi_start_gen) % ssi_freq == 0) and all_subsurrogate_metrics is not None:
        # returns pop dict
        selection_pool = copy.deepcopy(elites + GaPipeline.current_deap_pop)
        selection_pool = {GaPipeline.get_hash_public(str(x)):x for x in selection_pool}
        unsustainable_pop = GaPipeline.simulated_surrogate_injection_ablation(selection_pool, *ablation_args)
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
    else:
        selected_parents = GaPipeline.select_parents(elites + GaPipeline.current_deap_pop) 
        unsustainable_pop = GaPipeline.overpopulate(selected_parents)
    # takes in pop dict
    GaPipeline.downselect(unsustainable_pop)
    

    GaPipeline.step_gen()

print('====================')
print(f'total genome fails: {GaPipeline.num_genome_fails}/{GaPipeline.total_evaluated_individuals}')
