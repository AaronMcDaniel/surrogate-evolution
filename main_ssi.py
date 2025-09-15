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
parser.add_argument('--seed', type=int, required=False, help='Random Seed')

args = parser.parse_args()

output_dir = args.outputs
config_dir = args.configuration
force_flag = args.force
num_gen = args.num_generations
clean = args.remove
seed_file = args.seed_file

DEFAULT_SEED = 93
if args.seed is not None:
    SEED = int(args.seed)
else:
    SEED = DEFAULT_SEED
    
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

REMOVE_PARTITIONED_POPULATION_ABLATION = False

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
        
        # Using ablation method instead of simulated_surrogate_injection_new
        print("=== USING SIMULATED_SURROGATE_INJECTION_ABLATION ===")
        unsustainable_pop = GaPipeline.simulated_surrogate_injection_ablation(
            selection_pool,
            override_fitnesses=True,
            mix_elites=True,
            partitioned_population=True
        )
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
