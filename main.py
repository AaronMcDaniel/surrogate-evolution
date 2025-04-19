"""
Launches an evolution.
"""


import argparse
from pipeline import Pipeline
import toml
import numpy as np
import random
import time
import hashlib

parser = argparse.ArgumentParser()
    
parser.add_argument('-o', '--outputs', type=str, required=True, help='The output directory')
parser.add_argument('-f', '--force', action='store_true', help='Force overwrite if output directory exists. Will attempt to resume run without this flag')
parser.add_argument('-n', '--num_generations', type=int, required=True, help='The number of generations to run the evolution for')
parser.add_argument('-r', '--remove', action='store_true', help='Cleans output directory of non-pareto-optimal individual weights')
parser.add_argument('-conf', '--configuration', type=str, required=True, help='The path to the configuration file')
parser.add_argument('-s', '--seed_file', type=str, required=False, help='The path to seeding .txt file')

args = parser.parse_args()

output_dir = args.outputs
config_dir = args.configuration
force_flag = args.force
num_gen = args.num_generations
clean = args.remove
seed_file = args.seed_file

configs = toml.load(config_dir)
pipeline_config = configs["pipeline"]
surrogate_enabled = pipeline_config['surrogate_enabled']
num_evals = 0


def print_random_state_fingerprint(random_state, np_random_state):
    print('RANDOM STATE INFO', flush=True)
    py_state_bytes = str(random_state[1]).encode()
    py_hash = hashlib.md5(py_state_bytes).hexdigest()
    
    np_state_bytes = np_random_state[1].tobytes()
    np_hash = hashlib.md5(np_state_bytes).hexdigest()
    print("RANDOM HASH", py_hash)
    print("NP RANDOM HASH", np_hash)

GaPipeline = Pipeline(output_dir, config_dir, force_flag, clean)
GaPipeline.initialize(seed_file)
SEED = 60
while GaPipeline.gen_count <= num_gen:
    print(f'---------- Generation {GaPipeline.gen_count} ----------')
    if not GaPipeline.attempt_resume:
        GaPipeline.evaluate_gen()
        num_evals += 1
    else:
        # just train the surrogate, don't evaluate generation on resume
        if surrogate_enabled:
            all_subsurrogate_metrics = GaPipeline.prepare_surrogate()
    
    random.seed(int(SEED*(GaPipeline.gen_count+1)))
    np.random.seed(int(SEED*(GaPipeline.gen_count+1)))
    
    if not GaPipeline.attempt_resume:
        elites = GaPipeline.update_elite_pool() # elites are selected from existing elite pool and current pop
    else :
        elites = GaPipeline.elite_pool
    if not GaPipeline.attempt_resume:
        GaPipeline.update_hof()
        GaPipeline.log_info()

    print_random_state_fingerprint(random.getstate(), np.random.get_state())

    selected_parents = GaPipeline.select_parents(list(set(elites + GaPipeline.current_deap_pop))) 
    unsustainable_pop = GaPipeline.overpopulate(selected_parents) # returns dict {hash: genome}

    GaPipeline.downselect(unsustainable_pop) # population is replaced by a completely new one
    

    GaPipeline.step_gen()

print('====================')
print(f'total genome fails: {GaPipeline.num_genome_fails}/{GaPipeline.total_evaluated_individuals}')
