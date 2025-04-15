"""
Launches an evolution.
"""


import argparse
from pipeline import Pipeline
import toml
import numpy as np
import random
import time

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

def advance_random_states(random_state, numpy_random_state, n_rolls):
    temp_random = random.Random()
    temp_random.setstate(random_state)
    _ = temp_random.sample(range(n_rolls+1), n_rolls)
    advanced_random_state = temp_random.getstate()
    
    temp_np_random = np.random.RandomState()
    temp_np_random.set_state(numpy_random_state)
    _ = temp_np_random.random(n_rolls)
    advanced_np_state = temp_np_random.get_state()
    
    return advanced_random_state, advanced_np_state

def print_random_state_fingerprint(random_state, np_random_state):
    print('RANDOM STATE INFO', flush=True)
    print({
        'version': random_state[0],
        'state_head': random_state[1][:3],  # First 3 elements
        'state_tail': random_state[1][-3:],  # Last 3 elements
        'gauss_next': random_state[2]
    }, flush=True)
    
    print({
        'algorithm': np_random_state[0],
        'keys_head': np_random_state[1][:3].tolist(),  # First 3 elements
        'keys_tail': np_random_state[1][-3:].tolist(),  # Last 3 elements
        'position': np_random_state[2],
        'has_gauss': np_random_state[3],
        'cached_gaussian': np_random_state[4]
    }, flush=True)

GaPipeline = Pipeline(output_dir, config_dir, force_flag, clean)
GaPipeline.initialize(seed_file)
random.seed(60)
np.random.seed(60)
while GaPipeline.gen_count <= num_gen:
    print(f'---------- Generation {GaPipeline.gen_count} ----------')
    if not GaPipeline.attempt_resume:
        GaPipeline.evaluate_gen()
        num_evals += 1
    else:
        # just train the surrogate, don't evaluate generation on resume
        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        if surrogate_enabled:
            all_subsurrogate_metrics = GaPipeline.prepare_surrogate()
        random_state, numpy_random_state = advance_random_states(random_state, numpy_random_state, 100)
        random.setstate(random_state)
        np.random.set_state(numpy_random_state)
    
    print_random_state_fingerprint(random.getstate(), np.random.get_state())

    if not GaPipeline.attempt_resume:
        elites = GaPipeline.update_elite_pool() # elites are selected from existing elite pool and current pop
    else :
        elites = GaPipeline.elite_pool
    if not GaPipeline.attempt_resume:
        GaPipeline.update_hof()
        GaPipeline.log_info()
    random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    selected_parents = GaPipeline.select_parents(list(set(elites + GaPipeline.current_deap_pop))) 
    unsustainable_pop = GaPipeline.overpopulate(selected_parents) # returns dict {hash: genome}
    random_state, numpy_random_state = advance_random_states(random_state, numpy_random_state, 100)
    random.setstate(random_state)
    np.random.set_state(numpy_random_state)
    GaPipeline.downselect(unsustainable_pop) # population is replaced by a completely new one
    
    print_random_state_fingerprint(random.getstate(), np.random.get_state())

    GaPipeline.step_gen()

print('====================')
print(f'total genome fails: {GaPipeline.num_genome_fails}/{GaPipeline.total_evaluated_individuals}')
