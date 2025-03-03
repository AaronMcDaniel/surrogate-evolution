"""
Launches an evolution.
"""

import toml
import argparse
from pipeline import Pipeline


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
ssi = pipeline_config['ssi']
ssi_start_gen = pipeline_config['ssi_start_gen']
ssi_freq = pipeline_config['ssi_freq']

num_evals = 0

GaPipeline = Pipeline(output_dir, config_dir, force_flag, clean)
GaPipeline.initialize(seed_file)
while GaPipeline.gen_count <= num_gen:
    print(f'---------- Generation {GaPipeline.gen_count} ----------')
    if not GaPipeline.attempt_resume:
        GaPipeline.evaluate_gen()
        num_evals += 1
    else:
        # just train the surrogate, don't evaluate generation on resume
        all_subsurrogate_metrics = GaPipeline.prepare_surrogate()
    if not GaPipeline.attempt_resume:
        elites = GaPipeline.update_elite_pool() # elites are selected from existing elite pool and current pop
    else :
        elites = GaPipeline.elite_pool
    if not GaPipeline.attempt_resume:
        GaPipeline.update_hof()
        GaPipeline.log_info()
    selected_parents = GaPipeline.select_parents(elites + GaPipeline.current_deap_pop) 
    unsustainable_pop = GaPipeline.overpopulate(selected_parents) # returns pop dict {hash: genome}
    if ssi and (GaPipeline.gen_count >= ssi_start_gen) and ((GaPipeline.gen_count - ssi_start_gen) % ssi_freq == 0):
        # returns pop dict
        unsustainable_pop = GaPipeline.simulated_surrogate_injection_new(unsustainable_pop)
    # takes in pop dict
    GaPipeline.downselect(unsustainable_pop) # population is replaced by a completely new one
    GaPipeline.step_gen()

print('====================')
print(f'total genome fails: {GaPipeline.num_genome_fails}/{GaPipeline.total_evaluated_individuals}')
