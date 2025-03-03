

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

GaPipeline = Pipeline(output_dir, config_dir, force_flag, clean)
GaPipeline.initialize(seed_file)

# all_subsurrogate_metrics = GaPipeline.prepare_surrogate(False)
elites = GaPipeline.elite_pool
# selected_parents = GaPipeline.select_parents(elites + GaPipeline.current_deap_pop) 
# unsustainable_pop = GaPipeline.overpopulate(selected_parents) # returns pop dict {hash: genome}

print(GaPipeline.sub_surrogates)
# {GaPipeline.__get_hash(str(x)):x for x in elites + GaPipeline.current_deap_pop}
unsustainable_pop = GaPipeline.simulated_surrogate_injection_new({GaPipeline.__get_hash(str(x)):x for x in elites + GaPipeline.current_deap_pop})
# takes in pop dict
GaPipeline.downselect(unsustainable_pop) # population is replaced by a completely new one
GaPipeline.surrogate_enabled = False
GaPipeline.evaluate_gen()
elites = GaPipeline.update_elite_pool()
GaPipeline.update_hof()
GaPipeline.log_info()