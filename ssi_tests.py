

import toml
import argparse
from pipeline import Pipeline
import os
import glob
import shutil
import numpy as np
import random
import time

print("Started script!", flush=True)

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

ROOT_DIR = output_dir
TRUTH_DIR = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/gen4_base/testing_baseline"
FILES_TO_COPY = [
    "elites_history.pkl", "elites.csv", "hall_of_fame.csv",
    "hof_history.pkl", "out.csv"
]

# TESTS = ['islands', 'downselect', 'old', 'pure_nsga', 'low_sustain', 'high_unsustain', 'elitism_high_sustain', ]
# TESTS = ['final_parents_elitism_static']
TESTS = ['spea']
TRIALS = 10
for test in TRIALS:
    cur_records_path = os.path.join(ROOT_DIR, test)
    if not os.path.exists(cur_records_path):
        os.makedirs(cur_records_path)
    
    for i in range(RERUNS):
        print(test)
        GaPipeline = Pipeline(output_dir, config_dir, force_flag, clean)
        GaPipeline.initialize(seed_file)
        seed = int(time.time())
        np.random.seed(seed)
        random.seed(seed)
        GaPipeline.surrogate.reg_trust = 0.625
        GaPipeline.surrogate.cls_trust = 0.916666666
        if test == 'low_sustain':
            GaPipeline.ssi_sus_pop_size = 250
        elif test == 'elitism_low_sustain':
            GaPipeline.ssi_sus_pop_size = 200
        elif test == 'elitism_high_sustain' or 'final_parents_elitism_static':
            GaPipeline.ssi_sus_pop_size = 1000
        elif test == 'high_unsustain':
            GaPipeline.ssi_unsus_pop_size = 4000
        elif test == "high_gens":
            GaPipeline.num_gens_ssi = 20

        # all_subsurrogate_metrics = GaPipeline.prepare_surrogate(False)
        elites = GaPipeline.elite_pool
        # selected_parents = GaPipeline.select_parents(elites + GaPipeline.current_deap_pop) 
        # unsustainable_pop = GaPipeline.overpopulate(selected_parents) # returns pop dict {hash: genome}

        print(GaPipeline.sub_surrogates)
        ssi_func = None
        routing = {
            'old': GaPipeline.simulated_surrogate_injection,
            'low_sustain': GaPipeline.simulated_surrogate_injection,
            'high_unsustain': GaPipeline.simulated_surrogate_injection,
            'high_gens': GaPipeline.simulated_surrogate_injection,
            'downselect': GaPipeline.simulated_surrogate_injection_nsga,
            'pure_nsga': GaPipeline.simulated_surrogate_injection_pure_nsga,
            'islands': GaPipeline.simulated_surrogate_injection_islands,
            'elitism': GaPipeline.simulated_surrogate_injection_elitism,
            'elitism_high_sustain': GaPipeline.simulated_surrogate_injection_elitism,
            'elitism_low_sustain': GaPipeline.simulated_surrogate_injection_elitism,
            'parents': GaPipeline.simulated_surrogate_injection_parents,
            'less_parents': GaPipeline.simulated_surrogate_injection_parents,
            'elitism_no_downselect': GaPipeline.simulated_surrogate_injection_elitism_no_downselect,
            'elitism_static': GaPipeline.simulated_surrogate_injection_elitism_static,
            'final_parents': GaPipeline.simulated_surrogate_injection_final_parents,
            # 'dbea': GaPipeline.simulated_surrogate_injection_dbea,
            'dbea_p': GaPipeline.simulated_surrogate_injection_dbea_p,
            'final_parents_elitism_static': GaPipeline.simulated_surrogate_injection_final_parents_elitism_static,
            'final_parents_elitism_static_p': GaPipeline.simulated_surrogate_injection_final_parents_elitism_static,
            # 'nsga_euclidean': GaPipeline.simulated_surrogate_injection_nsga_euclidean,
            'nsga_euclidean_p': GaPipeline.simulated_surrogate_injection_nsga_euclidean_p,
            'lexicase_p': GaPipeline.simulated_surrogate_injection_lexicase_p,
            'tsdea': GaPipeline.simulated_surrogate_injection_tsdea,
            'tsdea_short': GaPipeline.simulated_surrogate_injection_tsdea_short,
            'tsdea_elitism': GaPipeline.simulated_surrogate_injection_tsdea_elitism,
            'spea': GaPipeline.simulated_surrogate_injection_spea,
            'no_downselect': GaPipeline.simulated_surrogate_injection_no_downselect,
        }

        print(test)
        unsustainable_pop = routing[test]({GaPipeline.get_hash_public(str(x)):x for x in elites + GaPipeline.current_deap_pop})
        # takes in pop dict
        GaPipeline.downselect(unsustainable_pop) # population is replaced by a completely new one
        GaPipeline.surrogate_enabled = False
        GaPipeline.gen_count += 1
        GaPipeline.evaluate_gen()
        elites = GaPipeline.update_elite_pool()
        GaPipeline.update_hof()
        GaPipeline.log_info()

        new_out_name = os.path.join(ROOT_DIR, f"out{i}.csv")
        os.rename(os.path.join(ROOT_DIR, "out.csv"), new_out_name)
        shutil.copy(new_out_name, cur_records_path)
        os.remove(new_out_name)
        shutil.rmtree(os.path.join(ROOT_DIR, "generation_5"), ignore_errors=True)
        shutil.rmtree(os.path.join(ROOT_DIR, "logs"), ignore_errors=True)
        shutil.copytree(os.path.join(TRUTH_DIR, "checkpoint"), os.path.join(ROOT_DIR, "checkpoint"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(TRUTH_DIR, "eval_inputs"), os.path.join(ROOT_DIR, "eval_inputs"), dirs_exist_ok=True)
        for file in FILES_TO_COPY:
            shutil.copy2(os.path.join(TRUTH_DIR, file), ROOT_DIR)