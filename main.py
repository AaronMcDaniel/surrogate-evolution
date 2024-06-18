import argparse
from pipeline import Pipeline


parser = argparse.ArgumentParser()
    
parser.add_argument('-o', '--outputs', type=str, required=True, help='The output directory')
parser.add_argument('-f', '--force', action='store_true', help='Force overwrite if output directory exists')
parser.add_argument('-n', '--num_generations', type=int, required=True, help='The number of generations to run the evolution for')
parser.add_argument('-c', '--clean', action='store_true', help='Cleans output directory of non pareto optimal individuals')

args = parser.parse_args()

output_dir = args.outputs
force_flag = args.force
num_gen = args.num_generations
clean = args.clean

GaPipeline = Pipeline(output_dir, force_flag, clean)
GaPipeline.init_pop()
for i in range (num_gen):
    print(f'---------- Generation {GaPipeline.gen_count} ----------')
    GaPipeline.evaluate_gen()
    selected_parents = GaPipeline.select_parents()
    elites = GaPipeline.update_elite_pool() # elites are selected from existing elite pool and current pop
    GaPipeline.update_hof()
    GaPipeline.log_info() 
    unsustainable_pop = GaPipeline.overpopulate(selected_parents + elites) # returns dict {hash: genome}
    GaPipeline.downselect(unsustainable_pop) # population is replaced by a completely new one
    GaPipeline.step_gen()

print('====================')
print(f'total genome fails: {GaPipeline.num_genome_fails}/{num_gen*GaPipeline.population_size}')
