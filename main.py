from pipeline import Pipeline
import primitives

NUM_GENERATIONS = 10

GaPipeline = Pipeline()
GaPipeline.clear_outputs()
GaPipeline.init_pop()
for i in range (NUM_GENERATIONS):
    print(f'---------- Generation {GaPipeline.gen_count} ----------')
    GaPipeline.evaluate_gen()
    selected_parents = GaPipeline.select_parents()
    elites = GaPipeline.update_elite_pool() # elites are selected from existing elite pool and current pop
    GaPipeline.update_hof()
    GaPipeline.log_info() 
    unsustainable_pop = GaPipeline.overpopulate(selected_parents + elites) # returns dict {hash: genome}
    GaPipeline.downselect(unsustainable_pop) # population is replaced by a completely new one
    GaPipeline.step_gen()
