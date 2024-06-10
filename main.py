from pipeline import Pipeline
import primitives

NUM_GENERATIONS = 1

GaPipeline = Pipeline()
GaPipeline.init_pop()
for i in range (NUM_GENERATIONS):
    GaPipeline.evaluate_gen()
    selected_parents = GaPipeline.select_parents()
    elites = GaPipeline.update_elite_pool()
    GaPipeline.update_hof()
    unsustainable_pop = GaPipeline.overpopulate(selected_parents + elites) # returns dict {hash: genome}