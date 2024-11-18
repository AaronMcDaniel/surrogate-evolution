import copy
import hashlib
import os
import pickle
from surrogates.surrogate import Surrogate
from pipeline import Pipeline
import pandas as pd
from deap import creator, gp, base, tools
import random
from codec import Codec


pl = Pipeline('/storage/ice-shared/vip-vvk/data/AOT/ttest_ssi', '/home/hice1/tthakur9/scratch/surrogate-evolution/conf.toml', force_wipe=False, clean=False)
codec = pl.codec
surrogate = Surrogate('/home/hice1/tthakur9/scratch/surrogate-evolution/conf.toml', '/home/hice1/tthakur9/scratch/surrogate-evolution/surrogate_dataset')

cls_train_df = pd.read_pickle('surrogate_dataset/pretrain_cls_train.pkl')
cls_val_df = pd.read_pickle('surrogate_dataset/surr_cls_val.pkl')
reg_train_df = pd.read_pickle('surrogate_dataset/pretrain_reg_train.pkl')
reg_val_df = pd.read_pickle('surrogate_dataset/surr_reg_val.pkl')

cls_scaler_path = os.path.join('surrogate_dataset', 'cls_genome_scaler.pkl')
reg_scaler_path = os.path.join('surrogate_dataset', 'reg_genome_scaler.pkl')

scores, cls_genome_scaler, reg_genome_scaler = surrogate.train(cls_train_df, cls_val_df, reg_train_df, reg_val_df, train_reg=True)
print(scores)
# with open(cls_scaler_path, 'wb') as f:
#     pickle.dump(cls_genome_scaler, f)
# with open(reg_scaler_path, 'wb') as f:
#     pickle.dump(reg_genome_scaler, f)

# sub_surrogates = []
# cls_trust = 0
# max_cls_model = ''
# for key, val in scores['classifiers'].items():
#     if val['acc'] > cls_trust:
#         cls_trust = val['acc']
#         max_cls_model = key
# cls_to_dict = {d['name']: d for d in surrogate.classifier_models}
# max_cls_model_idx = list(cls_to_dict.keys()).index(max_cls_model)
# sub_surrogates.append(max_cls_model_idx)
# result = surrogate.optimize_trust(cls_genome_scaler, reg_genome_scaler, cls_val_df, reg_val_df)
# print(result)
# reg_trust = result[0]
# sub_surrogates += result[1]
# print(sub_surrogates)

# result = [0.7647058823529411, [1, 5, 3]]

with open(cls_scaler_path, 'rb') as f:
    cls_genome_scaler = pickle.load(f)
with open(reg_scaler_path, 'rb') as f:
    reg_genome_scaler = pickle.load(f)
sub_surrogates = [0, 1, 5, 3]

curr_pop = {}
curr_deap_pop = pl.toolbox.population(n=500)

for genome in curr_deap_pop:
    genome_string = str(genome)
    layer_list = pl.codec.get_layer_list(genome_string)
    hash = hashlib.shake_256(str(layer_list).encode()).hexdigest(5)
    curr_pop[hash] = genome

for i in range(6):
    print(f'Generation {i + 1} Population Size: {len(curr_deap_pop)}')
    # need curr_pop.values to be deap individuals hash -> genome
    _, valid = surrogate.set_fitnesses(sub_surrogates, cls_genome_scaler, reg_genome_scaler, list(curr_pop.values()))
    parents = pl.select_parents(valid) 
    print(f'Generation {i + 1} Parent Size: {len(parents)}')
    unsustainable_pop = pl.overpopulate(parents)
    print(f'Generation {i + 1} Overpopulated Size: {len(unsustainable_pop.keys())}')
    if i == 5:
        curr_pop = unsustainable_pop
    else:
        new_hashes = random.sample(list(unsustainable_pop.keys()), 30)
        new_pop = {}
        new_deap_pop = []
        for hash in new_hashes:
            new_deap_pop.append(unsustainable_pop[hash])
            new_pop[hash] = unsustainable_pop[hash]
        curr_pop = new_pop
        curr_deap_pop = new_deap_pop
    
for ind in curr_pop.items():
    print(ind)