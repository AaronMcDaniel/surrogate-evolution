import os
import pandas as pd
import toml
import hashlib
import matplotlib.pyplot as plt
from codec import Codec

outdir = '/gv1/projects/GRIP_Precog_Opt/unseeded_baseline_evolution'
plotdir = 'test/duplicates.png'

configs = toml.load(os.path.join(outdir, "conf.toml"))
model_config = configs["model"]
codec_config = configs["codec"]
data_config = configs["data"]
all_config = model_config | codec_config | data_config
num_classes = all_config['num_classes']
genome_encoding = all_config['genome_encoding_strat']
codec = Codec(num_classes, genome_encoding_strat=genome_encoding)

def get_effective_hash(genome):
    layer_list = codec.get_layer_list(genome)
    return hashlib.shake_256(str(layer_list).encode()).hexdigest(5)

directory = os.path.join(outdir, 'eval_inputs')


# To get number of duplicates within each generation
#
# result_dict = {}
# for file_name in os.listdir(directory):
#     file_path = os.path.join(directory, file_name)
#     if os.path.isfile(file_path) and file_name.endswith('.csv'):
#         df = pd.read_csv(file_path)
#         for gen in df['gen'].unique():
#             subset = df[df['gen'] == gen]
#             hash_counts = {}
#             for genome in subset['genome']:
#                 hash_value = get_effective_hash(genome)
#                 if hash_value in hash_counts:
#                     print(genome)
#                     print()
#                     hash_counts[hash_value] += 1
#                 else:
#                     hash_counts[hash_value] = 1
#             count = sum(value-1 for value in hash_counts.values() if value > 1)
#             result_dict[gen] = count


def plot_dict(data_dict, output_file):
    # Ensure the keys are sorted to maintain proper x-axis sequence
    x_values = sorted(data_dict.keys())
    y_values = [data_dict[key] for key in x_values]

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Gen')
    plt.ylabel('Duplicates')
    plt.title('Cumulative Effective Duplicates over Generations')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

result_dict = {}
plotting_dict = {}
expected_len = 0
i = 1
for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
        for genome in df['genome']:
            expected_len += 1
            hash_value = get_effective_hash(genome)
            if hash_value in result_dict:
                result_dict[hash_value] += 1
            else:
                result_dict[hash_value] = 1
        plotting_dict[i] = count = sum(value-1 for value in result_dict.values() if value > 1)
        i+=1
length = len(result_dict)
count = sum(value-1 for value in result_dict.values() if value > 1)
plot_dict(plotting_dict, plotdir)
print(f'found {length} unique. Expected {expected_len}. Total duplicates: {count}')