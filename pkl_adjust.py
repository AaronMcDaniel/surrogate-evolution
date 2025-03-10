import pandas as pd
import numpy as np

file_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/old_codec/old_codec_reg_val.pkl'  
df = pd.read_pickle(file_path)

# Randomly remove 200 rows
rows_to_remove = 220
df = df.drop(np.random.choice(df.index, rows_to_remove, replace=False))
# df = df.reset_index(drop=True)

# Save the modified DataFrame back to the same .pkl file
df.to_pickle(file_path)