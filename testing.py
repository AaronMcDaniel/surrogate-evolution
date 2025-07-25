import pandas as pd

# strpre = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/test/surrogate_train_data/reg_train"
strpre = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/test/temp_surrogate_datasets/surr_evolution_reg_train"
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_pickle(f'{strpre}.pkl')
# iterate over each column in the DataFrame of the first row and print the item's type
# for col in df.columns:
#     print(f"Column: {col}, Type: {type(df[col][0])}")
df.to_csv(f'{strpre}.csv', index=False)