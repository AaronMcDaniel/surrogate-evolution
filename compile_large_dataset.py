import pickle
import pandas as pd
from surrogates.surrogate_dataset import build_dataset
import os

paths = [
    'full_vae_30',
    'full_vae_30_2',
    'full_vae_30_3',
    'full_vae_30_4',
    'full_vae_30_5',
    'full_baseline_30',
    'full_baseline_30_2',
    'full_baseline_30_3',
    'full_baseline_30_4',
    'full_ssi_30',
    'full_ssi_30_2',
    'full_ssi_30_3',
    'full_ssi_30_4',
    'full_ssi_30_5',
    'full_no_pretrain_30',
    'full_30'
]
more_paths = [
    'light_baseline_30',
    'light_baseline_30_2',
    'light_baseline_30_3',
    'light_baseline_30_4',
    'light_samemut_30',
    'light_samemut_30_2',
    'light_samemut_30_3',
    'light_samemut_30_4',
    'light_simplify_30',
    'light_simplify_30_2',
    'light_simplify_30_3',
    'light_simplify_30_4',
]
output_dir = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/'
temp_output_dir = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/temp/'
def create_full_30_csv():
    global paths
    full_df = None
    for path in paths:
        df = pd.read_csv(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/{path}/out.csv')
        if full_df is None:
            full_df = df
        else:
            full_df = pd.concat([full_df, df], ignore_index=True)

    full_df = full_df.drop_duplicates(subset=["hash"])

    full_out = os.path.join(output_dir, 'full_out.csv')
    full_df.to_csv(full_out, index=False)

    print(f"Total unique entries: {len(full_df)}")


def create_full_30_dataset():
    global paths
    for path in paths:
        working_dir = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/' + path
        csv = os.path.join(working_dir, 'out.csv')
        build_dataset("temp_full_30_dataset", csv, working_dir, temp_output_dir, val_ratio=0.2)
        print(f"Created dataset for {path}", flush=True)
        if os.path.exists(os.path.join(output_dir, 'full_30_dataset_reg_val.pkl')):
            for name in ['reg_train', 'reg_val', 'cls_train', 'cls_val']:
                df1 = pd.read_pickle(os.path.join(output_dir, f'full_30_dataset_{name}.pkl'))
                df2 = pd.read_pickle(os.path.join(temp_output_dir, f'temp_full_30_dataset_{name}.pkl'))
                combined_df = pd.concat([df1, df2], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["hash"])
                combined_df.to_pickle(os.path.join(output_dir, f'full_30_dataset_{name}.pkl'))
                print(f"Updated full_30_dataset_{name}.pkl with {len(df2)} new samples", flush=True)
        else:
            for name in ['reg_train', 'reg_val', 'cls_train', 'cls_val']:
                df = pd.read_pickle(os.path.join(temp_output_dir, f'temp_full_30_dataset_{name}.pkl'))
                df.to_pickle(os.path.join(output_dir, f'full_30_dataset_{name}.pkl'))
                print(f"Copied full_30_dataset_{name}.pkl with {len(df)} samples", flush=True)

def create_mix_dataset():
    global paths, more_paths
    more_paths.extend(paths)
    for path in more_paths:
        working_dir = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/' + path
        csv = os.path.join(working_dir, 'out.csv')
        build_dataset("temp_mix_dataset", csv, working_dir, temp_output_dir, val_ratio=0.3, return_raw_genomes=True)
        print(f"Created dataset for {path}", flush=True)
        if os.path.exists(os.path.join(output_dir, 'mix_dataset_reg_val.pkl')):
            for name in ['reg_train', 'reg_val', 'cls_train', 'cls_val']:
                df1 = pd.read_pickle(os.path.join(output_dir, f'mix_dataset_{name}.pkl'))
                df2 = pd.read_pickle(os.path.join(temp_output_dir, f'temp_mix_dataset_{name}.pkl'))
                combined_df = pd.concat([df1, df2], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["hash"])
                combined_df.to_pickle(os.path.join(output_dir, f'mix_dataset_{name}.pkl'))
                print(f"Updated mix_dataset_{name}.pkl with {len(df2)} new samples", flush=True)
        else:
            for name in ['reg_train', 'reg_val', 'cls_train', 'cls_val']:
                df = pd.read_pickle(os.path.join(temp_output_dir, f'temp_mix_dataset_{name}.pkl'))
                df.to_pickle(os.path.join(output_dir, f'mix_dataset_{name}.pkl'))
                print(f"Copied mix_dataset_{name}.pkl with {len(df)} samples", flush=True)

def view_dataset(name):
    reg_train_set = pd.read_pickle(os.path.join(output_dir, f'{name}_dataset_reg_train.pkl'))
    reg_val_set = pd.read_pickle(os.path.join(output_dir, f'{name}_dataset_reg_val.pkl'))
    cls_train_set = pd.read_pickle(os.path.join(output_dir, f'{name}_dataset_cls_train.pkl'))
    cls_val_set = pd.read_pickle(os.path.join(output_dir, f'{name}_dataset_cls_val.pkl'))

    print(f"Regression train set size: {len(reg_train_set)}")
    print(f"Regression val set size: {len(reg_val_set)}")
    print(f"Classification train set size: {len(cls_train_set)}")
    print(f"Classification val set size: {len(cls_val_set)}")
    
create_mix_dataset()
view_dataset("mix")
