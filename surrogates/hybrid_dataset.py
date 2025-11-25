# Create a new file: surrogates/hybrid_dataset.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from codec import Codec # Your existing Codec
import traceback

class HybridGenomeDataset(Dataset):
    def __init__(self, reg_df: pd.DataFrame, cls_df: pd.DataFrame, codec: Codec, max_seq_len: int, mode: str = 'all'):
        """
        mode: 'all' (for SeqFlow training), 'reg' (for Surrogate training)
        """
        self.codec = codec
        self.max_seq_len = max_seq_len
        self.codec._build_hybrid_vocab() # Ensure vocab is built
        self.vocab_size = len(self.codec.vocab)
        self.pad_token_id = self.codec.vocab["<PAD>"]

        if mode == 'all':
            self.df = pd.concat([reg_df, cls_df], ignore_index=True)
            self.has_fitness = False
        elif mode == 'reg':
            self.df = reg_df
            self.has_fitness = True
            # Define your target objectives here
            self.objectives = ['ciou_loss', 'average_precision']
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        genome_str = row['str_genome']

        # --- 1. Encode ---
        try:
            tokens, values = self.codec.encode_hybrid(genome_str)

        except Exception as e:
            print(f"Warning: Failed to encode genome {idx}: {e}")
        
            # Return dummy data on failure
            tokens = np.array([self.pad_token_id] * self.max_seq_len)
            values = np.zeros(self.max_seq_len)

        # --- 2. Pad / Truncate ---
        seq_len = len(tokens)
        
        # Pad
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            padded_tokens = np.pad(tokens, (0, pad_len), 'constant', constant_values=self.pad_token_id)
            padded_values = np.pad(values, (0, pad_len), 'constant', constant_values=0.0)
        # Truncate
        else:
            padded_tokens = tokens[:self.max_seq_len]
            padded_values = values[:self.max_seq_len]

        output = {
            'tokens': torch.tensor(padded_tokens, dtype=torch.long),
            'values': torch.tensor(padded_values, dtype=torch.float32),
            'genome_str': genome_str
        }

        # --- 3. Get Fitness (for surrogate training) ---
        if self.has_fitness:
            fitness = [row[obj] for obj in self.objectives]
            output['fitness'] = torch.tensor(fitness, dtype=torch.float32)

        return output