import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import sys
import os
import numpy as np
import torch.nn.functional as F

# Import custom modules
from codec import Codec
from surrogates.hybrid_seqflow import DiscreteSeqFlow


class DiscreteGenomeDataset(Dataset):
    def __init__(self, reg_df: pd.DataFrame, codec: Codec, max_seq_len: int, mode: str = 'reg'):
        """
        Dataset for fully discretized genomes (no continuous values).
        
        Args:
            reg_df: DataFrame with genome strings
            codec: Codec instance with _build_discrete_vocab
            max_seq_len: Maximum sequence length
            mode: 'reg' for regression mode (includes fitness)
        """
        self.codec = codec
        self.max_seq_len = max_seq_len
        # self.codec._build_discrete_vocab()
        if SPLIT_FLOAT:
            self.codec._build_discrete_float_split_vocab()
        else:
            self.codec._build_discrete_vocab()
        self.vocab_size = len(self.codec.vocab)
        self.pad_token_id = self.codec.vocab["<PAD>"]
        
        self.df = reg_df
        self.has_fitness = (mode == 'reg')
        
        if self.has_fitness:
            self.objectives = ['ciou_loss', 'average_precision']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        genome_str = row['str_genome']
        
        # Encode using discrete encoding (no values array)
        try:
            # tokens = self.codec.encode_discrete(genome_str)
            tokens = self.codec.encode_discrete_float_split(genome_str)
        except Exception as e:
            print(f"Warning: Failed to encode genome {idx}: {e}")
            tokens = np.array([self.pad_token_id] * self.max_seq_len)
        
        # Pad / Truncate
        seq_len = len(tokens)
        
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            padded_tokens = np.pad(tokens, (0, pad_len), 'constant', constant_values=self.pad_token_id)
        else:
            padded_tokens = tokens[:self.max_seq_len]
        
        output = {
            'tokens': torch.tensor(padded_tokens, dtype=torch.long),
            'genome_str': genome_str
        }
        
        if self.has_fitness:
            fitness = [row[obj] for obj in self.objectives]
            output['fitness'] = torch.tensor(fitness, dtype=torch.float32)
        
        return output

SPLIT_FLOAT = True  # Use float split discrete encoding
def train_seqflow():
    # --- Configuration ---
    REG_DATA_PATH = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_train.pkl"
    MAX_SEQ_LEN = 350
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    
    # Model Architecture
    EMBED_DIM = 128
    FLOW_HIDDEN_DIM = 256
    FLOW_NUM_LAYERS = 4
    SIGMA = 0.1
    
    # Loss weights
    WEIGHT_NLL = 1.0
    WEIGHT_SIM = 1.0
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # --- Load Data & Codec ---
    print(f"Loading data from {REG_DATA_PATH}...")
    try:
        reg_df = pd.read_pickle(REG_DATA_PATH)
        print(f"Loaded {len(reg_df)} samples")
        print(f"Columns: {reg_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    codec = Codec(num_classes=1)
    if SPLIT_FLOAT:
        codec._build_discrete_float_split_vocab()
    else:
        codec._build_discrete_vocab()
    vocab_size = len(codec.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    dataset = DiscreteGenomeDataset(
        reg_df=reg_df,
        codec=codec,
        max_seq_len=MAX_SEQ_LEN,
        mode='reg'
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    print(f"Created DataLoader with {len(dataloader)} batches")
    
    # --- Initialize Model ---
    print("Initializing DiscreteSeqFlow model...")
    model = DiscreteSeqFlow(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        seq_len=MAX_SEQ_LEN,
        flow_hidden_dim=FLOW_HIDDEN_DIM,
        flow_num_layers=FLOW_NUM_LAYERS,
        sigma=SIGMA
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    print("\nStarting training...")
    print(f"Loss weights - NLL: {WEIGHT_NLL}, Sim: {WEIGHT_SIM}")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        epoch_nll_loss = 0.0
        epoch_sim_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            tokens = batch['tokens'].to(DEVICE)  # [B, L]
            
            # Forward pass - get both loss components
            L_NLL, L_sim = model(tokens)
            
            # Combined weighted loss
            loss = WEIGHT_NLL * L_NLL + WEIGHT_SIM * L_sim
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_nll_loss += L_NLL.item()
            epoch_sim_loss += L_sim.item()
            epoch_total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
                      f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"NLL: {L_NLL.item():.4f}, "
                      f"Sim: {L_sim.item():.4f}")
        
        # Epoch summary
        avg_nll = epoch_nll_loss / num_batches
        avg_sim = epoch_sim_loss / num_batches
        avg_total = epoch_total_loss / num_batches
        
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Average NLL Loss:   {avg_nll:.4f}")
        print(f"  Average Sim Loss:   {avg_sim:.4f}")
        print(f"  Average Total Loss: {avg_total:.4f}")
        print(f"{'='*70}\n")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            if SPLIT_FLOAT:
                checkpoint_path = f"discrete_seqflow_float_split_epoch{epoch+1}.pt"
            else:
                checkpoint_path = f"discrete_seqflow_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total,
                'loss_components': {
                    'nll': avg_nll,
                    'sim': avg_sim
                }
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n")
    
    print("Training complete!")
    
    # --- Test Encoding/Decoding ---
    print("\nTesting encode/decode on sample data points...")
    
    # Create reverse vocab mapping (id -> token_name)
    id_to_token = {v: k for k, v in codec.vocab.items()}
    
    model.eval()
    with torch.no_grad():
        # Test on 3 random samples
        import random
        test_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
        
        for i, idx in enumerate(test_indices):
            print(f"\n{'='*70}")
            print(f"Test Sample {i+1} (Dataset index: {idx})")
            print(f"{'='*70}")
            
            sample = dataset[idx]
            tokens = sample['tokens'].unsqueeze(0).to(DEVICE)  # [1, L]
            
            # Encode to z
            z = model.encode(tokens)
            print(f"Encoded z shape: {z.shape}")
            print(f"Latent z statistics: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
            
            # Decode back
            decoded_tokens = model.decode(z)
            
            # Calculate reconstruction metrics
            token_acc = (decoded_tokens == tokens).float().mean().item()
            print(f"Token reconstruction accuracy: {token_acc*100:.2f}%")
            
            # Show token comparisons
            print(f"\nFirst 30 tokens comparison:")
            print(f"Original:  {tokens[0, :30].cpu().numpy()}")
            print(f"Decoded:   {decoded_tokens[0, :30].cpu().numpy()}")
            
            # Show token names
            print(f"\nFirst 30 token names:")
            orig_names = [id_to_token.get(tok.item(), f"UNK_{tok.item()}") for tok in tokens[0, :30]]
            dec_names = [id_to_token.get(tok.item(), f"UNK_{tok.item()}") for tok in decoded_tokens[0, :30]]
            
            print("Original tokens:")
            for j, name in enumerate(orig_names):
                print(f"  [{j}] {name}")
            
            print("\nDecoded tokens:")
            for j, name in enumerate(dec_names):
                marker = "✓" if name == orig_names[j] else "✗"
                print(f"  [{j}] {name} {marker}")
            
            # Count mismatches
            mismatches = (decoded_tokens != tokens).sum().item()
            print(f"\nTotal mismatches: {mismatches}/{tokens.shape[1]}")
            
            # Show genome string (truncated)
            print(f"\nOriginal genome (first 150 chars):")
            print(f"{sample['genome_str'][:150]}...")
            
            if 'fitness' in sample:
                print(f"\nFitness: {sample['fitness'].numpy()}")
    
    # Save final model
    if SPLIT_FLOAT:
        final_path = "discrete_seqflow_float_split_final.pt"
    else:
        final_path = "discrete_seqflow_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'config': {
            'embed_dim': EMBED_DIM,
            'seq_len': MAX_SEQ_LEN,
            'flow_hidden_dim': FLOW_HIDDEN_DIM,
            'flow_num_layers': FLOW_NUM_LAYERS,
            'sigma': SIGMA
        }
    }, final_path)
    print(f"\nSaved final model to {final_path}")


def test_from_checkpoint(checkpoint_path, data_path=None, num_samples=5):
    """
    Load a trained model checkpoint and test encoding/decoding on random samples.
    
    Args:
        checkpoint_path: Path to the saved checkpoint .pt file
        data_path: Path to the pickle file. If None, uses the default
        num_samples: Number of random samples to test
    """
    # --- Configuration (should match training) ---
    if data_path is None:
        data_path = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_train.pkl"
    
    MAX_SEQ_LEN = 350
    EMBED_DIM = 128
    FLOW_HIDDEN_DIM = 256
    FLOW_NUM_LAYERS = 4
    SIGMA = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # --- Load Data ---
    print(f"Loading data from {data_path}...")
    try:
        reg_df = pd.read_pickle(data_path)
        print(f"Loaded {len(reg_df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # --- Initialize Codec ---
    print("Initializing Codec...")
    codec = Codec(num_classes=1)
    # codec._build_discrete_vocab()
    if SPLIT_FLOAT:
        codec._build_discrete_float_split_vocab()
    else:
        codec._build_discrete_vocab()
    vocab_size = len(codec.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # --- Create Dataset ---
    print("Creating dataset...")
    dataset = DiscreteGenomeDataset(
        reg_df=reg_df,
        codec=codec,
        max_seq_len=MAX_SEQ_LEN,
        mode='reg'
    )
    
    # --- Initialize Model ---
    print("Initializing model...")
    model = DiscreteSeqFlow(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        seq_len=MAX_SEQ_LEN,
        flow_hidden_dim=FLOW_HIDDEN_DIM,
        flow_num_layers=FLOW_NUM_LAYERS,
        sigma=SIGMA
    ).to(DEVICE)
    
    # --- Load Checkpoint ---
    print("Loading model weights...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    # print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    if 'loss_components' in checkpoint:
        comps = checkpoint['loss_components']
        print(f"  NLL: {comps['nll']:.4f}, Sim: {comps['sim']:.4f}")
    
    model.eval()
    
    # Create reverse vocab mapping (id -> token_name)
    id_to_token = {v: k for k, v in codec.vocab.items()}
    
    # --- Test on Random Samples ---
    print(f"\n{'='*80}")
    print(f"Testing on {num_samples} random samples")
    print(f"{'='*80}\n")
    
    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    total_token_acc = 0.0
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            print(f"\n--- Sample {i+1}/{num_samples} (Dataset index: {idx}) ---")
            
            sample = dataset[idx]
            tokens = sample['tokens'].unsqueeze(0).to(DEVICE)  # [1, L]
            genome_str = sample['genome_str']
            
            # Encode to latent space
            z = model.encode(tokens)
            print(f"Encoded to latent space z with shape: {z.shape}")
            print(f"Latent z statistics: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
            
            # Decode back
            decoded_tokens = model.decode(z)
            
            # Calculate reconstruction metrics
            token_acc = (decoded_tokens == tokens).float().mean().item()
            total_token_acc += token_acc
            
            print(f"Token reconstruction accuracy: {token_acc*100:.2f}%")
            
            # Show some token comparisons
            print(f"\nFirst 30 tokens comparison:")
            print(f"Original:  {tokens[0, :30].cpu().numpy()}")
            print(f"Decoded:   {decoded_tokens[0, :30].cpu().numpy()}")
            
            # Show token names
            print(f"\nFirst 30 token names:")
            orig_names = [id_to_token.get(tok.item(), f"UNK_{tok.item()}") for tok in tokens[0, :30]]
            dec_names = [id_to_token.get(tok.item(), f"UNK_{tok.item()}") for tok in decoded_tokens[0, :30]]
            
            print("Original tokens:")
            for j, name in enumerate(orig_names):
                print(f"  [{j}] {name}")
            
            print("\nDecoded tokens:")
            for j, name in enumerate(dec_names):
                marker = "✓" if name == orig_names[j] else "✗"
                print(f"  [{j}] {name} {marker}")
            
            # Count mismatches
            mismatches = (decoded_tokens != tokens).sum().item()
            print(f"\nTotal mismatches: {mismatches}/{tokens.shape[1]}")
            
            # Show genome string (truncated)
            print(f"\nOriginal genome (first 150 chars):")
            print(f"{genome_str[:150]}...")
            
            if 'fitness' in sample:
                print(f"\nFitness: {sample['fitness'].numpy()}")
    
    # --- Summary Statistics ---
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Average token reconstruction accuracy: {(total_token_acc/num_samples)*100:.2f}%")
    print(f"{'='*80}\n")


def validate_on_full_dataset(checkpoint_path, val_data_path=None):
    """
    Run validation on the entire validation dataset to compute reconstruction accuracy.
    
    Args:
        checkpoint_path: Path to the saved checkpoint .pt file
        val_data_path: Path to validation pickle file. If None, uses default
    """
    # --- Configuration (should match training) ---
    if val_data_path is None:
        val_data_path = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_val.pkl"
    
    MAX_SEQ_LEN = 350
    EMBED_DIM = 128
    FLOW_HIDDEN_DIM = 256
    FLOW_NUM_LAYERS = 4
    SIGMA = 0.1
    BATCH_SIZE = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # --- Load Validation Data ---
    print(f"Loading validation data from {val_data_path}...")
    try:
        val_df = pd.read_pickle(val_data_path)
        print(f"Loaded {len(val_df)} validation samples")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return
    
    # --- Initialize Codec ---
    print("Initializing Codec...")
    codec = Codec(num_classes=1)
    if SPLIT_FLOAT:
        codec._build_discrete_float_split_vocab()
    else:
        codec._build_discrete_vocab()
    vocab_size = len(codec.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # --- Create Dataset ---
    print("Creating validation dataset...")
    val_dataset = DiscreteGenomeDataset(
        reg_df=val_df,
        codec=codec,
        max_seq_len=MAX_SEQ_LEN,
        mode='reg'
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # --- Initialize Model ---
    print("Initializing model...")
    model = DiscreteSeqFlow(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        seq_len=MAX_SEQ_LEN,
        flow_hidden_dim=FLOW_HIDDEN_DIM,
        flow_num_layers=FLOW_NUM_LAYERS,
        sigma=SIGMA
    ).to(DEVICE)
    
    # --- Load Checkpoint ---
    print("Loading model weights...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint training loss: {checkpoint['loss']:.4f}")
    if 'loss_components' in checkpoint:
        comps = checkpoint['loss_components']
        print(f"  NLL: {comps['nll']:.4f}, Sim: {comps['sim']:.4f}")
    
    model.eval()
    
    # --- Run Validation ---
    print(f"\n{'='*80}")
    print(f"Running validation on {len(val_dataset)} samples")
    print(f"{'='*80}\n")
    
    total_samples = 0
    total_correct_tokens = 0
    total_tokens = 0
    perfect_reconstructions = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            tokens = batch['tokens'].to(DEVICE)  # [B, L]
            batch_size = tokens.shape[0]
            
            # Encode to latent space
            z = model.encode(tokens)
            
            # Decode back
            decoded_tokens = model.decode(z)
            
            # Calculate metrics
            correct = (decoded_tokens == tokens).float()
            total_correct_tokens += correct.sum().item()
            total_tokens += tokens.numel()
            
            # Count perfect reconstructions (all tokens match)
            perfect_batch = (decoded_tokens == tokens).all(dim=1).sum().item()
            perfect_reconstructions += perfect_batch
            
            total_samples += batch_size
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                current_acc = (total_correct_tokens / total_tokens) * 100
                current_perfect = (perfect_reconstructions / total_samples) * 100
                print(f"Batch [{batch_idx+1}/{len(val_loader)}] | "
                      f"Samples: {total_samples}/{len(val_dataset)} | "
                      f"Token Acc: {current_acc:.2f}% | "
                      f"Perfect: {current_perfect:.2f}%")
    
    # --- Final Statistics ---
    overall_token_acc = (total_correct_tokens / total_tokens) * 100
    overall_perfect_acc = (perfect_reconstructions / total_samples) * 100
    
    print(f"\n{'='*80}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Correct tokens: {total_correct_tokens:,}")
    print(f"Token-level accuracy: {overall_token_acc:.4f}%")
    print(f"Perfect reconstructions: {perfect_reconstructions}/{total_samples}")
    print(f"Perfect reconstruction rate: {overall_perfect_acc:.4f}%")
    print(f"{'='*80}\n")
    
    # Save validation results
    results = {
        'checkpoint_path': checkpoint_path,
        'checkpoint_epoch': checkpoint['epoch'],
        'val_samples': total_samples,
        'total_tokens': total_tokens,
        'correct_tokens': total_correct_tokens,
        'token_accuracy': overall_token_acc,
        'perfect_reconstructions': perfect_reconstructions,
        'perfect_reconstruction_rate': overall_perfect_acc
    }
    
    results_path = checkpoint_path.replace('.pt', '_validation_results.txt')
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved validation results to: {results_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            # Validation mode: python train_discrete_seqflow.py validate checkpoint_path
            if len(sys.argv) < 3:
                print("Usage: python train_discrete_seqflow.py validate checkpoint_path")
                sys.exit(1)
            checkpoint_path = sys.argv[2]
            validate_on_full_dataset(checkpoint_path)
        else:
            # Test mode: python train_discrete_seqflow.py checkpoint_path [num_samples]
            checkpoint_path = sys.argv[1]
            num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            test_from_checkpoint(checkpoint_path, num_samples=num_samples)
    else:
        # Training mode
        train_seqflow()
