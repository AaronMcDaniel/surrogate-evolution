"""
Test script to train the HybridSeqFlow model using hybrid_dataset.py
on the regression dataframe.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

# Import custom modules
from codec import Codec
from surrogates.hybrid_dataset import HybridGenomeDataset
from surrogates.dual_arch_seqflow import HybridSeqFlow

def train_seqflow():
    # --- Configuration ---
    REG_DATA_PATH = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_train.pkl"
    MAX_SEQ_LEN = 350  # Adjust based on your genome lengths
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    EMBED_DIM = 128
    VALUE_DIM = 32
    FLOW_HIDDEN_DIM = 256
    FLOW_NUM_LAYERS = 2
    SIGMA = 0.1
    
    # Loss Weights
    W_NLL = 1.0
    W_SIM = 1.0
    W_VAL = 10.0 # Give higher weight to value reconstruction to ensure it converges
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # --- Load Data ---
    print(f"Loading regression data from {REG_DATA_PATH}...")
    try:
        reg_df = pd.read_pickle(REG_DATA_PATH)
        print(f"Loaded {len(reg_df)} samples")
        print(f"Columns: {reg_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # --- Initialize Codec ---
    print("Initializing Codec...")
    codec = Codec(num_classes=7)  # Adjust num_classes if needed
    codec._build_hybrid_vocab()
    vocab_size = len(codec.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # --- Create Dataset ---
    print("Creating dataset...")
    dataset = HybridGenomeDataset(
        reg_df=reg_df,
        cls_df=pd.DataFrame(),  # Empty cls_df
        codec=codec,
        max_seq_len=MAX_SEQ_LEN,
        mode='reg'
    )
    
    # --- Create DataLoader ---
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    print(f"Created DataLoader with {len(dataloader)} batches")
    
    # --- Initialize Model ---
    print("Initializing HybridSeqFlow model...")
    model = HybridSeqFlow(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        value_dim=VALUE_DIM,
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
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        epoch_nll_loss = 0.0
        epoch_sim_loss = 0.0
        epoch_val_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            tokens = batch['tokens'].to(DEVICE)  # [B, L]
            values = batch['values'].to(DEVICE)  # [B, L]
            
            # Forward pass - Now returns 3 losses
            L_NLL, L_sim, L_val = model(tokens, values)
            
            # Combined loss
            loss = (W_NLL * L_NLL) + (W_SIM * L_sim) + (W_VAL * L_val)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_nll_loss += L_NLL.item()
            epoch_sim_loss += L_sim.item()
            epoch_val_loss += L_val.item()
            epoch_total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
                      f"Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"NLL: {L_NLL.item():.4f}, "
                      f"Sim: {L_sim.item():.4f}, "
                      f"Val: {L_val.item():.4f}")
        
        # Epoch summary
        avg_nll = epoch_nll_loss / num_batches
        avg_sim = epoch_sim_loss / num_batches
        avg_val = epoch_val_loss / num_batches
        avg_total = epoch_total_loss / num_batches
        
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Average NLL Loss: {avg_nll:.4f}")
        print(f"  Average Sim Loss: {avg_sim:.4f}")
        print(f"  Average Val Loss: {avg_val:.4f}")
        print(f"  Average Total Loss: {avg_total:.4f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"hybrid_seqflow_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n")
    
    print("Training complete!")
    
    # --- Test Encoding/Decoding ---
    print("\nTesting encode/decode on a sample...")
    model.eval()
    with torch.no_grad():
        # Get a sample
        sample = dataset[0]
        tokens = sample['tokens'].unsqueeze(0).to(DEVICE)  # [1, L]
        values = sample['values'].unsqueeze(0).to(DEVICE)  # [1, L]
        
        # Encode to z
        z = model.encode(tokens, values)
        print(f"Encoded z shape: {z.shape}")
        print(f"Encoded z sample values: {z[0, :5, :5]}")
        
        # Decode back
        decoded_tokens, decoded_values = model.decode(z)
        print(f"Decoded tokens shape: {decoded_tokens.shape}")
        print(f"Decoded values shape: {decoded_values.shape}")
        
        # Check reconstruction accuracy
        token_acc = (decoded_tokens == tokens).float().mean().item()
        print(f"\nToken reconstruction accuracy: {token_acc*100:.2f}%")
        
        # Check value reconstruction MSE (only for VAL_NUM tokens)
        mask = (tokens == 3) # Assuming VAL_NUM is 3
        if mask.sum() > 0:
            val_mse = nn.functional.mse_loss(decoded_values[mask], values[mask]).item()
            print(f"Value reconstruction MSE (on valid tokens): {val_mse:.6f}")
        
        # Print original genome
        print(f"\nOriginal genome (first 100 chars):")
        print(sample['genome_str'][:100])

def test_from_checkpoint(checkpoint_path, data_path=None, num_samples=5):
    """
    Load a trained model checkpoint and test encoding/decoding on random samples.
    
    Args:
        checkpoint_path: Path to the saved checkpoint .pt file
        data_path: Path to the pickle file. If None, uses the default from train_seqflow
        num_samples: Number of random samples to test
    """
    # --- Configuration (should match training) ---
    if data_path is None:
        data_path = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_train.pkl"
    
    MAX_SEQ_LEN = 350
    EMBED_DIM = 128
    VALUE_DIM = 32
    FLOW_HIDDEN_DIM = 256
    FLOW_NUM_LAYERS = 2
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
    codec._build_hybrid_vocab()
    vocab_size = len(codec.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # --- Create Dataset ---
    print("Creating dataset...")
    dataset = HybridGenomeDataset(
        reg_df=reg_df,
        cls_df=pd.DataFrame(),
        codec=codec,
        max_seq_len=MAX_SEQ_LEN,
        mode='reg'
    )
    
    # --- Initialize Model ---
    print("Initializing model...")
    model = HybridSeqFlow(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        value_dim=VALUE_DIM,
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
    print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    
    model.eval()
    
    # --- Test on Random Samples ---
    print(f"\n{'='*80}")
    print(f"Testing on {num_samples} random samples")
    print(f"{'='*80}\n")
    
    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    total_token_acc = 0.0
    total_value_mse = 0.0
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            print(f"\n--- Sample {i+1}/{num_samples} (Dataset index: {idx}) ---")
            
            sample = dataset[idx]
            tokens = sample['tokens'].unsqueeze(0).to(DEVICE)  # [1, L]
            print(f"Original tokens", tokens)
            values = sample['values'].unsqueeze(0).to(DEVICE)  # [1, L]
            print(f"Original values", values)
            genome_str = sample['genome_str']
            
            # Encode to latent space
            z = model.encode(tokens, values)
            print(f"Encoded to latent space z with shape: {z.shape}")
            print(f"Latent z statistics: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
            
            # Decode back
            decoded_tokens, decoded_values = model.decode(z)
            
            # Calculate reconstruction metrics
            token_acc = (decoded_tokens == tokens).float().mean().item()
            value_mse = ((decoded_values - values) ** 2).mean().item()
            
            total_token_acc += token_acc
            total_value_mse += value_mse
            
            print(f"Token reconstruction accuracy: {token_acc*100:.2f}%")
            print(f"Value reconstruction MSE: {value_mse:.6f}")
            
            # Show some token comparisons
            print(f"\nFirst 20 tokens comparison:")
            print(f"Original:  {tokens[0, :20].cpu().numpy()}")
            print(f"Decoded:   {decoded_tokens[0, :20].cpu().numpy()}")
            
            print(f"\nFirst 10 values comparison:")
            print(f"Original:  {values[0, :10].cpu().numpy()}")
            print(f"Decoded:   {decoded_values[0, :10].cpu().numpy()}")
            
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
    print(f"Average value reconstruction MSE: {total_value_mse/num_samples:.6f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test mode: python test2.py checkpoint_path [num_samples]
        checkpoint_path = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_from_checkpoint(checkpoint_path, num_samples=num_samples)
    else:
        # Training mode
        train_seqflow()
