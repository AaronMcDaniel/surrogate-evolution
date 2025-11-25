import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os
import numpy as np
import torch.nn.functional as F

# Import custom modules
from codec import Codec
from surrogates.hybrid_dataset import HybridGenomeDataset
from surrogates.valued_hybrid_seqflow import HybridSeqFlow

def train_seqflow():
    # --- Configuration ---
    REG_DATA_PATH = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_train.pkl"
    MAX_SEQ_LEN = 350
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    
    # Model Architecture
    EMBED_DIM = 128
    VALUE_DIM = 64
    FLOW_HIDDEN_DIM = 256
    FLOW_NUM_LAYERS = 4
    SIGMA = 0.1
    
    # Loss weights
    WEIGHT_NLL = 1.0
    WEIGHT_SIM = 1.0
    WEIGHT_VAL = 10.0  # Higher weight for value reconstruction
    
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
    codec._build_hybrid_vocab()
    vocab_size = len(codec.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    dataset = HybridGenomeDataset(
        reg_df=reg_df,
        cls_df=pd.DataFrame(),
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
    print(f"Loss weights - NLL: {WEIGHT_NLL}, Sim: {WEIGHT_SIM}, Val: {WEIGHT_VAL}")
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
            
            # Forward pass - get all three loss components
            L_NLL, L_sim, L_val = model(tokens, values)
            
            # Combined weighted loss
            loss = WEIGHT_NLL * L_NLL + WEIGHT_SIM * L_sim + WEIGHT_VAL * L_val
            
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
        
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Average NLL Loss:   {avg_nll:.4f}")
        print(f"  Average Sim Loss:   {avg_sim:.4f}")
        print(f"  Average Val Loss:   {avg_val:.4f}")
        print(f"  Average Total Loss: {avg_total:.4f}")
        print(f"{'='*70}\n")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"valued_seqflow_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total,
                'loss_components': {
                    'nll': avg_nll,
                    'sim': avg_sim,
                    'val': avg_val
                }
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n")
    
    print("Training complete!")
    
    # --- Test Encoding/Decoding ---
    print("\nTesting encode/decode on sample data points...")
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
            values = sample['values'].unsqueeze(0).to(DEVICE)  # [1, L]
            
            # Encode to z
            z = model.encode(tokens, values)
            print(f"Encoded z shape: {z.shape}")
            print(f"Latent z statistics: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
            
            # Decode back
            decoded_tokens, decoded_values = model.decode(z)
            
            # Calculate reconstruction metrics
            token_acc = (decoded_tokens == tokens).float().mean().item()
            
            # Value MSE only for VAL_NUM tokens
            val_mask = (tokens == model.val_token_idx)
            if val_mask.sum() > 0:
                value_mse = F.mse_loss(decoded_values[val_mask], values[val_mask]).item()
                print(f"Token reconstruction accuracy: {token_acc*100:.2f}%")
                print(f"Value reconstruction MSE (VAL tokens): {value_mse:.6f}")
                
                # Show value comparisons
                print(f"\nValue comparison (first 10 VAL tokens):")
                val_indices = val_mask[0].nonzero(as_tuple=True)[0][:10]
                if len(val_indices) > 0:
                    print(f"Original:  {values[0, val_indices].cpu().numpy()}")
                    print(f"Decoded:   {decoded_values[0, val_indices].cpu().numpy()}")
            else:
                print(f"Token reconstruction accuracy: {token_acc*100:.2f}%")
                print("No VAL_NUM tokens found in this sample")
            
            # Show token comparisons
            print(f"\nFirst 20 tokens comparison:")
            print(f"Original:  {tokens[0, :20].cpu().numpy()}")
            print(f"Decoded:   {decoded_tokens[0, :20].cpu().numpy()}")
            
            # Show genome string (truncated)
            print(f"\nOriginal genome (first 150 chars):")
            print(f"{sample['genome_str'][:150]}...")
            
            if 'fitness' in sample:
                print(f"\nFitness: {sample['fitness'].numpy()}")
    
    # Save final model
    final_path = "valued_seqflow_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'config': {
            'embed_dim': EMBED_DIM,
            'value_dim': VALUE_DIM,
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
    if 'loss_components' in checkpoint:
        comps = checkpoint['loss_components']
        print(f"  NLL: {comps['nll']:.4f}, Sim: {comps['sim']:.4f}, Val: {comps['val']:.4f}")
    
    model.eval()
    
    # --- Test on Random Samples ---
    print(f"\n{'='*80}")
    print(f"Testing on {num_samples} random samples")
    print(f"{'='*80}\n")
    
    import random
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    total_token_acc = 0.0
    total_value_mse = 0.0
    num_with_values = 0
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            print(f"\n--- Sample {i+1}/{num_samples} (Dataset index: {idx}) ---")
            
            sample = dataset[idx]
            tokens = sample['tokens'].unsqueeze(0).to(DEVICE)  # [1, L]
            values = sample['values'].unsqueeze(0).to(DEVICE)  # [1, L]
            genome_str = sample['genome_str']
            
            # Encode to latent space
            z = model.encode(tokens, values)
            print(f"Encoded to latent space z with shape: {z.shape}")
            print(f"Latent z statistics: mean={z.mean().item():.4f}, std={z.std().item():.4f}")
            
            # Decode back
            decoded_tokens, decoded_values = model.decode(z)
            
            # Calculate reconstruction metrics
            token_acc = (decoded_tokens == tokens).float().mean().item()
            total_token_acc += token_acc
            
            print(f"Token reconstruction accuracy: {token_acc*100:.2f}%")
            
            # Value MSE only for VAL_NUM tokens
            val_mask = (tokens == model.val_token_idx)
            if val_mask.sum() > 0:
                value_mse = F.mse_loss(decoded_values[val_mask], values[val_mask]).item()
                total_value_mse += value_mse
                num_with_values += 1
                print(f"Value reconstruction MSE (VAL tokens): {value_mse:.6f}")
                
                # Show value comparisons
                print(f"\nValue comparison (first 10 VAL tokens):")
                val_indices = val_mask[0].nonzero(as_tuple=True)[0][:10]
                if len(val_indices) > 0:
                    print(f"Original:  {values[0, val_indices].cpu().numpy()}")
                    print(f"Decoded:   {decoded_values[0, val_indices].cpu().numpy()}")
            else:
                print("No VAL_NUM tokens found in this sample")
            
            # Show some token comparisons
            print(f"\nFirst 20 tokens comparison:")
            print(f"Original:  {tokens[0, :20].cpu().numpy()}")
            print(f"Decoded:   {decoded_tokens[0, :20].cpu().numpy()}")
            
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
    if num_with_values > 0:
        print(f"Average value reconstruction MSE: {total_value_mse/num_with_values:.6f}")
        print(f"Samples with VAL tokens: {num_with_values}/{num_samples}")
    else:
        print("No samples contained VAL tokens")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test mode: python train_valued_seqflow.py checkpoint_path [num_samples]
        checkpoint_path = sys.argv[1]
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_from_checkpoint(checkpoint_path, num_samples=num_samples)
    else:
        # Training mode
        train_seqflow()