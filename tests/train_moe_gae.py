"""
Train MoE Grammar AE on mix_dataset_reg files and save latent representations.

This script:
1. Loads train and validation datasets from mix_dataset_reg_{train,val}.pkl
2. Creates "Block-Expanded" versions of these datasets (rows with 68-vectors instead of 1021)
3. Trains the MoEGrammarAE on the block data
4. Applies the trained AE to encode the datasets in two ways:
   a. Block-wise encoding (saving to moe_block_mix_dataset_reg_{train,val}.pkl)
   b. Full 1021 global encoding (saving to moe_global_mix_dataset_reg_{train,val}.pkl)
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

from surrogates.gae_unvectorized import (
    MoEGrammarAE, 
    grammar_loss_function, 
    preprocess_encoding,
    expand_dataframe_to_blocks,
    get_latent_representation,
    weights_init,
    collapse_to_physical
)

# Constants
LATENT_DIM = 16 
INPUT_DIM = 68
NUM_EXPERTS = 54
PARAM_DIM = 14

def train_ae(ae, train_loader, val_loader, epochs=200, lr=1e-3, device=None, alpha_type=1.0, alpha_param=1.0, alpha_reg=0.1):
    """Train the MoEGrammarAE model with validation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae = ae.to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    
    ae.train()
    for epoch in range(epochs):
        data_iter = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        total_loss = 0
        ctrt = 0
        for raw_encoding_batch, _ in data_iter:
            # raw_encoding_batch: [B, 1021]
            raw_encoding_batch = raw_encoding_batch.to(device)
            
            # 1. Preprocess
            epochs_batch, blocks = preprocess_encoding(raw_encoding_batch)
            # blocks: [B, 15, 68]
            
            optimizer.zero_grad()
            
            # 2. Forward
            recon_types, recon_params, z = ae(blocks)
            
            # 3. Loss
            loss = grammar_loss_function(recon_types, recon_params, blocks, 
                                        alpha_type=alpha_type, 
                                        alpha_param=alpha_param, 
                                        alpha_reg=alpha_reg)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            data_iter.set_postfix(loss=loss.item())
            ctrt += 1

        # Validation Loss Calculation
        ae.eval()
        val_loss = 0
        ctrv = 0
        with torch.no_grad():
            for raw_encoding_batch, _ in val_loader:
                raw_encoding_batch = raw_encoding_batch.to(device)
                
                # Preprocess
                epochs_batch, blocks = preprocess_encoding(raw_encoding_batch)
                
                # Forward
                recon_types, recon_params, z = ae(blocks)
                
                # Loss
                loss = grammar_loss_function(recon_types, recon_params, blocks,
                                            alpha_type=alpha_type,
                                            alpha_param=alpha_param,
                                            alpha_reg=alpha_reg)
                val_loss += loss.item()
                ctrv += 1
        ae.train()
        
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/ctrt:.6f}")
        print(f"Epoch {epoch+1}: Validation Loss = {val_loss/ctrv:.6f}")
    
    return ae

def load_data(train_path, val_path):
    """Load training and validation dataframes."""
    print(f"Loading training data from {train_path}...")
    with open(train_path, 'rb') as f:
        train_df = pickle.load(f)
    
    print(f"Loading validation data from {val_path}...")
    with open(val_path, 'rb') as f:
        val_df = pickle.load(f)
    
    print(f"Train dataset size: {len(train_df)}")
    print(f"Val dataset size: {len(val_df)}")
    print(f"Genome shape: {train_df['genome'].iloc[0].shape}")
    
    return train_df, val_df

def prepare_dataloaders(train_df, val_df, batch_size=16):
    """
    Prepare DataLoaders from original 1021-vector dataframes.
    Note: The train_ae loop handles the block expansion/preprocessing internally
    so we can just pass the raw 1021 vectors here.
    """
    # Stack genomes (No scaling for MoE AE - it expects raw physical values/one-hots)
    train_genomes = np.stack(train_df['genome'].values)
    val_genomes = np.stack(val_df['genome'].values)
    
    print(f"Training genomes shape: {train_genomes.shape}")
    print(f"Validation genomes shape: {val_genomes.shape}")
    
    # Convert to tensors
    train_tensors = torch.from_numpy(train_genomes).float()
    val_tensors = torch.from_numpy(val_genomes).float()
    
    # Create dummy labels (not used)
    train_labels = torch.zeros(len(train_tensors))
    val_labels = torch.zeros(len(val_tensors))
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_tensors, train_labels)
    val_dataset = TensorDataset(val_tensors, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def prepare_block_dataframe(df):
    """Wrapper to expand DF to blocks."""
    print("Expanding DataFrame to 68-vector blocks...")
    return expand_dataframe_to_blocks(df)

def save_transformed_datasets(ae, train_df, val_df, 
                            train_block_out, val_block_out,
                            train_global_out, val_global_out, 
                            device):
    """Apply AE encoding in two modes and save."""
    
    # 1. Block-Level Encoding (Applies AE on 68-vectors)
    print("\n--- Generating Block-Level Encodings ---")
    
    # Expand to blocks first
    train_df_blocks = prepare_block_dataframe(train_df)
    val_df_blocks = prepare_block_dataframe(val_df)
    
    print(f"Expanded Train Blocks: {len(train_df_blocks)}")
    print(f"Expanded Val Blocks: {len(val_df_blocks)}")
    
    print("Encoding Train Blocks...")
    # use_full_encoding=False triggers the 68-vector logic
    train_blocks_encoded = get_latent_representation(ae, train_df_blocks, device=device, use_full_encoding=False)
    
    print("Encoding Val Blocks...")
    val_blocks_encoded = get_latent_representation(ae, val_df_blocks, device=device, use_full_encoding=False)
    
    print(f"Saving Block Encodings to {train_block_out}...")
    with open(train_block_out, 'wb') as f:
        pickle.dump(train_blocks_encoded, f)
    with open(val_block_out, 'wb') as f:
        pickle.dump(val_blocks_encoded, f)
        
    # 2. Global Encoding (Applies AE on 1021-vectors to get global latents)
    print("\n--- Generating Global Encodings (1021 -> Global Latent) ---")
    
    print("Encoding Train Global...")
    # use_full_encoding=True triggers the 1021-vector logic
    train_global_encoded = get_latent_representation(ae, train_df, device=device, use_full_encoding=True)
    
    print("Encoding Val Global...")
    val_global_encoded = get_latent_representation(ae, val_df, device=device, use_full_encoding=True)
    
    print(f"Saving Global Encodings to {train_global_out}...")
    with open(train_global_out, 'wb') as f:
        pickle.dump(train_global_encoded, f)
    with open(val_global_out, 'wb') as f:
        pickle.dump(val_global_encoded, f)
    
    print("\nTransformation complete!")

def reconstruct_from_model(ae, raw_encoding, device):
    """
    Reconstruct a 1021-vector through the autoencoder.
    
    Args:
        ae: Trained MoEGrammarAE model
        raw_encoding: Tensor of shape [B, 1021] or [1021]
        device: torch device
        
    Returns:
        reconstructed: Tensor of shape [B, 1021] or [1021] (reconstructed encoding)
    """
    ae.eval()
    
    # Handle single sample
    if raw_encoding.dim() == 1:
        raw_encoding = raw_encoding.unsqueeze(0)
        single_sample = True
    else:
        single_sample = False
    
    raw_encoding = raw_encoding.to(device)
    
    with torch.no_grad():
        # Preprocess to blocks
        epochs_batch, blocks = preprocess_encoding(raw_encoding)
        # blocks: [B, 15, 68]
        
        # Forward pass
        # recon_types: [B, 15, 54] (Logits)
        # param_stack: [B, 15, 54, 14] (All experts)
        # z: [B, 15, latent_dim]
        recon_types, param_stack, z = ae(blocks)
        
        # CRITICAL FIX: Collapse MoE stack to physical representation
        # physical_blocks: [B, 15, 68]
        physical_blocks = collapse_to_physical(recon_types, param_stack)
        
        # MASKING Logic (Ghost Layer Strategy)
        # We must zero out blocks that correspond to zero-inputs to match 1021 format
        # Check original blocks for zeros
        is_valid = ~torch.all(torch.abs(blocks) < 1e-6, dim=2) # [B, 15]
        mask = is_valid.float().unsqueeze(2) # [B, 15, 1]
        
        physical_blocks = physical_blocks * mask
        
        # Reconstruct the 1021 vector
        batch_size = physical_blocks.shape[0]
        reconstructed = torch.zeros(batch_size, 1021, device=device)
        
        # First element is epoch count
        reconstructed[:, 0] = epochs_batch.squeeze(1)
        
        # For each block (15 blocks)
        # We flatten the (B, 15, 68) back to (B, 1020)
        # Note: The Codec format is Column-Major flattening of (68, 15)
        # blocks is (B, 15, 68). Transpose to (B, 68, 15) then flatten.
        
        flat_genome = physical_blocks.transpose(1, 2).reshape(batch_size, -1) # [B, 1020]
        
        reconstructed[:, 1:] = flat_genome
    
    if single_sample:
        return reconstructed.squeeze(0)
    return reconstructed


def visualize_reconstruction(model_path, val_path, num_samples=5, device=None):
    """
    Load a trained model and visualize original vs reconstructed samples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"RECONSTRUCTION VISUALIZATION")
    print(f"{'='*80}")
    print(f"Using device: {device}")
    
    # Load validation data
    print(f"\nLoading validation data from {val_path}...")
    with open(val_path, 'rb') as f:
        val_df = pickle.load(f)
    
    print(f"Validation dataset size: {len(val_df)}")
    
    # Initialize and load model
    print(f"\nLoading model from {model_path}...")
    ae = MoEGrammarAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, 
                                num_experts=NUM_EXPERTS, param_dim=PARAM_DIM)
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae = ae.to(device)
    ae.eval()
    
    # Select random samples
    sample_indices = np.random.choice(len(val_df), size=min(num_samples, len(val_df)), replace=False)
    
    print(f"\n{'='*80}")
    print(f"Visualizing {len(sample_indices)} random samples")
    print(f"{'='*80}\n")
    np.set_printoptions(threshold=np.inf)
    for idx, sample_idx in enumerate(sample_indices):
        original = torch.from_numpy(val_df['genome'].iloc[sample_idx]).float()
        
        # Pass to reconstruct_from_model (it handles device internally, but input should be on CPU initially)
        reconstructed = reconstruct_from_model(ae, original, device).cpu()
        
        # Calculate metrics
        mse = torch.mean((original - reconstructed) ** 2).item()
        mae = torch.mean(torch.abs(original - reconstructed)).item()
        
        print(f"\n--- Sample {idx + 1} (Dataset Index: {sample_idx}) ---")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # Show first 20 and last 20 elements for comparison
        print(f"\nFirst 20 elements:")
        print(f"  Original:      {original.numpy()}")
        print(f"  Reconstructed: {reconstructed.numpy()}")
        
        # Show comparison of random non-zero block
        # Find index where original is non-zero
        non_zero_indices = torch.nonzero(original[1:]).squeeze()
        if len(non_zero_indices) > 0:
            rand_idx = non_zero_indices[torch.randint(0, len(non_zero_indices), (1,)).item()] + 1
            print(f"\nRandom Active Element [{rand_idx}]:")
            print(f"  Original:      {original[rand_idx].item():.6f}")
            print(f"  Reconstructed: {reconstructed[rand_idx].item():.6f}")

        
        # Show max absolute difference
        max_diff_idx = torch.argmax(torch.abs(original - reconstructed))
        max_diff = (original - reconstructed)[max_diff_idx].item()
        print(f"\nMax absolute difference: {abs(max_diff):.6f} at index {max_diff_idx}")
        print(f"  Original[{max_diff_idx}]: {original[max_diff_idx].item():.6f}")
        print(f"  Reconstructed[{max_diff_idx}]: {reconstructed[max_diff_idx].item():.6f}")
        print(f"{'-'*80}")
    
    print(f"\n{'='*80}")
    print(f"Visualization complete!")
    print(f"{'='*80}\n")


def train_and_process(train_path, val_path, 
                      train_block_out, val_block_out,
                      train_global_out, val_global_out,
                      model_output_path, 
                      batch_size=16, epochs=200, learning_rate=1e-3):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Processing dataset: {os.path.basename(train_path)}")
    print(f"{'='*80}")
    print(f"Using device: {device}")
    
    # Load data
    train_df, val_df = load_data(train_path, val_path)
    
    # Prepare dataloaders (Using raw 1021 vectors, loop handles preprocessing)
    train_loader, val_loader = prepare_dataloaders(train_df, val_df, batch_size=batch_size)
    
    # Initialize MoE AE
    print(f"\nInitializing MoEGrammarAE...")
    ae = MoEGrammarAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, num_experts=NUM_EXPERTS, param_dim=PARAM_DIM)
    ae.apply(weights_init)
    total_params = sum(p.numel() for p in ae.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train
    print(f"\nTraining AE for {epochs} epochs with learning rate {learning_rate}...")
    ae = train_ae(ae, train_loader, val_loader, epochs=epochs, lr=learning_rate, device=device)
    
    # Save model
    print(f"\nSaving trained model to {model_output_path}...")
    torch.save(ae.state_dict(), model_output_path)
    
    # Transform and save datasets (Both Block and Global versions)
    save_transformed_datasets(ae, train_df, val_df, 
                            train_block_out, val_block_out,
                            train_global_out, val_global_out,
                            device)
    
    print(f"\n{'='*80}")
    print(f"Completed processing")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Train or visualize MoE Grammar Autoencoder')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'visualize'],
                        help='Mode: train the model or visualize reconstructions')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model weights (required for visualize mode)')
    parser.add_argument('--val_path', type=str, default=None,
                        help='Path to validation dataset (required for visualize mode)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Default data directory
    data_dir = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset"
    
    if args.mode == 'visualize':
        # Visualization mode
        model_path = args.model_path or os.path.join(data_dir, "moe_ae_model.pt")
        val_path = args.val_path or os.path.join(data_dir, "mix_dataset_reg_val.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please provide --model_path or train the model first")
            sys.exit(1)
        
        if not os.path.exists(val_path):
            print(f"Error: Validation dataset not found at {val_path}")
            print("Please provide --val_path")
            sys.exit(1)
        
        visualize_reconstruction(model_path, val_path, num_samples=args.num_samples)
        
    else:
        # Training mode
        print("\n" + "="*80)
        print("DATASET: mix_dataset_reg")
        print("="*80)
        
        train_path = os.path.join(data_dir, "mix_dataset_reg_train.pkl")
        val_path = os.path.join(data_dir, "mix_dataset_reg_val.pkl")
        
        # Outputs for Block-Level Encodings
        train_block_out = os.path.join(data_dir, f"moe_{LATENT_DIM}_block_mix_dataset_reg_train.pkl")
        val_block_out = os.path.join(data_dir, f"moe_{LATENT_DIM}_block_mix_dataset_reg_val.pkl")
        
        # Outputs for Global Encodings
        train_global_out = os.path.join(data_dir, f"moe_{LATENT_DIM}_global_mix_dataset_reg_train.pkl")
        val_global_out = os.path.join(data_dir, f"moe_{LATENT_DIM}_global_mix_dataset_reg_val.pkl")
        
        model_path = os.path.join(data_dir, "moe_ae_model.pt")
        
        train_and_process(
            train_path, val_path,
            train_block_out, val_block_out,
            train_global_out, val_global_out,
            model_path,
            args.batch_size, args.epochs, args.learning_rate
        )
        
        # After training, optionally show some visualizations
        print("\n" + "="*80)
        print("Training complete! Showing reconstruction samples...")
        print("="*80)
        visualize_reconstruction(model_path, val_path, num_samples=3)

if __name__ == "__main__":
    main()