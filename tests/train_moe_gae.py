"""
Train MoE Grammar AE on mix_dataset_reg files and save latent representations.

This script:
1. Loads train and validation datasets from mix_dataset_reg_{train,val}.pkl
2. Creates "Block-Expanded" versions of these datasets (rows with 68-vectors instead of 1021)
   - BALANCED with padding blocks (ratio=0.1) for training stability.
3. Trains the MoEGrammarAE on the block data
4. Applies the trained AE to encode the datasets in two ways:
   a. Block-wise encoding (saving to moe_block_mix_dataset_reg_{train,val}.pkl)
   b. Full 1021 global encoding (saving to moe_global_mix_dataset_reg_{train,val}.pkl)
      - NOTE: Global encoding is now 240-dim (15x16) pure latents. No Epoch/Length.
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

# Import the updated unvectorized AE
from surrogates.gae import (
    MoEGrammarAE, 
    grammar_loss_function, 
    preprocess_encoding,
    expand_dataframe_to_blocks,
    get_latent_representation,
    weights_init,
    collapse_to_physical,
    reconstruct_samples # Use the AE's reconstruct_samples for consistent logic
)

# Constants
LATENT_DIM = 16 
INPUT_DIM = 68
NUM_EXPERTS = 55 # Updated to 55 (54 Real + 1 Padding)
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
            
            # 1. Preprocess (Handles Padding Logic internally now)
            # Returns blocks [B, 15, 68] with zeros preserved
            blocks = preprocess_encoding(raw_encoding_batch)
            
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
                
                blocks = preprocess_encoding(raw_encoding_batch)
                
                recon_types, recon_params, z = ae(blocks)
                
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
    """Wrapper to expand DF to blocks with Padding Balancing."""
    print("Expanding DataFrame to 68-vector blocks (Balanced with Padding)...")
    # Using padding_ratio=0.1 as per strategy to teach Void Cluster
    return expand_dataframe_to_blocks(df, padding_ratio=0.1)

def save_transformed_datasets(ae, train_df, val_df, 
                            train_block_out, val_block_out,
                            train_global_out, val_global_out, 
                            device):
    """Apply AE encoding in two modes and save."""
    
    # 1. Block-Level Encoding (Applies AE on 68-vectors)
    print("\n--- Generating Block-Level Encodings ---")
    
    # Expand to blocks first (This creates the balanced dataset)
    # Note: For encoding the DATASET, we probably want all blocks?
    # Actually, for Flow training later, we usually rely on Global Encodings.
    # The 'block' dataset here is mostly for analysis or debugging.
    # Let's keep the balancing logic for consistency with training distribution.
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
    # use_full_encoding=True triggers the 1021-vector logic (Returns 240-dim latents)
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
    """
    ae.eval()
    
    if raw_encoding.dim() == 1:
        raw_encoding = raw_encoding.unsqueeze(0)
        single_sample = True
    else:
        single_sample = False
    
    raw_encoding = raw_encoding.to(device)
    
    with torch.no_grad():
        # Call the AE's reconstruct_samples directly
        # It handles preprocess -> encode -> decode -> collapse -> flatten
        reconstructed_blocks, _ = reconstruct_samples(ae, raw_encoding, device=device, is_1021=True)
        
        # reconstruct_samples returns [B, 15, 68] numpy
        # We need to flatten back to 1021 format for comparison
        # Original: [Epoch, Flattened(15x68)]
        
        rec_tensor = torch.from_numpy(reconstructed_blocks).to(device)
        batch_size = rec_tensor.shape[0]
        
        # Reshape to 1020
        # Blocks are [B, 15, 68]. Codec format is Column-Major of (68, 15).
        # We transpose to (B, 68, 15) then flatten.
        flat_genome = rec_tensor.transpose(1, 2).reshape(batch_size, -1) # [B, 1020]
        
        # Get original Epochs to prepend
        epochs = raw_encoding[:, 0:1]
        
        reconstructed = torch.cat([epochs, flat_genome], dim=1)
    
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
        
        # Pass to reconstruct_from_model
        reconstructed = reconstruct_from_model(ae, original, device).cpu()
        
        # Calculate metrics (Skip epoch index 0)
        orig_arch = original[1:]
        rec_arch = reconstructed[1:]
        
        mse = torch.mean((orig_arch - rec_arch) ** 2).item()
        mae = torch.mean(torch.abs(orig_arch - rec_arch)).item()
        
        print(f"\n--- Sample {idx + 1} (Dataset Index: {sample_idx}) ---")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # Show first 20 and last 20 elements for comparison
        print(f"\nFirst 20 elements (Architecture):")
        print(f"  Original:      {orig_arch.numpy()}")
        print(f"  Reconstructed: {rec_arch.numpy()}")
        
        # Show comparison of random non-zero block
        non_zero_indices = torch.nonzero(orig_arch).squeeze()
        if len(non_zero_indices) > 0:
            rand_idx = non_zero_indices[torch.randint(0, len(non_zero_indices), (1,)).item()]
            print(f"\nRandom Active Element [{rand_idx}]:")
            print(f"  Original:      {orig_arch[rand_idx].item():.6f}")
            print(f"  Reconstructed: {rec_arch[rand_idx].item():.6f}")

        
        # Show max absolute difference
        max_diff_idx = torch.argmax(torch.abs(orig_arch - rec_arch))
        max_diff = (orig_arch - rec_arch)[max_diff_idx].item()
        print(f"\nMax absolute difference: {abs(max_diff):.6f} at index {max_diff_idx}")
        print(f"  Original[{max_diff_idx}]: {orig_arch[max_diff_idx].item():.6f}")
        print(f"  Reconstructed[{max_diff_idx}]: {rec_arch[max_diff_idx].item():.6f}")
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
    
    # Prepare dataloaders
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
    parser.add_argument('--epochs', type=int, default=30,
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
        train_block_out = os.path.join(data_dir, f"moe_new_{LATENT_DIM}_block_mix_dataset_reg_train.pkl")
        val_block_out = os.path.join(data_dir, f"moe_new_{LATENT_DIM}_block_mix_dataset_reg_val.pkl")
        
        # Outputs for Global Encodings
        train_global_out = os.path.join(data_dir, f"moe_new_{LATENT_DIM}_global_mix_dataset_reg_train.pkl")
        val_global_out = os.path.join(data_dir, f"moe_new_{LATENT_DIM}_global_mix_dataset_reg_val.pkl")
        
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