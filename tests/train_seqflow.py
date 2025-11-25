import torch
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
from surrogates.dual_arch_seqflow import HybridSeqFlow

def train_seqflow_phases():
    # --- Configuration ---
    REG_DATA_PATH = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset/mix_dataset_reg_train.pkl"
    MAX_SEQ_LEN = 350
    BATCH_SIZE = 32
    
    # Phase 1 (Structure) Config
    EPOCHS_STRUCT = 10
    LR_STRUCT = 1e-3
    
    # Phase 2 (Value) Config
    EPOCHS_VAL = 100
    LR_VAL = 3e-3
    
    # Model Architecture
    EMBED_DIM = 128
    STRUCT_HIDDEN_DIM = 256
    VAL_HIDDEN_DIM = 256
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # --- Load Data & Codec ---
    print(f"Loading data from {REG_DATA_PATH}...")
    try:
        reg_df = pd.read_pickle(REG_DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    codec = Codec(num_classes=7)
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
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # --- Initialize Model ---
    model = HybridSeqFlow(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        struct_hidden_dim=STRUCT_HIDDEN_DIM,
        val_hidden_dim=VAL_HIDDEN_DIM,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)
    
    # ==============================================================================
    # PHASE 1: Train Discrete Structure Flow
    # ==============================================================================
    print("\n" + "="*40)
    print("PHASE 1: Training Discrete Structure Flow")
    print("="*40)
    
    optimizer_struct = optim.Adam(model.struct_flow.parameters(), lr=LR_STRUCT)
    model.struct_flow.train()
    # Ensure Value flow is frozen just in case
    for param in model.val_flow.parameters():
        param.requires_grad = False
        
    for epoch in range(EPOCHS_STRUCT):
        total_nll = 0
        total_sim = 0
        batches = 0
        
        for batch in dataloader:
            tokens = batch['tokens'].to(DEVICE, dtype=torch.long)
            
            # Forward Pass (Structure Only)
            loss_nll, loss_sim, _ = model.struct_flow(tokens)
            loss = loss_nll + loss_sim
            
            optimizer_struct.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.struct_flow.parameters(), 1.0)
            optimizer_struct.step()
            
            total_nll += loss_nll.item()
            total_sim += loss_sim.item()
            batches += 1
            
        print(f"Epoch {epoch+1}/{EPOCHS_STRUCT} | NLL: {total_nll/batches:.4f} | Sim: {total_sim/batches:.4f}")

    # --- Phase 1 Testing ---
    print("\nTesting Discrete Reconstruction (Phase 1)...")
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        tokens = sample['tokens'].unsqueeze(0).to(DEVICE)
        
        # Encode -> Decode
        z_struct, _ = model.struct_flow.encode(tokens)
        rec_tokens, _ = model.struct_flow.decode(z_struct)
        
        acc = (tokens == rec_tokens).float().mean().item()
        print(f"Structure Reconstruction Accuracy: {acc*100:.4f}%")
        print(f"Original (first 20): {tokens[0, :20].cpu().numpy()}")
        print(f"Reconst  (first 20): {rec_tokens[0, :20].cpu().numpy()}")

    # ==============================================================================
    # PHASE 2: Train Continuous Value Flow
    # ==============================================================================
    print("\n" + "="*40)
    print("PHASE 2: Training Continuous Value Flow")
    print("="*40)
    
    # Freeze Structure Model
    for param in model.struct_flow.parameters():
        param.requires_grad = False
    
    # Unfreeze Value Model
    for param in model.val_flow.parameters():
        param.requires_grad = True
        
    optimizer_val = optim.Adam(model.val_flow.parameters(), lr=LR_VAL)
    model.val_flow.train()
    model.struct_flow.eval() # Set structure to eval mode (affects dropout etc)
    
    for epoch in range(EPOCHS_VAL):
        total_val_loss = 0
        batches = 0
        
        for batch in dataloader:
            tokens = batch['tokens'].to(DEVICE, dtype=torch.long)
            values = batch['values'].to(DEVICE, dtype=torch.float).unsqueeze(-1)
            
            # 1. Get Structure Context (No Grad)
            with torch.no_grad():
                _, _, context = model.struct_flow(tokens)
            
            # 2. Forward Value Flow
            # We detach context to be double sure gradients don't flow back
            loss_val, _ = model.val_flow(values, context.detach())
            
            optimizer_val.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.val_flow.parameters(), 1.0)
            optimizer_val.step()
            
            total_val_loss += loss_val.item()
            batches += 1
            
        print(f"Epoch {epoch+1}/{EPOCHS_VAL} | Value NLL: {total_val_loss/batches:.4f}")

    # --- Phase 2 Testing ---
    print("\nTesting Value Reconstruction (Phase 2)...")
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        tokens = sample['tokens'].unsqueeze(0).to(DEVICE)
        values = sample['values'].unsqueeze(0).to(DEVICE).unsqueeze(-1)
        
        # Get Z and Context
        z_struct, context = model.struct_flow.encode(tokens)
        z_val = model.val_flow.encode(values, context)
        
        # Reconstruct Values
        rec_values = model.val_flow.decode(z_val, context)
        
        # Calculate MSE only for valid value tokens (Token ID 3)
        mask = (tokens == model.val_token_id)
        if mask.sum() > 0:
            mse = F.mse_loss(values[mask], rec_values[mask])
            print(f"Value Reconstruction MSE (on VAL tokens): {mse.item():.6f}")
            print(f"Original Values (sample): {values[0, mask[0]][:5].cpu().numpy().flatten()}")
            print(f"Reconst Values (sample):  {rec_values[0, mask[0]][:5].cpu().numpy().flatten()}")
        else:
            print("No value tokens found in sample.")

    # Save Full Model
    torch.save(model.state_dict(), "hybrid_seqflow_final.pt")
    print("\nSaved final model to hybrid_seqflow_final.pt")

if __name__ == "__main__":
    train_seqflow_phases()