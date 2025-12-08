"""
Latent Normalizing Flow for Inverse Design of Neural Architectures.
STABILITY FIXED: Adds Tanh gating, Data Scaling, and Seeded Initialization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import copy
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler

# Import existing modules
# Ensure surrogates folder is in your python path
from surrogates.surrogate import Surrogate 
from surrogates.gae import MoEGrammarAE, collapse_to_physical, PRIM_SCHEMA

# ============================================================================
# Robust RealNVP Implementation
# ============================================================================
class CouplingLayer(nn.Module):
    def __init__(self, num_inputs, num_hidden, mask):
        super().__init__()
        self.num_inputs = num_inputs
        self.mask = nn.Parameter(mask, requires_grad=False)
        
        # Scale and Translate networks
        self.scale_net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_inputs)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_inputs)
        )

    def forward(self, inputs, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask
        
        # STABILITY FIX: Tanh Gating to prevent explosion
        log_s = self.scale_net(masked_inputs) * (1 - mask)
        log_s = torch.tanh(log_s) * 2.0 # Clamp to [-2, 2]
        
        t = self.translate_net(masked_inputs) * (1 - mask)
        
        if mode == 'direct':
            # x -> z (Forward/Training)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1)
        else:
            # z -> x (Inverse/Sampling)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1)

class LatentFlow(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_layers, device='cuda'):
        super().__init__()
        self.num_inputs = num_inputs
        self.device = device
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            mask = torch.arange(num_inputs) % 2 == i % 2
            mask = mask.float().to(device)
            self.layers.append(CouplingLayer(num_inputs, num_hidden, mask))
            
        self.prior = torch.distributions.Normal(
            torch.tensor(0.0).to(device), 
            torch.tensor(1.0).to(device)
        )

    def forward(self, inputs):
        """Latent -> Noise"""
        log_det_sum = torch.zeros(inputs.shape[0]).to(inputs.device)
        z = inputs
        for layer in self.layers:
            z, log_det = layer(z, mode='direct')
            log_det_sum += log_det
        return z, log_det_sum

    def inverse(self, z):
        """Noise -> Latent"""
        x = z
        for layer in reversed(self.layers):
            x, _ = layer(x, mode='inverse')
        return x

    def log_prob(self, inputs):
        z, log_det = self.forward(inputs)
        prior_log_prob = self.prior.log_prob(z).sum(-1)
        return prior_log_prob + log_det

# ============================================================================
# Latent Inverse Designer
# ============================================================================
class LatentInverseDesigner:
    def __init__(self, surrogate_config_path, surrogate_weights_dir, gae_model_path, reg_train_df, device='cuda'):
        self.device = device
        self.reg_train_df = reg_train_df
        
        # 1. Load GAE (For final decoding)
        print("Loading Grammar Autoencoder...")
        self.gae = MoEGrammarAE(input_dim=68, latent_dim=16, num_experts=55, param_dim=14)
        self.gae.load_state_dict(torch.load(gae_model_path, map_location=device))
        self.gae.to(device)
        self.gae.eval()
        
        # 2. Load Surrogate
        print("Loading Surrogate Ensemble...")
        self.surrogate = Surrogate(surrogate_config_path, surrogate_weights_dir)
        
        # 3. Setup Scalers
        # Surrogate Scaler: Used for surrogate.predict() on 240-dim latents
        print("Fitting Surrogate Scaler...")
        self.surrogate_scaler = self._fit_scaler(reg_train_df)
        
        # Flow Scaler: Used to normalize GAE Latents before entering Flow
        # CRITICAL: Flows fail if inputs are not ~Normal(0,1)
        print("Fitting Flow Scaler...")
        # reg_train_df['genome'] is 240-dim (Indices 2:242 were saved earlier as 'genome')
        # If the file contains 242 (with Epoch/Length), we must strip them here for Flow.
        # But your latest scripts imply the saved 'moe_...global...pkl' is purely 240 latents.
        full_data = np.stack(self.reg_train_df['genome'].values) 
        
        self.flow_scaler = StandardScaler()
        self.flow_scaler.fit(full_data)
        
        # 4. Initialize Flow
        self.flow_dim = 240
        self.flow = LatentFlow(num_inputs=self.flow_dim, num_hidden=512, num_layers=12, device=device).to(device)

    def _fit_scaler(self, df):
        data = np.stack(df['genome'].values)
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler

    def train_flow(self, epochs=50, lr=1e-3, batch_size=64):
        """Trains the Flow on the Scaled 240-dim latents."""
        # Get Data
        full_data = np.stack(self.reg_train_df['genome'].values) # [N, 240]
        
        # SCALE DATA BEFORE TRAINING
        scaled_data = self.flow_scaler.transform(full_data)
        
        tensor_data = torch.from_numpy(scaled_data).float()
        dataset = torch.utils.data.TensorDataset(tensor_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Use lower LR for stability
        optimizer = optim.Adam(self.flow.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        self.flow.train()
        
        print(f"Starting Flow Training on {self.flow_dim} latent dimensions...")
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(loader, desc=f"Flow Epoch {epoch+1}")
            
            for batch in pbar:
                batch_x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                
                # NLL Loss
                log_prob = self.flow.log_prob(batch_x)
                loss = -torch.mean(log_prob)
                
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.2f}")
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

    def optimize(self, target_fitness, start_from_best=True, num_steps=200, lr=0.1, lambda_prior=0.01):
        """
        Inverse Design Optimization.
        """
        targets = torch.tensor(target_fitness, device=self.device).float()
        
        # 1. Initialization
        if start_from_best:
            print("Initializing optimization from best existing architecture...")
            # Find index of min ciou_loss
            best_idx = self.reg_train_df['ciou_loss'].idxmin()
            best_genome_raw = self.reg_train_df.loc[best_idx, 'genome'] # [240] numpy
            
            # Scale it using Flow Scaler so it's in the Flow's domain
            best_genome_scaled = self.flow_scaler.transform(best_genome_raw.reshape(1, -1))
            best_tensor = torch.from_numpy(best_genome_scaled).float().to(self.device)
            
            # Project to Noise Space using Flow
            with torch.no_grad():
                self.flow.eval()
                initial_noise, _ = self.flow(best_tensor) # Latent -> Noise
            
            z_noise = initial_noise.clone().detach().requires_grad_(True)
        else:
            print("Initializing optimization from random noise...")
            z_noise = torch.randn(1, self.flow_dim, device=self.device, requires_grad=True)

        optimizer = optim.Adam([z_noise], lr=lr)
        
        # Get Flow Scaler params for differentiable unscaling
        flow_mean = torch.tensor(self.flow_scaler.mean_, device=self.device).float()
        flow_scale = torch.tensor(self.flow_scaler.scale_, device=self.device).float()
        
        print(f"Optimizing for Target: {target_fitness}")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # A. Flow Inverse: Noise -> Scaled Latents
            z_scaled_struct = self.flow.inverse(z_noise)
            
            # B. Differentiable Unscaling: Scaled Latents -> Raw GAE Latents
            # We must do this so the Surrogate receives the correct magnitude inputs
            z_raw_struct = (z_scaled_struct * flow_scale) + flow_mean
            
            # C. Surrogate Prediction (Using its own Scaler)
            preds = self.surrogate.predict(z_raw_struct, genome_scaler=self.surrogate_scaler)
            
            # D. Loss
            fitness_loss = torch.nn.functional.mse_loss(preds, targets.unsqueeze(0))
            
            # Prior constraint: Keep noise close to origin
            prior_loss = torch.mean(z_noise ** 2)
            
            total_loss = fitness_loss + (lambda_prior * prior_loss)
            
            total_loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"Step {step}: Loss={fitness_loss.item():.4f}, Preds={preds.detach().cpu().numpy()}")
        
        # Final Extraction
        with torch.no_grad():
            final_scaled = self.flow.inverse(z_noise)
            final_raw = (final_scaled * flow_scale) + flow_mean
            
        return final_raw.cpu().numpy().flatten()

    def decode_to_physical(self, z_latent_flat):
        """
        Decodes a flat 240-dim latent vector into the 1021-dim physical vector.
        """
        self.gae.eval()
        with torch.no_grad():
            tensor_in = torch.from_numpy(z_latent_flat).float().to(self.device).view(1, 15, 16)
            
            type_logits, param_stack = self.gae.decode(tensor_in)
            physical_blocks = collapse_to_physical(type_logits, param_stack) # [1, 15, 68]
            
            flat_genome = physical_blocks.transpose(1, 2).reshape(1, -1) # [1, 1020]
            
            # Prepend Dummy Epoch (0.0) for visualization tool compatibility
            dummy_epoch = torch.tensor([[0.0]], device=self.device)
            final_vector = torch.cat([dummy_epoch, flat_genome], dim=1) # [1, 1021]
            
            return final_vector.cpu().numpy().flatten()

# ============================================================================
# Execution Block
# ============================================================================
if __name__ == "__main__":
    # Configuration
    dataset_dir = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset"
    reg_train_path = os.path.join(dataset_dir, "moe_new_16_global_mix_dataset_reg_train.pkl")
    gae_model_path = os.path.join(dataset_dir, "moe_ae_model.pt")
    
    surr_config = "conf.toml"
    surr_weights = os.path.join("/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/surrogate_training/surrogate_weights")
    
    # Load Data
    print(f"Loading Latent Data: {reg_train_path}")
    reg_train_df = pd.read_pickle(reg_train_path)
    
    # Initialize Designer
    designer = LatentInverseDesigner(surr_config, surr_weights, gae_model_path, reg_train_df)
    
    # Train Flow
    designer.train_flow(epochs=30) # Increased epochs since we have stability now
    
    # Optimize (Start from Best CIoU architecture)
    # Target: CIOU=1.2, AP=0.7
    target = [1.2, 0.70]
    optimized_latent = designer.optimize(target_fitness=target, start_from_best=True)
    
    # Decode
    physical_genome = designer.decode_to_physical(optimized_latent)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULT - FULL GENOME")
    print("="*80)
    
    # Ensure complete print without truncation
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    print(physical_genome)
    
    print("\n" + "="*80)
    print(f"Shape: {physical_genome.shape}")