"""
Inverse Design Formulation (IDF) module for goal-conditioned architecture generation.

This module implements a goal-conditioned generative model that learns to produce
neural architecture latent representations (z_arch) conditioned on desired fitness values.

The module supports two generator architectures:
1. ConditionalVAERepresentation: A CVAE that learns the inverse mapping fitness -> z_arch
2. ConditionalDiffusionRepresentation: A diffusion model with classifier-free guidance

The InverseDesigner orchestrates training and optimization using a frozen surrogate
as a differentiable critic to refine the generator.
"""

from abc import ABC, abstractmethod
import os
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
from torchdiffeq import odeint_adjoint as odeint

class IGenerator(ABC):
    """
    Abstract base class for conditional generators.
    
    All generators must implement:
    - initial_train: Pre-train on (z_arch, fitness) pairs from archive
    - sample: Generate z_arch vectors conditioned on desired fitness
    - parameters property: Return trainable parameters for optimization
    """
    
    @abstractmethod
    def initial_train(
        self, 
        z_arch_vectors: torch.Tensor, 
        fitness_values: torch.Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Pre-train the generator on existing archive data.
        
        Args:
            z_arch_vectors: Tensor of shape [N, z_dim] - latent architecture vectors
            fitness_values: Tensor of shape [N, num_objectives] - true fitness values
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            Dict containing training statistics (losses per epoch)
        """
        pass
    
    @abstractmethod
    def sample(
        self, 
        c: torch.Tensor, 
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate latent architecture vectors conditioned on desired fitness.
        
        Args:
            c: Condition tensor of shape [batch_size, num_objectives] or scalar
            batch_size: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated z_arch tensor of shape [batch_size, z_dim]
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self):
        """Return trainable parameters for the optimizer."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save generator weights."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load generator weights."""
        pass


class ArchiveFitnessDataset(Dataset):
    """Dataset for (z_arch, fitness) pairs."""
    
    def __init__(self, z_vectors: torch.Tensor, fitness_values: torch.Tensor):
        self.z = z_vectors
        self.fitness = fitness_values
        
    def __len__(self):
        return len(self.z)
    
    def __getitem__(self, idx):
        return self.z[idx], self.fitness[idx]


class ConditionalVAERepresentation(IGenerator):
    """
    Conditional VAE generator for inverse design.
    
    Architecture:
    - Encoder: [z_arch + fitness_embed] -> [mu, logvar] of z_latent2
    - Decoder: [z_latent2 + fitness_embed] -> z_arch_reconstructed
    
    The conditioning is done via concatenation with embedded fitness values.
    """
    
    def __init__(
        self, 
        z_dim: int,
        num_objectives: int,
        latent2_dim: int = 128,
        hidden_sizes: List[int] = [512, 256],
        fitness_embed_dim: int = 32,
        dropout: float = 0.1,
        device: torch.device = None
    ):
        """
        Args:
            z_dim: Dimension of architecture latent space (from VAE encoder)
            num_objectives: Number of fitness objectives
            latent2_dim: Dimension of second-level latent space
            hidden_sizes: Hidden layer sizes for encoder/decoder MLPs
            fitness_embed_dim: Embedding dimension for fitness conditioning
            dropout: Dropout probability
            device: Device to run on
        """
        self.z_dim = z_dim
        self.num_objectives = num_objectives
        self.latent2_dim = latent2_dim
        self.fitness_embed_dim = fitness_embed_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fitness embedding network (shared by encoder and decoder)
        self.fitness_embedder = nn.Sequential(
            nn.Linear(num_objectives, fitness_embed_dim),
            nn.ReLU(),
            nn.Linear(fitness_embed_dim, fitness_embed_dim)
        ).to(self.device)
        
        # Encoder: [z_arch + fitness_embed] -> [mu, logvar]
        encoder_layers = []
        input_dim = z_dim + fitness_embed_dim
        for h_dim in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = h_dim
        
        self.encoder_shared = nn.Sequential(*encoder_layers).to(self.device)
        self.encoder_mu = nn.Linear(hidden_sizes[-1], latent2_dim).to(self.device)
        self.encoder_logvar = nn.Linear(hidden_sizes[-1], latent2_dim).to(self.device)
        
        # Decoder: [z_latent2 + fitness_embed] -> z_arch
        decoder_layers = []
        input_dim = latent2_dim + fitness_embed_dim
        for h_dim in reversed(hidden_sizes):
            decoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_sizes[0], z_dim))
        
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        
    def encode(self, z_arch: torch.Tensor, fitness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode z_arch conditioned on fitness to get z_latent2 distribution."""
        fitness_embed = self.fitness_embedder(fitness)
        x = torch.cat([z_arch, fitness_embed], dim=-1)
        h = self.encoder_shared(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z_latent2: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        """Decode z_latent2 conditioned on fitness to reconstruct z_arch."""
        fitness_embed = self.fitness_embedder(fitness)
        x = torch.cat([z_latent2, fitness_embed], dim=-1)
        return self.decoder(x)
    
    def forward(self, z_arch: torch.Tensor, fitness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for training."""
        mu, logvar = self.encode(z_arch, fitness)
        z_latent2 = self.reparameterize(mu, logvar)
        z_recon = self.decode(z_latent2, fitness)
        return z_recon, mu, logvar
    
    def initial_train(
        self, 
        z_arch_vectors: torch.Tensor, 
        fitness_values: torch.Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        beta: float = 1.0,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Pre-train the CVAE on archive data."""
        
        # Ensure tensors are on correct device
        z_arch_vectors = z_arch_vectors.to(self.device)
        fitness_values = fitness_values.to(self.device)
        
        # Create dataset and loader
        dataset = ArchiveFitnessDataset(z_arch_vectors, fitness_values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Optimizer
        optimizer = optim.Adam(self.parameters, lr=lr)
        
        # Training loop
        history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        
        print(f"Starting CVAE pre-training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for z_batch, f_batch in pbar:
                z_batch = z_batch.to(self.device)
                f_batch = f_batch.to(self.device)
                
                # Forward pass
                z_recon, mu, logvar = self.forward(z_batch, f_batch)
                
                # Compute losses
                recon_loss = F.mse_loss(z_recon, z_batch, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = recon_loss + beta * kl_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
                optimizer.step()
                
                # Track losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'recon': recon_loss.item(),
                    'kl': kl_loss.item()
                })
            
            # Average losses
            n_batches = len(loader)
            history['total_loss'].append(epoch_total_loss / n_batches)
            history['recon_loss'].append(epoch_recon_loss / n_batches)
            history['kl_loss'].append(epoch_kl_loss / n_batches)
            
            print(f"Epoch {epoch+1}: Loss={history['total_loss'][-1]:.4f}, "
                  f"Recon={history['recon_loss'][-1]:.4f}, KL={history['kl_loss'][-1]:.4f}")
            
            # Save checkpoint periodically
            if save_dir and (epoch + 1) % 10 == 0:
                os.makedirs(save_dir, exist_ok=True)
                self.save_checkpoint(os.path.join(save_dir, f'cvae_epoch_{epoch+1}.pth'))
        
        print("CVAE pre-training complete!")
        return history
    
    def sample(
        self, 
        c: torch.Tensor, 
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate z_arch vectors conditioned on desired fitness.
        
        Args:
            c: Desired fitness, shape [batch_size, num_objectives] or [num_objectives]
            batch_size: Number of samples
            
        Returns:
            Generated z_arch of shape [batch_size, z_dim]
        """
        # Handle scalar or single-vector conditioning
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(batch_size, -1)
        c = c.to(self.device)
        
        # Sample from prior N(0, I)
        z_latent2 = torch.randn(batch_size, self.latent2_dim, device=self.device)
        
        # Decode conditioned on desired fitness
        z_arch_generated = self.decode(z_latent2, c)
        
        return z_arch_generated
    
    @property
    def parameters(self):
        """Return all trainable parameters."""
        return list(self.fitness_embedder.parameters()) + \
               list(self.encoder_shared.parameters()) + \
               list(self.encoder_mu.parameters()) + \
               list(self.encoder_logvar.parameters()) + \
               list(self.decoder.parameters())
    
    def save_checkpoint(self, path: str):
        """Save model state."""
        torch.save({
            'fitness_embedder': self.fitness_embedder.state_dict(),
            'encoder_shared': self.encoder_shared.state_dict(),
            'encoder_mu': self.encoder_mu.state_dict(),
            'encoder_logvar': self.encoder_logvar.state_dict(),
            'decoder': self.decoder.state_dict(),
            'z_dim': self.z_dim,
            'num_objectives': self.num_objectives,
            'latent2_dim': self.latent2_dim,
            'fitness_embed_dim': self.fitness_embed_dim
        }, path)
        print(f"CVAE checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.fitness_embedder.load_state_dict(checkpoint['fitness_embedder'])
        self.encoder_shared.load_state_dict(checkpoint['encoder_shared'])
        self.encoder_mu.load_state_dict(checkpoint['encoder_mu'])
        self.encoder_logvar.load_state_dict(checkpoint['encoder_logvar'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        print(f"CVAE checkpoint loaded from {path}")


class UnconditionalVAERepresentation(IGenerator):
    """
    Unconditional VAE generator for inverse design with fixed target fitness.
    
    Unlike the conditional VAE, this model is trained with a FIXED target fitness
    value and learns to generate z_arch vectors that produce that specific fitness.
    During sampling, no conditioning is provided - the model has internalized the
    target fitness during training.
    
    This is useful when you want to train a dedicated generator for a specific
    fitness target rather than a general conditional model.
    
    Architecture:
    - Encoder: z_arch -> [mu, logvar] of z_latent2 (NO fitness input)
    - Decoder: z_latent2 -> z_arch_reconstructed (NO fitness input)
    - During training: Uses fixed target_fitness to compute surrogate loss
    """
    
    def __init__(
        self, 
        z_dim: int,
        num_objectives: int,
        target_fitness: torch.Tensor,  # Fixed target fitness for this generator
        latent2_dim: int = 128,
        hidden_sizes: List[int] = [512, 256],
        dropout: float = 0.1,
        device: torch.device = None
    ):
        """
        Args:
            z_dim: Dimension of architecture latent space (from VAE encoder)
            num_objectives: Number of fitness objectives
            target_fitness: Fixed target fitness tensor of shape [num_objectives]
            latent2_dim: Dimension of second-level latent space
            hidden_sizes: Hidden layer sizes for encoder/decoder MLPs
            dropout: Dropout probability
            device: Device to run on
        """
        self.z_dim = z_dim
        self.num_objectives = num_objectives
        self.latent2_dim = latent2_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store fixed target fitness
        if isinstance(target_fitness, np.ndarray):
            target_fitness = torch.from_numpy(target_fitness).float()
        self.target_fitness = target_fitness.to(self.device)
        
        # Encoder: z_arch -> [mu, logvar] (NO conditioning)
        encoder_layers = []
        input_dim = z_dim
        for h_dim in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = h_dim
        
        self.encoder_shared = nn.Sequential(*encoder_layers).to(self.device)
        self.encoder_mu = nn.Linear(hidden_sizes[-1], latent2_dim).to(self.device)
        self.encoder_logvar = nn.Linear(hidden_sizes[-1], latent2_dim).to(self.device)
        
        # Decoder: z_latent2 -> z_arch (NO conditioning)
        decoder_layers = []
        input_dim = latent2_dim
        for h_dim in reversed(hidden_sizes):
            decoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_sizes[0], z_dim))
        
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        
    def encode(self, z_arch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode z_arch to get z_latent2 distribution (unconditional)."""
        h = self.encoder_shared(z_arch)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z_latent2: torch.Tensor) -> torch.Tensor:
        """Decode z_latent2 to reconstruct z_arch (unconditional)."""
        return self.decoder(z_latent2)
    
    def forward(self, z_arch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for training."""
        mu, logvar = self.encode(z_arch)
        z_latent2 = self.reparameterize(mu, logvar)
        z_recon = self.decode(z_latent2)
        return z_recon, mu, logvar
    
    def initial_train(
        self, 
        z_arch_vectors: torch.Tensor, 
        fitness_values: torch.Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        beta: float = 1.0,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Pre-train the unconditional VAE on archive data.
        
        Note: fitness_values are provided for compatibility but are not used
        during the VAE training itself (only reconstruction loss + KL).
        The fixed target_fitness is stored in self.target_fitness.
        """
        
        # Ensure tensors are on correct device
        z_arch_vectors = z_arch_vectors.to(self.device)
        
        # Create dataset and loader (fitness values not used in training)
        dataset = ArchiveFitnessDataset(z_arch_vectors, fitness_values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Optimizer
        optimizer = optim.Adam(self.parameters, lr=lr)
        
        # Training loop
        history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        
        print(f"Starting Unconditional VAE pre-training for {num_epochs} epochs...")
        print(f"Target fitness: {self.target_fitness.cpu().numpy()}")
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for z_batch, _ in pbar:  # Ignore fitness values
                z_batch = z_batch.to(self.device)
                
                # Forward pass (no conditioning)
                z_recon, mu, logvar = self.forward(z_batch)
                
                # Compute VAE loss
                recon_loss = F.mse_loss(z_recon, z_batch, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = recon_loss + beta * kl_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
                optimizer.step()
                
                # Track losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
            
            # Average losses
            n_batches = len(loader)
            history['total_loss'].append(epoch_total_loss / n_batches)
            history['recon_loss'].append(epoch_recon_loss / n_batches)
            history['kl_loss'].append(epoch_kl_loss / n_batches)
            
            print(f"Epoch {epoch+1}: Loss={history['total_loss'][-1]:.4f}, "
                  f"Recon={history['recon_loss'][-1]:.4f}, KL={history['kl_loss'][-1]:.4f}")
            
            # Save checkpoint periodically
            if save_dir and (epoch + 1) % 10 == 0:
                os.makedirs(save_dir, exist_ok=True)
                self.save_checkpoint(os.path.join(save_dir, f'uncond_vae_epoch_{epoch+1}.pt'))
        
        print("Unconditional VAE pre-training complete!")
        return history
    
    def sample(
        self, 
        c: torch.Tensor,  # Ignored for unconditional VAE
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate z_arch vectors (unconditional sampling).
        
        Args:
            c: Ignored (kept for interface compatibility)
            batch_size: Number of samples
            
        Returns:
            Generated z_arch of shape [batch_size, z_dim]
        """
        # Sample from prior N(0, I)
        z_latent2 = torch.randn(batch_size, self.latent2_dim, device=self.device)
        
        # Decode (no conditioning)
        z_arch_generated = self.decode(z_latent2)
        
        return z_arch_generated
    
    @property
    def parameters(self):
        """Return all trainable parameters."""
        return list(self.encoder_shared.parameters()) + \
               list(self.encoder_mu.parameters()) + \
               list(self.encoder_logvar.parameters()) + \
               list(self.decoder.parameters())
    
    def save_checkpoint(self, path: str):
        """Save model state."""
        torch.save({
            'encoder_shared': self.encoder_shared.state_dict(),
            'encoder_mu': self.encoder_mu.state_dict(),
            'encoder_logvar': self.encoder_logvar.state_dict(),
            'decoder': self.decoder.state_dict(),
            'target_fitness': self.target_fitness.cpu(),
            'z_dim': self.z_dim,
            'num_objectives': self.num_objectives,
            'latent2_dim': self.latent2_dim
        }, path)
        print(f"Unconditional VAE checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder_shared.load_state_dict(checkpoint['encoder_shared'])
        self.encoder_mu.load_state_dict(checkpoint['encoder_mu'])
        self.encoder_logvar.load_state_dict(checkpoint['encoder_logvar'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.target_fitness = checkpoint['target_fitness'].to(self.device)
        print(f"Unconditional VAE checkpoint loaded from {path}")


class ConditionalDiffusionRepresentation(IGenerator):
    """
    Conditional Diffusion Model for inverse design.
    
    Uses a U-Net-like denoising network with cross-attention conditioning
    on desired fitness values. Implements DDPM/DDIM sampling.
    """
    
    def __init__(
        self,
        z_dim: int,
        num_objectives: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        fitness_embed_dim: int = 64,
        timesteps: int = 1000,
        dropout: float = 0.1,
        device: torch.device = None
    ):
        """
        Args:
            z_dim: Dimension of architecture latent space
            num_objectives: Number of fitness objectives
            hidden_dim: Hidden dimension for U-Net
            num_layers: Number of denoising layers
            num_heads: Number of attention heads
            fitness_embed_dim: Embedding dimension for fitness conditioning
            timesteps: Number of diffusion timesteps
            dropout: Dropout probability
            device: Device to run on
        """
        self.z_dim = z_dim
        self.num_objectives = num_objectives
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.timesteps = timesteps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fitness embedding
        self.fitness_embedder = nn.Sequential(
            nn.Linear(num_objectives, fitness_embed_dim),
            nn.ReLU(),
            nn.Linear(fitness_embed_dim, fitness_embed_dim)
        ).to(self.device)
        
        # Time embedding (sinusoidal)
        self.time_embed_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)
        
        # Input projection
        self.input_proj = nn.Linear(z_dim, hidden_dim).to(self.device)
        
        # U-Net-like denoising network with cross-attention
        self.denoise_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.denoise_layers.append(
                DenoisingBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=fitness_embed_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ).to(self.device)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, z_dim).to(self.device)
        
        # Beta schedule for diffusion (linear)
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = self.time_embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_mlp(emb)
    
    def forward(
        self, 
        z_t: torch.Tensor, 
        t: torch.Tensor, 
        fitness_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise given noisy latent z_t at timestep t, conditioned on fitness.
        
        Args:
            z_t: Noisy latent, shape [B, z_dim]
            t: Timestep indices, shape [B]
            fitness_cond: Fitness condition, shape [B, num_objectives]
            
        Returns:
            Predicted noise, shape [B, z_dim]
        """
        # Embed time and fitness
        t_emb = self.get_time_embedding(t)
        f_emb = self.fitness_embedder(fitness_cond)
        
        # Project input
        h = self.input_proj(z_t)
        
        # Add time embedding
        h = h + t_emb
        
        # Denoising layers with cross-attention to fitness
        for layer in self.denoise_layers:
            h = layer(h, f_emb)
        
        # Project to noise prediction
        noise_pred = self.output_proj(h)
        
        return noise_pred
    
    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise to z_0 at timestep t."""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample_step(
        self, 
        z_t: torch.Tensor, 
        t: int, 
        fitness_cond: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Single reverse diffusion step (DDPM)."""
        batch_size = z_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        noise_pred = self.forward(z_t, t_tensor, fitness_cond)
        
        # Compute z_{t-1}
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        
        # Mean of p(z_{t-1} | z_t)
        z_0_pred = sqrt_recip_alpha_t * (z_t - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred)
        
        if clip_denoised:
            z_0_pred = torch.clamp(z_0_pred, -10, 10)  # Prevent extreme values
        
        if t > 0:
            noise = torch.randn_like(z_t)
            posterior_variance_t = self.posterior_variance[t]
            z_prev = z_0_pred + torch.sqrt(posterior_variance_t) * noise
        else:
            z_prev = z_0_pred
        
        return z_prev
    
    def initial_train(
        self, 
        z_arch_vectors: torch.Tensor, 
        fitness_values: torch.Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Pre-train the diffusion model on archive data."""
        
        # Ensure tensors are on correct device
        z_arch_vectors = z_arch_vectors.to(self.device)
        fitness_values = fitness_values.to(self.device)
        
        # Create dataset and loader
        dataset = ArchiveFitnessDataset(z_arch_vectors, fitness_values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Optimizer
        optimizer = optim.AdamW(self.parameters, lr=lr)
        
        # Training loop
        history = {'loss': []}
        
        print(f"Starting Diffusion pre-training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for z_batch, f_batch in pbar:
                z_batch = z_batch.to(self.device)
                f_batch = f_batch.to(self.device)
                
                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (z_batch.shape[0],), device=self.device, dtype=torch.long)
                
                # Sample noise
                noise = torch.randn_like(z_batch)
                
                # Forward diffusion
                z_t = self.q_sample(z_batch, t, noise)
                
                # Predict noise
                noise_pred = self.forward(z_t, t, f_batch)
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # Average loss
            n_batches = len(loader)
            history['loss'].append(epoch_loss / n_batches)
            
            print(f"Epoch {epoch+1}: Loss={history['loss'][-1]:.4f}")
            
            # Save checkpoint periodically
            if save_dir and (epoch + 1) % 10 == 0:
                os.makedirs(save_dir, exist_ok=True)
                self.save_checkpoint(os.path.join(save_dir, f'diffusion_epoch_{epoch+1}.pth'))
        
        print("Diffusion pre-training complete!")
        return history
    
    def sample(
        self, 
        c: torch.Tensor, 
        batch_size: int,
        ddim: bool = False,
        ddim_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate z_arch vectors using reverse diffusion.
        
        Args:
            c: Desired fitness, shape [batch_size, num_objectives] or [num_objectives]
            batch_size: Number of samples
            ddim: Use DDIM sampling (faster) instead of DDPM
            ddim_steps: Number of steps for DDIM (ignored if ddim=False)
            
        Returns:
            Generated z_arch of shape [batch_size, z_dim]
        """
        # Handle scalar or single-vector conditioning
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(batch_size, -1)
        c = c.to(self.device)
        
        # Start from pure noise
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        
        if ddim:
            # DDIM sampling (deterministic, faster)
            timesteps_to_use = torch.linspace(self.timesteps - 1, 0, ddim_steps, dtype=torch.long, device=self.device)
            for t in tqdm(timesteps_to_use, desc="DDIM Sampling"):
                z = self.p_sample_step(z, t.item(), c, clip_denoised=True)
        else:
            # DDPM sampling (stochastic)
            for t in tqdm(reversed(range(self.timesteps)), desc="DDPM Sampling", total=self.timesteps):
                z = self.p_sample_step(z, t, c, clip_denoised=True)
        
        return z
    
    @property
    def parameters(self):
        """Return all trainable parameters."""
        params = list(self.fitness_embedder.parameters()) + \
                 list(self.time_mlp.parameters()) + \
                 list(self.input_proj.parameters()) + \
                 list(self.denoise_layers.parameters()) + \
                 list(self.output_proj.parameters())
        return params
    
    def save_checkpoint(self, path: str):
        """Save model state."""
        torch.save({
            'fitness_embedder': self.fitness_embedder.state_dict(),
            'time_mlp': self.time_mlp.state_dict(),
            'input_proj': self.input_proj.state_dict(),
            'denoise_layers': self.denoise_layers.state_dict(),
            'output_proj': self.output_proj.state_dict(),
            'z_dim': self.z_dim,
            'num_objectives': self.num_objectives,
            'hidden_dim': self.hidden_dim,
            'timesteps': self.timesteps
        }, path)
        print(f"Diffusion checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.fitness_embedder.load_state_dict(checkpoint['fitness_embedder'])
        self.time_mlp.load_state_dict(checkpoint['time_mlp'])
        self.input_proj.load_state_dict(checkpoint['input_proj'])
        self.denoise_layers.load_state_dict(checkpoint['denoise_layers'])
        self.output_proj.load_state_dict(checkpoint['output_proj'])
        print(f"Diffusion checkpoint loaded from {path}")


class DenoisingBlock(nn.Module):
    """
    Single denoising block with self-attention and cross-attention to fitness condition.
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention to fitness condition
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, fitness_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden state, shape [B, hidden_dim]
            fitness_cond: Fitness embedding, shape [B, cond_dim]
        """
        # Add sequence dimension for attention
        x_seq = x.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Self-attention
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        x = self.norm1(x + attn_out.squeeze(1))
        
        # Cross-attention to fitness
        x_seq = x.unsqueeze(1)
        cond_seq = self.cond_proj(fitness_cond).unsqueeze(1)  # [B, 1, hidden_dim]
        cross_out, _ = self.cross_attn(x_seq, cond_seq, cond_seq)
        x = self.norm2(x + cross_out.squeeze(1))
        
        # Feed-forward
        x = self.norm3(x + self.ff(x))
        
        return x


class InverseDesigner:
    """
    Main orchestrator for inverse design optimization.
    
    This class manages the entire inverse design pipeline:
    1. Pre-training the generator on archive data
    2. Running optimization steps using the frozen surrogate as a critic
    """
    
    def __init__(
        self,
        generator_type: str,
        surrogate_model,
        vae_encoder,
        z_dim: int,
        num_objectives: int,
        device: torch.device = None,
        generator_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            generator_type: 'cvae' or 'diffusion'
            surrogate_model: Trained surrogate model (will be frozen)
            vae_encoder: Trained VAE encoder (will be frozen)
            z_dim: Dimension of architecture latent space
            num_objectives: Number of fitness objectives
            device: Device to run on
            generator_kwargs: Additional kwargs for generator initialization
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.surrogate = surrogate_model
        self.vae_encoder = vae_encoder
        self.z_dim = z_dim
        self.num_objectives = num_objectives
        
        # Freeze surrogate and VAE encoder
        self.surrogate.eval()
        for param in self.surrogate.parameters():
            param.requires_grad = False
        
        if hasattr(self.vae_encoder, 'eval'):
            self.vae_encoder.eval()
            for param in self.vae_encoder.parameters():
                param.requires_grad = False
        
        # Initialize generator
        generator_kwargs = generator_kwargs or {}
        if generator_type.lower() == 'cvae':
            self.generator = ConditionalVAERepresentation(
                z_dim=z_dim,
                num_objectives=num_objectives,
                device=self.device,
                **generator_kwargs
            )
        elif generator_type.lower() == 'uncond_vae' or generator_type.lower() == 'unconditional_vae':
            # For unconditional VAE, target_fitness must be in generator_kwargs
            if 'target_fitness' not in generator_kwargs:
                raise ValueError("UnconditionalVAE requires 'target_fitness' in generator_kwargs")
            self.generator = UnconditionalVAERepresentation(
                z_dim=z_dim,
                num_objectives=num_objectives,
                device=self.device,
                **generator_kwargs
            )
        elif generator_type.lower() == 'diffusion':
            self.generator = ConditionalDiffusionRepresentation(
                z_dim=z_dim,
                num_objectives=num_objectives,
                device=self.device,
                **generator_kwargs
            )
        else:
            raise ValueError(f"Unknown generator_type: {generator_type}. Choose 'cvae', 'uncond_vae', or 'diffusion'.")
        
        self.generator_type = generator_type.lower()
        
        # Will be set during optimization
        self.optimizer = None
    
    def initial_train(
        self,
        archive_data: pd.DataFrame,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Phase 1: Pre-train generator on existing archive data.
        
        Args:
            archive_data: DataFrame with 'genome' (latent vectors) and objective columns
            num_epochs: Number of pre-training epochs
            batch_size: Batch size
            lr: Learning rate
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dict
        """
        print("=" * 80)
        print("Phase 1: Initial Training of Inverse Generator")
        print("=" * 80)
        
        # Extract z_arch vectors and fitness values
        # Assume archive_data has 'genome' column containing latent vectors
        # and objective columns matching self.surrogate.objectives.keys()
        
        z_arch_list = []
        fitness_list = []
        
        for idx, row in archive_data.iterrows():
            z_arch = row['genome']
            if isinstance(z_arch, np.ndarray):
                z_arch_list.append(z_arch)
            else:
                # If genome is not already a vector, need to encode it
                # This assumes vae_encoder can handle it
                print("Warning: 'genome' not a numpy array, attempting to encode...")
                # You may need to adjust this based on your VAE encoder API
                continue
            
            # Extract fitness values
            fitness_vals = [row[obj_name] for obj_name in self.surrogate.objectives.keys()]
            fitness_list.append(fitness_vals)
        
        z_arch_vectors = torch.tensor(np.stack(z_arch_list), dtype=torch.float32)
        fitness_values = torch.tensor(fitness_list, dtype=torch.float32)
        
        print(f"Archive data: {len(z_arch_vectors)} samples")
        print(f"z_arch shape: {z_arch_vectors.shape}")
        print(f"fitness shape: {fitness_values.shape}")
        
        # Train generator
        history = self.generator.initial_train(
            z_arch_vectors=z_arch_vectors,
            fitness_values=fitness_values,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=save_dir
        )
        
        print("Initial training complete!")
        return history
    
    def run_optimization_step(
        self,
        desired_fitness: torch.Tensor,
        batch_size: int = 64,
        lr: float = 1e-4,
        objective_indices: Optional[List[int]] = None,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Phase 2: Single optimization step using surrogate as critic.
        
        This method:
        1. Generates z_arch samples conditioned on desired_fitness
        2. Evaluates them with the frozen surrogate
        3. Computes loss between predicted and desired fitness
        4. Backpropagates to update generator weights
        
        Args:
            desired_fitness: Target fitness value(s), shape [num_objectives] or scalar
            batch_size: Number of samples to generate
            lr: Learning rate (used if optimizer not provided)
            objective_indices: Which objectives to optimize (None = all)
            optimizer: Optional custom optimizer (otherwise uses Adam)
            
        Returns:
            Tuple of (generated_z_batch, predicted_fitness, loss_value)
        """
        # Initialize optimizer on first call
        if optimizer is None:
            if self.optimizer is None:
                self.optimizer = optim.Adam(self.generator.parameters, lr=lr)
            optimizer = self.optimizer
        
        # Ensure desired_fitness is a tensor
        if not isinstance(desired_fitness, torch.Tensor):
            desired_fitness = torch.tensor([desired_fitness], dtype=torch.float32)
        desired_fitness = desired_fitness.to(self.device)
        
        # Expand to match batch size if needed
        if desired_fitness.dim() == 1:
            desired_fitness_batch = desired_fitness.unsqueeze(0).expand(batch_size, -1)
        else:
            desired_fitness_batch = desired_fitness
        
        # Set generator to train mode
        self.generator.fitness_embedder.train()
        if hasattr(self.generator, 'encoder_shared'):
            self.generator.encoder_shared.train()
        if hasattr(self.generator, 'decoder'):
            self.generator.decoder.train()
        
        # Generate z_arch samples (differentiable)
        z_arch_batch = self.generator.sample(c=desired_fitness_batch, batch_size=batch_size)
        
        # Get surrogate predictions (must be differentiable!)
        # This calls the new predict method we'll add to surrogate.py
        f_pred_batch = self.surrogate.predict(z_arch_batch)
        
        # Compute loss
        if objective_indices is not None:
            # Optimize only specific objectives
            f_pred_selected = f_pred_batch[:, objective_indices]
            desired_selected = desired_fitness_batch[:, objective_indices]
            loss = F.mse_loss(f_pred_selected, desired_selected)
        else:
            # Optimize all objectives
            loss = F.mse_loss(f_pred_batch, desired_fitness_batch)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters, 1.0)
        optimizer.step()
        
        return z_arch_batch.detach(), f_pred_batch.detach(), loss.item()
    
    def save_generator(self, path: str):
        """Save generator checkpoint."""
        self.generator.save_checkpoint(path)
    
    def load_generator(self, path: str):
        """Load generator checkpoint."""
        self.generator.load_checkpoint(path)

class ConditionedODENet(nn.Module):
    """
    Neural network parametrizing the dynamics dz/dt = f(t, z, c).
    Conditioned on fitness c.
    """
    def __init__(self, z_dim, cond_dim, hidden_dims=[64, 64], nonlinearity='tanh'):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        
        # Activation function
        if nonlinearity == 'tanh':
            self.act = nn.Tanh()
        elif nonlinearity == 'relu':
            self.act = nn.ReLU()
        elif nonlinearity == 'softplus':
            self.act = nn.Softplus()
        else:
            self.act = nn.Tanh()

        # Time embedding
        self.time_embed_dim = 16
        self.time_net = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            self.act,
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        # Main network
        # Input: z_dim (state) + time_embed + cond_dim (fitness)
        layers = []
        input_dim = z_dim + self.time_embed_dim + cond_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(self.act)
            input_dim = h_dim
            
        # Output: dz/dt (same dim as z)
        layers.append(nn.Linear(input_dim, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, z, c):
        # Handle time tensor broadcasting
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.expand(z.shape[0], 1)
        
        # Embed time
        t_emb = self.time_net(t)
        
        # Concatenate state, time, and condition
        # z: [B, z_dim], t_emb: [B, t_dim], c: [B, c_dim]
        x = torch.cat([z, t_emb, c], dim=1)
        
        return self.net(x)

class ConditionalNormalizingFlow(IGenerator):
    """
    Conditional Continuous Normalizing Flow (CNF) for Inverse Design.
    
    Models the conditional distribution p(z_arch | fitness) using a Neural ODE.
    
    Training Modes:
    1. track_divergence=True (Default/Paper): Maximizes exact log-likelihood. 
       Uses Hutchinson's trace estimator for O(1) memory cost via adjoint method.
    2. track_divergence=False (Fast): Minimizes reconstruction/cycle-consistency loss.
       Maps z -> latent (Gaussian) -> z_recon. 
       Loss = ||z - z_recon|| + ||latent|| (prior regularization).
    """
    
    def __init__(
        self, 
        z_dim: int, 
        num_objectives: int,
        hidden_dims: List[int] = [128, 128],
        nonlinearity: str = 'tanh',
        time_net: bool = False, 
        track_divergence: bool = True,
        device: torch.device = None,
        **kwargs
    ):
        self.z_dim = z_dim
        self.num_objectives = num_objectives
        self.track_divergence = track_divergence
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fitness embedding
        self.fitness_embed_dim = 32
        self.fitness_embedder = nn.Sequential(
            nn.Linear(num_objectives, self.fitness_embed_dim),
            nn.Tanh(),
            nn.Linear(self.fitness_embed_dim, self.fitness_embed_dim)
        ).to(self.device)
        
        # ODE Dynamics Network
        self.ode_net = ConditionedODENet(
            z_dim=z_dim, 
            cond_dim=self.fitness_embed_dim,
            hidden_dims=hidden_dims,
            nonlinearity=nonlinearity
        ).to(self.device)
        
        # Integration times
        self.t0 = 0.0
        self.t1 = 1.0
        
        # Trace estimator noise distribution
        self.trace_noise_dist = torch.distributions.Bernoulli(torch.tensor(0.5).to(self.device))

    @property
    def parameters(self):
        return list(self.fitness_embedder.parameters()) + list(self.ode_net.parameters())

    # --- HELPER CLASSES FOR ODEINT ---
    class LikelihoodWrapper(nn.Module):
        """
        Wrapper for exact likelihood training.
        State: (z, log_det, c_emb)
        """
        def __init__(self, cnf):
            super().__init__()
            self.cnf = cnf
            
        def forward(self, t, states):
            z = states[0]
            c_emb = states[2] # Use c_emb from state to ensure adjoint gradients flow
            
            # CRITICAL FIX: Enable grad for z locally to compute divergence
            with torch.set_grad_enabled(True):
                z.requires_grad_(True)
                
                # Compute dz/dt
                dz_dt = self.cnf.ode_net(t, z, c_emb)
                
                # Compute divergence via Hutchinson's estimator
                # Sample random vector v
                epsilon = (self.cnf.trace_noise_dist.sample(z.shape) * 2 - 1).detach()
                
                # Compute vector-Jacobian product: v^T * (df/dz)
                dz_dt_epsilon = torch.autograd.grad(
                    dz_dt, z, epsilon, create_graph=True, retain_graph=True
                )[0]
                
                # Estimate trace: v^T * (df/dz) * v
                divergence = torch.sum(dz_dt_epsilon * epsilon, dim=1, keepdim=True)
            
            # Return derivatives: (dz/dt, -divergence, dc/dt=0)
            return dz_dt, -divergence, torch.zeros_like(c_emb)

    class CycleWrapper(nn.Module):
        """
        Wrapper for cycle consistency training.
        State: (z, c_emb)
        """
        def __init__(self, ode_net):
            super().__init__()
            self.ode_net = ode_net
            
        def forward(self, t, states):
            z = states[0]
            c_emb = states[1]
            dz_dt = self.ode_net(t, z, c_emb)
            return dz_dt, torch.zeros_like(c_emb)

    def initial_train(
        self, 
        z_arch_vectors: torch.Tensor, 
        fitness_values: torch.Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the CNF on archive data."""
        
        z_arch_vectors = z_arch_vectors.to(self.device)
        fitness_values = fitness_values.to(self.device)
        
        dataset = ArchiveFitnessDataset(z_arch_vectors, fitness_values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        optimizer = optim.Adam(self.parameters, lr=lr)
        
        history = {'nll_loss': [], 'recon_loss': []} if self.track_divergence else {'recon_loss': []}
        
        print(f"Starting CNF training (Mode: {'Exact Likelihood' if self.track_divergence else 'Cycle Consistency'})")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for z_batch, f_batch in pbar:
                z_batch = z_batch.to(self.device)
                f_batch = f_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Embed condition
                c_emb = self.fitness_embedder(f_batch)
                
                if self.track_divergence:
                    # --- Likelihood Training ---
                    batch_size = z_batch.shape[0]
                    log_det_0 = torch.zeros(batch_size, 1).to(self.device)
                    
                    func = self.LikelihoodWrapper(self)
                    
                    # Integrate forward: t0 -> t1
                    state_t = odeint(
                        func,
                        (z_batch, log_det_0, c_emb),
                        torch.tensor([self.t0, self.t1]).to(self.device),
                        atol=1e-5, rtol=1e-5, method='dopri5'
                    )
                    
                    z_T = state_t[0][-1]
                    delta_log_det = state_t[1][-1]
                    
                    # Prior log-likelihood
                    log_prob_prior = -0.5 * (z_T.pow(2).sum(1, keepdim=True) + 
                                           self.z_dim * np.log(2 * np.pi))
                    
                    # Total log-likelihood
                    log_prob = log_prob_prior + delta_log_det
                    loss = -torch.mean(log_prob)
                    
                    history['nll_loss'].append(loss.item())
                    
                else:
                    # --- Fast Cycle-Consistency Training ---
                    
                    # FIX: Removing c_emb from init arguments here!
                    func_fwd = self.CycleWrapper(self.ode_net)
                    
                    # Forward integration: z -> z_prior
                    z_prior = odeint(
                        func_fwd,
                        (z_batch, c_emb),
                        torch.tensor([self.t0, self.t1]).to(self.device),
                        atol=1e-4, rtol=1e-4, method='dopri5'
                    )[0][-1]
                    
                    # Backward integration: z_prior -> z_recon
                    z_recon = odeint(
                        func_fwd,
                        (z_prior, c_emb),
                        torch.tensor([self.t1, self.t0]).to(self.device),
                        atol=1e-4, rtol=1e-4, method='dopri5'
                    )[0][-1]
                    
                    # Loss
                    recon_loss = F.mse_loss(z_recon, z_batch)
                    prior_loss = torch.mean(z_prior.pow(2))
                    
                    loss = recon_loss + 0.01 * prior_loss
                    history['recon_loss'].append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if save_dir and (epoch + 1) % 10 == 0:
                os.makedirs(save_dir, exist_ok=True)
                self.save_checkpoint(os.path.join(save_dir, f'cnf_epoch_{epoch+1}.pt'))
                
        print("CNF training complete.")
        return history

    def sample(self, c: torch.Tensor, batch_size: int, **kwargs) -> torch.Tensor:
        self.ode_net.eval()
        self.fitness_embedder.eval()
        
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(batch_size, -1)
        c = c.to(self.device)
        
        with torch.no_grad():
            z_prior = torch.randn(batch_size, self.z_dim).to(self.device)
            c_emb = self.fitness_embedder(c)
            
            # FIX: Removing c_emb from init arguments here as well!
            func = self.CycleWrapper(self.ode_net)
            
            # Integrate backwards (t1 -> t0)
            z_gen = odeint(
                func,
                (z_prior, c_emb),
                torch.tensor([self.t1, self.t0]).to(self.device),
                atol=1e-5, rtol=1e-5, method='dopri5'
            )[0][-1]
            
        self.ode_net.train()
        self.fitness_embedder.train()
        return z_gen

    def save_checkpoint(self, path: str):
        torch.save({
            'ode_net': self.ode_net.state_dict(),
            'fitness_embedder': self.fitness_embedder.state_dict(),
            'z_dim': self.z_dim,
            'num_objectives': self.num_objectives
        }, path)
        print(f"CNF saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.ode_net.load_state_dict(checkpoint['ode_net'])
        self.fitness_embedder.load_state_dict(checkpoint['fitness_embedder'])
        print(f"CNF loaded from {path}")