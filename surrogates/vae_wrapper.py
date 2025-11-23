"""
VAE Encoder Wrapper for Inverse Design.

Provides utilities to load and use the trained VAE encoder to convert
architecture representations to latent vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional


class LargeVAE(nn.Module):
    """
    Large VAE architecture matching the one in vae.py.
    This is used to load the pre-trained encoder.
    """
    
    def __init__(self, input_dim=6151, latent_dim=512, dropout_p=0.3):
        super(LargeVAE, self).__init__()
        
        # --- Encoder Layers ---
        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        
        self.fc3_mu = nn.Linear(512, latent_dim)
        self.fc3_logvar = nn.Linear(512, latent_dim)

        # --- Decoder Layers ---
        self.fc4 = nn.Linear(latent_dim, 512)
        self.ln4 = nn.LayerNorm(512)
        self.fc5 = nn.Linear(512, 1024)
        self.ln5 = nn.LayerNorm(1024)
        self.fc6 = nn.Linear(1024, input_dim)
        
        # --- Dropout Layer ---
        self.dropout = nn.Dropout(p=dropout_p)

    def encode(self, x):
        h = self.dropout(F.gelu(self.ln1(self.fc1(x))))
        h = self.dropout(F.gelu(self.ln3(self.fc3(h))))
        
        mu = self.fc3_mu(h)
        logvar = self.fc3_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dropout(F.gelu(self.ln4(self.fc4(z))))
        h = self.dropout(F.gelu(self.ln5(self.fc5(h))))
        return self.fc6(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEEncoder:
    """
    Wrapper for the VAE encoder to provide a clean interface for inverse design.
    """
    
    def __init__(
        self, 
        checkpoint_path: Optional[str] = None,
        input_dim: int = 1021,
        latent_dim: int = 512,
        dropout_p: float = 0.3,
        device: torch.device = None
    ):
        """
        Args:
            checkpoint_path: Path to trained VAE checkpoint (.pth file)
            input_dim: Input dimension (architecture encoding size)
            latent_dim: Latent dimension
            dropout_p: Dropout probability
            device: Device to run on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # Initialize VAE
        self.vae = LargeVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout_p=dropout_p
        ).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Set to eval mode
        self.vae.eval()
        
    def load_checkpoint(self, path: str):
        """Load VAE weights from checkpoint."""
        self.vae.load_state_dict(torch.load(path, map_location=self.device))
        print(f"VAE encoder loaded from {path}")
    
    def encode(
        self, 
        x: Union[torch.Tensor, np.ndarray],
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Encode architectures to latent space.
        
        Args:
            x: Architecture encodings, shape [B, input_dim] or [input_dim]
            deterministic: If True, returns mean (mu); if False, samples from distribution
            
        Returns:
            Latent vectors, shape [B, latent_dim] or [latent_dim]
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Add batch dimension if needed
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        
        x = x.to(self.device)
        
        # Encode
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            
            if deterministic:
                z = mu
            else:
                z = self.vae.reparameterize(mu, logvar)
        
        # Remove batch dimension if we added it
        if squeeze:
            z = z.squeeze(0)
        
        return z
    
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convenience method for encoding."""
        return self.encode(x, deterministic=True)
    
    def eval(self):
        """Set to eval mode (for compatibility with InverseDesigner)."""
        self.vae.eval()
    
    def parameters(self):
        """Return parameters (for compatibility with InverseDesigner freeze)."""
        return self.vae.parameters()


def load_vae_encoder(
    checkpoint_path: str,
    input_dim: int = 6151,
    latent_dim: int = 512,
    device: torch.device = None
) -> VAEEncoder:
    """
    Convenience function to load a trained VAE encoder.
    
    Args:
        checkpoint_path: Path to VAE checkpoint
        input_dim: Architecture encoding dimension
        latent_dim: Latent dimension
        device: Device to run on
        
    Returns:
        VAEEncoder instance
    """
    return VAEEncoder(
        checkpoint_path=checkpoint_path,
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device
    )
