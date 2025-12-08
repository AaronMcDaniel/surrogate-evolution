import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from surrogates.surrogate_eval import prepare_data
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
import pyro.distributions as dist
from torch.utils.data import ConcatDataset

LATENT_DIM = 512

# Base VAE Class
class BaseVAE(nn.Module):
    def __init__(self, input_dim=6151, latent_dim=LATENT_DIM):
        super(BaseVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2_mu = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x):
        h = F.relu(self.ln1(self.fc1(x)))
        # h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-8)  # Prevent zero std
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.ln3(self.fc3(z)))
        # h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        if torch.isnan(x).any().item():
            raise ValueError("Input contains NaN values.")
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def weights_init(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization keeps variance consistent across layers
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# 6. Just a larger VAE with more depth and width in layers - 3 layers
class LargeVAE(BaseVAE):
    def __init__(self, input_dim=6151, latent_dim=LATENT_DIM, dropout_p=0.3):
        super(LargeVAE, self).__init__(input_dim, latent_dim)
        
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
        # self.ln6 = nn.LayerNorm(2048)
        # self.fc7 = nn.Linear(2048, input_dim)
        
        # --- Dropout Layer ---
        self.dropout = nn.Dropout(p=dropout_p)

    def encode(self, x):
        h = self.dropout(F.gelu(self.ln1(self.fc1(x))))
        # h = self.dropout(F.gelu(self.fc1(x)))
        # h = F.gelu(self.ln2(self.fc2(h)))
        h = self.dropout(F.gelu(self.ln3(self.fc3(h))))
        
        mu = self.fc3_mu(h)
        logvar = self.fc3_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.dropout(F.gelu(self.ln4(self.fc4(z))))
        h = self.dropout(F.gelu(self.ln5(self.fc5(h))))
        # h = self.dropout(F.gelu(self.fc5(h)))
        
        # No activation/dropout on the final reconstruction layer
        return self.fc6(h)


# 2. Hierarchical VAE
class HVAE(BaseVAE):
    def __init__(self, input_dim=6151, latent_dim=LATENT_DIM):
        super(HVAE, self).__init__(input_dim, latent_dim)
        self.fc2_z2 = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        z1 = super().reparameterize(mu, logvar)
        z2 = self.fc2_z2(z1)
        return z2

# 3. Normalizing Flow-based VAE
class NFVAE(BaseVAE):
    def __init__(self, input_dim=6151, latent_dim=LATENT_DIM, flow_steps=4):
        super(NFVAE, self).__init__(input_dim, latent_dim)
        self.transforms = nn.ModuleList([AffineAutoregressive(AutoRegressiveNN(latent_dim, [latent_dim])) for _ in range(flow_steps)])

    def reparameterize(self, mu, logvar):
        base_dist = Normal(mu, torch.exp(0.5 * logvar))
        for transform in self.transforms:
            base_dist = dist.TransformedDistribution(base_dist, [transform])
        return base_dist.rsample()

# 4. MoG-VAE
class MoGVAE(BaseVAE):
    def __init__(self, input_dim=6151, latent_dim=32, num_components=5):
        super(MoGVAE, self).__init__(input_dim, latent_dim)
        self.num_components = num_components
        self.mixture_weights = nn.Linear(512, num_components)
        self.mu_components = nn.Linear(512, num_components * latent_dim)
        self.logvar_components = nn.Linear(512, num_components * latent_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        weights = F.softmax(self.mixture_weights(h), dim=1)  # Shape: (batch_size, num_components)
        mu = self.mu_components(h).view(-1, self.num_components, self.latent_dim)  # (batch, num_comp, latent_dim)
        logvar = self.logvar_components(h).view(-1, self.num_components, self.latent_dim)
        
        # Return posterior parameters instead of sampling directly
        return weights, mu, logvar
        
    def reparameterize(self, weights, mu, logvar):
        """Reparameterization for Mixture of Gaussians"""
        # Sample from the categorical distribution to determine which Gaussian to use
        batch_size = weights.size(0)
        component_indices = Categorical(weights).sample()  # Shape: (batch_size,)
        
        # Extract the corresponding mus and logvars for each sample
        batch_indices = torch.arange(batch_size, device=weights.device)
        selected_mu = mu[batch_indices, component_indices]  # Shape: (batch_size, latent_dim)
        selected_logvar = logvar[batch_indices, component_indices]  # Shape: (batch_size, latent_dim)
        
        # Perform reparameterization on the selected Gaussian
        std = torch.exp(0.5 * selected_logvar)
        eps = torch.randn_like(std)
        return selected_mu + eps * std
        
    def forward(self, x):
        weights, mu, logvar = self.encode(x)
        z = self.reparameterize(weights, mu, logvar)
        return self.decode(z), mu, logvar

def train_vae(vae, train_loader, val_loader, epochs=200, lr=5e-4, device=None):
    """Train a VAE model with validation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae = vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    vae.train()
    for epoch in range(epochs):
        data_iter = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        total_loss = 0
        total_recon_loss = 0
        total_kl_div = 0
        ctrt = 0
        for vector, _ in data_iter:
            vector = vector.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(vector)
            loss = loss_function(recon, vector, mu, logvar)
            recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss
            total_kl_div += kl_div
            
            data_iter.set_postfix(loss=loss.item())
            ctrt += 1

        # Validation Loss Calculation
        vae.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_div = 0
        ctrv = 0
        with torch.no_grad():
            for vector, _ in val_loader:
                vector = vector.to(device)
                recon, mu, logvar = vae(vector)
                loss = loss_function(recon, vector, mu, logvar)
                recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
                val_loss += loss.item()
                val_recon_loss += recon_loss
                val_kl_div += kl_div
                ctrv += 1
        vae.train()
        
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/ctrt:.6f}, Recon Loss = {total_recon_loss/ctrt:.6f}, KL Divergence = {total_kl_div/ctrt:.6f}")
        print(f"Epoch {epoch+1}: Validation Loss = {val_loss/ctrv:.6f}, Val Recon Loss = {val_recon_loss/ctrv:.6f}, Val KL Divergence = {val_kl_div/ctrv:.6f}")
    
    return vae

def get_latent_representation(vae, data_df, genomes_scaler, device=None, use_provided_scaler=True):
    """Extract latent representations from a trained VAE using a pre-fitted scaler."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vae.eval()
    total_recon_loss = 0
    genomes = np.stack(data_df['genome'].values)
    if use_provided_scaler:
        genomes = genomes_scaler.transform(genomes)
    else:
        genomes_scaler_local = StandardScaler()
        genomes = genomes_scaler_local.fit_transform(genomes)
    
    # Check for NaN/Inf after scaling
    if np.isnan(genomes).any() or np.isinf(genomes).any():
        print("WARNING: NaN or Inf in genomes after scaling in get_latent_representation!")
        genomes = np.nan_to_num(genomes, nan=0.0, posinf=0.0, neginf=0.0)
    
    latent_vectors = []
    with torch.no_grad():
        for i in range(genomes.shape[0]):
            vector = torch.from_numpy(genomes[i,:]).float().unsqueeze(0)  # Add batch dimension
            vector = vector.to(device)
            recon, mu, _ = vae(vector)
            recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
            total_recon_loss += recon_loss
            latent_vectors.append(mu.squeeze(0).cpu().numpy())  # Remove batch dimension
    
    print(f"Reconstruction Loss on Dataset: {total_recon_loss:.2f}")
    data_df_copy = data_df.copy()
    data_df_copy['genome'] = latent_vectors
    return data_df_copy

