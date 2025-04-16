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

mode = 'old'
print("Mode", mode)
# Load Data
with open(f'/storage/ice-shared/vip-vvk/data/AOT/surrogate_dataset/pretrain_cls_train.pkl', 'rb') as f:
    train_df = pickle.load(f)
with open(f'/storage/ice-shared/vip-vvk/data/AOT/surrogate_dataset/surr_cls_val.pkl', 'rb') as f:
    val_df = pickle.load(f)

all_df = pd.concat([train_df, val_df])
# DataLoader Preparation
batch_size = 16
train_loader, val_loader, _, _ = prepare_data({'metrics_subset': [0,1,2,3]}, batch_size, all_df, val_df)
LATENT_DIM = 256

# Base VAE Class
class BaseVAE(nn.Module):
    def __init__(self, input_dim=1021, latent_dim=LATENT_DIM):
        super(BaseVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2_mu = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# 2. Hierarchical VAE
class HVAE(BaseVAE):
    def __init__(self, input_dim=1021, latent_dim=LATENT_DIM):
        super(HVAE, self).__init__(input_dim, latent_dim)
        self.fc2_z2 = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        z1 = super().reparameterize(mu, logvar)
        z2 = self.fc2_z2(z1)
        return z2

# 3. Normalizing Flow-based VAE
class NFVAE(BaseVAE):
    def __init__(self, input_dim=1021, latent_dim=LATENT_DIM, flow_steps=4):
        super(NFVAE, self).__init__(input_dim, latent_dim)
        self.transforms = nn.ModuleList([AffineAutoregressive(AutoRegressiveNN(latent_dim, [latent_dim])) for _ in range(flow_steps)])

    def reparameterize(self, mu, logvar):
        base_dist = Normal(mu, torch.exp(0.5 * logvar))
        for transform in self.transforms:
            base_dist = dist.TransformedDistribution(base_dist, [transform])
        return base_dist.rsample()

# 4. MoG-VAE
class MoGVAE(BaseVAE):
    def __init__(self, input_dim=1021, latent_dim=32, num_components=5):
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
# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = BaseVAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=5e-4)

epochs = 40
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

# Extract Latent Representations
vae.eval()
def get_latent_representation(data_df, loader):
    total_recon_loss = 0
    # ctr = 1
    genomes_scaler = StandardScaler()
    genomes = np.stack(data_df['genome'].values)
    genomes = genomes_scaler.fit_transform(genomes)
    latent_vectors = []
    with torch.no_grad():
        for i in range(genomes.shape[0]):
            vector = torch.from_numpy(genomes[i,:]).float()
            vector = vector.to(device)
            recon, mu, _ = vae(vector)
            recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
            total_recon_loss += recon_loss
            latent_vectors.append(mu.cpu().numpy())
            # ctr += 1
    
    print(f"Reconstruction Loss on Dataset: {total_recon_loss:.2f}")
    data_df['genome'] = latent_vectors
    return data_df

train_df = get_latent_representation(train_df, train_loader)
val_df = get_latent_representation(val_df, val_loader)

# Save Updated DataFrames
with open(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/surrogate_dataset/{mode}_codec_cls_train_latent_256.pkl', 'wb') as f:
    pickle.dump(train_df, f)
with open(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/surrogate_dataset/{mode}_codec_cls_val_latent_256.pkl', 'wb') as f:
    pickle.dump(val_df, f)

print("Mode", mode)
print("Latent representations saved successfully.")