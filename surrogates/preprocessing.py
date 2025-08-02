import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


# Base VAE Class
class BaseVAE(nn.Module):
    latent_dim = 256

    def __init__(self, input_dim=1021, latent_dim=latent_dim):
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
    
class VAEPreprocessor():
    def __init__(self, train_df, val_df, train_loader, val_loader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = BaseVAE().to(self.device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=5e-4)
        self.epochs = 40
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_df = train_df
        self.val_df = val_df
    
    def vae_loss_function(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def train_vae(self):
        self.vae.train()

        for epoch in range(self.epochs):
            data_iter = tqdm(self.train_loader, desc=f'Training Epoch {epoch+1}')
            total_loss = 0
            total_recon_loss = 0
            total_kl_div = 0
            ctrt = 0
            for vector, _ in data_iter:
                vector = vector.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.vae(vector)
                loss = VAEPreprocessor.vae_loss_function(recon, vector, mu, logvar)
                recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss
                total_kl_div += kl_div
                
                data_iter.set_postfix(loss=loss.item())
                ctrt += 1

            # Validation Loss Calculation
            if self.val_loader:
                self.vae.eval()
                val_loss = 0
                val_recon_loss = 0
                val_kl_div = 0
                ctrv = 0
                with torch.no_grad():
                    for vector, _ in self.val_loader:
                        vector = vector.to(self.device)
                        recon, mu, logvar = self.vae(vector)
                        loss = self.vae_loss_function(recon, vector, mu, logvar)
                        recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
                        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
                        val_loss += loss.item()
                        val_recon_loss += recon_loss
                        val_kl_div += kl_div
                        ctrv += 1
                self.vae.train()
            
            print(f"Epoch {epoch+1}: Train Loss = {total_loss/ctrt:.6f}, Recon Loss = {total_recon_loss/ctrt:.6f}, KL Divergence = {total_kl_div/ctrt:.6f}")
            if self.val_loader:
                print(f"Epoch {epoch+1}: Validation Loss = {val_loss/ctrv:.6f}, Val Recon Loss = {val_recon_loss/ctrv:.6f}, Val KL Divergence = {val_kl_div/ctrv:.6f}")

    def apply_vae(self, data_df):
        self.vae.eval()
        total_recon_loss = 0
        # ctr = 1
        genomes = np.stack(data_df['genome'].values)
        latent_vectors = []
        with torch.no_grad():
            for i in range(genomes.shape[0]):
                vector = torch.from_numpy(genomes[i,:]).float()
                vector = vector.to(self.device)
                recon, mu, _ = self.vae(vector)
                recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
                total_recon_loss += recon_loss
                latent_vectors.append(mu.cpu().numpy())
                # ctr += 1
        
        print(f"Reconstruction Loss on Dataset: {total_recon_loss:.2f}")
        data_df['genome'] = latent_vectors
        return data_df
    
    def process(self):
        self.train_vae()
        mod_train_df = self.apply_vae(self.train_df)
        mod_val_df = self.apply_vae(self.val_df)
        return mod_train_df, mod_val_df
