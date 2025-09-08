import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os


# Base VAE Class
class BaseVAE(nn.Module):
    latent_dim = 256

    def __init__(self, input_dim=1021, latent_dim=latent_dim):
        super(BaseVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder with dropout for regularization
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2_mu = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)

        # Decoder with dropout for regularization
        self.fc3 = nn.Linear(latent_dim, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(512, input_dim)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization for better stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout1(h)
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = self.dropout2(h)
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
class VAEPreprocessor():
    def __init__(self, train_df, val_df, train_loader, val_loader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = BaseVAE().to(self.device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-4)  # Reduced learning rate
        self.epochs = 80
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_df = train_df
        self.val_df = val_df
    
    @staticmethod
    def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_div

    def train_vae(self):
        self.vae.train()

        for epoch in range(self.epochs):
            # Beta scheduling for KL divergence weight
            beta = min(1.0, epoch / 20.0)  # Gradually increase KL weight over first 20 epochs
            
            data_iter = tqdm(self.train_loader, desc=f'Training Epoch {epoch+1}')
            total_loss = 0
            total_recon_loss = 0
            total_kl_div = 0
            ctrt = 0
            for vector, _ in data_iter:
                vector = vector.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.vae(vector)
                
                # Check for NaN/inf in model outputs
                if torch.isnan(recon).any() or torch.isinf(recon).any():
                    print(f"NaN/inf detected in reconstruction at epoch {epoch+1}, batch {ctrt}")
                    break
                    
                loss = VAEPreprocessor.vae_loss_function(recon, vector, mu, logvar, beta)
                recon_loss = F.mse_loss(recon, vector, reduction='mean').item()
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
                
                # Check for NaN/inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/inf detected in loss at epoch {epoch+1}, batch {ctrt}")
                    break
                
                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
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
                        loss = VAEPreprocessor.vae_loss_function(recon, vector, mu, logvar, beta)
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
                total_recon_loss += recon_loss / genomes.shape[0]
                latent_vectors.append(mu.cpu().numpy())
                # ctr += 1
        
        print(f"Reconstruction Loss on Dataset: {total_recon_loss:.6f}")
        data_df['genome'] = latent_vectors
    
    def process(self):
        self.train_vae()
        self.apply_vae(self.train_df)
        self.apply_vae(self.val_df)
    
    def save_vae_weights(self, save_path):
        """Save VAE model weights to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.vae.state_dict(), save_path)
        print(f"VAE weights saved to {save_path}")
    
    def load_vae_weights(self, load_path):
        """Load VAE model weights from disk."""
        if os.path.exists(load_path):
            self.vae.load_state_dict(torch.load(load_path, map_location=self.device))
            self.vae.eval()
            print(f"VAE weights loaded from {load_path}")
            return True
        else:
            print(f"VAE weights file not found at {load_path}")
            return False
    
    def preprocess_inference_data(self, inference_df):
        """Apply VAE preprocessing to inference data (genome column only)."""
        
        copy_df = inference_df.copy()
        self.apply_vae(copy_df)

        return copy_df
