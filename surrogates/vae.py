import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
from surrogates.surrogate_eval import prepare_data
from sklearn.preprocessing import StandardScaler, RobustScaler

mode = 'old'
print("Mode", mode)
# Load Data
with open(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/{mode}_codec/{mode}_codec_reg_train.pkl', 'rb') as f:
    train_df = pickle.load(f)
with open(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/{mode}_codec/{mode}_codec_reg_val.pkl', 'rb') as f:
    val_df = pickle.load(f)

# DataLoader Preparation
batch_size = 16
train_loader, val_loader, _, _ = prepare_data({'metrics_subset': [0,1,2,3]}, batch_size, train_df, val_df)
LATENT_DIM = 64

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=1021, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        
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

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
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
with open(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/{mode}_codec/{mode}_codec_reg_train_latent_64.pkl', 'wb') as f:
    pickle.dump(train_df, f)
with open(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/strong_codec_test/{mode}_codec/{mode}_codec_reg_val_latent_64.pkl', 'wb') as f:
    pickle.dump(val_df, f)

print("Mode", mode)
print("Latent representations saved successfully.")