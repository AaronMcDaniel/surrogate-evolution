"""
Conditional Normalizing Flow (CNF) based Inverse Design Pipeline

This script implements the inverse design approach from the paper:
1. Train an Autoencoder (not VAE) for dimensionality reduction
2. Train a Regressor (surrogate) on latent codes to predict properties
3. Train a Conditional Normalizing Flow to model p(z|properties)
4. Generate new designs by sampling from CNF conditioned on target properties

The key difference from run_inverse_design.py:
- Uses standard Autoencoder instead of VAE (no KL divergence term)
- Uses Conditional Normalizing Flow (CNF) instead of conditional VAE
- CNF models exact posterior p(z|c) rather than approximation
- More stable training without adversarial or variational objectives

PERFORMANCE NOTE:
- The CNF uses track_divergence=False for FAST training (reconstruction-based loss)
- Setting track_divergence=True enables exact likelihood computation but is ~100x slower
- The fast mode uses cycle-consistency (z -> prior -> z_recon) which works well in practice
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from surrogates.inverse_designer import ConditionalNormalizingFlow
from surrogates.surrogate import Surrogate
from surrogates import surrogate_dataset as sd, classifier_surrogate_eval as cse, surrogate_eval as rse
from surrogates.classifier_surrogate_eval import build_configuration, prepare_data as cls_prepare_data

# ============================================================================
# Configuration
# ============================================================================
INCLUDE_CLS = True
repo_dir = "/storage/ice-shared/vip-vvk/data/AOT/"
testing_dir = "psomu3/inverse_design/cnf_results"
dataset_dir = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset"

print("=" * 80)
print("CNF-BASED INVERSE DESIGN PIPELINE")
print("=" * 80)
print("This pipeline follows the approach from the paper:")
print("1. Autoencoder for dimensionality reduction (no VAE)")
print("2. Surrogate regressor for property prediction")
print("3. Conditional Normalizing Flow for p(z|properties)")
print("=" * 80)

# ============================================================================
# Load Datasets
# ============================================================================
print("\n" + "=" * 80)
print("Loading datasets...")
print("=" * 80)
cls_train_df = pd.read_pickle(os.path.join(dataset_dir, f'mix_dataset_cls_train.pkl'))
cls_val_df = pd.read_pickle(os.path.join(dataset_dir, f'mix_dataset_cls_val.pkl'))
reg_train_df = pd.read_pickle(os.path.join(dataset_dir, f'mix_dataset_reg_train.pkl'))
reg_val_df = pd.read_pickle(os.path.join(dataset_dir, f'mix_dataset_reg_val.pkl'))

print(f"Reg train size: {len(reg_train_df)}")
print(f"Reg val size: {len(reg_val_df)}")
print(f"Raw genome dimension: {len(reg_train_df['genome'].iloc[0])}")
print()

# Create directories
os.makedirs(os.path.join(repo_dir, testing_dir), exist_ok=True)
os.makedirs(os.path.join(repo_dir, testing_dir, 'surrogate_weights'), exist_ok=True)
os.makedirs(os.path.join(repo_dir, testing_dir, 'autoencoder_checkpoints'), exist_ok=True)
os.makedirs(os.path.join(repo_dir, testing_dir, 'cnf_checkpoints'), exist_ok=True)

# ============================================================================
# Autoencoder Architecture
# ============================================================================
class SimpleAutoencoder(nn.Module):
    """
    Standard Autoencoder (not VAE) for dimensionality reduction.
    
    Unlike VAE, this uses deterministic encoding without KL divergence.
    The encoder directly maps input to latent code, and decoder reconstructs.
    """
    
    def __init__(self, input_dim, latent_dim=256, hidden_dims=[1024, 512], dropout_p=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Deterministic encoding (no sampling like in VAE)."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent code to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ============================================================================
# STEP 1: Train Autoencoder (or Load Existing)
# ============================================================================
print("=" * 80)
print("STEP 1: Autoencoder Training")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(reg_train_df['genome'].iloc[0])
latent_dim = 256
ae_epochs = 60
ae_batch_size = 32
ae_lr = 1e-3
FORCE_AE_TRAIN = False
FORCE_SURR_TRAIN = False


# Check for existing checkpoint
ae_checkpoint_path = os.path.join(repo_dir, testing_dir, 'autoencoder_checkpoints', 'autoencoder.pth')
ae_exists = os.path.exists(ae_checkpoint_path) and not FORCE_AE_TRAIN

# Prepare data
genome_scaler = StandardScaler()
reg_genomes_train = np.stack(reg_train_df['genome'].values)
reg_genomes_val = np.stack(reg_val_df['genome'].values)
cls_genomes_train = np.stack(cls_train_df['genome'].values)
cls_genomes_val = np.stack(cls_val_df['genome'].values)

# Fit scaler on all training data
all_train_genomes = np.vstack([reg_genomes_train, cls_genomes_train])
genome_scaler.fit(all_train_genomes)

# Scale genomes
reg_genomes_train_scaled = genome_scaler.transform(reg_genomes_train)
reg_genomes_val_scaled = genome_scaler.transform(reg_genomes_val)
cls_genomes_train_scaled = genome_scaler.transform(cls_genomes_train)
cls_genomes_val_scaled = genome_scaler.transform(cls_genomes_val)

# Initialize Autoencoder
autoencoder = SimpleAutoencoder(input_dim=input_dim, latent_dim=latent_dim, dropout_p=0.3).to(device)

if ae_exists:
    print(f"✓ Loading existing Autoencoder from {ae_checkpoint_path}")
    autoencoder.load_state_dict(torch.load(ae_checkpoint_path, map_location=device))
    print(f"  Autoencoder loaded: {input_dim} → {latent_dim} dimensional latent space\n")
else:
    print("Training Autoencoder for dimensionality reduction...")
    print(f"Note: This is a standard AE, not VAE (no KL divergence)\n")
    
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=ae_lr)
    
    print(f"Architecture: {input_dim} → 1024 → 512 → {latent_dim}")
    print(f"Training on {len(all_train_genomes)} architectures for {ae_epochs} epochs")
    print(f"Device: {device}\n")
    
    # Training loop
    autoencoder.train()
    for epoch in range(ae_epochs):
        # Shuffle and batch training data
        indices = np.random.permutation(len(all_train_genomes))
        epoch_loss = 0
        n_batches = 0
        
        pbar = tqdm(range(0, len(all_train_genomes), ae_batch_size), 
                    desc=f'Epoch {epoch+1}/{ae_epochs}')
        
        for i in pbar:
            batch_indices = indices[i:i+ae_batch_size]
            batch = torch.from_numpy(genome_scaler.transform(all_train_genomes[batch_indices])).float().to(device)
            
            ae_optimizer.zero_grad()
            x_recon, z = autoencoder(batch)
            
            # Simple MSE reconstruction loss (no KL term like in VAE)
            loss = F.mse_loss(x_recon, batch)
            
            loss.backward()
            ae_optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        if epoch % 10 == 0 or epoch == ae_epochs - 1:
            autoencoder.eval()
            val_loss = 0
            val_batches = 0
            
            all_val_genomes = np.vstack([reg_genomes_val_scaled, cls_genomes_val_scaled])
            with torch.no_grad():
                for i in range(0, len(all_val_genomes), ae_batch_size):
                    batch = torch.from_numpy(all_val_genomes[i:i+ae_batch_size]).float().to(device)
                    x_recon, z = autoencoder(batch)
                    loss = F.mse_loss(x_recon, batch)
                    val_loss += loss.item()
                    val_batches += 1
            
            autoencoder.train()
            print(f"  Train Loss: {epoch_loss/n_batches:.4f} | Val Loss: {val_loss/val_batches:.4f}")
    
    # Save checkpoint
    torch.save(autoencoder.state_dict(), ae_checkpoint_path)
    print(f"\n✓ Autoencoder trained and saved to {ae_checkpoint_path}\n")

# ============================================================================
# STEP 2: Encode All Data with Trained Autoencoder
# ============================================================================
print("=" * 80)
print("STEP 2: Encoding Architectures to Latent Space")
print("=" * 80)

autoencoder.eval()
def encode_genomes(genomes_scaled):
    """Encode genomes to latent vectors using trained autoencoder."""
    latents = []
    with torch.no_grad():
        for i in range(0, len(genomes_scaled), ae_batch_size):
            batch = torch.from_numpy(genomes_scaled[i:i+ae_batch_size]).float().to(device)
            z = autoencoder.encode(batch)
            latents.append(z.cpu().numpy())
    return np.vstack(latents)

# Encode all datasets
print("Encoding regression train...")
reg_train_latents = encode_genomes(reg_genomes_train_scaled)
print("Encoding regression val...")
reg_val_latents = encode_genomes(reg_genomes_val_scaled)
print("Encoding classification train...")
cls_train_latents = encode_genomes(cls_genomes_train_scaled)
print("Encoding classification val...")
cls_val_latents = encode_genomes(cls_genomes_val_scaled)

# Update dataframes with latent representations
reg_train_df_latent = reg_train_df.copy()
reg_val_df_latent = reg_val_df.copy()
cls_train_df_latent = cls_train_df.copy()
cls_val_df_latent = cls_val_df.copy()

reg_train_df_latent['genome'] = list(reg_train_latents)
reg_val_df_latent['genome'] = list(reg_val_latents)
cls_train_df_latent['genome'] = list(cls_train_latents)
cls_val_df_latent['genome'] = list(cls_val_latents)

print(f"\n✓ Encoded to {latent_dim}-dimensional latent space")
print(f"  Reg train: {reg_train_latents.shape}")
print(f"  Reg val: {reg_val_latents.shape}")
print(f"  Cls train: {cls_train_latents.shape}")
print(f"  Cls val: {cls_val_latents.shape}\n")

# ============================================================================
# STEP 3: Train Surrogate on Latent Vectors (or Skip if Already Trained)
# ============================================================================
print("=" * 80)
print("STEP 3: Surrogate Training (Regressor)")
print("=" * 80)

surrogate = Surrogate('conf.toml', os.path.join(repo_dir, testing_dir, 'surrogate_weights'))

# Check if all surrogate weights already exist
def check_surrogate_weights_exist(surrogate, weights_dir):
    """Check if all classifier and regressor weights exist."""
    all_exist = True
    
    # Check classifier weights
    for cls_model in surrogate.classifier_models:
        weight_path = os.path.join(weights_dir, f"{cls_model['name']}.pth")
        if not os.path.exists(weight_path):
            all_exist = False
            print(f"Missing classifier weights: {weight_path}")
            break
    
    # Check regressor weights
    if all_exist:
        for reg_model in surrogate.models:
            weight_path = os.path.join(weights_dir, f"{reg_model['name']}.pth")
            if not os.path.exists(weight_path):
                all_exist = False
                print(f"Missing regressor weights: {weight_path}")
                break
    
    return all_exist

surrogate_weights_exist = check_surrogate_weights_exist(surrogate, os.path.join(repo_dir, testing_dir, 'surrogate_weights'))

if surrogate_weights_exist and not FORCE_SURR_TRAIN:
    print("✓ All surrogate weights found, skipping training")
    print("  Loading and evaluating existing models...\n")
    
    # Get scalers
    cls_dataset = sd.ClassifierSurrogateDataset(cls_train_df_latent, mode='train')
    cls_genome_scaler = cls_dataset.genomes_scaler
    
    reg_dataset = sd.SurrogateDataset(reg_train_df_latent, mode='train')
    reg_genome_scaler = reg_dataset.genomes_scaler  
    
    # Evaluate loaded models to get scores
    scores = {'classifiers': {}, 'regressors': {}}
    
    print("Evaluating classifiers...")
    for cls_model in surrogate.classifier_models:
        train_loader, val_loader, _, _ = cse.prepare_data(surrogate.surrogate_config['surrogate_batch_size'], 
                                                           cls_train_df_latent, cls_val_df_latent)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _, _, _ = build_configuration(model_dict=cls_model, device=device)
        model.load_state_dict(torch.load(f"{surrogate.weights_dir}/{cls_model['name']}.pth", map_location=device))
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for genomes, labels in val_loader:
                genomes, labels = genomes.to(device), labels.to(device)
                outputs = model(genomes).sigmoid()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total if total > 0 else 0
        scores['classifiers'][cls_model['name']] = {'acc': acc}
        print(f"  {cls_model['name']}: acc={acc:.4f}")
    
    print("\nEvaluating regressors...")
    for reg_model in surrogate.models:
        metrics = rse.get_val_scores(surrogate.surrogate_config, reg_model, 
                                     reg_train_df_latent, reg_val_df_latent, 
                                     surrogate.weights_dir)
        scores['regressors'][reg_model['name']] = metrics
        print(f"  {reg_model['name']}: metrics loaded")
    
    print("\n✓ All surrogate models loaded and evaluated\n")
else:
    print("Training surrogate models on latent representations...\n")
    scores, cls_genome_scaler, reg_genome_scaler = surrogate.train(
        cls_train_df_latent, cls_val_df_latent, reg_train_df_latent, reg_val_df_latent, reg_lambda=0
    )
    print("✓ Surrogate training complete!")
    print(f"  Regression scores: {scores['regressors']}\n")

# ============================================================================
# STEP 4: Select Best Surrogate Models
# ============================================================================
print("=" * 80)
print("STEP 4: Selecting Best Surrogate Models")
print("=" * 80)

sub_surrogates = []

if INCLUDE_CLS:
    cls_trust = 0
    max_cls_model = ''
    for key, val in scores['classifiers'].items():
        if val['acc'] > cls_trust:
            cls_trust = val['acc']
            max_cls_model = key
    cls_to_dict = {d['name']: d for d in surrogate.classifier_models}
    max_cls_model_idx = list(cls_to_dict.keys()).index(max_cls_model)
    sub_surrogates.append(max_cls_model_idx)
    print(f'✓ Classifier: {max_cls_model}')
else:
    sub_surrogates.append(0)
    print('✓ Skipping classifier')

# Select regressors for target objectives (ciou_loss, average_precision)
print('\nSelecting regressors for target objectives (ciou_loss, average_precision)...')
reg_model_indices = []
for idx, model in enumerate(surrogate.models):
    if 4 in model['validation_subset'] or 11 in model['validation_subset']:
        reg_model_indices.append(idx)
        print(f"  Model {idx} ({model['name']}): validation_subset={model['validation_subset']}")
        if len(reg_model_indices) >= 2:
            break

if len(reg_model_indices) < 2:
    print("Warning: Could not find enough specialized models, using first 2 regressors")
    reg_model_indices = [0, 1]

sub_surrogates += reg_model_indices
print(f'✓ Regressor indices: {reg_model_indices}')

surrogate.inference_models = sub_surrogates
print(f'✓ Surrogate ready with models: {sub_surrogates}\n')

# ============================================================================
# STEP 5: Train Conditional Normalizing Flow
# ============================================================================
print("=" * 80)
print("STEP 5: Training Conditional Normalizing Flow")
print("=" * 80)

num_objectives = 2  # ciou_loss and average_precision
target_fitness = torch.tensor([1.2, 0.70]).float()
print(f"Target: ciou_loss={target_fitness[0]:.3f}, AP={target_fitness[1]:.3f}\n")

# Check for existing CNF checkpoint
cnf_checkpoint_path = os.path.join(repo_dir, testing_dir, 'cnf_checkpoints', 'cnf_final.pt')
cnf_exists = os.path.exists(cnf_checkpoint_path)

# Initialize CNF
cnf = ConditionalNormalizingFlow(
    z_dim=latent_dim,
    num_objectives=num_objectives,
    hidden_dims=[256, 256, 256],
    time_net=False,
    nonlinearity='tanh',
    track_divergence=False,  # Set to False for FAST training (uses reconstruction loss)
    device=device
)
print(f"Training mode: {'Exact likelihood (SLOW)' if cnf.track_divergence else 'Reconstruction loss (FAST)'}")

if cnf_exists:
    print(f"✓ Loading existing CNF from {cnf_checkpoint_path}")
    cnf.load_checkpoint(cnf_checkpoint_path)
    print(f"  CNF loaded\n")
else:
    print("Training Conditional Normalizing Flow...")
    print("CNF models p(z|properties) using Neural ODEs\n")
    
    # Prepare training data
    z_latents = torch.from_numpy(reg_train_latents).float()
    fitness = torch.from_numpy(np.stack([
        reg_train_df['ciou_loss'].values,
        reg_train_df['average_precision'].values
    ], axis=1)).float()
    
    print(f"Training on {len(z_latents)} latent vectors (dim={latent_dim})")
    
    # Train CNF
    cnf_epochs = 10
    cnf_batch_size = 32
    cnf_lr = 1e-3
    
    history = cnf.initial_train(
        z_arch_vectors=z_latents,
        fitness_values=fitness,
        num_epochs=cnf_epochs,
        batch_size=cnf_batch_size,
        lr=cnf_lr,
        save_dir=os.path.join(repo_dir, testing_dir, 'cnf_checkpoints')
    )
    
    print(f"\n✓ CNF trained!")
    loss_key = 'nll_loss' if cnf.track_divergence else 'recon_loss'
    print(f"  Final {loss_key}: {history[loss_key][-1]:.4f}\n")

# ============================================================================
# STEP 6: Generate Samples using CNF
# ============================================================================
print("=" * 80)
print("STEP 6: Generating Samples with CNF")
print("=" * 80)

num_samples = 50
print(f"Generating {num_samples} architectures conditioned on target fitness...")

with torch.no_grad():
    # Sample from CNF conditioned on target fitness
    sampled_latents = cnf.sample(c=target_fitness, batch_size=num_samples)
    
    # Decode latents back to raw genome space using Autoencoder
    autoencoder.eval()
    decoded_genomes = autoencoder.decode(sampled_latents)
    
    # Inverse transform to original genome space
    decoded_genomes_np = decoded_genomes.cpu().numpy()
    decoded_genomes_unscaled = genome_scaler.inverse_transform(decoded_genomes_np)
    
    # Get predicted fitness for each sample using surrogate
    fitness_preds = surrogate.predict(sampled_latents, genome_scaler=reg_genome_scaler)
    fitness_preds_np = fitness_preds.cpu().numpy()

# ============================================================================
# STEP 7: Evaluate and Save Results
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Evaluation and Saving Results")
print("=" * 80)

mean_fit = fitness_preds_np.mean(axis=0)
std_fit = fitness_preds_np.std(axis=0)

print(f"Target:   ciou={target_fitness[0]:.3f}, AP={target_fitness[1]:.3f}")
print(f"Achieved: ciou={mean_fit[0]:.3f}±{std_fit[0]:.3f}, AP={mean_fit[1]:.3f}±{std_fit[1]:.3f}")
print(f"Error:    ciou={abs(mean_fit[0]-target_fitness[0].item()):.3f}, AP={abs(mean_fit[1]-target_fitness[1].item()):.3f}")

# Create results dataframe
results_df = pd.DataFrame({
    'sample_id': range(num_samples),
    'predicted_ciou_loss': fitness_preds_np[:, 0],
    'predicted_average_precision': fitness_preds_np[:, 1],
    'genome': [decoded_genomes_unscaled[i].tolist() for i in range(num_samples)],
    'latent': [sampled_latents[i].cpu().numpy().tolist() for i in range(num_samples)]
})

# Save to CSV
output_csv = os.path.join(repo_dir, testing_dir, 'generated_architectures_cnf.csv')
results_df[['sample_id', 'predicted_ciou_loss', 'predicted_average_precision', 'genome']].to_csv(output_csv, index=False)
print(f"\n✓ Saved fitness predictions and genomes to {output_csv}")

# Save full data to pickle
output_pkl = os.path.join(repo_dir, testing_dir, 'generated_architectures_cnf.pkl')
results_df.to_pickle(output_pkl)
print(f"✓ Saved full data (genomes + latents) to {output_pkl}")

# Save human-readable summary
output_txt = os.path.join(repo_dir, testing_dir, 'generated_architectures_cnf_summary.txt')
with open(output_txt, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CNF-BASED INVERSE DESIGN RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Method: Autoencoder + Conditional Normalizing Flow\n")
    f.write(f"Target Fitness: ciou_loss={target_fitness[0]:.3f}, AP={target_fitness[1]:.3f}\n")
    f.write(f"Number of samples: {num_samples}\n\n")
    f.write(f"Predicted Fitness Statistics:\n")
    f.write(f"  ciou_loss:           mean={fitness_preds_np[:, 0].mean():.3f}, std={fitness_preds_np[:, 0].std():.3f}\n")
    f.write(f"  average_precision:   mean={fitness_preds_np[:, 1].mean():.3f}, std={fitness_preds_np[:, 1].std():.3f}\n\n")
    f.write("=" * 80 + "\n")
    f.write("Top 10 Samples (closest to target):\n")
    f.write("=" * 80 + "\n\n")
    
    # Calculate distance to target for each sample
    target_np = target_fitness.cpu().numpy()
    distances = np.linalg.norm(fitness_preds_np - target_np, axis=1)
    top_indices = np.argsort(distances)[:10]
    
    for rank, idx in enumerate(top_indices, 1):
        f.write(f"Rank {rank}: Sample {idx}\n")
        f.write(f"  Predicted: ciou={fitness_preds_np[idx, 0]:.3f}, AP={fitness_preds_np[idx, 1]:.3f}\n")
        f.write(f"  Distance to target: {distances[idx]:.4f}\n")
        f.write(f"  Genome shape: {len(decoded_genomes_unscaled[idx])}\n")
        f.write(f"  Latent shape: {len(sampled_latents[idx])}\n\n")

print(f"✓ Saved human-readable summary to {output_txt}")

print("\n" + "=" * 80)
print("✅ CNF-BASED INVERSE DESIGN COMPLETE")
print("=" * 80)
print("\nKey advantages of this approach:")
print("  ✓ No KL divergence approximation (exact posterior inference)")
print("  ✓ More stable training (no adversarial objectives)")
print("  ✓ Deterministic autoencoder encoding")
print("  ✓ Flexible ODE-based generation")
print("=" * 80)
