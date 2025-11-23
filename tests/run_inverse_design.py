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

from surrogates.inverse_designer import (
    ConditionalVAERepresentation,
    UnconditionalVAERepresentation,
    ConditionalDiffusionRepresentation,
    InverseDesigner
)
from surrogates.surrogate import Surrogate
from surrogates.vae_wrapper import LargeVAE
from surrogates import surrogate_dataset as sd, classifier_surrogate_eval as cse, surrogate_eval as rse
from surrogates.classifier_surrogate_eval import build_configuration, prepare_data as cls_prepare_data

INCLUDE_CLS = True
repo_dir = "/storage/ice-shared/vip-vvk/data/AOT/"
testing_dir = "psomu3/inverse_design/training_results"
dataset_dir = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codestral/large_dataset"
scores_file = os.path.join(repo_dir, testing_dir, f"scores_base.txt")

print("=" * 80)
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

"""
Regression dataset preview:
         hash                                             genome                                         str_genome  epoch_num  uw_val_epoch_loss  iou_loss  giou_loss  diou_loss  ciou_loss  center_loss  size_loss  obj_loss  precision    recall  f1_score  average_precision
0  ff33826a6e  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...  RetinaNet_Head(Tanh_2D(EfficientNet_V2(Adaptiv...          1           7.550578  0.999666   1.962304   1.865209   1.867929     0.004646   0.005002  0.845823   0.000000  0.000000  0.000000           0.000475
1  01e596f543  [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...  RetinaNet_Head(Tanh_2D(EfficientNet_V2(Adaptiv...          3           7.526393  0.999604   1.954491   1.857533   1.858476     0.004205   0.002173  0.849911   0.000000  0.000000  0.000000           0.000129
2  ef843a09d0  [7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...  RetinaNet_Head(ReLU_2D(ConvNeXt(Dropout_2D(Lea...          7           7.831706  0.999780   1.967598   1.866444   1.868240     0.005717   0.018846  1.105081   0.000000  0.000000  0.000000           0.000106
3  4a0051742f  [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...  RetinaNet_Head(Tanh_2D(ReLU_2D(ConvNeXt(IN0, d...          6           6.977745  0.962034   1.742825   1.668853   1.670035     0.005493   0.011654  0.916851   0.537594  0.516245  0.526703           0.497264
4  2b64526bb2  [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...  RetinaNet_Head(ReLU_2D(ConvNeXt(Dropout_2D(Lea...          8           7.758899  0.999843   1.971125   1.876738   1.877447     0.005440   0.004495  1.023811   0.000000  0.000000  0.000000           0.000025
"""

# Create directories
os.makedirs(os.path.join(repo_dir, testing_dir), exist_ok=True)
os.makedirs(os.path.join(repo_dir, testing_dir, 'surrogate_weights'), exist_ok=True)
os.makedirs(os.path.join(repo_dir, testing_dir, 'generator_checkpoints'), exist_ok=True)
os.makedirs(os.path.join(repo_dir, testing_dir, 'vae_checkpoints'), exist_ok=True)

# Check for existing checkpoints
vae_checkpoint_path = os.path.join(repo_dir, testing_dir, 'vae_checkpoints', 'vae_encoder.pth')
vae_exists = os.path.exists(vae_checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(reg_train_df['genome'].iloc[0])
latent_dim = 256
vae_epochs = 30
vae_batch_size = 32
vae_lr = 5e-4

# Prepare data for VAE
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

# ============================================================================
# STEP 1: Train VAE on Raw Genome Encodings (or Load Existing)
# ============================================================================
print("=" * 80)
print("STEP 1: VAE Encoder")
print("=" * 80)

# Initialize VAE
vae = LargeVAE(input_dim=input_dim, latent_dim=latent_dim, dropout_p=0.3).to(device)

if vae_exists:
    print(f"✓ Loading existing VAE from {vae_checkpoint_path}")
    vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    print(f"  VAE loaded: {input_dim} → {latent_dim} dimensional latent space\n")
else:
    print("Training VAE to learn latent representations...")
    print(f"The VAE will compress raw architecture encodings into latent space\n")
    
    vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    
    print(f"VAE Architecture: {input_dim} → 1024 → 512 → {latent_dim}")
    print(f"Training on {len(all_train_genomes)} architectures for {vae_epochs} epochs")
    print(f"Device: {device}\n")
    
    def vae_loss_function(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div, recon_loss, kl_div
    
    # Training loop
    vae.train()
    for epoch in range(vae_epochs):
        # Shuffle and batch training data
        indices = np.random.permutation(len(all_train_genomes))
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        n_batches = 0
        
        pbar = tqdm(range(0, len(all_train_genomes), vae_batch_size), 
                    desc=f'Epoch {epoch+1}/{vae_epochs}')
        
        for i in pbar:
            batch_indices = indices[i:i+vae_batch_size]
            batch = torch.from_numpy(genome_scaler.transform(all_train_genomes[batch_indices])).float().to(device)
            
            vae_optimizer.zero_grad()
            recon, mu, logvar = vae(batch)
            loss, recon_loss, kl_loss = vae_loss_function(recon, batch, mu, logvar)
            loss.backward()
            vae_optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        if epoch % 10 == 0 or epoch == vae_epochs - 1:
            vae.eval()
            val_loss = 0
            val_recon = 0
            val_kl = 0
            val_batches = 0
            
            all_val_genomes = np.vstack([reg_genomes_val_scaled, cls_genomes_val_scaled])
            with torch.no_grad():
                for i in range(0, len(all_val_genomes), vae_batch_size):
                    batch = torch.from_numpy(all_val_genomes[i:i+vae_batch_size]).float().to(device)
                    recon, mu, logvar = vae(batch)
                    loss, recon_loss, kl_loss = vae_loss_function(recon, batch, mu, logvar)
                    val_loss += loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_batches += 1
            
            vae.train()
            print(f"  Train: Loss={epoch_loss/n_batches:.4f}, Recon={epoch_recon/n_batches:.4f}, KL={epoch_kl/n_batches:.4f}")
            print(f"  Val:   Loss={val_loss/val_batches:.4f}, Recon={val_recon/val_batches:.4f}, KL={val_kl/val_batches:.4f}")
    
    # Save VAE checkpoint
    torch.save(vae.state_dict(), vae_checkpoint_path)
    print(f"\n✓ VAE trained and saved to {vae_checkpoint_path}\n")

# ============================================================================
# STEP 2: Encode All Data with Trained VAE
# ============================================================================
print("=" * 80)
print("STEP 2: Encoding Architectures to Latent Space")
print("=" * 80)

vae.eval()
def encode_genomes(genomes_scaled):
    """Encode genomes to latent vectors using trained VAE."""
    latents = []
    with torch.no_grad():
        for i in range(0, len(genomes_scaled), vae_batch_size):
            batch = torch.from_numpy(genomes_scaled[i:i+vae_batch_size]).float().to(device)
            mu, _ = vae.encode(batch)
            latents.append(mu.cpu().numpy())
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
print("STEP 3: Surrogate Training")
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

if surrogate_weights_exist:
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
        # Load and evaluate
        train_loader, val_loader, _, _ = cse.prepare_data(surrogate.surrogate_config['surrogate_batch_size'], 
                                                           cls_train_df_latent, cls_val_df_latent)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _, _, _ = build_configuration(model_dict=cls_model, device=device)
        model.load_state_dict(torch.load(f"{surrogate.weights_dir}/{cls_model['name']}.pth", map_location=device))
        
        # Quick validation to get acc metric
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
        # Use the built-in get_val_scores function
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

# For inverse design, we'll use the first 2 regressors that predict our target objectives
# Target: ciou_loss (index 4) and average_precision (index 11)
print('\nSelecting regressors for target objectives (ciou_loss, average_precision)...')
reg_model_indices = []
for idx, model in enumerate(surrogate.models):
    # Check if model predicts ciou_loss (metric index 4) or average_precision (metric index 11)
    if 4 in model['validation_subset'] or 11 in model['validation_subset']:
        reg_model_indices.append(idx)
        print(f"  Model {idx} ({model['name']}): validation_subset={model['validation_subset']}")
        if len(reg_model_indices) >= 2:  # We only need 2 objectives
            break

if len(reg_model_indices) < 2:
    print("Warning: Could not find enough specialized models, using first 2 regressors")
    reg_model_indices = [0, 1]

sub_surrogates += reg_model_indices

print(f'✓ Regressor indices: {reg_model_indices}', flush=True)

surrogate.inference_models = sub_surrogates
print(f'✓ Surrogate ready with models: {sub_surrogates}\n', flush=True)

# ============================================================================
# STEP 5: Train UnconditionalVAERepresentation Generator
# ============================================================================
print("=" * 80)
print("STEP 5: Training Inverse Design Generator")
print("=" * 80)

z_dim = latent_dim  # Use the VAE latent dimension
num_objectives = 2  # ciou_loss and average_precision
batch_size = 32
num_epochs = 20
num_opt_steps = 100

target_fitness = torch.tensor([1.2, 0.70]).float()
print(f"Target: ciou_loss={target_fitness[0]:.3f}, AP={target_fitness[1]:.3f}\n")

uncond_vae = UnconditionalVAERepresentation(
    z_dim=z_dim,
    num_objectives=num_objectives,
    target_fitness=target_fitness,
    latent2_dim=128,
    hidden_sizes=[512, 256],
    dropout=0.1
)

# Prepare training data: latent vectors + fitness values
z_latents = torch.from_numpy(reg_train_latents).float()
fitness = torch.from_numpy(np.stack([
    reg_train_df['ciou_loss'].values,
    reg_train_df['average_precision'].values
], axis=1)).float()

print(f"Training on {len(z_latents)} latent vectors (dim={z_dim})")
print(f"Pre-training generator for {num_epochs} epochs...\n")

history = uncond_vae.initial_train(
    z_arch_vectors=z_latents,
    fitness_values=fitness,
    num_epochs=num_epochs,
    batch_size=batch_size,
    lr=1e-4,
    beta=1.0,
    save_dir=os.path.join(repo_dir, testing_dir, 'generator_checkpoints')
)

print(f"✓ Generator trained!")
print(f"  Final loss: {history['total_loss'][-1]:.4f}")
print(f"  Recon: {history['recon_loss'][-1]:.4f}, KL: {history['kl_loss'][-1]:.4f}\n")

# ============================================================================
# STEP 6: Optimize Generator with Frozen Surrogate
# ============================================================================
print("=" * 80)
print("STEP 6: Optimizing Generator (Frozen Surrogate)")
print("=" * 80)
print(f"Optimization steps: {num_opt_steps}\n")

optimizer = torch.optim.Adam(uncond_vae.parameters, lr=1e-4)
surrogate.device = uncond_vae.device

losses = []
recon_losses = []
best_loss = float('inf')
best_z = None

for step in range(num_opt_steps):
    z_gen = uncond_vae.sample(c=None, batch_size=batch_size)
    f_pred = surrogate.predict(z_gen, genome_scaler=reg_genome_scaler)
    
    target_batch = target_fitness.unsqueeze(0).expand(batch_size, -1).to(uncond_vae.device)
    loss = torch.nn.functional.mse_loss(f_pred, target_batch)
    
    # Compute reconstruction loss to monitor reconstructability
    with torch.no_grad():
        z_recon, mu, logvar = uncond_vae.forward(z_gen)
        recon_loss = torch.nn.functional.mse_loss(z_recon, z_gen)
        recon_losses.append(recon_loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(uncond_vae.parameters, 1.0)
    optimizer.step()
    
    losses.append(loss.item())
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_z = z_gen.detach()
    
    if step % 10 == 0 or step == num_opt_steps - 1:
        mean_pred = f_pred.mean(dim=0).detach().cpu()
        std_pred = f_pred.std(dim=0).detach().cpu()
        print(f"Step {step:3d}: Loss={loss.item():.4f} | Recon={recon_loss.item():.4f} | "
              f"Pred=[{mean_pred[0]:.3f}±{std_pred[0]:.3f}, {mean_pred[1]:.3f}±{std_pred[1]:.3f}]")


print(f"\n✓ Optimization complete!")
print(f"  Fitness Loss: Initial={losses[0]:.4f} → Final={losses[-1]:.4f} → Best={best_loss:.4f}")
print(f"  Reconstruction Loss: Initial={recon_losses[0]:.4f} → Final={recon_losses[-1]:.4f}")
print(f"  Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%\n")

# Save optimized generator
final_ckpt = os.path.join(repo_dir, testing_dir, 'generator_checkpoints', 'optimized_generator.pt')
uncond_vae.save_checkpoint(final_ckpt)
print(f"✓ Saved to {final_ckpt}\n")

# ============================================================================
# STEP 5: Evaluate Final Generator
# ============================================================================
print("=" * 80)
print("STEP 5: Final Evaluation")
print("=" * 80)

final_samples = uncond_vae.sample(c=None, batch_size=100)
final_preds = surrogate.predict(final_samples, genome_scaler=reg_genome_scaler)

mean_fit = final_preds.mean(dim=0).cpu().detach().numpy()
std_fit = final_preds.std(dim=0).cpu().detach().numpy()

print(f"Target:   ciou={target_fitness[0]:.3f}, AP={target_fitness[1]:.3f}")
print(f"Achieved: ciou={mean_fit[0]:.3f}±{std_fit[0]:.3f}, AP={mean_fit[1]:.3f}±{std_fit[1]:.3f}")
print(f"Error:    ciou={abs(mean_fit[0]-target_fitness[0].item()):.3f}, AP={abs(mean_fit[1]-target_fitness[1].item()):.3f}")

# ============================================================================
# Decode Samples and Save to File
# ============================================================================
print("\n" + "=" * 80)
print("Decoding and Saving Generated Architectures")
print("=" * 80)

# Sample from the optimized generator
num_samples = 50
print(f"Generating {num_samples} architectures from optimized generator...")
with torch.no_grad():
    sampled_latents = uncond_vae.sample(c=None, batch_size=num_samples)
    
    # Decode latents back to raw genome space using VAE
    vae.eval()
    decoded_genomes = vae.decode(sampled_latents)
    
    # Inverse transform to original genome space
    decoded_genomes_np = decoded_genomes.cpu().numpy()
    decoded_genomes_unscaled = genome_scaler.inverse_transform(decoded_genomes_np)
    
    # Get predicted fitness for each sample
    fitness_preds = surrogate.predict(sampled_latents, genome_scaler=reg_genome_scaler)
    fitness_preds_np = fitness_preds.cpu().numpy()

# Create results dataframe
results_df = pd.DataFrame({
    'sample_id': range(num_samples),
    'predicted_ciou_loss': fitness_preds_np[:, 0],
    'predicted_average_precision': fitness_preds_np[:, 1],
    'genome': [decoded_genomes_unscaled[i].tolist() for i in range(num_samples)],
    'latent': [sampled_latents[i].cpu().numpy().tolist() for i in range(num_samples)]
})

# Save to CSV (including genome vector)
output_csv = os.path.join(repo_dir, testing_dir, 'generated_architectures.csv')
results_df[['sample_id', 'predicted_ciou_loss', 'predicted_average_precision', 'genome']].to_csv(output_csv, index=False)
print(f"✓ Saved fitness predictions and genomes to {output_csv}")

# Save full data (with genomes and latents) to pickle
output_pkl = os.path.join(repo_dir, testing_dir, 'generated_architectures.pkl')
results_df.to_pickle(output_pkl)
print(f"✓ Saved full data (genomes + latents) to {output_pkl}")

# Save a human-readable summary
output_txt = os.path.join(repo_dir, testing_dir, 'generated_architectures_summary.txt')
with open(output_txt, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("GENERATED ARCHITECTURES SUMMARY\n")
    f.write("=" * 80 + "\n\n")
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
print("✅ INVERSE DESIGN TEST COMPLETE")
print("=" * 80)
