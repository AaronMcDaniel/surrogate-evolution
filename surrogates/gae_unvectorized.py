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
from torch.utils.data import ConcatDataset
from grammar_utils import PRIM_SCHEMA

class MoEGrammarAE(nn.Module):
    def __init__(self, input_dim=68, latent_dim=16, num_experts=55, param_dim=14):
        # NOTE: num_experts changed to 55 (54 real + 1 padding)
        super(MoEGrammarAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.param_dim = param_dim
        
        # --- Encoder (Shared) ---
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
        
        # --- Decoder Trunk (Shared) ---
        self.dec_trunk = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # --- Head 1: Type Classifier ---
        self.type_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_experts)
        )
        
        # --- Head 2: Separate Parameter Experts (ModuleList) ---
        # 55 distinct FFNs. Index 54 is the "Padding Expert".
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, param_dim)
            ) for _ in range(num_experts)
        ])
        
        # --- Pre-compute Grammar Constraints ---
        self._init_bounds_buffers()

    def _init_bounds_buffers(self):
        # We need bounds for 55 experts. 
        # The first 54 come from schema. The 55th (Padding) we set to 0.
        bounds_tensor = torch.zeros(self.num_experts, self.param_dim)
        self.enum_patches = []
        
        for expert_idx, schema in PRIM_SCHEMA.items():
            for rule in schema:
                start = rule['start_idx']
                end = rule['end_idx']
                
                if rule['type'] == 'enum':
                    self.enum_patches.append((expert_idx, start, end))
                    bounds_tensor[expert_idx, start:end] = 1.0
                else:
                    bounds_tensor[expert_idx, start:end] = rule['max_val']
                    
        self.register_buffer('bounds_tensor', bounds_tensor)

    def encode(self, x):
        is_seq = x.dim() == 3
        if is_seq:
            B, L, F = x.shape
            x = x.reshape(B*L, F)
            
        z = self.enc_net(x)
        
        if is_seq:
            z = z.reshape(B, L, -1)
        return z

    def decode(self, z):
        """
        MoE Decode with Loop.
        Iterates through all 55 experts.
        """
        is_seq = z.dim() == 3
        if is_seq:
            B, L, D = z.shape
            z = z.reshape(B*L, D)
            
        # 1. Shared Features
        h = self.dec_trunk(z) # [N, 256]
        
        # 2. Type Prediction (55 classes)
        type_logits = self.type_head(h) # [N, 55]
        
        # 3. Expert Execution
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            raw_out = expert(h)
            
            # Apply specific bounds
            bounds = self.bounds_tensor[i]
            processed = torch.sigmoid(raw_out) * bounds
            
            # Apply Enum Softmax Patches
            for (e_idx, start, end) in self.enum_patches:
                if e_idx == i:
                    subset = raw_out[:, start:end]
                    processed[:, start:end] = F.softmax(subset, dim=1)
            
            expert_outputs.append(processed)
            
        # Stack: [N, 55, 14]
        stacked_params = torch.stack(expert_outputs, dim=1)
        
        if is_seq:
            type_logits = type_logits.view(B, L, -1)
            stacked_params = stacked_params.view(B, L, 55, 14)
            
        return type_logits, stacked_params

    def forward(self, x):
        z = self.encode(x)
        types, params = self.decode(z)
        return types, params, z

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def preprocess_encoding(batch_encoding):
    """
    Transforms raw 1021-length vector into 15x68 block format.
    DISCARDS Epoch number (Index 0).
    Input: [B, 1021]
    Output: [B, 15, 68]
    """
    # 1. Discard Epoch (Index 0) and take Genome part (1020)
    genome_flat = batch_encoding[:, 1:] 
    
    # 2. Reshape [B, 1020] -> [B, 15, 68]
    # No zero filtering. We keep zeros as Type 54.
    matrix = genome_flat.view(-1, 68, 15)
    blocks = matrix.transpose(1, 2) 
    
    return blocks

def flatten_encoding(blocks):
    """
    Reverses preprocess_encoding to 1020 vector.
    Does NOT prepend epoch number.
    Input: [B, 15, 68]
    Output: [B, 1020]
    """
    matrix = blocks.transpose(1, 2)
    genome_flat = matrix.contiguous().view(-1, 1020)
    return genome_flat

def grammar_loss_function(recon_types, recon_params, target_blocks, alpha_type=1.0, alpha_param=1.0, alpha_reg=0.1):
    """
    Calculates loss. Handles 55 classes.
    """
    # 1. Split Target [B, 15, 68]
    target_types_onehot = target_blocks[:, :, :54]
    target_params = target_blocks[:, :, 54:]
    
    # 2. Determine Ground Truth Indices (0-54)
    # Check for zero blocks (Padding -> Index 54)
    is_padding = torch.all(torch.abs(target_blocks) < 1e-6, dim=2) # [B, 15]
    
    target_type_indices = torch.argmax(target_types_onehot, dim=2)
    target_type_indices[is_padding] = 54
    
    # --- Loss A: Layer Type Classification ---
    loss_types = F.cross_entropy(
        recon_types.reshape(-1, 55), 
        target_type_indices.reshape(-1)
    )
    
    # --- Loss B: Parameter Regression ---
    # Gather output from the active expert
    gather_idx = target_type_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, 14)
    selected_recon_params = torch.gather(recon_params, 2, gather_idx).squeeze(2)
    
    # MSE Loss
    loss_params = F.mse_loss(selected_recon_params, target_params)
    
    # --- Loss C: Integer Regularization ---
    device = recon_params.device
    int_mask_template = torch.zeros(55, 14, device=device)
    
    for t_idx, schema in PRIM_SCHEMA.items():
        for rule in schema:
            if rule['type'] == 'int':
                int_mask_template[t_idx, rule['start_idx']:rule['end_idx']] = 1.0
                
    batch_int_mask = F.embedding(target_type_indices, int_mask_template)
    
    preds = selected_recon_params
    round_error = (preds - torch.round(preds).detach()) ** 2
    loss_reg_int = (round_error * batch_int_mask).mean()

    return (alpha_type * loss_types) + (alpha_param * loss_params) + (alpha_reg * loss_reg_int)


# -------------------------------------------------------------------------
# Helper: Logic to collapse MoE outputs into physical 68-dim vectors
# -------------------------------------------------------------------------
def collapse_to_physical(type_logits, param_stack):
    """
    Collapses 55-expert output to 68-dim physical vector.
    If Type 54 (Padding) is predicted, output ALL ZEROS.
    """
    # 1. Predict Type (0-54)
    pred_type_indices = torch.argmax(type_logits, dim=2) # [B, 15]
    
    # 2. Gather Params
    gather_idx = pred_type_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, 14)
    selected_params = torch.gather(param_stack, 2, gather_idx).squeeze(2) # [B, 15, 14]
    
    # 3. Create One-Hot (54 dim)
    # F.one_hot gives size 55. Slice to 54. 
    # If index=54, this slice becomes all zeros. Correct.
    one_hot_55 = F.one_hot(pred_type_indices, num_classes=55).float()
    one_hot_54 = one_hot_55[:, :, :54] # [B, 15, 54]
    
    # 4. Concatenate
    physical_blocks = torch.cat([one_hot_54, selected_params], dim=2) # [B, 15, 68]
    
    # 5. HARD ZERO MASKING for Padding Class
    # If pred index is 54, force everything to 0.
    is_padding = (pred_type_indices == 54).unsqueeze(2).float()
    physical_blocks = physical_blocks * (1.0 - is_padding)
    
    return physical_blocks


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def expand_dataframe_to_blocks(data_df, padding_ratio=0.1):
    """
    Expands a dataframe into blocks.
    DROPS epoch_num from the new dataframe.
    """
    valid_rows = []
    padding_rows = []
    
    for idx, row in data_df.iterrows():
        genome_1021 = row['genome']
        # Epoch at [0] is ignored
        genome_1020 = genome_1021[1:]
        
        # Reshape to (15, 68)
        genome_matrix = genome_1020.reshape(68, 15).T
        
        for layer_idx in range(15):
            layer_block = genome_matrix[layer_idx]
            
            new_row = row.copy()
            new_row['genome'] = layer_block
            # Drop epoch_num from block DF
            if 'epoch_num' in new_row:
                del new_row['epoch_num']
            new_row['layer_idx'] = layer_idx
            
            if np.allclose(layer_block, 0):
                padding_rows.append(new_row)
            else:
                valid_rows.append(new_row)
    
    # Balancing Logic
    num_valid = len(valid_rows)
    if padding_ratio > 0 and len(padding_rows) > 0:
        num_padding_needed = int(num_valid * padding_ratio / (1 - padding_ratio))
        num_padding_needed = min(len(padding_rows), num_padding_needed)
        import random
        selected_padding = random.sample(padding_rows, num_padding_needed)
    else:
        selected_padding = []
        
    print(f"Dataset Balancing Stats:")
    print(f"  Valid Blocks:   {num_valid}")
    print(f"  Padding Blocks: {len(selected_padding)}")
    print(f"  Final Ratio:    {len(selected_padding) / (num_valid + max(1, len(selected_padding))):.2%}")
    
    final_rows = valid_rows + selected_padding
    final_df = pd.DataFrame(final_rows).sample(frac=1).reset_index(drop=True)
    
    return final_df


def encode_block(ae, block_68, device=None):
    """Encodes a single 68-length block."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    with torch.no_grad():
        if isinstance(block_68, np.ndarray):
            block_68 = torch.from_numpy(block_68).float()
        
        if block_68.dim() == 1:
            block_68 = block_68.unsqueeze(0)  # (1, 68)
        
        block_68 = block_68.to(device)
        z = ae.encode(block_68) 
        return z.cpu()


def encode_full_1021(ae, encoding_1021, device=None):
    """
    Vectorized encoding of 1021-length vectors.
    Produces: [15*Latent] -> 240 dimensions.
    NO Epoch, NO Length.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    with torch.no_grad():
        if isinstance(encoding_1021, np.ndarray):
            encoding_1021 = torch.from_numpy(encoding_1021).float()
        
        is_single = encoding_1021.dim() == 1
        if is_single:
            encoding_1021 = encoding_1021.unsqueeze(0)
            
        encoding_1021 = encoding_1021.to(device)
        batch_size = encoding_1021.shape[0]
        
        # Discard Epoch
        genome_1020 = encoding_1021[:, 1:]
        blocks = genome_1020.view(batch_size, 68, 15).transpose(1, 2)
        
        # Encode EVERYTHING, including zeros (Padding Cluster)
        z_seq = ae.encode(blocks) # (B, 15, 16)
        
        # Flatten -> 240 dims
        z_flat = z_seq.reshape(batch_size, -1)
        
        # Return only latents
        if is_single:
            return z_flat.squeeze(0).cpu()
        return z_flat.cpu()


def generate_samples(ae, num_samples, num_layers=15, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, num_layers, ae.latent_dim).to(device)
        type_logits, param_stack = ae.decode(z)
        physical_blocks = collapse_to_physical(type_logits, param_stack)
    
    return physical_blocks.cpu().numpy()


def reconstruct_samples(ae, data, device=None, is_1021=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    with torch.no_grad():
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        data = data.to(device)
        
        if is_1021:
            # preprocess now returns just blocks [B, 15, 68]
            blocks = preprocess_encoding(data)
        else:
            blocks = data
            
        type_logits, param_stack, z = ae(blocks)
        physical_blocks = collapse_to_physical(type_logits, param_stack)
        
    return physical_blocks.cpu().numpy(), z.cpu().numpy()


def get_latent_representation(ae, data_df, device=None, use_full_encoding=True):
    """Extract latent representations."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    genomes = np.stack(data_df['genome'].values)
    
    BATCH_SIZE = 256
    latent_results = []
    
    with torch.no_grad():
        for i in range(0, len(genomes), BATCH_SIZE):
            batch = genomes[i : i + BATCH_SIZE]
            
            if genomes.shape[1] == 1021 and use_full_encoding:
                # encode_full_1021 now returns [B, 240]
                z_batch = encode_full_1021(ae, batch, device=device)
            elif genomes.shape[1] == 68:
                batch_tensor = torch.from_numpy(batch).float().to(device)
                z_batch = ae.encode(batch_tensor)
                if z_batch.dim() == 3: 
                    z_batch = z_batch.squeeze(1)
            else:
                raise ValueError("Unexpected dimension")
                
            latent_results.append(z_batch.cpu().numpy())
            
    all_latents = np.concatenate(latent_results, axis=0)
    
    data_df_copy = data_df.copy()
    data_df_copy['genome'] = list(all_latents)
    return data_df_copy


def train_ae(ae, train_loader, val_loader, epochs=200, lr=1e-3, device=None, alpha_type=1.0, alpha_param=1.0, alpha_reg=0.1):
    """Train the MoEVectorizedGrammarAE model with validation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae = ae.to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    
    ae.train()
    for epoch in range(epochs):
        data_iter = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        total_loss = 0
        ctrt = 0
        for raw_encoding_batch, _ in data_iter:
            raw_encoding_batch = raw_encoding_batch.to(device)
            
            # 1. Preprocess: returns just blocks [B, 15, 68]
            blocks = preprocess_encoding(raw_encoding_batch)
            
            optimizer.zero_grad()
            
            # 2. Forward
            recon_types, recon_params, z = ae(blocks)
            
            # 3. Loss
            loss = grammar_loss_function(recon_types, recon_params, blocks, 
                                        alpha_type=alpha_type, 
                                        alpha_param=alpha_param, 
                                        alpha_reg=alpha_reg)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            data_iter.set_postfix(loss=loss.item())
            ctrt += 1

        # Validation Loss
        ae.eval()
        val_loss = 0
        ctrv = 0
        with torch.no_grad():
            for raw_encoding_batch, _ in val_loader:
                raw_encoding_batch = raw_encoding_batch.to(device)
                
                blocks = preprocess_encoding(raw_encoding_batch)
                
                recon_types, recon_params, z = ae(blocks)
                
                loss = grammar_loss_function(recon_types, recon_params, blocks,
                                            alpha_type=alpha_type,
                                            alpha_param=alpha_param,
                                            alpha_reg=alpha_reg)
                val_loss += loss.item()
                ctrv += 1
        ae.train()
        
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/ctrt:.6f}")
        print(f"Epoch {epoch+1}: Validation Loss = {val_loss/ctrv:.6f}")
    
    return ae