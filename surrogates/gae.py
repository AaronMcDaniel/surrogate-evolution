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

class MoEVectorizedGrammarAE(nn.Module):
    def __init__(self, input_dim=68, latent_dim=16, num_experts=54, param_dim=14):
        super(MoEVectorizedGrammarAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.param_dim = param_dim
        
        # --- Encoder (Shared) ---
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
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
        
        # --- Head 2: Vectorized Parameter Experts ---
        self.all_experts = nn.Sequential(
            nn.Linear(256, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1), # Slight dropout for regularization
            nn.Linear(1024, num_experts * param_dim)
        )
        
        # --- Pre-compute Grammar Constraints ---
        self._init_bounds_buffers()

    def _init_bounds_buffers(self):
        bounds_tensor = torch.ones(self.num_experts, self.param_dim)
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
        Vectorized decode. 
        REMOVED: The manual zero-check. Now outputs 'Ghost Layers' for zero inputs.
        This allows gradients to flow if the optimizer wants to turn a zero-block into a layer.
        """
        is_seq = z.dim() == 3
        if is_seq:
            B, L, D = z.shape
            z = z.reshape(B*L, D)
            
        h = self.dec_trunk(z) # [N, 128]
        
        # 1. Type Prediction
        type_logits = self.type_head(h) # [N, 54]
        
        # 2. Expert Projection (Vectorized)
        raw_logits = self.all_experts(h)
        
        # Reshape to expert grid: [N, 54, 14]
        raw_logits = raw_logits.view(-1, self.num_experts, self.param_dim)
        
        # 3. Apply Global Bounds (Sigmoid Scaling)
        processed_params = torch.sigmoid(raw_logits) * self.bounds_tensor
        
        # 4. Apply Enum Patches (Softmax)
        for (idx, start, end) in self.enum_patches:
            subset_logits = raw_logits[:, idx, start:end]
            processed_params[:, idx, start:end] = F.softmax(subset_logits, dim=1)
        
        if is_seq:
            type_logits = type_logits.view(B, L, -1)
            processed_params = processed_params.view(B, L, 54, 14)
            
        return type_logits, processed_params

    def forward(self, x):
        z = self.encode(x)
        types, params = self.decode(z)
        return types, params, z

def weights_init(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization keeps variance consistent across layers
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def preprocess_encoding(batch_encoding):
    """
    Transforms the raw 1021-length vector into the 15x68 block format.
    
    Args:
        batch_encoding (torch.Tensor): Shape [B, 1021]
        
    Returns:
        epochs (torch.Tensor): Shape [B, 1]
        blocks (torch.Tensor): Shape [B, 15, 68]
    """
    # 1. Separate Epoch (Index 0)
    epochs = batch_encoding[:, 0:1]
    
    # 2. Get Genome part
    genome_flat = batch_encoding[:, 1:] # [B, 1020]
    
    # 3. Reshape
    # The Codec flattens column-major (Fortran style) or row-major?
    # Looking at Codec.encode_surrogate: 
    # encoded_genome = np.zeros((68, 15)) ... flattened_encoding = encoded_genome.flatten()
    # Numpy flatten is 'C' (row-major) by default, meaning it reads row 0, then row 1...
    # BUT wait, Codec code: `encoded_genome[0:len(optimizer_layer), 0] = ...`
    # It fills columns.
    # If flattened default, it goes index 0,0 -> 0,1 -> 0,2... 
    # That mixes features across layers immediately. 
    # Let's assume standard reshaping [B, 68, 15] then transpose to [B, 15, 68]
    
    # Reversing numpy default flatten on a (68, 15) matrix:
    # We need to ensure we reconstruct the (68, 15) matrix correctly.
    # Since numpy flatten is row-major, and the data was (68, 15),
    # the vector is [feat0_layer0, feat0_layer1... feat0_layer14, feat1_layer0...]
    # So we reshape to (68, 15) first.
    
    matrix = genome_flat.view(-1, 68, 15)
    
    # We want [Batch, 15 Layers, 68 Features] for sequential/block processing
    blocks = matrix.transpose(1, 2) # [B, 15, 68]
    
    return epochs, blocks

def flatten_encoding(epochs, blocks):
    """
    Reverses preprocess_encoding.
    """
    # blocks: [B, 15, 68] -> [B, 68, 15]
    matrix = blocks.transpose(1, 2)
    genome_flat = matrix.contiguous().view(-1, 1020)
    return torch.cat([epochs, genome_flat], dim=1)

def grammar_loss_function(recon_types, recon_params, target_blocks, alpha_type=1.0, alpha_param=1.0, alpha_reg=0.1):
    """
    Calculates loss ensuring specific bounds and integer constraints are met.
    """
    # 1. Split Target
    target_types_onehot = target_blocks[:, :, :54]
    target_params = target_blocks[:, :, 54:]
    
    # Get target type indices [B, 15]
    target_type_indices = torch.argmax(target_types_onehot, dim=2)
    
    # --- Loss A: Layer Type Classification ---
    loss_types = F.cross_entropy(
        recon_types.reshape(-1, 54), 
        target_type_indices.reshape(-1)
    )
    
    # --- Loss B: Parameter Regression (Active Expert Only) ---
    # Gather the outputs from the expert corresponding to the Ground Truth type
    gather_idx = target_type_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, 14)
    selected_recon_params = torch.gather(recon_params, 2, gather_idx).squeeze(2)
    
    # MSE Loss
    loss_params = F.mse_loss(selected_recon_params, target_params)
    
    # --- Loss C: Integer Rounding Regularization ---
    # We only penalize non-integers if the schema says it SHOULD be an int.
    # We construct a dynamic mask based on the batch's target types.
    
    device = recon_params.device
    
    # 1. Precompute integer mask for all 54 types (Shape: [54, 14])
    # This should ideally be cached in the model, but construction is fast enough.
    int_mask_template = torch.zeros(54, 14, device=device)
    
    for t_idx, schema in PRIM_SCHEMA.items():
        for rule in schema:
            if rule['type'] == 'int': # Defined in grammar_utils based on primitives types
                int_mask_template[t_idx, rule['start_idx']:rule['end_idx']] = 1.0
    
    # 2. Look up masks for the current batch [B, 15, 14]
    batch_int_mask = F.embedding(target_type_indices, int_mask_template)
    
    # 3. Calculate Distance to nearest integer
    # (x - round(x))^2
    preds = selected_recon_params
    # detach round() so we pull x towards the integer, not move the integer towards x
    round_error = (preds - torch.round(preds).detach()) ** 2
    
    # 4. Apply Mask
    loss_reg_int = (round_error * batch_int_mask).mean()

    return (alpha_type * loss_types) + (alpha_param * loss_params) + (alpha_reg * loss_reg_int)


# -------------------------------------------------------------------------
# Helper: Logic to collapse MoE outputs into physical 68-dim vectors
# -------------------------------------------------------------------------
def collapse_to_physical(type_logits, param_stack):
    """
    Converts raw MoE outputs into the physical [B, 15, 68] block format
    by selecting the parameters corresponding to the predicted type.
    
    Args:
        type_logits: [B, 15, 54]
        param_stack: [B, 15, 54, 14]
        
    Returns:
        physical_blocks: [B, 15, 68] (54-dim One-Hot + 14-dim Params)
    """
    # 1. Determine predicted type (Hard Argmax for inference)
    pred_type_indices = torch.argmax(type_logits, dim=2) # [B, 15]
    
    # 2. Create One-Hot encoding of types
    # F.one_hot returns Long, cast to Float
    one_hot_types = F.one_hot(pred_type_indices, num_classes=54).float() # [B, 15, 54]
    
    # 3. Gather specific expert parameters
    # We need to gather along dim=2 (the expert dimension 54)
    # pred_type_indices is [B, 15]. Expand to [B, 15, 1, 14] for gather
    gather_idx = pred_type_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, 14)
    
    # Gather: result is [B, 15, 1, 14] -> Squeeze to [B, 15, 14]
    selected_params = torch.gather(param_stack, 2, gather_idx).squeeze(2)
    
    # 4. Concatenate to form 68-dim blocks
    physical_blocks = torch.cat([one_hot_types, selected_params], dim=2) # [B, 15, 68]
    
    return physical_blocks


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def expand_dataframe_to_blocks(data_df):
    """
    Expands a dataframe with 1021-length encodings into rows of 68-length blocks.
    """
    expanded_rows = []
    
    for idx, row in data_df.iterrows():
        genome_1021 = row['genome']
        
        # Extract epoch number (first value)
        epoch_num = genome_1021[0]
        
        # Get the 1020-length genome part
        genome_1020 = genome_1021[1:]
        
        # Reshape: The Codec is essentially (68 features x 15 layers) flattened.
        # So we reshape to (68, 15) and transpose to iterate layers.
        genome_matrix = genome_1020.reshape(68, 15).T  # Shape: (15, 68)
        
        for layer_idx in range(15):
            layer_block = genome_matrix[layer_idx]  # Shape: (68,)
            
            # Skip if all zeros (padding layer)
            if np.allclose(layer_block, 0):
                continue
            
            # Create new row
            new_row = row.copy()
            new_row['genome'] = layer_block
            new_row['epoch_num'] = epoch_num
            new_row['layer_idx'] = layer_idx
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows).reset_index(drop=True)


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
        
        # Note: MoE AE expects [B, 15, 68] usually, or reshapes internally.
        # If passed [1, 68], encode logic `x.reshape(B*L, F)` handles it.
        z = ae.encode(block_68) 
        
        # z output is usually [B, 15, D] if input was 3D, or [N, D] if 2D
        # MoE AE implementation returns [B, L, D] if input dim is 3.
        # If input is [1, 68], AE treats it as 2D batch of 1. Output [1, 16]
        
        return z.cpu()


def encode_full_1021(ae, encoding_1021, device=None):
    """
    Vectorized encoding of 1021-length vectors.
    Produces: [Epoch, Length, 15*Latent]
    KEEPS: Manual zeroing of latents for clean storage/canonical representation.
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
        
        # 1. Preprocess
        epochs = encoding_1021[:, 0:1] # (B, 1)
        genome_1020 = encoding_1021[:, 1:]
        
        blocks = genome_1020.view(batch_size, 68, 15).transpose(1, 2)
        
        # 2. Encode
        z_seq = ae.encode(blocks) # (B, 15, 16)
        
        # 3. Handle Zero-Padding (Masking)
        # We manually zero out latents for clean storage.
        is_zero_block = torch.all(torch.abs(blocks) < 1e-6, dim=2) # (B, 15)
        valid_mask = (~is_zero_block).float().unsqueeze(2)
        
        z_seq = z_seq * valid_mask
        
        # Count valid blocks for the Length field
        valid_counts = valid_mask.sum(dim=1) # (B, 1)
        
        # 4. Flatten and Assemble
        z_flat = z_seq.reshape(batch_size, -1) # (B, 15*16)
        
        # [Epoch, Count, Latents]
        global_repr = torch.cat([epochs, valid_counts, z_flat], dim=1)
        
        if is_single:
            return global_repr.squeeze(0).cpu()
        return global_repr.cpu()


def generate_samples(ae, num_samples, num_layers=15, device=None):
    """
    Generates PHYSICAL samples (collapsed 68-dim blocks).
    
    Returns:
        physical_blocks: [num_samples, 15, 68] numpy array
                         Contains One-Hots and parameters collapsed from the MoE.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    with torch.no_grad():
        # Sample latent
        z = torch.randn(num_samples, num_layers, ae.latent_dim).to(device)
        
        # Decode -> Returns raw stack [B, 15, 54, 14]
        type_logits, param_stack = ae.decode(z)
        
        # Collapse to physical representation
        physical_blocks = collapse_to_physical(type_logits, param_stack)
    
    return physical_blocks.cpu().numpy()


def reconstruct_samples(ae, data, device=None, is_1021=True):
    """
    Reconstructs inputs into PHYSICAL samples.
    Apply Masking based on Length (Ghost Layer Strategy) if reconstructing from global latent.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    with torch.no_grad():
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        data = data.to(device)
        
        # Logic Fork: Are we reconstructing from [B, 1021] (Raw input) OR [B, 242] (Latent)?
        # The prompt implies 'is_1021' means Raw Input.
        # But if 'data' is the output of 'encode_full_1021' (Global Latent), it has size 242.
        
        # Let's assume standard behavior:
        # If is_1021=True, input is [B, 1021] -> Preprocess -> Encode -> Decode
        # If is_1021=False, input is [B, 15, 68] -> Encode -> Decode
        
        # NOTE: If you want to reconstruct from LATENT (e.g. from file), you need a different function
        # or logic here. I will assume this function behaves as a full Autoencoder pass (Input -> Output).
        
        if is_1021:
            epochs, blocks = preprocess_encoding(data)
            
            # Forward pass
            type_logits, param_stack, z = ae(blocks)
            
            # Collapse to physical
            physical_blocks = collapse_to_physical(type_logits, param_stack)
            
            # MASKING: Since we started with 1021 raw, we know the ground truth zeros.
            # But the AE might have hallucinated ghost layers for the zero-blocks.
            # We should mask them out to match input structure.
            
            # Calculate length from input blocks
            is_valid = ~torch.all(torch.abs(blocks) < 1e-6, dim=2) # [B, 15]
            mask = is_valid.float().unsqueeze(2) # [B, 15, 1]
            
            physical_blocks = physical_blocks * mask
            
        else:
            blocks = data
            type_logits, param_stack, z = ae(blocks)
            physical_blocks = collapse_to_physical(type_logits, param_stack)
            # If blocks had zeros, we mask them
            mask = (~torch.all(torch.abs(blocks) < 1e-6, dim=2)).float().unsqueeze(2)
            physical_blocks = physical_blocks * mask
    
    return physical_blocks.cpu().numpy(), z.cpu().numpy()


def get_latent_representation(ae, data_df, device=None, use_full_encoding=True):
    """Extract latent representations."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae.eval()
    genomes = np.stack(data_df['genome'].values)
    
    # Vectorized Batch Processing is much faster than looping
    # We process in chunks to avoid OOM if dataframe is huge
    BATCH_SIZE = 256
    latent_results = []
    
    with torch.no_grad():
        for i in range(0, len(genomes), BATCH_SIZE):
            batch = genomes[i : i + BATCH_SIZE]
            
            if genomes.shape[1] == 1021 and use_full_encoding:
                z_batch = encode_full_1021(ae, batch, device=device)
            elif genomes.shape[1] == 68:
                # Assuming batch is [B, 68] -> unsqueeze to [B, 1, 68] or let ae handle 2D
                # ae.encode handles 2D input [B, 68] -> returns [B, 16]
                batch_tensor = torch.from_numpy(batch).float().to(device)
                z_batch = ae.encode(batch_tensor)
                # If encode returned [B, 1, 16], squeeze
                if z_batch.dim() == 3: 
                    z_batch = z_batch.squeeze(1)
            else:
                raise ValueError("Unexpected dimension")
                
            latent_results.append(z_batch.cpu().numpy())
            
    all_latents = np.concatenate(latent_results, axis=0)
    
    data_df_copy = data_df.copy()
    # Convert numpy array rows to lists/arrays in the dataframe cell
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
            # raw_encoding_batch: [B, 1021]
            raw_encoding_batch = raw_encoding_batch.to(device)
            
            # 1. Preprocess
            epochs_batch, blocks = preprocess_encoding(raw_encoding_batch)
            # blocks: [B, 15, 68]
            
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

        # Validation Loss Calculation
        ae.eval()
        val_loss = 0
        ctrv = 0
        with torch.no_grad():
            for raw_encoding_batch, _ in val_loader:
                raw_encoding_batch = raw_encoding_batch.to(device)
                
                # Preprocess
                epochs_batch, blocks = preprocess_encoding(raw_encoding_batch)
                
                # Forward
                recon_types, recon_params, z = ae(blocks)
                
                # Loss
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