import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from surrogates.surrogate_eval import prepare_data
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import math


# print("TRANSFORMER")
# # Load Data
# with open(f'/storage/ice-shared/vip-vvk/data/AOT/surrogate_dataset/pretrain_cls_train.pkl', 'rb') as f:
#     train_df = pickle.load(f)
# with open(f'/storage/ice-shared/vip-vvk/data/AOT/surrogate_dataset/surr_cls_val.pkl', 'rb') as f:
#     val_df = pickle.load(f)

# # train_data = np.stack(train_df['genome'].values)
# # new_train_data = np.zeros(train_data.shape[0], train_data.shape[1]+1)



# # DataLoader Preparation
# batch_size = 16
# train_loader, val_loader, _, _ = prepare_data({'metrics_subset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}, batch_size, train_df, val_df)



class NASTransformer(nn.Module):
    def __init__(
        self,
        input_dim=68,        # Total dimension of a component (includes categorical + continuous)
        global_metadata_dim=1, # Dimension of global metadata
        max_components=15,   # Maximum number of components (layers)
        embed_dim=256,       # Embedding dimension
        num_heads=4,         # Number of attention heads
        ff_dim=1024,         # Feed-forward dimension
        num_encoder_layers=3, # Number of encoder layers in each tower
        output_size=3,    # Number of fitness objectives to predict
        dropout=0.1
    ):
        super().__init__()
        
        # Simple projection for component features
        self.component_projection = nn.Linear(input_dim, embed_dim)
        
        # Global metadata projection
        self.global_projection = nn.Linear(global_metadata_dim, embed_dim)
        
        # Position embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, max_components + 1, embed_dim))
        
        # Component tower
        self.component_layers = nn.ModuleList([
            ComponentEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Pairwise tower
        self.pairwise_layers = nn.ModuleList([
            PairwiseEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Cross-tower communication
        self.comp_to_pair = ComponentToPair(embed_dim)
        self.pair_to_comp = PairToComponent(embed_dim)
        
        # Output prediction head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pooler = nn.Linear(embed_dim, embed_dim)
        self.predictor = nn.Linear(embed_dim, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_components = max_components
        
    def forward(self, component_features, global_metadata, component_mask=None):
        """
        Args:
            component_features: [batch_size, actual_components, input_dim] Combined categorical+continuous features
            global_metadata: [batch_size, global_metadata_dim]
            component_mask: [batch_size, actual_components] (1 for valid components, 0 for padding)
        """
        batch_size = component_features.shape[0]
        actual_components = component_features.shape[1]
        
        # Project component features
        comp_repr = self.component_projection(component_features)
        
        # Project global metadata
        global_repr = self.global_projection(global_metadata).unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Concatenate global token with component representations
        comp_repr = torch.cat([global_repr, comp_repr], dim=1)  # [batch_size, 1+actual_components, embed_dim]
        
        # Add position embeddings
        comp_repr = comp_repr + self.position_embedding[:, :actual_components+1, :]
        
        # Create masks for attention and valid components
        if component_mask is None:
            component_mask = torch.ones(batch_size, actual_components, dtype=torch.bool, 
                                        device=comp_repr.device)
        
        # Add mask for global token (always valid)
        global_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=component_mask.device)
        valid_mask = torch.cat([global_mask, component_mask], dim=1)
        
        # Create attention mask
        attn_mask = valid_mask.unsqueeze(1).unsqueeze(2)
        
        # Pad to max sequence length if needed
        if actual_components < self.max_components:
            padding_length = self.max_components - actual_components
            padding = torch.zeros(batch_size, padding_length, comp_repr.shape[2], 
                                  device=comp_repr.device)
            comp_repr = torch.cat([comp_repr, padding], dim=1)
            
            # Update mask for padding
            mask_padding = torch.zeros(batch_size, padding_length, 
                                       dtype=torch.bool, device=valid_mask.device)
            valid_mask = torch.cat([valid_mask, mask_padding], dim=1)
        
        # Initialize pairwise representations
        pair_repr = self.comp_to_pair(comp_repr)
        
        # Iterative refinement through layers
        for i in range(len(self.component_layers)):
            # Component tower
            comp_repr = self.component_layers[i](comp_repr, valid_mask)
            
            # Update pairwise representation from component representation
            if i > 0:  # Skip first update to allow initial processing
                pair_contrib = self.comp_to_pair(comp_repr)
                pair_repr = pair_repr + pair_contrib
            
            # Pairwise tower
            pair_repr = self.pairwise_layers[i](pair_repr)
            
            # Update component representation from pairwise representation
            comp_contrib = self.pair_to_comp(pair_repr)
            comp_repr = comp_repr + comp_contrib
        
        # Use global token for prediction
        pooled = comp_repr[:, 0]  # Use the global token representation
        pooled = self.layer_norm(pooled)
        pooled = torch.tanh(self.pooler(pooled))
        predictions = self.predictor(pooled)
        
        return predictions

class ComponentEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        # Self attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=~key_padding_mask if key_padding_mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class PairwiseEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.col_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _, embed_dim = x.shape
        
        # Row-wise attention (attend along rows)
        row_x = x.reshape(batch_size * seq_len, seq_len, embed_dim)
        row_out, _ = self.row_attn(row_x, row_x, row_x)
        row_out = row_out.reshape(batch_size, seq_len, seq_len, embed_dim)
        x = self.norm1(x + self.dropout(row_out))
        
        # Column-wise attention (attend along columns)
        col_x = x.transpose(1, 2).reshape(batch_size * seq_len, seq_len, embed_dim)
        col_out, _ = self.col_attn(col_x, col_x, col_x)
        col_out = col_out.reshape(batch_size, seq_len, seq_len, embed_dim).transpose(1, 2)
        x = self.norm2(x + self.dropout(col_out))
        
        # Feed forward
        ff_x = x.reshape(batch_size * seq_len * seq_len, embed_dim)
        ff_out = self.ff(ff_x).reshape(batch_size, seq_len, seq_len, embed_dim)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

class ComponentToPair(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(2 * embed_dim, embed_dim)
        
    def forward(self, component_repr):
        batch_size, seq_len, embed_dim = component_repr.shape
        
        # Create outer product of embeddings
        comp_i = component_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)
        comp_j = component_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Concatenate embeddings from position i and j
        pair_repr = torch.cat([comp_i, comp_j], dim=-1)
        
        # Project to pair embedding dimension
        pair_repr = self.proj(pair_repr)
        
        return pair_repr

class PairToComponent(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, pair_repr):
        # Average over all pairs involving each component
        comp_update = pair_repr.mean(dim=2)
        return self.proj(comp_update)



"""
def __init__(
        self,
        input_dim=68,         # Total dimension of a component (includes categorical + continuous)
        global_metadata_dim=1, # Dimension of global metadata
        max_components=15,    # Maximum number of components (layers)
        embed_dim=68,        # Embedding dimension
        num_heads=2,          # Number of attention heads
        ff_dim=512,          # Feed-forward dimension
        num_encoder_layers=4, # Number of encoder layers
        output_size=3,     # Number of fitness objectives to predict
        dropout=0.1,
        hidden_sizes=[2048, 1024, 512]
    ):
        super().__init__()
        
        # Feature projections
        self.component_projection = nn.Linear(input_dim, embed_dim)
        self.global_projection = nn.Linear(global_metadata_dim, embed_dim)
        
        # Position embeddings
        self.register_buffer(
            "position_embedding", 
            self._create_position_embedding(max_components + 1, embed_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm architecture for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Output prediction head
        out_layers = []
        in_dim = embed_dim
        self.layer_norm = nn.LayerNorm(in_dim)
        for hidden_dim in hidden_sizes:
            # linear -> norm -> activation -> dropout
            out_layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            out_layers.append(nn.ReLU())
            # out_layers.append(nn.Dropout(dropout))
            # output size of hidden layer is input size of next hidden layer
            in_dim = hidden_dim
        out_layers.append(nn.Linear(in_dim, output_size, bias=True))
        self.predictor = nn.Sequential(*out_layers)
        
        # self.predictor = nn.Linear(embed_dim*(max_components+1), 2048)
        # self.predictor2 = nn.Linear(2048, 1024)
        # self.predictor3 = nn.Linear(1024, 512)
        # self.predictor4 = nn.Linear(512, output_size)
        # self.activation = nn.ReLU()
        
        # Initialize weights
        self.predictor.apply(self._init_weights_mlp)
        self._init_weights()
        
        self.max_components = max_components
"""



class SimpleNASTransformer(nn.Module):
    def __init__(
        self,
        input_dim=68,         # Total dimension of a component (includes categorical + continuous)
        global_metadata_dim=1, # Dimension of global metadata
        max_components=15,    # Maximum number of components (layers)
        embed_dim=68,        # Embedding dimension
        num_heads=1,          # Number of attention heads
        ff_dim=256,          # Feed-forward dimension
        num_encoder_layers=9, # Number of encoder layers
        output_size=3,     # Number of fitness objectives to predict
        dropout=0.1
    ):
        super().__init__()
        
        # Feature projections
        self.component_projection = nn.Linear(input_dim, embed_dim)
        self.global_projection = nn.Linear(global_metadata_dim, embed_dim)
        
        # Position embeddings
        self.register_buffer(
            "position_embedding", 
            self._create_position_embedding(max_components + 1, embed_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm architecture for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Output prediction head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.predictor = nn.Linear(embed_dim, output_size)
        
        # Initialize weights
        self._init_weights()
        
        self.max_components = max_components
    
    def _create_position_embedding(self, seq_length, embed_dim):
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pos_embed = torch.zeros(1, seq_length, embed_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize projection layers with smaller weights
        # nn.init.xavier_uniform_(self.component_projection.weight, gain=0.1)
        nn.init.xavier_uniform_(self.global_projection.weight, gain=0.1)
        # nn.init.zeros_(self.component_projection.bias)
        nn.init.zeros_(self.global_projection.bias)
        
        # Initialize prediction head
        nn.init.xavier_uniform_(self.predictor.weight, gain=0.1)
        nn.init.zeros_(self.predictor.bias)
    
    def _init_weights_mlp(self, module):
        if isinstance(module, nn.Linear):
            # if relu, use he norm
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, component_features, global_metadata, component_mask=None):
        """
        Args:
            component_features: [batch_size, actual_components, input_dim] Combined categorical+continuous features
            global_metadata: [batch_size, global_metadata_dim]
            component_mask: [batch_size, actual_components] (1 for valid components, 0 for padding)
        """
        batch_size = component_features.shape[0]
        actual_components = component_features.shape[1]
        
        # Project component features
        # comp_repr = self.component_projection(component_features)
        comp_repr = component_features
        
        # Project global metadata
        global_repr = self.global_projection(global_metadata).unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Concatenate global token with component representations
        sequence = torch.cat([global_repr, comp_repr], dim=1)  # [batch_size, 1+actual_components, embed_dim]
        
        # Add position embeddings
        sequence = sequence + self.position_embedding[:, :sequence.size(1), :]
        
        # Create mask for transformer
        if component_mask is None:
            component_mask = torch.ones(batch_size, actual_components, dtype=torch.bool, 
                                        device=sequence.device)
            
        # Add mask for global token (always valid)
        global_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=component_mask.device)
        src_key_padding_mask = ~torch.cat([global_mask, component_mask], dim=1)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(sequence, src_key_padding_mask=src_key_padding_mask)
        
        # Use global token for prediction
        pooled = encoded[:, 0]  # Use the global token representation
        # pooled = encoded.reshape((encoded.shape[0], -1))
        pooled = self.layer_norm(pooled)
        predictions = self.predictor(pooled)
        
        return predictions









class EnhancedNASTransformer(nn.Module):
    def __init__(
        self,
        input_dim=68,           # Total dimension of a component
        global_metadata_dim=1,  # Dimension of global metadata
        max_components=15,      # Maximum number of components
        embed_dim=512,           # Embedding dimension (changed to power of 2)
        pair_dim=512,           # Dimension for pair representation
        num_heads=8,            # Number of attention heads
        ff_dim=256,             # Feed-forward dimension
        num_blocks=4,           # Number of Evoformer-like blocks
        output_size=3,          # Number of fitness objectives to predict
        dropout=0.1
    ):
        super().__init__()
        
        # Feature projections
        self.component_projection = nn.Linear(input_dim, embed_dim)
        self.global_projection = nn.Linear(global_metadata_dim, embed_dim)
        
        # Position embeddings
        self.register_buffer(
            "position_embedding", 
            self._create_position_embedding(max_components + 1, embed_dim)
        )
        
        # Initial pair representation through outer product
        self.pair_init = nn.Sequential(
            nn.Linear(2 * embed_dim, pair_dim),
            nn.GELU()
        )
        
        # Evoformer-like blocks
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(
                embed_dim=embed_dim, 
                pair_dim=pair_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        # Output prediction head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.predictor = nn.Linear(embed_dim, output_size)
        
        # Initialize weights
        self._init_weights()
        
        self.max_components = max_components
    
    def _create_position_embedding(self, seq_length, embed_dim):
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pos_embed = torch.zeros(1, seq_length, embed_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Initialize projection layers with smaller weights
        nn.init.xavier_uniform_(self.component_projection.weight, gain=0.1)
        nn.init.xavier_uniform_(self.global_projection.weight, gain=0.1)
        nn.init.zeros_(self.component_projection.bias)
        nn.init.zeros_(self.global_projection.bias)
        
        nn.init.xavier_uniform_(self.predictor.weight, gain=0.1)
        nn.init.zeros_(self.predictor.bias)
    
    def forward(self, component_features, global_metadata, component_mask=None):
        """
        Args:
            component_features: [batch_size, actual_components, input_dim] Combined categorical+continuous features
            global_metadata: [batch_size, global_metadata_dim]
            component_mask: [batch_size, actual_components] (1 for valid components, 0 for padding)
        """
        batch_size = component_features.shape[0]
        actual_components = component_features.shape[1]
        
        # Project component features
        comp_repr = self.component_projection(component_features)
        # comp_repr = component_features
        
        # Project global metadata
        global_repr = self.global_projection(global_metadata).unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Concatenate global token with component representations
        sequence = torch.cat([global_repr, comp_repr], dim=1)  # [batch_size, 1+actual_components, embed_dim]
        
        # Add position embeddings
        sequence = sequence + self.position_embedding[:, :sequence.size(1), :]
        
        # Create mask for transformer
        if component_mask is None:
            component_mask = torch.ones(batch_size, actual_components, dtype=torch.bool, 
                                        device=sequence.device)
            
        # Add mask for global token (always valid)
        global_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=component_mask.device)
        full_mask = torch.cat([global_mask, component_mask], dim=1)
        
        # Initialize pair representation from outer product
        pair_repr = self._init_pair_representation(sequence)
        
        # Process through Evoformer blocks
        for block in self.evoformer_blocks:
            sequence, pair_repr = block(sequence, pair_repr, mask=full_mask)
        
        # Use global token for prediction
        pooled = sequence[:, 0]  # Use the global token representation
        pooled = self.layer_norm(pooled)
        predictions = self.predictor(pooled)
        
        return predictions
    
    def _init_pair_representation(self, sequence):
        """Create initial pair representation using outer product-like operation."""
        batch_size, seq_len, embed_dim = sequence.shape
        
        # Create pairwise features by combining all pairs of components
        # [batch, seq_len, seq_len, 2*embed_dim]
        seq_i = sequence.unsqueeze(2).repeat(1, 1, seq_len, 1)
        seq_j = sequence.unsqueeze(1).repeat(1, seq_len, 1, 1)
        pair_features = torch.cat([seq_i, seq_j], dim=-1)
        
        # Project to pair dimension
        pair_repr = self.pair_init(pair_features)
        
        return pair_repr


class EvoformerBlock(nn.Module):
    """
    AlphaFold 2-inspired Evoformer block with bidirectional information flow
    between sequence (row) and pair representations.
    """
    def __init__(self, embed_dim, pair_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # --- MSA/Row representation modules ---
        # Row self-attention (component-to-component)
        self.row_attn = MSARowAttention(
            embed_dim=embed_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.row_attn_norm = nn.LayerNorm(embed_dim)
        
        # Row feed-forward
        self.row_ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # --- Pair representation modules ---
        # Triangle multiplication (higher-order interactions)
        self.triangle_mult_outgoing = TriangleMultiplication(
            pair_dim=pair_dim,
            orientation="outgoing",
            dropout=dropout
        )
        self.triangle_mult_incoming = TriangleMultiplication(
            pair_dim=pair_dim,
            orientation="incoming",
            dropout=dropout
        )
        
        # Pair self-attention
        self.pair_attn = PairwiseSelfAttention(
            pair_dim=pair_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pair_attn_norm = nn.LayerNorm(pair_dim)
        
        # Pair feed-forward
        self.pair_ff = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, pair_dim),
            nn.Dropout(dropout)
        )
        
        # Row to pair transition - communication from MSA to pair
        self.row_to_pair = MSAColumnAttention(
            embed_dim=embed_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.row_to_pair_norm = nn.LayerNorm(pair_dim)
    
    def forward(self, msa_repr, pair_repr, mask=None):
        """
        Args:
            msa_repr: [batch_size, seq_len, embed_dim] - Sequence/MSA representation
            pair_repr: [batch_size, seq_len, seq_len, pair_dim] - Pair representation
            mask: [batch_size, seq_len] - Sequence validity mask
        """
        # Create attention mask for transformer
        attn_mask = None
        if mask is not None:
            # For pair attention, we need to create a 2D mask from the 1D sequence mask
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [batch, seq_len, seq_len]
        else:
            pair_mask = None
        
        # --- Update sequence representation ---
        # MSA row attention with pair bias
        residual = msa_repr
        msa_repr = self.row_attn_norm(msa_repr)
        msa_repr = residual + self.row_attn(msa_repr, pair_repr, mask)
        
        # MSA feed-forward
        msa_repr = msa_repr + self.row_ff(msa_repr)
        
        # --- Update pair representation ---
        # Row -> pair communication (attention from columns to rows)
        residual = pair_repr
        pair_repr = self.row_to_pair_norm(pair_repr)
        pair_repr = residual + self.row_to_pair(msa_repr, pair_repr, mask)
        
        # Triangle multiplication - outgoing
        pair_repr = pair_repr + self.triangle_mult_outgoing(pair_repr, pair_mask)
        
        # Triangle multiplication - incoming
        pair_repr = pair_repr + self.triangle_mult_incoming(pair_repr, pair_mask)
        
        # Pair self-attention
        residual = pair_repr
        pair_repr = self.pair_attn_norm(pair_repr)
        pair_repr = residual + self.pair_attn(pair_repr, pair_mask)
        
        # Pair feed-forward
        pair_repr = pair_repr + self.pair_ff(pair_repr)
        
        return msa_repr, pair_repr


class MSARowAttention(nn.Module):
    """
    Row-wise attention that incorporates pair bias.
    Similar to AlphaFold 2's MSA row attention with pair bias.
    """
    def __init__(self, embed_dim, pair_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Pair bias projection
        self.pair_bias_proj = nn.Linear(pair_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, pair_repr, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            pair_repr: [batch_size, seq_len, seq_len, pair_dim]
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add pair bias to attention scores
        # Project pair representation to attention bias [batch, seq_len, seq_len, num_heads]
        pair_bias = self.pair_bias_proj(pair_repr).permute(0, 3, 1, 2)
        scores = scores + pair_bias
        
        # Apply mask
        if mask is not None:
            # Create attention mask
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Project output
        output = self.output_proj(context)
        
        return output


class MSAColumnAttention(nn.Module):
    """
    Implementation of MSA column attention that updates the pair representation
    based on sequence information. Inspired by AlphaFold 2's outer product mean.
    """
    def __init__(self, embed_dim, pair_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Projections for row representation
        self.row_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection to pair dimension
        self.output_proj = nn.Linear(embed_dim, pair_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, msa_repr, pair_repr, mask=None):
        """
        Args:
            msa_repr: [batch_size, seq_len, embed_dim] - Sequence representation
            pair_repr: [batch_size, seq_len, seq_len, pair_dim] - Pair representation
            mask: [batch_size, seq_len] - Sequence validity mask
        """
        batch_size, seq_len, _ = msa_repr.shape
        
        # Project row representation
        row_projected = self.row_proj(msa_repr)  # [batch, seq_len, embed_dim]
        
        # Create outer product-like representation
        row_i = row_projected.unsqueeze(2)  # [batch, seq_len, 1, embed_dim]
        row_j = row_projected.unsqueeze(1)  # [batch, 1, seq_len, embed_dim]
        
        # Combine features (similar to outer product)
        # Using element-wise multiplication for interaction modeling
        outer_product = row_i * row_j  # [batch, seq_len, seq_len, embed_dim]
        
        # Project to pair dimension
        pair_update = self.output_proj(outer_product)  # [batch, seq_len, seq_len, pair_dim]
        
        # Apply mask if provided
        if mask is not None:
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [batch, seq_len, seq_len]
            pair_mask = pair_mask.unsqueeze(-1)  # [batch, seq_len, seq_len, 1]
            pair_update = pair_update * pair_mask.float()
        
        return self.dropout(pair_update)


class TriangleMultiplication(nn.Module):
    """
    AlphaFold 2's triangle multiplication for capturing higher-order interactions
    in the pair representation.
    """
    def __init__(self, pair_dim, orientation="outgoing", dropout=0.1):
        super().__init__()
        
        self.orientation = orientation
        
        # Projections for triangle multiplication
        self.proj_left = nn.Linear(pair_dim, pair_dim)
        self.proj_right = nn.Linear(pair_dim, pair_dim)
        self.gate = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(pair_dim, pair_dim)
        
        self.layer_norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, pair_repr, mask=None):
        """
        Args:
            pair_repr: [batch_size, seq_len, seq_len, pair_dim]
            mask: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _, pair_dim = pair_repr.shape
        
        # Apply layer norm
        pair_repr = self.layer_norm(pair_repr)
        
        # Left and right projections
        left_proj = self.proj_left(pair_repr)
        right_proj = self.proj_right(pair_repr)
        
        # Compute triangle multiplication
        if self.orientation == "outgoing":
            # Multiply along outgoing edges: i->k->j
            # For each pair i,j, gather information from all k connected to i
            product = torch.einsum('bikc,bjkc->bijc', left_proj, right_proj)
        else:  # "incoming"
            # Multiply along incoming edges: i<-k<-j
            # For each pair i,j, gather information from all k connected to j
            product = torch.einsum('bkic,bkjc->bijc', left_proj, right_proj)
        
        # Apply mask
        if mask is not None:
            product = product * mask.unsqueeze(-1).float()
        
        # Apply gating and output projection
        gate_values = self.gate(pair_repr)
        output = gate_values * self.output_proj(product)
        
        return self.dropout(output)


class PairwiseSelfAttention(nn.Module):
    """
    Self-attention over the pair representation, capturing relationships
    between pairs of components.
    """
    def __init__(self, pair_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        assert self.head_dim * num_heads == pair_dim, "pair_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(pair_dim, pair_dim)
        self.k_proj = nn.Linear(pair_dim, pair_dim)
        self.v_proj = nn.Linear(pair_dim, pair_dim)
        self.output_proj = nn.Linear(pair_dim, pair_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, pair_repr, mask=None):
        """
        Args:
            pair_repr: [batch_size, seq_len, seq_len, pair_dim]
            mask: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _, _ = pair_repr.shape
        
        # Reshape pair representation to [batch_size, seq_len*seq_len, pair_dim]
        flat_pair = pair_repr.reshape(batch_size, seq_len*seq_len, self.pair_dim)
        
        # Project queries, keys, values
        q = self.q_proj(flat_pair).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(flat_pair).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(flat_pair).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Reshape mask to match attention matrix
            flat_mask = mask.reshape(batch_size, -1)
            attn_mask = flat_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len*seq_len]
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len*seq_len, self.pair_dim)
        
        # Project output and reshape back to [batch, seq_len, seq_len, pair_dim]
        output = self.output_proj(context)
        output = output.view(batch_size, seq_len, seq_len, self.pair_dim)
        
        return output









# # Architecture details
# batch_size = 16
# max_components = 15
# actual_components = 10  # Variable number of components per example
# input_dim = 68         # Combined dimension of all features per component
# global_metadata_dim = 1 # Dimension of global metadata
# output_size = 3     # Number of objectives to predict

# # Create model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = NASTransformer(
#     input_dim=input_dim,
#     global_metadata_dim=global_metadata_dim,
#     max_components=max_components,
#     embed_dim=256,
#     num_heads=4,
#     ff_dim=1024,
#     num_encoder_layers=3,
#     output_size=output_size
# ).to(device)

# # Example inputs (batch_size, actual_components, input_dim)
# # component_features = torch.randn(batch_size, actual_components, input_dim)
# # global_metadata = torch.randn(batch_size, global_metadata_dim)

# # Component mask (1 for valid components, 0 for padding)
# # component_mask = torch.ones(batch_size, actual_components, dtype=torch.bool)

# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # Forward pass
# # predictions = model(component_features, global_metadata, component_mask)
# criterion = nn.L1Loss()

# epochs = 40
# model.train()
# for epoch in range(epochs):
#     data_iter = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
#     total_loss = 0
#     total_recon_loss = 0
#     total_kl_div = 0
#     ctrt = 0
#     for vector, metrics in data_iter:
#         global_metadata = vector[:,0]
#         component_features = vector[:,1:].reshape((batch_size, 15, 68))
#         global_metadata = global_metadata.to(device)
#         optimizer.zero_grad()
#         component_features = component_features.to(device)
#         predictions = model(component_features, global_metadata)

#         loss = criterion(predictions, metrics)
        
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#         data_iter.set_postfix(loss=loss.item())
#         ctrt += 1

#     # Validation Loss Calculation
#     model.eval()
#     val_loss = 0
#     ctrv = 0
#     with torch.no_grad():
#         for vector, metrics in val_loader:
#             global_metadata = vector[:,0]
#             component_features = vector[:,1:].reshape((batch_size, 15, 68))
#             global_metadata = global_metadata.to(device)
#             component_features = component_features.to(device)
#             predictions = model(component_features, global_metadata)

#             loss = criterion(predictions, metrics)

#             val_loss += loss.item()
#             ctrv += 1
#     model.train()
    
#     print(f"Epoch {epoch+1}: Train Loss = {total_loss/ctrt:.6f}, Recon Loss = {total_recon_loss/ctrt:.6f}, KL Divergence = {total_kl_div/ctrt:.6f}")
#     print(f"Epoch {epoch+1}: Validation Loss = {val_loss/ctrv:.6f}, Val Recon Loss = {val_recon_loss/ctrv:.6f}, Val KL Divergence = {val_kl_div/ctrv:.6f}")
