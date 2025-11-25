# Create a new file: surrogates/hybrid_seqflow.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# A placeholder for a real autoregressive flow.
# For a robust implementation, use a library like 'nflows' or 'pyro'
# This is a simplified version of an affine coupling layer
class AutoregressiveFlow(nn.Module):
    def __init__(self, features, hidden_features, num_layers):
        super().__init__()
        # This LSTM reads the sequence autoregressively to condition the flow
        self.lstm = nn.LSTM(features, hidden_features, num_layers, batch_first=True)
        # These layers compute the scale (s) and translation (t)
        self.t_net = nn.Linear(hidden_features, features)
        self.s_net = nn.Linear(hidden_features, features)

    def forward(self, v):
        # v shape: [B, L, F]
        # Create shifted input for autoregression
        # v_shifted: [B, L, F], where v_shifted[:, i, :] depends on v[:, :i, :]
        v_shifted = F.pad(v[:, :-1, :], (0, 0, 1, 0), "constant", 0.)
        
        # lstm_out: [B, L, H]
        lstm_out, _ = self.lstm(v_shifted)

        t = self.t_net(lstm_out)
        s = torch.tanh(self.s_net(lstm_out)) # Stabilize scale
        
        z = v * torch.exp(s) + t
        log_det_g = s.sum(dim=[1, 2]) # Sum over L and F
        return z, log_det_g

    def inverse(self, z):
        # Inverse of an autoregressive flow is efficient
        v = torch.zeros_like(z)
        # We must generate token by token (L times)
        h = None
        for i in range(z.shape[1]): # Iterate over sequence length L
            # Get LSTM output for this step
            lstm_in = v[:, i, :].unsqueeze(1) # [B, 1, F]
            lstm_out, h = self.lstm(lstm_in, h) # [B, 1, H]
            
            t = self.t_net(lstm_out.squeeze(1)) # [B, F]
            s = torch.tanh(self.s_net(lstm_out.squeeze(1))) # [B, F]
            
            v[:, i, :] = (z[:, i, :] - t) / torch.exp(s)
        return v

class HybridSeqFlow(nn.Module):
    def __init__(self, vocab_size, embed_dim, value_dim, seq_len, flow_hidden_dim, flow_num_layers, sigma=0.1):
        super().__init__()
        self.L = seq_len
        self.F_embed = embed_dim
        self.F_value = value_dim
        self.F_total = embed_dim + value_dim
        self.sigma = sigma
        
        # --- Mapping h: (Token, Value) -> v ---
        # 1. For discrete tokens
        self.h_map_token = nn.Embedding(vocab_size, embed_dim)
        # L2-normalize embeddings as suggested [cite: 165]
        self.h_map_token.weight.data = F.normalize(self.h_map_token.weight.data, p=2, dim=1)
        
        # 2. For continuous values
        self.h_map_value = nn.Linear(1, value_dim)
        
        # --- Mapping g: v <-> z ---
        self.g_map = AutoregressiveFlow(features=self.F_total, 
                                        hidden_features=flow_hidden_dim, 
                                        num_layers=flow_num_layers)
        
        # --- Inverse Mapping h_inverse: v -> (Token, Value) ---
        self.h_inv_value = nn.Linear(value_dim, 1)

        # Prior distribution p(z)
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def h(self, tokens, values):
        """Encodes (tokens, values) into the continuous vector v."""
        v_emb = self.h_map_token(tokens) # [B, L, F_embed]
        v_val = self.h_map_value(values.unsqueeze(-1)) # [B, L, F_value]
        v = torch.cat([v_emb, v_val], dim=-1) # [B, L, F_total]
        return v

    def sample_v(self, tokens, values):
        """Implements q'(v|x) from Eq. 11, simplified."""
        v = self.h(tokens, values)
        # Sample from N(v, sigma^2 * I)
        v_samples = v + self.sigma * torch.randn_like(v)
        return v_samples

    def loss_sim(self, v_samples, tokens):
        """Implements the contrastive similarity loss L_sim from Eq. 13."""
        B, L, _ = v_samples.shape
        v_emb = v_samples[:, :, :self.F_embed] # Get embedding part of v
        e_x = self.h_map_token(tokens) # [B, L, F_embed]
        
        pos_sim = F.cosine_similarity(v_emb, e_x, dim=2).mean()
        
        # Sample random negative tokens
        neg_ids = torch.randint(0, self.h_map_token.num_embeddings, (B, L), device=v_samples.device)
        e_j = self.h_map_token(neg_ids) # [B, L, F_embed]
        
        neg_sim = F.cosine_similarity(v_emb, e_j, dim=2).mean()
        
        return -pos_sim + neg_sim

    def forward(self, tokens, values):
        """Encodes x -> z and computes loss for training SeqFlow."""
        
        # 1. Sample v from q'(v|x)
        v_samples = self.sample_v(tokens, values) # [B, L, F_total]
        
        # 2. Get z = g(v)
        z, log_det_g = self.g_map(v_samples) # [B, L, F_total], [B]
        
        # 3. Calculate L_NLL (Eq. 12)
        log_p_z = self.prior.log_prob(z).sum(dim=[1, 2]) # sum over L and F
        L_NLL = -(log_p_z + log_det_g).mean() # mean over B
        
        # 4. Calculate L_sim (Eq. 13) [cite: 187-191]
        L_sim = self.loss_sim(v_samples, tokens)
        
        return L_NLL, L_sim

    def encode(self, tokens, values):
        """Encodes (tokens, values) -> z for surrogate training."""
        v = self.h(tokens, values) # Use mean, not sample
        z, _ = self.g_map(v)
        return z

    def decode(self, z):
        """Decodes z -> (tokens, values) for generation."""
        # z shape: [B, L, F_total]
        
        # 1. v = g^{-1}(z)
        v = self.g_map.inverse(z) # [B, L, F_total]
        
        # 2. Split v -> v_emb, v_val
        v_emb = v[:, :, :self.F_embed]
        v_val = v[:, :, self.F_embed:]
        
        # 3. x_tokens = h_inv_token(v_emb)
        e_all = self.h_map_token.weight.data # [V, F_embed]
        logits = F.cosine_similarity(v_emb.unsqueeze(2), e_all.unsqueeze(0).unsqueeze(0), dim=3)
        tokens = torch.argmax(logits, dim=2) # [B, L]
        
        # 4. x_values = h_inv_value(v_val)
        values = self.h_inv_value(v_val).squeeze(-1) # [B, L]
        
        return tokens, values