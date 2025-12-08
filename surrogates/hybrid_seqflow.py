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
        # Standard inverse (used for hard decode/validation)
        # Note: This does NOT do manifold projection, so it relies on z being perfect.
        batch_size, seq_len, features = z.shape
        inp = torch.zeros(batch_size, 1, features, device=z.device)
        h = None
        v_list = []
        for i in range(seq_len):
            lstm_out, h = self.lstm(inp, h)
            t = self.t_net(lstm_out.squeeze(1))
            s = torch.tanh(self.s_net(lstm_out.squeeze(1)))
            
            z_i = z[:, i, :]
            v_i = (z_i - t) / torch.exp(s)
            v_list.append(v_i)
            inp = v_i.unsqueeze(1) # Feeds raw v_i back
        return torch.stack(v_list, dim=1)

class DiscreteSeqFlow(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, flow_hidden_dim, flow_num_layers, sigma=0.1):
        super().__init__()
        self.L = seq_len
        self.F_embed = embed_dim
        self.sigma = sigma
        
        # --- Mapping h: Token -> v ---
        # For discrete tokens only
        self.h_map_token = nn.Embedding(vocab_size, embed_dim)
        # L2-normalize embeddings as suggested
        self.h_map_token.weight.data = F.normalize(self.h_map_token.weight.data, p=2, dim=1)
        
        # --- Mapping g: v <-> z ---
        self.g_map = AutoregressiveFlow(features=embed_dim, 
                                        hidden_features=flow_hidden_dim, 
                                        num_layers=flow_num_layers)

        # Prior distribution p(z)
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def h(self, tokens):
        """Encodes tokens into the continuous vector v."""
        v_emb = self.h_map_token(tokens) # [B, L, F_embed]
        return v_emb

    def sample_v(self, tokens):
        """Implements q'(v|x) from Eq. 11, simplified."""
        v = self.h(tokens)
        # Sample from N(v, sigma^2 * I)
        v_samples = v + self.sigma * torch.randn_like(v)
        return v_samples

    def loss_sim(self, v_samples, tokens):
        """Implements the contrastive similarity loss L_sim from Eq. 13."""
        B, L, _ = v_samples.shape
        v_emb = v_samples # All is embedding now
        e_x = self.h_map_token(tokens) # [B, L, F_embed]
        
        pos_sim = F.cosine_similarity(v_emb, e_x, dim=2).mean()
        
        # Sample random negative tokens
        neg_ids = torch.randint(0, self.h_map_token.num_embeddings, (B, L), device=v_samples.device)
        e_j = self.h_map_token(neg_ids) # [B, L, F_embed]
        
        neg_sim = F.cosine_similarity(v_emb, e_j, dim=2).mean()
        
        return -pos_sim + neg_sim

    def forward(self, tokens):
        """Encodes x -> z and computes loss for training SeqFlow."""
        
        # 1. Sample v from q'(v|x)
        v_samples = self.sample_v(tokens) # [B, L, F_embed]
        
        # 2. Get z = g(v)
        z, log_det_g = self.g_map(v_samples) # [B, L, F_embed], [B]
        
        # 3. Calculate L_NLL (Eq. 12)
        log_p_z = self.prior.log_prob(z).sum(dim=[1, 2]) # sum over L and F
        L_NLL = -(log_p_z + log_det_g).mean() # mean over B
        
        # 4. Calculate L_sim (Eq. 13)
        L_sim = self.loss_sim(v_samples, tokens)
        
        return L_NLL, L_sim

    def encode(self, tokens):
        """Encodes tokens -> z for surrogate training."""
        v = self.h(tokens) # Use mean, not sample
        z, _ = self.g_map(v)
        return z

    def decode(self, z):
        """
        Hard decode for final sampling.
        Memory Optimized: Uses MatMul instead of broadcasting CosineSimilarity.
        """
        # 1. Invert Flow to get continuous embeddings
        v = self.g_map.inverse(z) # [B, L, F]
        
        # 2. Normalize vectors (L2 norm)
        # Cosine Similarity(A, B) == DotProduct(Norm(A), Norm(B))
        v_norm = F.normalize(v, p=2, dim=2)
        e_all = self.h_map_token.weight.data
        e_norm = F.normalize(e_all, p=2, dim=1) # [V, F]
        
        # 3. Compute Logits via Matrix Multiplication
        # [B, L, F] @ [F, V] -> [B, L, V]
        # This avoids creating the massive 4D intermediate tensor
        logits = torch.matmul(v_norm, e_norm.T)
        
        # 4. Select best token
        tokens = torch.argmax(logits, dim=2) # [B, L]
        return tokens
    
    def decode_soft(self, z, temperature=1.0, hard=False):
        """
        Differentiable decoding WITH Manifold Projection.
        
        This unrolls the LSTM manually. At each step, it:
        1. Predicts the noisy vector v_raw from z.
        2. Calculates the Softmax distribution over the vocab.
        3. Projects v_raw onto the valid embedding manifold (Weighted Sum).
        4. Feeds the PROJECTED vector into the next LSTM step.
        """
        batch_size, seq_len, features = z.shape
        device = z.device
        
        # 1. Initialize Recurrence
        # Start with zero vector (which corresponds to padding/start context in the Flow's logic)
        # Note: If you forced z[:,0] to be SOS, the first output v will be SOS.
        inp = torch.zeros(batch_size, 1, features, device=device)
        
        h = None
        soft_tokens_list = []
        
        # Pre-fetch vocab weights for projection
        # [Vocab, F]
        w_emb = self.h_map_token.weight 
        # Normalized for Cosine Sim logic: [Vocab, F]
        w_norm = F.normalize(w_emb, p=2, dim=1) 

        # 2. Unrolled Loop
        for i in range(seq_len):
            # A. LSTM Step
            # inp is the PROJECTED embedding from previous step
            lstm_out, h = self.g_map.lstm(inp, h) # [B, 1, H]
            
            # B. Flow Transform (Inverse)
            t = self.g_map.t_net(lstm_out.squeeze(1)) # [B, F]
            s = torch.tanh(self.g_map.s_net(lstm_out.squeeze(1))) # [B, F]
            
            z_i = z[:, i, :]
            v_raw = (z_i - t) / torch.exp(s) # [B, F] -> Raw noisy vector
            
            # C. Softmax / Gumbel
            # Calculate similarity to valid tokens
            v_norm = F.normalize(v_raw, p=2, dim=1)
            logits = torch.matmul(v_norm, w_norm.T) # [B, V]
            
            # Sharpen logits before Gumbel to encourage discrete decisions
            scaled_logits = logits * 10.0
            probs = F.gumbel_softmax(scaled_logits, tau=temperature, hard=hard, dim=-1) # [B, V]
            soft_tokens_list.append(probs)
            
            # D. MANIFOLD PROJECTION (The Critical Fix)
            # Instead of feeding 'v_raw' (which might be garbage) into the next step,
            # we feed the "Cleaned" embedding based on the model's own confidence.
            # v_projected = Sum(Prob * Embedding)
            v_projected = torch.matmul(probs, w_emb) # [B, F]
            
            # Update input for next step
            inp = v_projected.unsqueeze(1) # [B, 1, F]

        # Stack outputs
        soft_tokens = torch.stack(soft_tokens_list, dim=1) # [B, L, V]
        return soft_tokens