import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressiveFlow(nn.Module):
    def __init__(self, features, hidden_features, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden_features, num_layers, batch_first=True)
        self.t_net = nn.Linear(hidden_features, features)
        self.s_net = nn.Linear(hidden_features, features)

    def forward(self, v):
        # v_shifted: [B, L, F]
        v_shifted = F.pad(v[:, :-1, :], (0, 0, 1, 0), "constant", 0.)
        lstm_out, _ = self.lstm(v_shifted)

        t = self.t_net(lstm_out)
        s = torch.tanh(self.s_net(lstm_out)) 
        
        z = v * torch.exp(s) + t
        log_det_g = s.sum(dim=[1, 2]) 
        return z, log_det_g

    def inverse(self, z):
        v = torch.zeros_like(z)
        h = None
        for i in range(z.shape[1]): 
            lstm_in = v[:, i, :].unsqueeze(1) 
            lstm_out, h = self.lstm(lstm_in, h) 
            
            t = self.t_net(lstm_out.squeeze(1)) 
            s = torch.tanh(self.s_net(lstm_out.squeeze(1))) 
            
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
        self.val_token_idx = 3 # Assuming VAL_NUM is index 3
        
        # --- Mapping h: (Token, Value) -> v ---
        # 1. For discrete tokens
        self.h_map_token = nn.Embedding(vocab_size, embed_dim)
        self.h_map_token.weight.data = F.normalize(self.h_map_token.weight.data, p=2, dim=1)
        
        # 2. For continuous values
        self.h_map_value = nn.Linear(1, value_dim)
        
        # --- Mapping g: v <-> z ---
        self.g_map = AutoregressiveFlow(features=self.F_total, 
                                        hidden_features=flow_hidden_dim, 
                                        num_layers=flow_num_layers)
        
        # --- Inverse Mapping h_inverse: v -> (Token, Value) ---
        # This layer was previously untrained. Now it will be trained via L_val.
        self.h_inv_value = nn.Linear(value_dim, 1)

        # Prior distribution p(z)
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def h(self, tokens, values):
        """Encodes (tokens, values) into the continuous vector v."""
        v_emb = self.h_map_token(tokens) 
        v_val = self.h_map_value(values.unsqueeze(-1)) 
        v = torch.cat([v_emb, v_val], dim=-1) 
        return v

    def sample_v(self, tokens, values):
        v = self.h(tokens, values)
        # Sample from N(v, sigma^2 * I)
        v_samples = v + self.sigma * torch.randn_like(v)
        return v_samples

    def loss_sim(self, v_samples, tokens):
        """Contrastive similarity loss for Discrete Tokens."""
        B, L, _ = v_samples.shape
        v_emb = v_samples[:, :, :self.F_embed] 
        e_x = self.h_map_token(tokens) 
        
        # Positive similarity (should match ground truth token)
        pos_sim = F.cosine_similarity(v_emb, e_x, dim=2).mean()
        
        # Negative similarity (should not match random tokens)
        neg_ids = torch.randint(0, self.h_map_token.num_embeddings, (B, L), device=v_samples.device)
        e_j = self.h_map_token(neg_ids) 
        
        neg_sim = F.cosine_similarity(v_emb, e_j, dim=2).mean()
        
        return -pos_sim + neg_sim

    def loss_val(self, v_samples, values, tokens):
        """
        Reconstruction loss for Continuous Values.
        Only computed where tokens == VAL_NUM (index 3).
        """
        # Extract the value component of the vector v
        v_val = v_samples[:, :, self.F_embed:] 
        
        # Attempt to reconstruct the value using the inverse mapping
        pred_values = self.h_inv_value(v_val).squeeze(-1) # [B, L]
        
        # Create mask: 1 where token is VAL_NUM, 0 otherwise
        mask = (tokens == self.val_token_idx)
        
        if mask.sum() > 0:
            # MSE between predicted value and ground truth, only for relevant tokens
            return F.mse_loss(pred_values[mask], values[mask])
        else:
            return torch.tensor(0.0, device=values.device)

    def forward(self, tokens, values):
        """Encodes x -> z and computes loss for training SeqFlow."""
        
        # 1. Sample v from q'(v|x)
        v_samples = self.sample_v(tokens, values) 
        
        # 2. Get z = g(v)
        z, log_det_g = self.g_map(v_samples) 
        
        # 3. Calculate L_NLL (Flow Loss)
        log_p_z = self.prior.log_prob(z).sum(dim=[1, 2]) 
        L_NLL = -(log_p_z + log_det_g).mean() 
        
        # 4. Calculate L_sim (Token Reconstruction Loss)
        L_sim = self.loss_sim(v_samples, tokens)
        
        # 5. Calculate L_val (Value Reconstruction Loss) - NEW
        L_val = self.loss_val(v_samples, values, tokens)
        
        return L_NLL, L_sim, L_val

    def encode(self, tokens, values):
        """Encodes (tokens, values) -> z for surrogate training."""
        v = self.h(tokens, values) 
        z, _ = self.g_map(v)
        return z

    def decode(self, z):
        """Decodes z -> (tokens, values) for generation."""
        # 1. v = g^{-1}(z)
        v = self.g_map.inverse(z) 
        
        # 2. Split v -> v_emb, v_val
        v_emb = v[:, :, :self.F_embed]
        v_val = v[:, :, self.F_embed:]
        
        # 3. x_tokens = h_inv_token(v_emb)
        e_all = self.h_map_token.weight.data 
        logits = F.cosine_similarity(v_emb.unsqueeze(2), e_all.unsqueeze(0).unsqueeze(0), dim=3)
        tokens = torch.argmax(logits, dim=2) 
        
        # 4. x_values = h_inv_value(v_val)
        values = self.h_inv_value(v_val).squeeze(-1) 
        
        return tokens, values