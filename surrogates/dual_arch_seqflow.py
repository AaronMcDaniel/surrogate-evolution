import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# Component 1: The Core Flow Logic
# ------------------------------------------------------------------------------
class AutoregressiveFlow(nn.Module):
    """
    Base Autoregressive Flow (Affine Coupling).
    Modified to return LSTM hidden states for conditioning downstream models.
    """
    def __init__(self, features, hidden_features, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden_features, num_layers, batch_first=True)
        self.t_net = nn.Linear(hidden_features, features)
        self.s_net = nn.Linear(hidden_features, features)

    def forward(self, v):
        # v shape: [B, L, F]
        # Shift input: input at time t is v_{t-1}
        v_shifted = F.pad(v[:, :-1, :], (0, 0, 1, 0), "constant", 0.)
        
        lstm_out, _ = self.lstm(v_shifted)

        t = self.t_net(lstm_out)
        s = torch.tanh(self.s_net(lstm_out)) 
        
        z = v * torch.exp(s) + t
        log_det_g = s.sum(dim=[1, 2]) 
        
        # RETURN LSTM_OUT: This is the "Structure Context" for the 2nd flow
        return z, log_det_g, lstm_out 

    def inverse(self, z):
        v = torch.zeros_like(z)
        h = None
        lstm_outs = []
        
        for i in range(z.shape[1]): 
            # Input to LSTM is the PREVIOUS generated value v_{i-1}
            if i == 0:
                lstm_in = torch.zeros(z.shape[0], 1, z.shape[2], device=z.device)
            else:
                lstm_in = v[:, i-1, :].unsqueeze(1)
                
            lstm_out, h = self.lstm(lstm_in, h) 
            lstm_outs.append(lstm_out)
            
            t = self.t_net(lstm_out.squeeze(1)) 
            s = torch.tanh(self.s_net(lstm_out.squeeze(1))) 
            
            v[:, i, :] = (z[:, i, :] - t) / torch.exp(s)
            
        # Return hidden states to condition the value flow during decoding
        return v, torch.cat(lstm_outs, dim=1)


# ------------------------------------------------------------------------------
# Component 2: Discrete Structure Model
# ------------------------------------------------------------------------------
class DiscreteSeqFlow(nn.Module):
    """
    Sole responsibility: P(Structure).
    """
    def __init__(self, vocab_size, embed_dim, seq_len, flow_hidden_dim, flow_num_layers, sigma=0.1):
        super().__init__()
        self.L = seq_len
        self.F_embed = embed_dim
        self.sigma = sigma
        
        # 1. Discrete Embedding
        self.h_map_token = nn.Embedding(vocab_size, embed_dim)
        self.h_map_token.weight.data = F.normalize(self.h_map_token.weight.data, p=2, dim=1)
        
        # 2. Autoregressive Flow (g_map)
        self.g_map = AutoregressiveFlow(features=embed_dim, 
                                        hidden_features=flow_hidden_dim, 
                                        num_layers=flow_num_layers)
        
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def h(self, tokens):
        """Encodes tokens into embedding space."""
        v_emb = self.h_map_token(tokens) # [B, L, F_embed]
        return v_emb

    def sample_v(self, tokens):
        """Dequantization: Adds noise for flow compatibility."""
        v = self.h(tokens)
        v_samples = v + self.sigma * torch.randn_like(v)
        return v_samples

    def loss_sim(self, v_samples, tokens):
        """Contrastive anchor loss."""
        B, L, _ = v_samples.shape
        v_emb = v_samples 
        e_x = self.h_map_token(tokens)
        
        pos_sim = F.cosine_similarity(v_emb, e_x, dim=2).mean()
        
        neg_ids = torch.randint(0, self.h_map_token.num_embeddings, (B, L), device=v_samples.device)
        e_j = self.h_map_token(neg_ids) 
        
        neg_sim = F.cosine_similarity(v_emb, e_j, dim=2).mean()
        
        return -pos_sim + neg_sim

    def forward(self, tokens):
        """Returns NLL, Sim Loss, AND Structure Context."""
        v_samples = self.sample_v(tokens)
        
        # g_map now returns lstm_out (Context)
        z, log_det_g, context = self.g_map(v_samples) 
        
        log_p_z = self.prior.log_prob(z).sum(dim=[1, 2])
        L_NLL = -(log_p_z + log_det_g).mean()
        
        L_sim = self.loss_sim(v_samples, tokens)
        
        return L_NLL, L_sim, context

    def encode(self, tokens):
        v = self.h(tokens)
        z, _, context = self.g_map(v) # Use mean embedding (no noise) for deterministic encoding
        return z, context

    def decode(self, z):
        # v: [B, L, F_embed], context: [B, L, H_flow]
        v, context = self.g_map.inverse(z) 
        
        e_all = self.h_map_token.weight.data
        logits = F.cosine_similarity(v.unsqueeze(2), e_all.unsqueeze(0).unsqueeze(0), dim=3)
        tokens = torch.argmax(logits, dim=2)
        
        return tokens, context


# ------------------------------------------------------------------------------
# Component 3: Continuous Value Model (Approach 2)
# ------------------------------------------------------------------------------
class ConditionalValueFlow(nn.Module):
    """
    Conditioned on the Context (Hidden States) from DiscreteSeqFlow.
    """
    def __init__(self, value_dim, context_dim, hidden_dim, num_layers):
        super().__init__()
        self.value_dim = value_dim
        self.context_dim = context_dim
        
        # LSTM Input: Previous Value + Current Structure Context
        self.lstm = nn.LSTM(value_dim + context_dim, hidden_dim, num_layers, batch_first=True)
        
        # Outputs parameters for Affine Transform
        self.t_net = nn.Linear(hidden_dim, value_dim)
        self.s_net = nn.Linear(hidden_dim, value_dim)
        
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, values, context):
        """
        values: [B, L, 1]
        context: [B, L, H_struct] (Hidden states from Discrete Flow)
        """
        # Shift values: input at t is v_{t-1}
        v_shifted = F.pad(values[:, :-1, :], (0, 0, 1, 0), "constant", 0.)
        
        # Condition on structure context
        rnn_input = torch.cat([v_shifted, context], dim=-1)
        
        lstm_out, _ = self.lstm(rnn_input)
        
        t = self.t_net(lstm_out)
        s = torch.tanh(self.s_net(lstm_out))
        
        z = values * torch.exp(s) + t
        log_det = s.sum(dim=[1, 2])
        
        log_p_z = self.prior.log_prob(z).sum(dim=[1, 2])
        L_NLL = -(log_p_z + log_det).mean()
        
        return L_NLL, z

    def encode(self, values, context):
        # Re-run forward logic to get Z
        _, z = self.forward(values, context)
        return z

    def decode(self, z, context):
        """
        Inverse flow. Conditioned on the structure context.
        """
        v = torch.zeros_like(z)
        h = None
        
        for i in range(z.shape[1]):
            # 1. Get inputs
            # Previous generated value
            if i == 0:
                v_prev = torch.zeros(z.shape[0], 1, self.value_dim, device=z.device)
            else:
                v_prev = v[:, i-1, :].unsqueeze(1)
            
            # Context for current step
            ctx_curr = context[:, i, :].unsqueeze(1)
            
            # 2. Run LSTM
            rnn_in = torch.cat([v_prev, ctx_curr], dim=-1)
            lstm_out, h = self.lstm(rnn_in, h)
            
            # 3. Invert Affine
            t = self.t_net(lstm_out.squeeze(1))
            s = torch.tanh(self.s_net(lstm_out.squeeze(1)))
            
            v[:, i, :] = (z[:, i, :] - t) / torch.exp(s)
            
        return v

# ------------------------------------------------------------------------------
# Component 4: Hybrid Wrapper
# ------------------------------------------------------------------------------
class HybridSeqFlow(nn.Module):
    def __init__(self, vocab_size, embed_dim, struct_hidden_dim, val_hidden_dim, 
                 struct_layers=2, val_layers=4, max_seq_len=350, sigma=0.1):
        super().__init__()
        
        self.struct_flow = DiscreteSeqFlow(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seq_len=max_seq_len,
            flow_hidden_dim=struct_hidden_dim,
            flow_num_layers=struct_layers,
            sigma=sigma
        )
        
        self.val_flow = ConditionalValueFlow(
            value_dim=1, # Scalar values
            context_dim=struct_hidden_dim, # Must match AutoregressiveFlow hidden_dim
            hidden_dim=val_hidden_dim,
            num_layers=val_layers
        )
        
        self.val_token_id = 3 # Default ID for VAL_NUM

    def forward(self, tokens, values):
        """
        Pass-through that runs both. 
        NOTE: In the 2-stage training script, you will likely call 
        self.struct_flow() and self.val_flow() directly.
        """
        loss_nll, loss_sim, context = self.struct_flow(tokens)
        
        # We detach context if we don't want gradients flowing back to structure model 
        # from value model (usually preferred in 2-stage training)
        loss_val, _ = self.val_flow(values.unsqueeze(-1), context)
        
        return loss_nll, loss_sim, loss_val
    
    def encode_to_z(self, tokens, values):
        z_struct, context = self.struct_flow.encode(tokens)
        z_val = self.val_flow.encode(values.unsqueeze(-1), context)
        return z_struct, z_val

    def decode_from_z(self, z_struct, z_val):
        tokens, context = self.struct_flow.decode(z_struct)
        values = self.val_flow.decode(z_val, context)
        return tokens, values.squeeze(-1)