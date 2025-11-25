import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# Model 1: Discrete Structure Flow
# Responsible for: Generating the sequence of tokens (Layers, Enums, Markers)
# ------------------------------------------------------------------------------
class DiscreteStructureFlow(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # L2-normalize embeddings as per NFBO paper recommendation
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):
        """
        Forward pass for training (Teacher Forcing).
        Input: tokens [B, L]
        Output: logits [B, L, V], hidden_states [B, L, H]
        """
        # Shift tokens for autoregression: input at t is token_{t-1}
        # Pad start with <SOS> or 0. Assuming codec handles SOS, we usually just 
        # need to ensure the input to LSTM is the sequence up to t-1 to predict t.
        # Here we assume input 'tokens' includes SOS.
        
        embeddings = self.embedding(tokens) # [B, L, E]
        
        # Run LSTM
        # hidden_states contains the context for each step
        hidden_states, _ = self.lstm(embeddings) # [B, L, H]
        
        # Predict logits for next token
        logits = self.head(hidden_states) # [B, L, V]
        
        return logits, hidden_states

    def sample(self, max_len, sos_token, eos_token, device):
        """
        Autoregressive generation of structure.
        Returns: generated_tokens [B, L], hidden_states [B, L, H]
        """
        batch_size = 1 # Optimization usually happens one at a time or modify for B
        current_token = torch.tensor([[sos_token]], device=device)
        hidden = None
        
        generated_tokens = [current_token]
        hidden_states_list = []
        
        for _ in range(max_len):
            emb = self.embedding(current_token)
            out, hidden = self.lstm(emb, hidden)
            
            logits = self.head(out)
            # Greedy decoding or sampling could be used here
            next_token = torch.argmax(logits, dim=-1)
            
            generated_tokens.append(next_token)
            hidden_states_list.append(out)
            
            if next_token.item() == eos_token:
                break
            current_token = next_token

        return torch.cat(generated_tokens, dim=1), torch.cat(hidden_states_list, dim=1)


# ------------------------------------------------------------------------------
# Model 2: Conditional Value Flow
# Responsible for: Generating continuous values given Structure + Previous Values
# ------------------------------------------------------------------------------
class ConditionalValueFlow(nn.Module):
    def __init__(self, context_dim, value_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        
        # Projects the scalar value v_{t-1} to a vector
        self.val_proj = nn.Linear(value_dim, hidden_dim // 2)
        
        # Projects the structural context h_t to a vector
        self.ctx_proj = nn.Linear(context_dim, hidden_dim // 2)
        
        # Autoregressive Core
        # Input size is sum of projected value and projected context
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Gaussian Heads: Predict Mean and Log-Scale
        self.mu_net = nn.Linear(hidden_dim, value_dim)
        self.log_sigma_net = nn.Linear(hidden_dim, value_dim)

    def forward(self, values, context):
        """
        Input: 
            values: [B, L] (Real continuous values)
            context: [B, L, H_struct] (Hidden states from Structure Model)
        Output: 
            mu, log_sigma for q(v_t | v_{<t}, structure)
        """
        B, L = values.shape
        values = values.unsqueeze(-1) # [B, L, 1]
        
        # Shift values for autoregression: Input at t is v_{t-1}
        # We pad the beginning with 0.0 (initial value condition)
        # Slice off the last value since we don't need to predict *after* the sequence ends
        # effectively aligned so that input[t] is used to predict target[t]
        prev_values = F.pad(values[:, :-1, :], (0, 0, 1, 0), "constant", 0.0)
        
        # 1. Project Inputs
        v_emb = F.relu(self.val_proj(prev_values)) # [B, L, H/2]
        c_emb = F.relu(self.ctx_proj(context))     # [B, L, H/2]
        
        # 2. Combine (Conditioning)
        lstm_input = torch.cat([v_emb, c_emb], dim=-1) # [B, L, H]
        
        # 3. Run LSTM
        out, _ = self.lstm(lstm_input)
        
        # 4. Predict Parameters
        mu = self.mu_net(out)             # [B, L, 1]
        log_sigma = self.log_sigma_net(out) # [B, L, 1]
        
        return mu.squeeze(-1), log_sigma.squeeze(-1)


# ------------------------------------------------------------------------------
# Wrapper: Hybrid SeqFlow
# Combines both models into the Latent Optimization framework
# ------------------------------------------------------------------------------
class HybridSeqFlow(nn.Module):
    def __init__(self, vocab_size, embed_dim, struct_hidden_dim, val_hidden_dim, 
                 struct_layers=2, val_layers=2, sigma_min=1e-3):
        super().__init__()
        
        self.struct_flow = DiscreteStructureFlow(vocab_size, embed_dim, struct_hidden_dim, struct_layers)
        self.val_flow = ConditionalValueFlow(context_dim=struct_hidden_dim, hidden_dim=val_hidden_dim, num_layers=val_layers)
        
        self.sigma_min = sigma_min
        self.prior = torch.distributions.Normal(0, 1)
        # Token index for 'VAL_NUM' which indicates a continuous value follows
        self.val_token_id = 3 

    def forward(self, tokens, values):
        """
        Training Pass.
        Returns composite loss.
        """
        # 1. Structure Forward
        # We pass full tokens. LSTM shifts internally or via input slicing implies 
        # we predict token[t] given token[0...t-1].
        # For simplicity in this snippet, we assume standard causal masking or input shifting is handled by the loop or data loader.
        # If using standard LSTM, we typically feed tokens[:, :-1] to predict tokens[:, 1:]
        
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        input_values = values[:, 1:] # Align values with targets
        
        # Run Structure Model
        logits, hidden_states = self.struct_flow(input_tokens)
        
        # Run Value Model
        # Context is the hidden state associated with predicting the *current* token
        mu, log_sigma = self.val_flow(input_values, hidden_states)
        sigma = torch.exp(log_sigma) + self.sigma_min
        
        # --- Losses ---
        
        # A. Discrete Loss (Cross Entropy)
        # Reshape for CE: [B*L, V] vs [B*L]
        loss_struct = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
        
        # B. Continuous Loss (Gaussian NLL)
        # We only care about loss where the *target* token was VAL_NUM
        mask = (target_tokens == self.val_token_id).float()
        
        # Calculate Latent Z_val for flow correctness checks
        z_val = (input_values - mu) / sigma
        log_det = -torch.log(sigma)
        
        # NLL = 0.5 * (log(2pi) + z^2) - log_det
        # We can just use the probability density directly or reconstruction MSE
        # Using Gaussian Log Prob:
        nll_val = -torch.distributions.Normal(mu, sigma).log_prob(input_values)
        
        # Masked Mean
        if mask.sum() > 0:
            loss_val = (nll_val * mask).sum() / mask.sum()
        else:
            loss_val = torch.tensor(0.0, device=tokens.device)
            
        return loss_struct, loss_val

    def encode_to_latent(self, tokens, values):
        """
        Maps X -> Z for optimization.
        Returns z_struct (not explicit in standard seqflow, usually handled by acquisition) 
        and z_val (explicit flow).
        """
        # In standard SeqFlow, z_struct isn't a single vector but the generative process.
        # However, we definitely need z_val for the continuous optimization.
        
        input_tokens = tokens[:, :-1]
        target_values = values[:, 1:]
        target_tokens = tokens[:, 1:]
        
        with torch.no_grad():
            _, hidden_states = self.struct_flow(input_tokens)
            mu, log_sigma = self.val_flow(target_values, hidden_states)
            sigma = torch.exp(log_sigma) + self.sigma_min
            
            # Get Z for values
            z_val = (target_values - mu) / sigma
            
            # Mask out non-value latents (replace with 0 or noise)
            mask = (target_tokens == self.val_token_id)
            z_val[~mask] = 0.0
            
        return z_val

    def decode_from_latent(self, z_val, max_len=20, sos=1, eos=2):
        """
        Full generation given a latent vector z_val (modifying the continuous parameters).
        Structure is generated greedily (or sampled), Values are generated using z_val.
        """
        device = z_val.device
        batch_size = z_val.shape[0]
        
        curr_token = torch.full((batch_size, 1), sos, dtype=torch.long, device=device)
        curr_value = torch.full((batch_size, 1), 0.0, dtype=torch.float, device=device)
        
        # LSTM states
        h_struct = None
        h_val = None
        
        tokens_out = []
        values_out = []
        
        for t in range(max_len):
            # 1. Structure Step
            emb = self.struct_flow.embedding(curr_token)
            out_struct, h_struct = self.struct_flow.lstm(emb, h_struct)
            
            logits = self.struct_flow.head(out_struct)
            next_token = torch.argmax(logits, dim=-1)
            
            # 2. Value Step
            # We need to project values and context (out_struct)
            # Value LSTM input: [v_{t-1}, h_struct_t]
            v_emb = F.relu(self.val_flow.val_proj(curr_value))
            c_emb = F.relu(self.val_flow.ctx_proj(out_struct))
            val_in = torch.cat([v_emb, c_emb], dim=-1)
            
            out_val, h_val = self.val_flow.lstm(val_in, h_val)
            
            mu = self.val_flow.mu_net(out_val)
            log_sigma = self.val_flow.log_sigma_net(out_val)
            sigma = torch.exp(log_sigma) + self.sigma_min
            
            # 3. Determine Value
            # If token is VAL_NUM, use the z_val latent to guide the value
            # If t < z_val length, use z_val[t], else sample 0
            if t < z_val.shape[1]:
                z = z_val[:, t:t+1].unsqueeze(-1) # Align dims
            else:
                z = torch.zeros_like(mu)
                
            # v = mu + z * sigma
            # We calculate this for every step, but only keep it if token is VAL_NUM
            pred_value = mu + z * sigma
            
            # Masking logic for output
            is_val_token = (next_token == self.val_token_id)
            
            # For next step input:
            # If discrete, value input is 0.0. If continuous, value is pred_value.
            # However, for autoregression, we feed the predicted value.
            final_value = torch.where(is_val_token, pred_value.squeeze(-1), torch.zeros_like(pred_value.squeeze(-1)))
            
            tokens_out.append(next_token)
            values_out.append(final_value)
            
            curr_token = next_token
            curr_value = final_value.unsqueeze(1)
            
            if (next_token == eos).all():
                break
                
        return torch.cat(tokens_out, dim=1), torch.cat(values_out, dim=1)