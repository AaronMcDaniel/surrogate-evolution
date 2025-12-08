import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

from surrogates.hybrid_seqflow import DiscreteSeqFlow
from tests.token_surrogate import Surrogate
from codec import Codec
from surrogates import surrogate_models as sm

# Configuration
REPO_DIR = "/storage/ice-shared/vip-vvk/data/AOT/"
TESTING_DIR = "psomu3/inverse_design/training_results"
SURROGATE_WEIGHTS_DIR = os.path.join(REPO_DIR, "psomu3/codestral/surrogate_training/surrogate_weights_inverse_design")
GENERATOR_CHECKPOINT = "discrete_seqflow_float_split_final.pt"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_surrogate_models(surrogate):
    """
    Helper to load weights into the surrogate wrapper and keep them in memory
    for the optimization loop.
    """
    print("Loading Surrogate Models into Memory...")
    for i, model_dict in enumerate(surrogate.models):
        print(f"Loading {model_dict['name']}...")
        model_class = model_dict['model']
        output_size = len(model_dict['metrics_subset'])
        
        # Instantiate
        model = model_class(
            vocab_size=surrogate.vocab_size,
            output_size=output_size,
            dropout=0.0 # No dropout for inference/optimization
        ).to(DEVICE)
        
        # Load Weights
        weights_path = os.path.join(surrogate.weights_dir, f"{model_dict['name']}.pth")
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        
        # Freeze
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        # Store in dict for predict_soft to access
        model_dict['loaded_model'] = model
    return surrogate

def main():
    # 1. Setup Codec and Vocab
    codec = Codec(num_classes=1)
    codec._build_discrete_float_split_vocab()
    vocab_size = len(codec.vocab)
    
    # 2. Load Surrogate Wrapper
    # Note: cls_steps/reg_steps are placeholders here as we aren't training surrogate
    surrogate = Surrogate('conf.toml', SURROGATE_WEIGHTS_DIR, cls_steps=100, reg_steps=100)
    surrogate.device = DEVICE
    surrogate = load_surrogate_models(surrogate)
    
    # 3. Load Generator (Flow)
    print("Loading Generator...")
    checkpoint = torch.load(GENERATOR_CHECKPOINT, map_location=DEVICE)
    config = checkpoint['config']
    
    generator = DiscreteSeqFlow(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        seq_len=config['seq_len'],
        flow_hidden_dim=config['flow_hidden_dim'],
        flow_num_layers=config['flow_num_layers'],
        sigma=config['sigma']
    ).to(DEVICE)
    
    generator.load_state_dict(checkpoint['model_state_dict'])
    
    # CRITICAL: Must be in train mode for LSTM backprop (cuDNN requirement)
    # Weights are frozen via requires_grad=False below
    generator.train() 
    
    for param in generator.parameters():
        param.requires_grad = False

    # ========================================================================
    # PREPARE ANCHOR (SOS TOKEN)
    # ========================================================================
    # We must force the first token to be <SOS> so the LSTM has valid history.
    sos_id = codec.vocab["<SOS>"]
    sos_tensor = torch.tensor([[sos_id]], device=DEVICE) # [1, 1]
    
    # Calculate the exact latent vector z that corresponds to <SOS>
    with torch.no_grad():
        z_sos = generator.encode(sos_tensor) # [1, 1, EmbedDim]
    
    print(f"Calculated anchor for <SOS> token. Shape: {z_sos.shape}")

    # ========================================================================
    # INVERSE DESIGN LOOP
    # ========================================================================
    
    BATCH_SIZE = 16
    NUM_STEPS = 500
    LR = 0.1
    
    # Targets: [CIoU (Lower is better), AP (Higher is better)]
    TARGET_FITNESS = torch.tensor([1.6, 0.75]).to(DEVICE)
    
    print(f"Target Fitness: CIoU={TARGET_FITNESS[0]}, AP={TARGET_FITNESS[1]}")
    
    # Initialize Latent Z randomly
    z_opt = torch.randn(BATCH_SIZE, config['seq_len'], config['embed_dim'], device=DEVICE)
    
    # FORCE ANCHOR: Overwrite the first timestep with z_sos for the whole batch
    with torch.no_grad():
        # FIX: Squeeze the sequence dimension [1, 1, D] -> [1, D] before repeating
        z_opt[:, 0, :] = z_sos.squeeze(1).repeat(BATCH_SIZE, 1)
        
    z_opt.requires_grad_(True)
    
    optimizer = optim.Adam([z_opt], lr=LR)
    
    # Cosine Annealing for Temperature (Sharpening)
    temp_schedule = np.linspace(2.0, 0.1, NUM_STEPS)
    
    pbar = tqdm(range(NUM_STEPS))
    for step in pbar:
        tau = temp_schedule[step]
        optimizer.zero_grad()
        
        # A. Generator Forward (Inverse) -> Soft Tokens
        soft_tokens = generator.decode_soft(z_opt, temperature=tau) # [B, L, V]
        
        # B. Surrogate Prediction (Soft Handoff)
        preds = surrogate.predict_soft(soft_tokens, model_indices=[2])
        
        # C. Loss Calculation
        target_batch = TARGET_FITNESS.unsqueeze(0).expand(BATCH_SIZE, -1)
        
        # 1. Fitness Loss
        fitness_loss = F.mse_loss(preds, target_batch)
        
        # 2. Prior Loss (Only on the non-anchor parts)
        prior_loss = torch.mean(z_opt[:, 1:, :] ** 2)
        
        # 3. Entropy Regularization
        dist = torch.distributions.Categorical(probs=soft_tokens)
        entropy = dist.entropy().mean()
        
        total_loss = fitness_loss + (0.01 * prior_loss) + (0.05 * entropy)
        
        # D. Backprop
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_([z_opt], 1.0)
        
        # Zero out gradients for the anchor position to ensure it doesn't move
        z_opt.grad[:, 0, :] = 0.0
        
        optimizer.step()
        
        # FORCE ANCHOR RESET (Safety measure)
        with torch.no_grad():
            # FIX: Squeeze then Repeat
            z_opt[:, 0, :] = z_sos.squeeze(1).repeat(BATCH_SIZE, 1)
        
        pbar.set_postfix({
            'Fit_L': f"{fitness_loss.item():.4f}", 
            'Prior_L': f"{prior_loss.item():.2f}",
            'Ent_L': f"{entropy.item():.2f}",
            'Pred_CIoU': f"{preds[:,0].mean().item():.3f}",
            'Pred_AP': f"{preds[:,1].mean().item():.3f}"
        })

    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print("\nOptimization Complete. Decoding Final Architectures...")
    
    # Hard Decode
    with torch.no_grad():
        final_tokens = generator.decode(z_opt)
        
    # Convert tokens back to strings
    id_to_token = {v: k for k, v in codec.vocab.items()}
    
    results = []
    for i in range(BATCH_SIZE):
        toks = final_tokens[i].cpu().numpy()
        
        # Filter padding
        pad_id = codec.vocab["<PAD>"]
        toks = [t for t in toks if t != pad_id]
        
        # Reconstruct string
        str_tokens = [id_to_token.get(t, "UNK") for t in toks]
        
        results.append({
            'pred_ciou': preds[i, 0].item(),
            'pred_ap': preds[i, 1].item(),
            'tokens': str_tokens
        })
        
    # Sort by proximity to target
    results.sort(key=lambda x: abs(x['pred_ap'] - TARGET_FITNESS[1].item()))
    
    print("\nTop Generated Architectures:")
    for r in results[:5]:
        print(f"Pred CIoU: {r['pred_ciou']:.4f} | Pred AP: {r['pred_ap']:.4f}")
        # Print nicely formatted
        print(f"Arch: {r['tokens'][:10]} ... [Length: {len(r['tokens'])}]")
        print("-" * 40)

if __name__ == "__main__":
    main()