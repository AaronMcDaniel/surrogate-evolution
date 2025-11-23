"""
Simple validation script to test inverse designer gradient flow.

This script creates synthetic data and tests:
1. CVAE generator sampling and gradient flow
2. Surrogate predict() method gradient flow
3. End-to-end optimization step

Run this before the full pipeline to ensure everything is working.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path

from surrogates.inverse_designer import (
    ConditionalVAERepresentation,
    UnconditionalVAERepresentation,
    ConditionalDiffusionRepresentation,
    InverseDesigner
)


class MockSurrogate(nn.Module):
    """
    Mock surrogate for testing.
    Simple MLP that predicts fitness from latent vectors.
    """
    
    def __init__(self, z_dim=128, num_objectives=3):
        super().__init__()
        self.objectives = {f'obj_{i}': 1.0 for i in range(num_objectives)}
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_objectives)
        )
    
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable predict method."""
        return self.net(z)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.predict(z)


class MockVAEEncoder:
    """Mock VAE encoder for testing."""
    
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim
    
    def eval(self):
        pass
    
    def parameters(self):
        return []


def test_cvae_gradient_flow():
    """Test 1: CVAE generator gradient flow."""
    print("=" * 80)
    print("Test 1: CVAE Generator Gradient Flow")
    print("=" * 80)
    
    z_dim = 128
    num_objectives = 3
    batch_size = 16
    
    # Create CVAE
    cvae = ConditionalVAERepresentation(
        z_dim=z_dim,
        num_objectives=num_objectives,
        latent2_dim=64,
        hidden_sizes=[256, 128]
    )
    
    # Create synthetic data
    z_arch_vectors = torch.randn(100, z_dim)
    fitness_values = torch.randn(100, num_objectives)
    
    # Test forward pass
    desired_fitness = torch.tensor([0.5, 0.3, 0.8]).float()
    z_generated = cvae.sample(desired_fitness, batch_size)
    
    print(f"✓ Generated z_arch shape: {z_generated.shape}")
    assert z_generated.shape == (batch_size, z_dim), "Wrong output shape!"
    
    # Test gradient flow
    z_generated = cvae.sample(desired_fitness, batch_size)
    loss = z_generated.mean()
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for param in cvae.parameters:
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients in CVAE parameters!"
    print("✓ Gradients flow through CVAE generator")
    
    # Test training
    print("\nTesting CVAE pre-training (5 epochs)...")
    history = cvae.initial_train(
        z_arch_vectors=z_arch_vectors,
        fitness_values=fitness_values,
        num_epochs=5,
        batch_size=16,
        lr=1e-3
    )
    
    print(f"✓ Training complete. Final loss: {history['total_loss'][-1]:.4f}")
    print()
    
    return cvae


def test_diffusion_gradient_flow():
    """Test 2: Diffusion generator gradient flow."""
    print("=" * 80)
    print("Test 2: Diffusion Generator Gradient Flow")
    print("=" * 80)
    
    z_dim = 128
    num_objectives = 3
    batch_size = 16
    
    # Create Diffusion model
    diffusion = ConditionalDiffusionRepresentation(
        z_dim=z_dim,
        num_objectives=num_objectives,
        hidden_dim=128,
        num_layers=2,
        timesteps=100  # Reduced for testing
    )
    
    # Create synthetic data
    z_arch_vectors = torch.randn(100, z_dim)
    fitness_values = torch.randn(100, num_objectives)
    
    # Test forward pass (noise prediction)
    z_t = torch.randn(batch_size, z_dim)
    t = torch.randint(0, 100, (batch_size,))
    fitness_cond = torch.randn(batch_size, num_objectives)
    
    noise_pred = diffusion.forward(z_t, t, fitness_cond)
    print(f"✓ Noise prediction shape: {noise_pred.shape}")
    assert noise_pred.shape == (batch_size, z_dim), "Wrong output shape!"
    
    # Test gradient flow
    loss = noise_pred.mean()
    loss.backward()
    
    has_grad = False
    for param in diffusion.parameters:
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients in Diffusion parameters!"
    print("✓ Gradients flow through Diffusion model")
    
    # Test training (brief)
    print("\nTesting Diffusion pre-training (3 epochs)...")
    history = diffusion.initial_train(
        z_arch_vectors=z_arch_vectors,
        fitness_values=fitness_values,
        num_epochs=3,
        batch_size=16,
        lr=1e-4
    )
    
    print(f"✓ Training complete. Final loss: {history['loss'][-1]:.4f}")
    print()
    
    return diffusion


def test_surrogate_gradient_flow():
    """Test 3: Mock surrogate gradient flow."""
    print("=" * 80)
    print("Test 3: Surrogate Predict Gradient Flow")
    print("=" * 80)
    
    z_dim = 128
    num_objectives = 3
    batch_size = 16
    
    # Create mock surrogate
    surrogate = MockSurrogate(z_dim, num_objectives)
    
    # Test forward pass
    z = torch.randn(batch_size, z_dim, requires_grad=True)
    f_pred = surrogate.predict(z)
    
    print(f"✓ Surrogate prediction shape: {f_pred.shape}")
    assert f_pred.shape == (batch_size, num_objectives), "Wrong output shape!"
    
    # Test gradient flow
    loss = f_pred.mean()
    loss.backward()
    
    assert z.grad is not None, "No gradient to input z!"
    assert z.grad.abs().sum() > 0, "Zero gradient to input z!"
    print("✓ Gradients flow through surrogate.predict()")
    print()
    
    return surrogate


def test_end_to_end_optimization():
    """Test 4: End-to-end optimization step."""
    print("=" * 80)
    print("Test 4: End-to-End Optimization (CVAE)")
    print("=" * 80)
    
    z_dim = 128
    num_objectives = 3
    batch_size = 16
    
    # Create components
    surrogate = MockSurrogate(z_dim, num_objectives)
    vae_encoder = MockVAEEncoder(z_dim)
    
    # Create InverseDesigner with CVAE
    designer = InverseDesigner(
        generator_type='cvae',
        surrogate_model=surrogate,
        vae_encoder=vae_encoder,
        z_dim=z_dim,
        num_objectives=num_objectives,
        generator_kwargs={'latent2_dim': 64, 'hidden_sizes': [256, 128]}
    )
    
    # Pre-train generator on synthetic data
    z_arch_vectors = torch.randn(100, z_dim)
    fitness_values = torch.randn(100, num_objectives)
    
    print("Pre-training generator (3 epochs)...")
    history = designer.generator.initial_train(
        z_arch_vectors=z_arch_vectors,
        fitness_values=fitness_values,
        num_epochs=3,
        batch_size=16,
        lr=1e-3
    )
    print(f"✓ Pre-training complete. Final loss: {history['total_loss'][-1]:.4f}")
    
    # Run optimization steps
    desired_fitness = torch.tensor([0.5, 0.3, 0.8]).float()
    
    print("\nRunning 10 optimization steps...")
    losses = []
    
    for i in range(10):
        z_gen, f_pred, loss = designer.run_optimization_step(
            desired_fitness=desired_fitness,
            batch_size=batch_size,
            lr=1e-3
        )
        losses.append(loss)
        
        if i % 3 == 0:
            mean_pred = f_pred.mean(dim=0)
            print(f"  Step {i}: Loss={loss:.4f}, Mean Pred={mean_pred.cpu().numpy()}")
    
    print(f"\n✓ Optimization steps complete")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    
    # Check that loss decreased (or at least didn't explode)
    if losses[-1] < losses[0] * 2:
        print("✓ Loss is reasonable (not exploding)")
    else:
        print("⚠ Warning: Loss may be unstable")
    
    print()


def test_unconditional_vae():
    """Test 5: Unconditional VAE gradient flow and sampling."""
    print("=" * 80)
    print("Test 5: Unconditional VAE Gradient Flow")
    print("=" * 80)
    
    z_dim = 128
    num_objectives = 3
    batch_size = 16
    
    # Define fixed target fitness
    target_fitness = torch.tensor([0.7, 0.5, 0.9]).float()
    
    # Create Unconditional VAE
    uncond_vae = UnconditionalVAERepresentation(
        z_dim=z_dim,
        num_objectives=num_objectives,
        target_fitness=target_fitness,
        latent2_dim=64,
        hidden_sizes=[256, 128]
    )
    
    print(f"Target fitness: {target_fitness.numpy()}")
    
    # Create synthetic data
    z_arch_vectors = torch.randn(100, z_dim)
    fitness_values = torch.randn(100, num_objectives)  # Not used in training
    
    # Test forward pass
    z_generated = uncond_vae.sample(c=None, batch_size=batch_size)  # c is ignored
    
    print(f"✓ Generated z_arch shape: {z_generated.shape}")
    assert z_generated.shape == (batch_size, z_dim), "Wrong output shape!"
    
    # Test gradient flow
    z_generated = uncond_vae.sample(c=None, batch_size=batch_size)
    loss = z_generated.mean()
    loss.backward()
    
    # Check gradients exist
    has_grad = False
    for param in uncond_vae.parameters:
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients in Unconditional VAE parameters!"
    print("✓ Gradients flow through Unconditional VAE generator")
    
    # Test training
    print("\nTesting Unconditional VAE pre-training (5 epochs)...")
    history = uncond_vae.initial_train(
        z_arch_vectors=z_arch_vectors,
        fitness_values=fitness_values,
        num_epochs=5,
        batch_size=16,
        lr=1e-3
    )
    
    print(f"✓ Training complete. Final loss: {history['total_loss'][-1]:.4f}")
    
    # Test that sampling works without conditioning
    print("\nTesting unconditional sampling...")
    z_sampled = uncond_vae.sample(c=None, batch_size=32)
    print(f"✓ Sampled {z_sampled.shape[0]} architectures without conditioning")
    
    # Test optimization with surrogate
    print("\nTesting optimization with frozen surrogate (10 steps)...")
    surrogate = MockSurrogate(z_dim, num_objectives)
    surrogate.eval()
    
    optimizer = torch.optim.Adam(uncond_vae.parameters, lr=1e-3)
    
    losses = []
    for step in range(10):
        # Generate samples
        z_gen = uncond_vae.sample(c=None, batch_size=batch_size)
        
        # Predict fitness with frozen surrogate
        f_pred = surrogate.predict(z_gen)
        
        # Compute loss to target fitness
        loss = torch.nn.functional.mse_loss(f_pred, target_fitness.unsqueeze(0).expand(batch_size, -1))
        
        # Update generator
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(uncond_vae.parameters, 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 3 == 0:
            mean_pred = f_pred.mean(dim=0).detach()
            print(f"  Step {step}: Loss={loss.item():.4f}, Mean Pred={mean_pred.cpu().numpy()}, Target={target_fitness.numpy()}")
    
    print(f"\n✓ Optimization complete")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    
    if losses[-1] < losses[0] * 2:
        print("✓ Loss is reasonable (not exploding)")
    else:
        print("⚠ Warning: Loss may be unstable")
    
    print()
    
    return uncond_vae


def test_diffusion_sampling():
    """Test 6: Diffusion sampling (quick DDIM)."""
    print("=" * 80)
    print("Test 6: Diffusion Sampling (DDIM)")
    print("=" * 80)
    
    z_dim = 128
    num_objectives = 3
    batch_size = 4  # Small batch
    
    # Create and train diffusion model
    diffusion = ConditionalDiffusionRepresentation(
        z_dim=z_dim,
        num_objectives=num_objectives,
        hidden_dim=128,
        num_layers=2,
        timesteps=100
    )
    
    # Brief training
    z_arch_vectors = torch.randn(50, z_dim)
    fitness_values = torch.randn(50, num_objectives)
    
    print("Pre-training diffusion (2 epochs)...")
    diffusion.initial_train(
        z_arch_vectors=z_arch_vectors,
        fitness_values=fitness_values,
        num_epochs=2,
        batch_size=8
    )
    
    # Test sampling
    print("\nTesting DDIM sampling (10 steps)...")
    desired_fitness = torch.tensor([0.5, 0.3, 0.8]).float()
    
    z_sampled = diffusion.sample(
        c=desired_fitness,
        batch_size=batch_size,
        ddim=True,
        ddim_steps=10
    )
    
    print(f"✓ Sampled z_arch shape: {z_sampled.shape}")
    assert z_sampled.shape == (batch_size, z_dim), "Wrong output shape!"
    print("✓ Diffusion sampling complete")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("#" * 80)
    print("# Inverse Designer Validation Tests")
    print("#" * 80)
    print()
    
    try:
        # Test 1: CVAE
        cvae = test_cvae_gradient_flow()
        
        # Test 2: Diffusion
        diffusion = test_diffusion_gradient_flow()
        
        # Test 3: Surrogate
        surrogate = test_surrogate_gradient_flow()
        
        # Test 4: End-to-end
        test_end_to_end_optimization()
        
        # Test 5: Unconditional VAE
        uncond_vae = test_unconditional_vae()
        
        # Test 6: Diffusion sampling
        test_diffusion_sampling()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe inverse designer implementation is working correctly.")
        print("You can now proceed to run the full pipeline with real data.")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
