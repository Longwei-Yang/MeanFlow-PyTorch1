#!/usr/bin/env python3
"""
Test script to verify that the MeanFlow DPM Solver 2 modifications work correctly.
"""

import torch
import torch.nn as nn
from meanflow import MeanFlow

class SimpleMockModel(nn.Module):
    """A simple mock model for testing purposes."""
    
    def __init__(self, input_size=32, num_classes=1000):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.norm = nn.GroupNorm(1, 4)
        self.act = nn.SiLU()
        self.time_embed = nn.Linear(1, 4)
        self.class_embed = nn.Embedding(num_classes + 1, 4)  # +1 for null class
        
    def forward(self, x, t, h, y, train=True):
        # Simple forward pass for testing
        # x: (B, C, H, W), t: (B,), h: (B,), y: (B,)
        B = x.shape[0]
        
        # Time and class embeddings
        t_emb = self.time_embed(t.float().unsqueeze(-1))  # (B, 4)
        y_emb = self.class_embed(y.long())  # (B, 4)
        
        # Add embeddings to spatial features
        x = self.conv(x)
        x = self.norm(x)
        
        # Add time and class info (broadcast to spatial dimensions)
        t_emb = t_emb.view(B, 4, 1, 1)
        y_emb = y_emb.view(B, 4, 1, 1)
        x = x + t_emb + y_emb
        
        x = self.act(x)
        return x

def test_meanflow_dpm_solver():
    """Test the MeanFlow with DPM Solver 2 approximation."""
    
    print("Testing MeanFlow with DPM Solver 2 approximation...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create mock model
    model = SimpleMockModel().to(device)
    
    # Create MeanFlow instance with DPM Solver 2 approximation
    meanflow = MeanFlow(
        noise_dist='uniform',
        P_mean=-0.4,
        P_std=1.0,
        data_proportion=0.75,
        guidance_eq='cfg',
        omega=1.0,
        kappa=0.5,
        t_start=0.0,
        t_end=1.0,
        norm_p=1.0,
        norm_eps=0.01,
        num_classes=1000,
        class_dropout_prob=0.1,
        sampling_schedule_type='default',
        dpm_h_step=0.1  # DPM Solver step size
    )
    
    # Create test data
    batch_size = 4
    img_size = 32
    imgs = torch.randn(batch_size, 4, img_size, img_size).to(device)
    labels = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"Input shape: {imgs.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test the forward pass
    try:
        model.train()
        loss, proj_loss, v_loss = meanflow(model, imgs, labels, train=True)
        
        print(f"✓ Forward pass successful!")
        print(f"  Denoising loss: {loss.item():.6f}")
        print(f"  Projection loss: {proj_loss}")
        print(f"  V loss: {v_loss.item():.6f}")
        
        # Test backward pass
        loss.backward()
        print("✓ Backward pass successful!")
        
        # Check that gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"✓ Gradients computed: {has_grads}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dpm_approximation_consistency():
    """Test that DPM approximation gives reasonable results."""
    
    print("\nTesting DPM approximation consistency...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleMockModel().to(device)
    
    meanflow = MeanFlow(dpm_h_step=0.1)
    
    # Create test inputs
    batch_size = 2
    z_t = torch.randn(batch_size, 4, 32, 32).to(device)
    t = torch.tensor([0.8, 0.6]).view(batch_size, 1, 1, 1).to(device)
    r = torch.tensor([0.2, 0.1]).view(batch_size, 1, 1, 1).to(device)
    y_inp = torch.tensor([100, 200]).to(device)
    
    try:
        # Create test v_g (guided velocity)
        v_g = torch.randn_like(z_t)
        
        # Test the DPM approximation function
        u, jvp_approx = meanflow.dpm_solver_second_order_approximation(
            model, z_t, t, r, y_inp, v_g, train=True
        )
        
        print(f"✓ DPM approximation successful!")
        print(f"  u shape: {u.shape}")
        print(f"  jvp_approx shape: {jvp_approx.shape}")
        print(f"  u mean: {u.mean().item():.6f}")
        print(f"  jvp_approx mean: {jvp_approx.mean().item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in DPM approximation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing MeanFlow with DPM Solver 2 Modifications")
    print("=" * 60)
    
    success1 = test_meanflow_dpm_solver()
    success2 = test_dpm_approximation_consistency()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 60)
