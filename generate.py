# Generation script for Seismic Diffusion Model

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from config import device, IMG_SIZE, NOISE_STEPS, BETA_START, BETA_END, DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
from dataset import get_dataloader
from diffusion_model import Diffusion

def load_model(checkpoint_path=None):
    """Load trained diffusion model"""
    model = Diffusion(img_size=IMG_SIZE, noise_steps=NOISE_STEPS, beta_start=BETA_START, beta_end=BETA_END).to(device)
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "diffusion_model_final.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Using untrained model")
    
    return model

def generate_samples(model, num_samples=4, save=True):
    """Generate velocity model samples"""
    print(f"Generating {num_samples} samples...")
    
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples)
    
    # Convert from [-1, 1] to [0, 1] for visualization
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    if save:
        save_samples(samples)
    
    return samples

def save_samples(samples, filename='generated_samples.png'):
    """Save generated samples as images"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, filename)
    
    num_samples = samples.shape[0]
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row, col = i // cols, i % cols
        axes[row, col].imshow(samples[i, 0].cpu().numpy(), cmap='seismic', vmin=0, vmax=1)
        axes[row, col].set_title(f'Generated {i+1}')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Samples saved: {save_path}")

def compare_with_real(model, num_samples=4):
    """Compare generated samples with real data"""
    print("Comparing with real data...")
    
    # Get real samples
    dataloader = get_dataloader(DATA_DIR, num_samples)
    real_batch = next(iter(dataloader)).to(device)
    
    # Generate samples
    generated_samples = generate_samples(model, num_samples, save=False)
    
    # Convert real samples from [-1, 1] to [0, 1]
    real_samples = (real_batch + 1) / 2
    real_samples = torch.clamp(real_samples, 0, 1)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        # Real sample
        axes[0, i].imshow(real_samples[i, 0].cpu().numpy(), cmap='seismic', vmin=0, vmax=1)
        axes[0, i].set_title(f'Real Model {i+1}')
        axes[0, i].axis('off')
        
        # Generated sample
        axes[1, i].imshow(generated_samples[i, 0].cpu().numpy(), cmap='seismic', vmin=0, vmax=1)
        axes[1, i].set_title(f'Generated {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(RESULTS_DIR, 'comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved: {comparison_path}")

def analyze_samples(samples):
    """Analyze statistics of generated samples"""
    samples_np = samples.cpu().numpy()
    
    print("\nGenerated Samples Analysis:")
    print(f"  Shape: {samples_np.shape}")
    print(f"  Min: {samples_np.min():.4f}")
    print(f"  Max: {samples_np.max():.4f}")
    print(f"  Mean: {samples_np.mean():.4f}")
    print(f"  Std: {samples_np.std():.4f}")

def main():
    """Main generation function"""
    print("=== Generating Seismic Velocity Models ===")
    
    # Load model
    model = load_model()
    
    # Generate samples
    num_samples = 8
    samples = generate_samples(model, num_samples)
    
    # Analyze
    analyze_samples(samples)
    
    # Compare with real data
    compare_with_real(model, num_samples=4)
    
    print("\nGeneration complete!")

if __name__ == "__main__":
    main()
