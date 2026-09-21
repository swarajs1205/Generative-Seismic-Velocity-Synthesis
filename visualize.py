# Visualization utilities for seismic velocity models

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import DATA_DIR, RESULTS_DIR
from dataset import get_dataloader

def visualize_dataset_samples(num_samples=6):
    """Visualize samples from the dataset"""
    print(f"Visualizing {num_samples} dataset samples...")
    
    # Load dataset
    dataloader = get_dataloader(DATA_DIR, batch_size=num_samples, shuffle=True)
    
    # Get samples
    batch = next(iter(dataloader))
    samples = batch[:num_samples]
    
    # Create plot
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        # Get velocity model
        velocity = samples[i, 0].cpu().numpy()
        
        # Plot
        im = axes[row, col].imshow(velocity, cmap='seismic', aspect='auto')
        axes[row, col].set_title(f'Dataset Sample {i+1}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'dataset_samples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset samples saved to: {save_path}")

def analyze_dataset():
    """Analyze dataset statistics"""
    print("Analyzing dataset statistics...")
    
    # Load dataset
    dataloader = get_dataloader(DATA_DIR, batch_size=16)
    
    # Collect statistics
    all_velocities = []
    shapes = []
    
    for batch in dataloader:
        velocities = batch.flatten().cpu().numpy()
        all_velocities.extend(velocities)
        shapes.append(batch.shape)
        
        # Limit to first 10 batches for speed
        if len(shapes) >= 10:
            break
    
    all_velocities = np.array(all_velocities)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples analyzed: {len(shapes)} batches")
    print(f"  Batch shape: {shapes[0]}")
    print(f"  Total velocity values: {len(all_velocities):,}")
    print(f"  Velocity range: [{all_velocities.min():.3f}, {all_velocities.max():.3f}]")
    print(f"  Mean velocity: {all_velocities.mean():.3f}")
    print(f"  Std velocity: {all_velocities.std():.3f}")
    print(f"  Median velocity: {np.median(all_velocities):.3f}")
    
    # Create histogram
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_velocities, bins=50, alpha=0.7, density=True)
    plt.title('Velocity Distribution')
    plt.xlabel('Normalized Velocity')
    plt.ylabel('Density')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(all_velocities)
    plt.title('Velocity Box Plot')
    plt.ylabel('Normalized Velocity')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'dataset_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset analysis saved to: {save_path}")

def visualize_training_progress():
    """Visualize training progress if available"""
    loss_file = os.path.join(RESULTS_DIR, 'training_loss.png')
    
    if os.path.exists(loss_file):
        print(f"Training loss plot available at: {loss_file}")
        # You can open this file to see training progress
    else:
        print("No training loss plot found. Please train the model first.")

def create_summary_report():
    """Create a summary report of the project"""
    print("Creating summary report...")
    
    # Check what files exist
    files_to_check = [
        ('Dataset Samples', 'dataset_samples.png'),
        ('Dataset Analysis', 'dataset_analysis.png'),
        ('Training Loss', 'training_loss.png'),
        ('Generated Samples', 'generated_samples.png'),
        ('Comparison', 'comparison.png'),
    ]
    
    print(f"\nProject Summary Report:")
    print(f"======================")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nAvailable Files:")
    
    for name, filename in files_to_check:
        filepath = os.path.join(RESULTS_DIR, filename)
        status = "✓" if os.path.exists(filepath) else "✗"
        print(f"  {status} {name}: {filename}")
    
    # Check model files
    model_files = ['best_model.pth', 'final_model.pth']
    print(f"\nModel Files:")
    for model_file in model_files:
        filepath = os.path.join(RESULTS_DIR, model_file)
        status = "✓" if os.path.exists(filepath) else "✗"
        size = ""
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        print(f"  {status} {model_file}{size}")

def main():
    """Main visualization function"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "dataset":
            visualize_dataset_samples()
        elif command == "analyze":
            analyze_dataset()
        elif command == "progress":
            visualize_training_progress()
        elif command == "report":
            create_summary_report()
        else:
            print("Available commands: dataset, analyze, progress, report")
    else:
        # Run all visualizations
        print("Running all visualizations...")
        visualize_dataset_samples()
        analyze_dataset()
        visualize_training_progress()
        create_summary_report()
        print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()
