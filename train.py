# Training script for Seismic Diffusion Model

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from config import device, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE, NOISE_STEPS, BETA_START, BETA_END, DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR, DATASET_LIMIT
from dataset import get_dataloader
from diffusion_model import Diffusion

class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

def train_model():
    """Full training loop with EMA"""
    print("=== Training Seismic Diffusion Model ===")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Noise steps: {NOISE_STEPS}")
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataloader = get_dataloader(DATA_DIR, BATCH_SIZE, limit=DATASET_LIMIT)
    print(f"Dataset loaded: {len(dataloader)} batches")
    
    # Initialize model
    print("Initializing model...")
    model = Diffusion(img_size=IMG_SIZE, noise_steps=NOISE_STEPS, beta_start=BETA_START, beta_end=BETA_END).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # EMA
    ema = EMA(model, decay=0.9999)
    
    # Training history
    losses = []
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, velocity_models in enumerate(pbar):
            velocity_models = velocity_models.to(device)
            
            # Forward pass
            loss = model(velocity_models)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update EMA
            ema.update()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            
            # Save EMA model
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            ema.restore()
            
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model with EMA
    ema.apply_shadow()
    final_path = os.path.join(CHECKPOINT_DIR, "diffusion_model_final.pt")
    torch.save(model.state_dict(), final_path)
    ema.restore()
    
    print(f"\nFinal model saved: {final_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved: {os.path.join(RESULTS_DIR, 'training_loss.png')}")
    
    return model

def quick_train():
    """Quick training for debugging (3 epochs, limited data)"""
    print("=== Quick Training (Debug Mode) ===")
    
    # Temporarily reduce dataset size
    dataloader = get_dataloader(DATA_DIR, BATCH_SIZE, limit=100)
    print(f"Using 100 samples for quick training")
    
    model = Diffusion(img_size=IMG_SIZE, noise_steps=100, beta_start=BETA_START, beta_end=BETA_END).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(3):
        losses = []
        for batch_idx, velocity_models in enumerate(dataloader):
            if batch_idx >= 2:  # Only 2 batches
                break
            
            velocity_models = velocity_models.to(device)
            loss = model(velocity_models)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss:.6f}")
        
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
    
    # Save quick model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "diffusion_model_quick.pt"))
    
    # Test sampling
    print("Testing sampling...")
    model.eval()
    with torch.no_grad():
        samples = model.sample(2)
    print(f"Generated samples shape: {samples.shape}")
    
    return model

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_train()
    else:
        train_model()
