# Dataset loader for seismic velocity models

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VelocityDataset(Dataset):
    """Dataset for seismic velocity models from OpenFWI CurveVel"""
    
    def __init__(self, data_dir, limit=None):
        self.samples = []
        
        # Load all .npy files and flatten individual samples
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        
        for file in files:
            file_path = os.path.join(data_dir, file)
            data = np.load(file_path)  # Shape: (500, 1, 70, 70)
            
            # Extract individual velocity models
            for i in range(data.shape[0]):
                velocity_model = data[i]  # Shape: (1, 70, 70)
                self.samples.append(velocity_model)
        
        # Optional limit for faster training
        if limit is not None and limit < len(self.samples):
            self.samples = self.samples[:limit]
            print(f"Using first {limit} samples from {len(self.samples) + limit} total")
        
        print(f"Loaded {len(self.samples)} velocity models")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        velocity = self.samples[idx]
        
        # Convert to tensor and normalize to [-1, 1]
        velocity = torch.tensor(velocity, dtype=torch.float32)
        
        # Normalize to [0, 1] first, then to [-1, 1]
        velocity = (velocity - velocity.min()) / (velocity.max() - velocity.min() + 1e-8)
        velocity = velocity * 2 - 1
        
        return velocity

def get_dataloader(data_dir, batch_size, limit=None, shuffle=True):
    """Create DataLoader for velocity dataset"""
    dataset = VelocityDataset(data_dir, limit=limit)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
