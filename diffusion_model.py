# Improved Diffusion Model with Residual U-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb

class ResidualBlock(nn.Module):
    """Residual block with group normalization and time embedding"""
    
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Use GroupNorm if channels divisible by 8, otherwise use BatchNorm
        if out_channels % 8 == 0:
            self.norm1 = nn.GroupNorm(8, out_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, t_emb):
        # First conv
        h = self.conv1(x)
        # Normalize and activate
        h = F.silu(self.norm1(h))
        
        # Add time embedding
        t_emb = self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = h + t_emb
        
        # Second conv with dropout
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    """Self-attention block for better feature extraction"""
    
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        
        # Attention
        attn = torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(b, c, h, w)
        
        return self.proj(out) + x

class UNet(nn.Module):
    """Improved U-Net with residual blocks and attention"""
    
    def __init__(self, channels=1, time_dim=256, base_channels=64):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Encoder (1 -> 64 -> 128 -> 256)
        # First layer: simple conv to expand channels
        self.input_conv = nn.Conv2d(channels, base_channels, 3, padding=1)
        
        self.enc1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        
        # Bottleneck with attention
        self.bottleneck1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.attn = AttentionBlock(base_channels * 4)
        self.bottleneck2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        
        # Decoder (256 -> 128 -> 64 -> 32)
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 2, time_dim)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_dim)
        
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_dim)
        
        # Output
        self.out = nn.Conv2d(base_channels, channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial conv to expand channels
        x = F.silu(self.input_conv(x))
        
        # Encoder with skip connections
        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.down1(x1), t_emb)
        x3 = self.enc3(self.down2(x2), t_emb)
        x4 = self.bottleneck1(self.down3(x3), t_emb)
        
        # Bottleneck with attention
        x4 = self.attn(x4)
        x4 = self.bottleneck2(x4, t_emb)
        
        # Decoder with skip connections
        x = self.up3(x4)
        # Resize to match skip connection if needed
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, x3], dim=1), t_emb)
        
        x = self.up2(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1), t_emb)
        
        x = self.up1(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x1], dim=1), t_emb)
        
        return self.out(x)

class Diffusion(nn.Module):
    """Improved diffusion model with cosine noise schedule"""
    
    def __init__(self, img_size=70, noise_steps=300, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.img_size = img_size
        self.noise_steps = noise_steps
        
        # Cosine noise schedule (better than linear)
        self.beta = self.cosine_beta_schedule(noise_steps, beta_start, beta_end)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # Improved model
        self.model = UNet(channels=1, time_dim=256, base_channels=64)
    
    def cosine_beta_schedule(self, timesteps, beta_start, beta_end, s=0.008):
        """Cosine noise schedule from Improved Denoising Diffusion Probabilistic Models"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
    
    def prepare_noise_schedule(self):
        """Move noise schedule to device"""
        device = next(self.model.parameters()).device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
    
    def add_noise(self, x, t):
        """Add noise according to timestep"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def sample_timesteps(self, n):
        """Sample random timesteps"""
        return torch.randint(0, self.noise_steps, (n,))
    
    def forward(self, x):
        """Training forward pass"""
        self.prepare_noise_schedule()
        
        t = self.sample_timesteps(x.shape[0]).to(x.device)
        x_noisy, noise = self.add_noise(x, t)
        predicted_noise = self.model(x_noisy, t)
        
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def sample(self, n, cfg_scale=0.0):
        """Generate samples using DDPM sampling"""
        self.model.eval()
        self.prepare_noise_schedule()
        
        # Start from pure noise
        x = torch.randn((n, 1, self.img_size, self.img_size), device=next(self.model.parameters()).device)
        
        # Reverse diffusion
        for i in reversed(range(self.noise_steps)):
            t = torch.full((n,), i, device=x.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(x, t)
            
            # Calculate alpha values
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            # DDPM sampling formula
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            x = (x - beta / torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha)
            x = x + torch.sqrt(beta) * noise
        
        self.model.train()
        return x
