# Seismic Diffusion Model

Generate realistic underground rock layer images using AI. This project uses a diffusion model trained on the OpenFWI CurveVel dataset to create synthetic seismic velocity models that look like real geological cross-sections.

## What This Does

Ever wondered what the ground beneath your feet looks like? This AI learns patterns from 30,000 real geological images and can generate new ones showing layered rock formations, curved boundaries, and depth-based velocity changes.

Perfect for:
- Students learning about geophysics
- Researchers needing synthetic data
- Anyone curious about AI and earth sciences

## Project Structure

```
seismic_diffusion_project/
├── config.py              # Settings (batch size, epochs, etc.)
├── dataset.py             # Loads and processes data
├── diffusion_model.py      # The neural network architecture
├── train.py               # Training script
├── generate.py            # Creates new images
├── data/raw/CurveVel/     # Dataset folder
├── checkpoints/           # Saved models
└── results/               # Generated outputs
```

## Quick Start

### 1. Setup Environment

```bash
conda create -n seismic_diffusion python=3.9 -y
conda activate seismic_diffusion
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm
```

### 2. Get the Data

Download the OpenFWI CurveVel dataset and put the .npy files in `data/raw/CurveVel/`.
link :- "  https://drive.google.com/drive/folders/1AL7z9TQVgzUdv3yrHB6f_wP2TRvVTKD0?usp=drive_link  "

### 3. Run the Project

Step-by-step commands:

```bash
# Check if everything works
python dataset.py

# Quick test (3 epochs, fast)
python train.py quick

# Full training (60 epochs, best results)
python train.py

# Generate new images
python generate.py
```

Training takes about 1-2 hours on a GPU. After training, your best model is saved at `checkpoints/diffusion_model_final.pt`.

## What to Expect

The model learns to generate images that look like this:
- Horizontal or curved layers
- Different colors representing different rock speeds
- Realistic depth-based patterns

After training, check `results/comparison.png` to see how your generated images compare to real ones.

## Tips for Success

- Make sure you have a GPU (training on CPU is very slow)
- If results look flat or uniform, train longer (increase epochs in config.py)
- For experimentation, use `DATASET_LIMIT = 1000` to train faster
- For best quality, use `DATASET_LIMIT = None` to train on all 30,000 samples

## How It Works (Simple Version)

1. The AI sees thousands of real underground images
2. It learns to remove noise from blurry versions
3. To generate: start with random noise, let the AI clean it up
4. Result: a realistic-looking geological cross-section

## How to Cite This Project

If you use this code in your research or projects, please mention it like this:

**Plain text:**
> Mohit (2026). Seismic Diffusion Model for Velocity Synthesis. GitHub repository. https://github.com/Mo-HIIT/Seismic-diffusion-model

**Or simply:**
> This work uses the seismic diffusion model by Mohit, available at https://github.com/Mo-HIIT/Seismic-diffusion-model

## Acknowledgments

Built with PyTorch. Trained on the OpenFWI CurveVel dataset.
