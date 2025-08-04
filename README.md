# UNet Polygon Coloring Project

This project implements a Conditional UNet architecture in PyTorch to generate colored polygon images. Given an input polygon shape (triangle, square, hexagon, etc.) and a color name, the model generates the polygon filled with the specified color.

## Features

- Full UNet implementation from scratch in PyTorch
- Color conditioning using embeddings
- Support for multiple polygon shapes and colors
- Automatic device detection (CUDA, MPS, CPU)
- Weights & Biases integration for experiment tracking
- Data augmentation for improved generalization
- Comprehensive inference tools

## Quick Start

### 1. Setup

```bash
# Clone the repository and navigate to it
cd AYNA-ML

# Or manually:
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch based on your system:
# For NVIDIA GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (MPS):
pip install torch torchvision torchaudio

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

### 2. Training

```bash
# Basic training with default parameters
python train.py

# Custom training configuration
python train.py --epochs no. of epoch  --batch_size batch size  --learning_rate learning rate  --img_size image size --wandb_project wandb project name   

# Training parameters:
#   --epochs: Number of training epochs (default: 50)
#   --batch_size: Batch size for training (default: 16)
#   --learning_rate: Learning rate (default: 0.001)
#   --img_size: Image size for training (default: 128)
#   --wandb_project: W&B project name (default: "unet-polygon-coloring")
```

### 3. Inference

```bash
# Single image inference
python inference.py --image_path dataset/validation/inputs/star.png --color blue

# With visualization
python inference.py --image_path dataset/validation/inputs/star.png --color red --visualize

# Batch inference on directory
python inference.py --batch --input_dir dataset/validation/inputs --output_dir results

# Generate all colors for one image
python inference.py --batch --image_path dataset/validation/inputs/triangle.png
```

### 4. Demo

```bash
# Run the demo to see a grid of all shapes with different colors
python demo.py
```

## Project Structure

```
AYNA-ML/
├── dataset/                # Training and validation data
│   ├── training/
│   │   ├── inputs/        # Input polygon images
│   │   ├── outputs/       # Target colored polygons
│   │   └── data.json      # Input-output mappings
│   └── validation/
├── unet_model.py          # UNet architecture implementation
├── dataset.py             # PyTorch Dataset class
├── train.py               # Training script
├── inference.py           # Inference utilities
├── demo.py                # Quick demo script
├── requirements.txt       # Python dependencies
└── checkpoints/           # Saved models (created during training)
```

## Model Architecture

The Conditional UNet consists of:
- **Encoder**: 4 downsampling blocks (64→128→256→512→512 channels)
- **Color Conditioning**: 32-dim embeddings injected at bottleneck
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Sigmoid activation for [0,1] range

## Tips for Best Results

1. **Training**:
   - Use a GPU for faster training (CUDA or MPS)
   - Monitor training on Weights & Biases
   - Early stopping is implemented via learning rate scheduling

2. **Inference**:
   - The model works best with images similar to training data
   - Use 128x128 image size for best results (or match training size)
   - Ensure input images have clear polygon shapes

3. **Customization**:
   - Adjust `color_embedding_dim` for different color representations
   - Modify augmentation in `dataset.py` for better generalization
   - Experiment with different learning rates and batch sizes