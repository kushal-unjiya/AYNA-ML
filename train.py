import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import json
from datetime import datetime
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
import math

from dataset import PolygonDataset
from unet_model import ConditionalUNet

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        x = (x - 0.5) * 2  # Normalize to [-1, 1]
        y = (y - 0.5) * 2
        
        x = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x
        y = y.repeat(1, 3, 1, 1) if y.size(1) == 1 else y
        
        x_relu1_2 = self.slice1(x)
        y_relu1_2 = self.slice1(y)
        
        x_relu2_2 = self.slice2(x_relu1_2)
        y_relu2_2 = self.slice2(y_relu1_2)
        
        x_relu3_3 = self.slice3(x_relu2_2)
        y_relu3_3 = self.slice3(y_relu2_2)
        
        loss = nn.MSELoss()(x_relu1_2, y_relu1_2) + \
               nn.MSELoss()(x_relu2_2, y_relu2_2) + \
               nn.MSELoss()(x_relu3_3, y_relu3_3)
        
        return loss

class CombinedLoss(nn.Module):
    """Combined L1, MSE and Perceptual Loss"""
    def __init__(self, l1_weight=1.0, mse_weight=0.5, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                     self.mse_weight * mse + 
                     self.perceptual_weight * perceptual)
        
        return total_loss, {
            'l1': l1.item(),
            'mse': mse.item(), 
            'perceptual': perceptual.item(),
            'total': total_loss.item()
        }

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-4):
    """Create a cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1 - min_lr) * cosine_decay + min_lr
        return decayed
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Apple MPS device detected")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def train_model(config):
    """Main training and validation loop."""
    device = get_device()
    print(f"Using device: {device}")

    # Initialize W&B
    run_name = f"unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=config.wandb_project, name=run_name, config=config)

    # Create training dataset first to get the color vocabulary
    train_dataset = PolygonDataset(
        root_dir='dataset/training', 
        image_size=(config.img_size, config.img_size),
        augment=True
    )
    
    # Use the training dataset's color vocabulary for validation dataset
    val_dataset = PolygonDataset(
        root_dir='dataset/validation', 
        image_size=(config.img_size, config.img_size),
        augment=False,
        color_vocab=train_dataset.colors
    )

    # Verify color maps are now consistent
    assert train_dataset.color_to_idx == val_dataset.color_to_idx, "Color maps mismatch between train/val sets"
    
    # Adjust batch size based on device to avoid OOM
    batch_size = config.batch_size
    if device.type == 'cuda':
        # Reduce batch size for GPU to avoid OOM
        batch_size = min(batch_size, 16)
    elif device.type == 'mps':
        # MPS might need smaller batch size
        batch_size = min(batch_size, 8)
    
    print(f"Using batch size: {batch_size}")
    
    # Use fewer workers for MPS to avoid issues
    num_workers = 4 if device.type != 'mps' else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0)
    )

    # Save color map for inference
    os.makedirs('checkpoints', exist_ok=True)
    color_map_path = 'checkpoints/color_map.json'
    with open(color_map_path, 'w') as f:
        json.dump({
            'color_to_idx': train_dataset.color_to_idx,
            'idx_to_color': train_dataset.idx_to_color,
            'colors': train_dataset.colors
        }, f, indent=2)
    print(f"Color map saved to {color_map_path}")
    print(f"Available colors: {train_dataset.colors}")
    
    # Initialize model, loss, and optimizer
    num_colors = len(train_dataset.colors)
    model = ConditionalUNet(
        n_channels=3, 
        n_classes=3, 
        num_colors=num_colors,
        color_embedding_dim=config.color_embedding_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    if device.type == 'mps':
        def criterion(pred, target):
            l1_loss = nn.L1Loss()(pred, target)
            mse_loss = nn.MSELoss()(pred, target)
            total_loss = l1_loss + 0.5 * mse_loss
            return total_loss, {
                'l1': l1_loss.item(),
                'mse': mse_loss.item(),
                'perceptual': 0.0,
                'total': total_loss.item()
            }
    else:
        criterion = CombinedLoss(
            l1_weight=1.0, 
            mse_weight=0.5, 
            perceptual_weight=0.1
        ).to(device)
    
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = len(train_loader) * 2
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=config.learning_rate * 0.01
    )
    
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    best_val_loss = float('inf')
    accumulation_steps = 4
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_loss_components = {'l1': 0, 'mse': 0, 'perceptual': 0, 'total': 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, (input_img, color_idx, target_img, color_names) in enumerate(pbar):
            input_img = input_img.to(device)
            color_idx = color_idx.to(device)
            target_img = target_img.to(device)

            if scaler and device.type == 'cuda':
                with autocast('cuda'):
                    outputs = model(input_img, color_idx)
                    loss, loss_components = criterion(outputs, target_img)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(input_img, color_idx)
                loss, loss_components = criterion(outputs, target_img)
                loss = loss / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            epoch_loss += loss.item() * accumulation_steps
            for key in loss_components:
                epoch_loss_components[key] += loss_components[key]
            
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log batch loss
            if batch_idx % 5 == 0:
                log_dict = {
                    "batch_train_loss": loss.item() * accumulation_steps,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1
                }
                log_dict.update({f"batch_{k}": v for k, v in loss_components.items()})
                wandb.log(log_dict)
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_loss_components = {k: v / len(train_loader) for k, v in epoch_loss_components.items()}

        # Validation Loop
        model.eval()
        val_loss = 0
        val_loss_components = {'l1': 0, 'mse': 0, 'perceptual': 0, 'total': 0}
        with torch.no_grad():
            for i, (input_img, color_idx, target_img, color_names) in enumerate(val_loader):
                input_img = input_img.to(device)
                color_idx = color_idx.to(device)
                target_img = target_img.to(device)
                
                outputs = model(input_img, color_idx)
                loss, loss_components = criterion(outputs, target_img)
                val_loss += loss.item()
                
                for key in loss_components:
                    val_loss_components[key] += loss_components[key]

                # Log example images to W&B on the first batch of every 3 epochs
                if i == 0 and epoch % 3 == 0:
                    # Log up to 4 examples
                    num_examples = min(4, input_img.size(0))
                    images_to_log = []
                    
                    for j in range(num_examples):
                        images_to_log.extend([
                            wandb.Image(input_img[j].cpu(), caption=f"Input ({color_names[j]})"),
                            wandb.Image(outputs[j].cpu(), caption=f"Generated ({color_names[j]})"),
                            wandb.Image(target_img[j].cpu(), caption=f"Target ({color_names[j]})")
                        ])
                    
                    wandb.log({"val_examples": images_to_log})
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
        
        # Log epoch metrics
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        log_dict.update({f"train_{k}": v for k, v in avg_loss_components.items()})
        log_dict.update({f"val_{k}": v for k, v in avg_val_loss_components.items()})
        wandb.log(log_dict)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = 'checkpoints/best_model.pth'
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
                'num_colors': num_colors
            }
            torch.save(checkpoint, model_path)
            wandb.save(model_path)
            print(f"✓ Best model saved (val_loss: {avg_val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = 'checkpoints/final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'num_colors': num_colors
    }, final_model_path)
    wandb.save(final_model_path)
    
    wandb.finish()
    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Enhanced Conditional UNet for coloring polygons.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (height and width).')
    parser.add_argument('--color_embedding_dim', type=int, default=64, help='Dimension of color embeddings.')
    parser.add_argument('--wandb_project', type=str, default="enhanced-unet-polygon-coloring", help="Weights & Biases project name.")
    
    args = parser.parse_args()
    train_model(args)