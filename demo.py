#!/usr/bin/env python3
"""
Quick demo script to test the trained UNet model.
Generates colored polygons for all available shapes and colors.
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from inference import load_model, predict, get_device
import json

def create_demo_grid(model, color_map_data, device, sample_dir='dataset/validation/inputs'):
    """Creates a grid visualization of all shapes with different colors."""
    
    # Get sample images
    image_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])[:4]
    
    # Get available colors
    if isinstance(color_map_data, dict) and 'color_to_idx' in color_map_data:
        colors = list(color_map_data['color_to_idx'].keys())[:6]  # Limit to 6 colors for display
    else:
        colors = list(color_map_data.keys())[:6]
    
    # Create figure
    fig, axes = plt.subplots(len(image_files) + 1, len(colors) + 1, 
                            figsize=(2.5 * (len(colors) + 1), 2.5 * (len(image_files) + 1)))
    
    # Hide all axes initially
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    
    # Add color labels
    for j, color in enumerate(colors):
        axes[0, j + 1].text(0.5, 0.5, color, ha='center', va='center', 
                           fontsize=12, transform=axes[0, j + 1].transAxes)
    
    # Process each shape
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(sample_dir, image_file)
        shape_name = os.path.splitext(image_file)[0]
        
        # Load and display original image
        original = Image.open(image_path).convert('RGB')
        axes[i + 1, 0].imshow(original)
        axes[i + 1, 0].set_title(shape_name, fontsize=10)
        
        # Generate colored versions
        for j, color in enumerate(colors):
            try:
                output_image, _ = predict(model, image_path, color, color_map_data, device)
                axes[i + 1, j + 1].imshow(output_image)
            except Exception as e:
                print(f"Error generating {shape_name} in {color}: {e}")
    
    plt.suptitle('UNet Polygon Coloring Demo', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save the grid
    output_path = 'demo_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Demo grid saved to {output_path}")
    plt.show()

def main():
    print("=== UNet Polygon Coloring Demo ===\n")
    
    # Check if model exists
    model_path = 'checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print("Error: No trained model found!")
        print("Please run 'python train.py' first to train the model.")
        return
    
    # Setup
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load color map
    try:
        with open('checkpoints/color_map.json', 'r') as f:
            color_map_data = json.load(f)
    except FileNotFoundError:
        print("Error: color_map.json not found. Please train the model first.")
        return
    
    # Extract number of colors
    if isinstance(color_map_data, dict) and 'color_to_idx' in color_map_data:
        num_colors = len(color_map_data['color_to_idx'])
        print(f"Available colors: {list(color_map_data['color_to_idx'].keys())}\n")
    else:
        num_colors = len(color_map_data)
        print(f"Available colors: {list(color_map_data.keys())}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, num_colors=num_colors, device=device)
    print("Model loaded successfully!\n")
    
    # Create demo grid
    print("Generating demo grid...")
    create_demo_grid(model, color_map_data, device)
    
    print("\nDemo complete! Check 'demo_grid.png' for results.")

if __name__ == "__main__":
    main()