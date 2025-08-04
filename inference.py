import argparse
import torch
from PIL import Image
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import os

from unet_model import ConditionalUNet

def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def load_model(model_path, num_colors, device, color_embedding_dim=64):
    """Loads the trained model weights."""
    model = ConditionalUNet(
        n_channels=3, 
        n_classes=3, 
        num_colors=num_colors,
        color_embedding_dim=color_embedding_dim
    )
    
    # Load checkpoint (handle PyTorch 2.6+ security changes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, color_name, color_map_data, device, img_size=128):
    """Generates a colored polygon from an input image and color name."""
    # Extract color mapping
    if isinstance(color_map_data, dict) and 'color_to_idx' in color_map_data:
        color_map = color_map_data['color_to_idx']
    else:
        color_map = color_map_data
    
    # Preprocess the input image (match training normalization)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Get color index
    if color_name not in color_map:
        available_colors = list(color_map.keys())
        raise ValueError(f"Color '{color_name}' not found. Available colors: {available_colors}")
    
    color_idx = torch.tensor([color_map[color_name]], dtype=torch.long).to(device)

    # Perform inference
    with torch.no_grad():
        output_tensor = model(image_tensor, color_idx)
    
    # Post-process the output tensor to a PIL image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    
    return output_image, image

def visualize_result(input_image, output_image, color_name, save_path=None):
    """Visualizes the input and output images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(input_image)
    axes[0].set_title('Input Polygon')
    axes[0].axis('off')
    
    axes[1].imshow(output_image)
    axes[1].set_title(f'Generated ({color_name})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def batch_inference(model, input_dir, colors, color_map_data, device, output_dir='outputs', img_size=128):
    """Performs batch inference on multiple images and colors."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        for color in colors:
            try:
                output_image, input_image = predict(model, image_path, color, color_map_data, device, img_size)
                
                # Save output image
                output_filename = f"{base_name}_{color}.png"
                output_path = os.path.join(output_dir, output_filename)
                output_image.save(output_path)
                print(f"Generated: {output_filename}")
                
            except Exception as e:
                print(f"Error processing {image_file} with color {color}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for the Conditional UNet.")
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', 
                       help='Path to the trained model checkpoint.')
    parser.add_argument('--image_path', type=str, help='Path to the input polygon image.')
    parser.add_argument('--color', type=str, help='Desired color name (e.g., "red", "blue").')
    parser.add_argument('--output_path', type=str, default='generated_image.png', 
                       help='Path to save the output image.')
    parser.add_argument('--img_size', type=int, default=128, help='Image size used during training.')
    parser.add_argument('--visualize', action='store_true', help='Show visualization of results.')
    parser.add_argument('--batch', action='store_true', help='Run batch inference on a directory.')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images for batch mode.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save batch outputs.')

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load the color map saved during training
    try:
        with open('checkpoints/color_map.json', 'r') as f:
            color_map_data = json.load(f)
    except FileNotFoundError:
        print("Error: color_map.json not found. Please run train.py first to generate it.")
        exit(1)

    # Extract available colors
    if isinstance(color_map_data, dict) and 'color_to_idx' in color_map_data:
        available_colors = list(color_map_data['color_to_idx'].keys())
        num_colors = len(available_colors)
    else:
        available_colors = list(color_map_data.keys())
        num_colors = len(available_colors)
    
    print(f"Available colors: {available_colors}")

    # Load model
    model = load_model(args.model_path, num_colors=num_colors, device=device)
    print("Model loaded successfully.")

    if args.batch:
        # Batch inference mode
        if not args.input_dir:
            print("Error: --input_dir required for batch mode")
            exit(1)
        
        # Use all available colors if not specified
        colors_to_use = [args.color] if args.color else available_colors
        batch_inference(model, args.input_dir, colors_to_use, color_map_data, device, 
                       args.output_dir, args.img_size)
    else:
        # Single inference mode
        if not args.image_path or not args.color:
            print("Error: --image_path and --color required for single inference mode")
            exit(1)
        
        # Generate image
        generated_image, input_image = predict(model, args.image_path, args.color, 
                                              color_map_data, device, args.img_size)
        
        # Save the output
        generated_image.save(args.output_path)
        print(f"Generated image saved to {args.output_path}")
        
        # Visualize if requested
        if args.visualize:
            visualize_result(input_image, generated_image, args.color)