import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class PolygonDataset(Dataset):
    """
    Custom PyTorch Dataset for loading polygon images, color names,
    and corresponding colored polygon images.
    """
    def __init__(self, root_dir, transform=None, image_size=(128, 128), augment=False, color_vocab=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.augment = augment
        self.data_map = self._load_json(os.path.join(root_dir, 'data.json'))
        
        # Create a vocabulary for color names
        if color_vocab is not None:
            # Use provided color vocabulary
            self.colors = color_vocab
            self.color_to_idx = {color: i for i, color in enumerate(self.colors)}
            self.idx_to_color = {i: color for i, color in enumerate(self.colors)}
        else:
            # Create vocabulary from this dataset
            self.colors = sorted(list(set(item['colour'] for item in self.data_map)))
            self.color_to_idx = {color: i for i, color in enumerate(self.colors)}
            self.idx_to_color = {i: color for i, color in enumerate(self.colors)}
        
        # Define standard transformations
        if self.transform is None:
            transform_list = [transforms.Resize(self.image_size)]
            
            if self.augment:
                # Enhanced augmentations for training
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomAffine(
                        degrees=0, 
                        translate=(0.1, 0.1), 
                        scale=(0.9, 1.1),
                        interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                ])
            
            transform_list.extend([
                transforms.ToTensor(),  # Scales images to [0.0, 1.0]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
            
            self.transform = transforms.Compose(transform_list)

    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data_map[idx]
        
        # Load input polygon image
        input_img_path = os.path.join(self.root_dir, 'inputs', item['input_polygon'])
        input_image = Image.open(input_img_path).convert('RGB')
        
        # Load output colored polygon image
        output_img_path = os.path.join(self.root_dir, 'outputs', item['output_image'])
        output_image = Image.open(output_img_path).convert('RGB')
        
        # Get color name and its index
        color_name = item['colour']
        color_idx = torch.tensor(self.color_to_idx[color_name], dtype=torch.long)
        
        # Apply same random seed for both input and output if augmenting
        if self.augment and self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            input_image = self.transform(input_image)
            torch.manual_seed(seed)
            output_image = self.transform(output_image)
        else:
            if self.transform:
                input_image = self.transform(input_image)
                output_image = self.transform(output_image)
            
        return input_image, color_idx, output_image, color_name