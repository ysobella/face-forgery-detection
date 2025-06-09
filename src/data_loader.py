"""
Face Forgery Dataset Loader

Provides a PyTorch-compatible dataset and data loader functions for
loading real and fake face images from a structured directory.

Expected data structure:
root_dir/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceForgeryDataset(Dataset):
    """
    Custom dataset for loading real and fake face images.

    Args:
        root_dir (str): Base directory containing 'train', 'val', or 'test' folders.
        split (str): One of 'train', 'val', or 'test'.
        transform (callable, optional): Optional image transformations.
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): One of 'train', 'test', or 'val'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['real', 'fake']
        
        self.image_paths = []
        self.labels = []
        
        # Load real images
        real_dir = os.path.join(self.root_dir, 'real')
        for img_name in os.listdir(real_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(real_dir, img_name))
                self.labels.append(0)  # 0 for real
                
        # Load fake images
        fake_dir = os.path.join(self.root_dir, 'fake')
        for img_name in os.listdir(fake_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)  # 1 for fake

    def __len__(self):
        """Return the total number of images."""
        
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return an image and its label.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image tensor, label)
        """
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(root_dir, batch_size=8, num_workers=8):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        root_dir (str): Directory containing 'train', 'val', and 'test' folders.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Define transforms - only normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = FaceForgeryDataset(root_dir, split='train', transform=transform)
    val_dataset = FaceForgeryDataset(root_dir, split='val', transform=transform)
    test_dataset = FaceForgeryDataset(root_dir, split='test', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader 