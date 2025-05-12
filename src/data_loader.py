import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceForgeryDataset(Dataset):
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(root_dir, batch_size=8, num_workers=8):
    """
    Create data loaders for train, validation and test sets
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