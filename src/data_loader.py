import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import BATCH_SIZE

# Define transformations for the dataset
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
    transforms.Resize((48, 48)),                  # Resize images to 48x48
    transforms.RandomHorizontalFlip(),            # Apply random horizontal flip
    transforms.RandomRotation(10),                # Apply random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),                        # Convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize images
])

def load_data(train_dir, test_dir):
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader
