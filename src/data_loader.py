import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import BATCH_SIZE

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((48, 48)),                  
    transforms.RandomHorizontalFlip(),            
    transforms.RandomRotation(10),                
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),                        
    transforms.Normalize(mean=[0.5], std=[0.5]),  
])

def load_data(train_dir, test_dir):
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, test_loader
