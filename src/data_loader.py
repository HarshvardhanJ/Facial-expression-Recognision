import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from PIL import Image
from src.config import *

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    data.columns = data.columns.str.strip()
    
    print("DataFrame columns:", data.columns)
    
    if 'pixels' not in data.columns:
        raise KeyError("'pixels' column is missing from the dataset")
    
    X = np.array([np.fromstring(image_string, sep=' ').reshape(48, 48) for image_string in data['pixels']])
    y = data['emotion'].values
    
    X = X.astype('float32') / 255.0 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.reshape(-1, 1, 48, 48)
    X_test = X_test.reshape(-1, 1, 48, 48)

    X_train = [data_transforms(Image.fromarray((image.squeeze(0) * 255).astype(np.uint8))) for image in X_train]
    X_train = torch.stack(X_train)
    
    X_test = [data_transforms(Image.fromarray((image.squeeze(0) * 255).astype(np.uint8))) for image in X_test]
    X_test = torch.stack(X_test)

    train_dataset = TensorDataset(X_train, torch.from_numpy(y_train).long())
    test_dataset = TensorDataset(X_test, torch.from_numpy(y_test).long())

    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE)
