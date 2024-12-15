import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from src.model import SimpleCNN, ResNetModel
from src.config import *
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader, val_loader, model_type='SimpleCNN'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'SimpleCNN':
        model = SimpleCNN().to(device)
    elif model_type == 'ResNet':
        model = ResNetModel().to(device)
    else:
        raise ValueError("Invalid model type. Choose 'SimpleCNN' or 'ResNet'.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler(device=device)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type=str(device)):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'Model saved to {MODEL_SAVE_PATH}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Final model saved to {MODEL_SAVE_PATH}')
    return model

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    model.train()
    return val_loss / len(val_loader)

def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['SimpleCNN', 'ResNet'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    
    model = SimpleCNN().to(device) if model_type == 'SimpleCNN' else ResNetModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler(device=device)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type=str(device)):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss

def optimize_hyperparameters():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print('Best trial:', study.best_trial.params)
    return study.best_trial.params

if __name__ == '__main__':
    best_params = optimize_hyperparameters()
    print('Best hyperparameters:', best_params)