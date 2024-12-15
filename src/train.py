import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from src.model import SimpleCNN, ResNetModel, EmotionCNN
from src.config import *
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader, val_loader):
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save the best model
        if val_accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_accuracy = val_accuracy
        
        # Step the scheduler
        scheduler.step()

    print("Training complete!")
    torch.save(model.state_dict(), MODEL_SAVE_PATH) 
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