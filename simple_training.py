#!/usr/bin/env python3
"""
Simple training script without transformers dependency
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import (
    BreakHisDataset,
    get_data_loaders,
    create_patient_splits
)
from efficientnet import EfficientNetB0Classifier

def main():
    # Configuration
    config = {
        'data_dir': 'data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast',
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"Using device: {config['device']}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    model = EfficientNetB0Classifier(num_classes=8, pretrained=True)
    model = model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{config["num_epochs"]}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config['device']), labels.to(config['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/efficientnet_b0_best.pth')
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
        
        print('-' * 50)
    
    print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    main()