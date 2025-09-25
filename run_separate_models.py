#!/usr/bin/env python3
"""
Run and evaluate models separately: EfficientNet, Swin Transformer, and Ensemble
Save results and models individually for comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import models and utilities
from src.efficientnet import EfficientNetB0Classifier
from src.advanced_models import SwinTransformerClassifier, EnsembleModel
from src.data_utils import create_breakhis_loaders

def train_model(model, train_loader, val_loader, model_name, num_epochs=50, device='cuda'):
    """Train a single model and return metrics"""
    print(f"\n🚀 Training {model_name}...")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if model_name == "Ensemble":
                output, _ = model(data)  # Ensemble returns tuple
            else:
                output = model(data)
                
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                if model_name == "Ensemble":
                    output, _ = model(data)
                else:
                    output = model(data)
                    
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_train_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name.lower()}_best.pth')
            print(f'✅ New best model saved: {val_acc:.2f}%')
    
    return {
        'best_val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader, model_name, device='cuda'):
    """Evaluate model on test set and return detailed metrics"""
    print(f"\n📊 Evaluating {model_name}...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if model_name == "Ensemble":
                output, _ = model(data)
            else:
                output = model(data)
                
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'targets': all_targets
    }

def save_confusion_matrix(cm, model_name, class_names):
    """Save confusion matrix plot"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_training_curves(history, model_name):
    """Save training loss and validation accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training loss
    ax1.plot(history['train_losses'])
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Validation accuracy
    ax2.plot(history['val_accuracies'])
    ax2.set_title(f'{model_name} - Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("📂 Loading BreakHis dataset...")
    train_loader, val_loader, test_loader = create_breakhis_loaders(
        data_root="data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
        batch_size=32,
        num_workers=4
    )
    
    # Class names for BreakHis
    class_names = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
                   'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
    
    # Models to train
    models_config = [
        {
            'name': 'EfficientNetB0',
            'model': EfficientNetB0Classifier(num_classes=8, pretrained=True)
        },
        {
            'name': 'SwinTransformer', 
            'model': SwinTransformerClassifier(num_classes=8, model_size='tiny')
        },
        {
            'name': 'Ensemble',
            'model': EnsembleModel(num_classes=8)
        }
    ]
    
    all_results = {}
    
    # Train and evaluate each model
    for config in models_config:
        model_name = config['name']
        model = config['model']
        
        print(f"\n{'='*60}")
        print(f"🔥 Processing {model_name}")
        print(f"{'='*60}")
        
        # Train model
        training_history = train_model(
            model, train_loader, val_loader, model_name, 
            num_epochs=30, device=device
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(f'models/{model_name.lower()}_best.pth'))
        model = model.to(device)
        
        # Evaluate model
        test_metrics = evaluate_model(model, test_loader, model_name, device)
        
        # Save results
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'training_history': training_history,
            'test_metrics': test_metrics,
            'model_params': sum(p.numel() for p in model.parameters()),
            'device': str(device)
        }
        
        all_results[model_name] = results
        
        # Save individual results
        with open(f'results/{model_name.lower()}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save visualizations
        save_confusion_matrix(
            np.array(test_metrics['confusion_matrix']), 
            model_name, class_names
        )
        save_training_curves(training_history, model_name)
        
        # Print summary
        print(f"\n📈 {model_name} Results:")
        print(f"   Best Val Accuracy: {training_history['best_val_accuracy']:.2f}%")
        print(f"   Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"   Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"   Model Parameters: {results['model_params']:,}")
    
    # Save combined results
    with open('results/all_models_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Val Acc':<10} {'Test Acc':<10} {'F1-Score':<10} {'Parameters':<12}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        val_acc = results['training_history']['best_val_accuracy']
        test_acc = results['test_metrics']['accuracy'] * 100
        f1_score = results['test_metrics']['f1_score']
        params = results['model_params']
        
        print(f"{model_name:<15} {val_acc:<10.2f} {test_acc:<10.2f} {f1_score:<10.4f} {params:<12,}")
    
    print(f"\n✅ All models trained and evaluated!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"🏆 Models saved in 'models/' directory")

if __name__ == "__main__":
    main()