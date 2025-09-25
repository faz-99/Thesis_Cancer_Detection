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
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

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

def evaluate_model(model, test_loader, model_name, class_names, device='cuda'):
    """Evaluate model on test set and return comprehensive metrics"""
    print(f"\n📊 Evaluating {model_name}...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if model_name == "Ensemble":
                output, _ = model(data)
            else:
                output = model(data)
            
            # Get probabilities for ROC-AUC
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)
    
    # Overall metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )
    
    # Class-wise accuracy
    cm = confusion_matrix(all_targets, all_predictions)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # ROC-AUC metrics (one-vs-rest for multiclass)
    try:
        # Overall ROC-AUC (macro average)
        roc_auc_macro = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='weighted')
        
        # Per-class ROC-AUC
        roc_auc_per_class = []
        for i in range(len(class_names)):
            # Binary classification for each class (one-vs-rest)
            binary_targets = (np.array(all_targets) == i).astype(int)
            class_probs = all_probabilities[:, i]
            try:
                auc = roc_auc_score(binary_targets, class_probs)
                roc_auc_per_class.append(auc)
            except ValueError:
                # Handle case where class might not be present in test set
                roc_auc_per_class.append(0.0)
    except ValueError as e:
        print(f"Warning: ROC-AUC calculation failed: {e}")
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0
        roc_auc_per_class = [0.0] * len(class_names)
    
    # Classification report
    class_report = classification_report(all_targets, all_predictions, 
                                       target_names=class_names, output_dict=True)
    
    # Create detailed class-wise metrics dictionary
    class_wise_metrics = {}
    for i, class_name in enumerate(class_names):
        class_wise_metrics[class_name] = {
            'accuracy': float(class_accuracy[i]) if i < len(class_accuracy) else 0.0,
            'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
            'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
            'f1_score': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
            'support': int(support_per_class[i]) if i < len(support_per_class) else 0,
            'roc_auc': float(roc_auc_per_class[i]) if i < len(roc_auc_per_class) else 0.0
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'confusion_matrix': cm.tolist(),
        'class_wise_metrics': class_wise_metrics,
        'classification_report': class_report,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities.tolist()
    }

def save_confusion_matrix(cm, model_name, class_names):
    """Save confusion matrix plot with percentages"""
    plt.figure(figsize=(12, 10))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
    
    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_roc_curves(targets, probabilities, model_name, class_names):
    """Save ROC curves for each class (one-vs-rest)"""
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Convert targets to binary format for each class
    targets_binary = np.zeros((len(targets), n_classes))
    for i, target in enumerate(targets):
        targets_binary[i, target] = 1
    
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(targets_binary[:, i], np.array(probabilities)[:, i])
            roc_auc[i] = roc_auc_score(targets_binary[:, i], np.array(probabilities)[:, i])
        except ValueError:
            # Handle case where class might not be present
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 1])
            roc_auc[i] = 0.5
    
    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def save_class_wise_metrics(class_wise_metrics, model_name, class_names):
    """Save class-wise metrics as bar plots"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [class_wise_metrics[class_name][metric] for class_name in class_names]
        
        bars = ax.bar(range(len(class_names)), values, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Classes', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide the last subplot
    axes[-1].axis('off')
    
    plt.suptitle(f'{model_name} - Class-wise Performance Metrics', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_training_curves(history, model_name):
    """Save training loss and validation accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training loss
    ax1.plot(history['train_losses'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation accuracy
    ax2.plot(history['val_accuracies'], 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title(f'{model_name} - Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_plot(all_results, class_names):
    """Create comprehensive model comparison visualization"""
    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_macro']
    
    # Extract metrics for comparison
    comparison_data = {}
    for metric in metrics:
        comparison_data[metric] = []
        for model in models:
            if metric == 'accuracy':
                value = all_results[model]['test_metrics']['accuracy']
            elif metric == 'roc_auc_macro':
                value = all_results[model]['test_metrics']['roc_auc_macro']
            else:
                value = all_results[model]['test_metrics'][metric]
            comparison_data[metric].append(value)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall metrics comparison
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, comparison_data[metric], width, 
               label=metric.replace('_', ' ').title())
    
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Class-wise F1-score comparison
    class_f1_data = {}
    for model in models:
        class_f1_data[model] = []
        for class_name in class_names:
            f1 = all_results[model]['test_metrics']['class_wise_metrics'][class_name]['f1_score']
            class_f1_data[model].append(f1)
    
    x_classes = np.arange(len(class_names))
    width_class = 0.25
    
    for i, model in enumerate(models):
        ax2.bar(x_classes + i*width_class, class_f1_data[model], width_class, label=model)
    
    ax2.set_xlabel('Classes', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('Class-wise F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_classes + width_class)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
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
            num_epochs=20, device=device
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(f'models/{model_name.lower()}_best.pth'))
        model = model.to(device)
        
        # Evaluate model
        test_metrics = evaluate_model(model, test_loader, model_name, class_names, device)
        
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
        
        # Save ROC curves
        save_roc_curves(
            test_metrics['targets'], 
            test_metrics['probabilities'], 
            model_name, class_names
        )
        
        # Save class-wise metrics
        save_class_wise_metrics(
            test_metrics['class_wise_metrics'], 
            model_name, class_names
        )
        
        # Print detailed summary
        print(f"\n📈 {model_name} Results:")
        print(f"   Best Val Accuracy: {training_history['best_val_accuracy']:.2f}%")
        print(f"   Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"   Test Precision: {test_metrics['precision']:.4f}")
        print(f"   Test Recall: {test_metrics['recall']:.4f}")
        print(f"   Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC (Macro): {test_metrics['roc_auc_macro']:.4f}")
        print(f"   ROC-AUC (Weighted): {test_metrics['roc_auc_weighted']:.4f}")
        print(f"   Model Parameters: {results['model_params']:,}")
        
        # Print class-wise summary
        print(f"\n   📊 Class-wise Performance:")
        for class_name, metrics in test_metrics['class_wise_metrics'].items():
            print(f"      {class_name:<18}: Acc={metrics['accuracy']:.3f}, "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                  f"F1={metrics['f1_score']:.3f}, AUC={metrics['roc_auc']:.3f}")
    
    # Save combined results
    with open('results/all_models_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Val Acc':<8} {'Test Acc':<8} {'F1':<8} {'ROC-AUC':<8} {'Parameters':<12}")
    print("-" * 85)
    
    for model_name, results in all_results.items():
        val_acc = results['training_history']['best_val_accuracy']
        test_acc = results['test_metrics']['accuracy'] * 100
        f1_score = results['test_metrics']['f1_score']
        roc_auc = results['test_metrics']['roc_auc_macro']
        params = results['model_params']
        
        print(f"{model_name:<15} {val_acc:<8.2f} {test_acc:<8.2f} {f1_score:<8.4f} {roc_auc:<8.4f} {params:<12,}")
    
    # Create comprehensive comparison visualization
    create_model_comparison_plot(all_results, class_names)
    
    print(f"\n✅ All models trained and evaluated!")
    print(f"📁 Results saved in 'results/' directory")
    print(f"🏆 Models saved in 'models/' directory")
    print(f"📊 Visualizations include:")
    print(f"   - Confusion matrices with percentages")
    print(f"   - ROC curves for each class")
    print(f"   - Class-wise performance metrics")
    print(f"   - Training curves")
    print(f"   - Model comparison plots")

if __name__ == "__main__":
    main()