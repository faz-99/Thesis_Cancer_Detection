#!/usr/bin/env python3
"""
Dataset comparison utilities
Provides tools to compare model performance across different datasets
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def compare_dataset_performance(breakhis_model_path, combined_model_path, 
                              test_loaders_dict, class_mappings_dict, device):
    """
    Compare performance of models trained on different datasets
    
    Args:
        breakhis_model_path (str): Path to BreakHis-only trained model
        combined_model_path (str): Path to combined dataset trained model
        test_loaders_dict (dict): Dictionary of test loaders for each dataset
        class_mappings_dict (dict): Dictionary of class mappings for each model
        device: PyTorch device
        
    Returns:
        dict: Comparison results
    """
    from src.efficientnet import EfficientNetB0Classifier
    from src.train import evaluate_model
    
    logger.info("Comparing dataset performance...")
    
    results = {}
    
    # Load models
    logger.info("Loading models...")
    
    # BreakHis model
    breakhis_classes = len(class_mappings_dict['breakhis'])
    breakhis_model = EfficientNetB0Classifier(num_classes=breakhis_classes)
    breakhis_model.load_state_dict(torch.load(breakhis_model_path, map_location=device))
    breakhis_model = breakhis_model.to(device)
    breakhis_model.eval()
    
    # Combined model
    combined_classes = len(class_mappings_dict['combined'])
    combined_model = EfficientNetB0Classifier(num_classes=combined_classes)
    combined_model.load_state_dict(torch.load(combined_model_path, map_location=device))
    combined_model = combined_model.to(device)
    combined_model.eval()
    
    # Evaluate on each test set
    for dataset_name, test_loader in test_loaders_dict.items():
        logger.info(f"Evaluating on {dataset_name} test set...")
        
        results[dataset_name] = {}
        
        # Evaluate BreakHis model
        if dataset_name == 'breakhis':
            class_names = {v: k for k, v in class_mappings_dict['breakhis'].items()}
            breakhis_results = evaluate_model(breakhis_model, test_loader, device, class_names)
            results[dataset_name]['breakhis_only'] = breakhis_results
        
        # Evaluate combined model
        class_names = {v: k for k, v in class_mappings_dict['combined'].items()}
        combined_results = evaluate_model(combined_model, test_loader, device, class_names)
        results[dataset_name]['combined'] = combined_results
    
    return results

def plot_performance_comparison(results, save_path="performance_comparison.png"):
    """
    Plot performance comparison between models
    
    Args:
        results (dict): Results from compare_dataset_performance
        save_path (str): Path to save the plot
    """
    logger.info("Creating performance comparison plot...")
    
    # Extract accuracy scores
    accuracies = []
    models = []
    datasets = []
    
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            accuracies.append(model_results['accuracy'])
            models.append(model_name)
            datasets.append(dataset_name)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Accuracy': accuracies,
        'Model': models,
        'Dataset': datasets
    })
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model')
    plt.title('Model Performance Comparison Across Datasets')
    plt.ylabel('Accuracy')
    plt.xlabel('Test Dataset')
    plt.legend(title='Training Strategy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, (dataset, model) in enumerate(zip(datasets, models)):
        plt.text(i, accuracies[i] + 0.01, f'{accuracies[i]:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Performance comparison plot saved to: {save_path}")

def analyze_cross_dataset_generalization(model, source_loader, target_loader, 
                                       source_name, target_name, device):
    """
    Analyze how well a model trained on one dataset generalizes to another
    
    Args:
        model: Trained PyTorch model
        source_loader: DataLoader for source dataset (training domain)
        target_loader: DataLoader for target dataset (test domain)
        source_name (str): Name of source dataset
        target_name (str): Name of target dataset
        device: PyTorch device
        
    Returns:
        dict: Generalization analysis results
    """
    logger.info(f"Analyzing generalization from {source_name} to {target_name}...")
    
    model.eval()
    
    def get_predictions(loader):
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    # Get predictions for both datasets
    source_preds, source_labels, source_probs = get_predictions(source_loader)
    target_preds, target_labels, target_probs = get_predictions(target_loader)
    
    # Calculate accuracies
    source_acc = (source_preds == source_labels).mean()
    target_acc = (target_preds == target_labels).mean()
    
    # Calculate confidence scores
    source_confidence = np.max(source_probs, axis=1).mean()
    target_confidence = np.max(target_probs, axis=1).mean()
    
    # Domain gap analysis
    domain_gap = source_acc - target_acc
    confidence_gap = source_confidence - target_confidence
    
    results = {
        'source_accuracy': source_acc,
        'target_accuracy': target_acc,
        'domain_gap': domain_gap,
        'source_confidence': source_confidence,
        'target_confidence': target_confidence,
        'confidence_gap': confidence_gap,
        'generalization_ratio': target_acc / source_acc if source_acc > 0 else 0
    }
    
    logger.info(f"Generalization Analysis Results:")
    logger.info(f"  {source_name} accuracy: {source_acc:.4f}")
    logger.info(f"  {target_name} accuracy: {target_acc:.4f}")
    logger.info(f"  Domain gap: {domain_gap:.4f}")
    logger.info(f"  Generalization ratio: {results['generalization_ratio']:.4f}")
    
    return results

def create_dataset_statistics_report(breakhis_metadata, bach_metadata):
    """
    Create comprehensive statistics report for both datasets
    
    Args:
        breakhis_metadata (pd.DataFrame): BreakHis metadata
        bach_metadata (pd.DataFrame): BACH metadata
        
    Returns:
        dict: Statistics report
    """
    logger.info("Creating dataset statistics report...")
    
    report = {
        'breakhis': {
            'total_images': len(breakhis_metadata),
            'classes': breakhis_metadata['subclass'].nunique(),
            'class_distribution': breakhis_metadata['subclass'].value_counts().to_dict(),
            'magnifications': breakhis_metadata['magnification'].value_counts().to_dict(),
            'patients': breakhis_metadata['path'].apply(
                lambda x: x.split('_')[2] if '_' in x else 'unknown'
            ).nunique()
        },
        'bach': {
            'total_images': len(bach_metadata),
            'classes': bach_metadata['class'].nunique(),
            'class_distribution': bach_metadata['class'].value_counts().to_dict(),
            'resolution': 'High (2048x1536)',
            'source': 'ICIAR 2018 Challenge'
        }
    }
    
    # Combined statistics
    report['combined'] = {
        'total_images': report['breakhis']['total_images'] + report['bach']['total_images'],
        'datasets': 2,
        'unique_classes': len(set(breakhis_metadata['subclass'].unique()) | 
                            set(bach_metadata['class'].unique()))
    }
    
    # Print report
    print("\n" + "="*50)
    print("DATASET STATISTICS REPORT")
    print("="*50)
    
    for dataset_name, stats in report.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print("-" * 30)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    return report

def plot_class_distribution_comparison(breakhis_metadata, bach_metadata, 
                                     save_path="class_distribution_comparison.png"):
    """
    Plot class distribution comparison between datasets
    
    Args:
        breakhis_metadata (pd.DataFrame): BreakHis metadata
        bach_metadata (pd.DataFrame): BACH metadata
        save_path (str): Path to save the plot
    """
    logger.info("Creating class distribution comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # BreakHis class distribution
    breakhis_counts = breakhis_metadata['subclass'].value_counts()
    ax1.pie(breakhis_counts.values, labels=breakhis_counts.index, autopct='%1.1f%%')
    ax1.set_title('BreakHis Class Distribution')
    
    # BACH class distribution
    bach_counts = bach_metadata['class'].value_counts()
    ax2.pie(bach_counts.values, labels=bach_counts.index, autopct='%1.1f%%')
    ax2.set_title('BACH Class Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Class distribution comparison saved to: {save_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be called from main training scripts
    print("Dataset comparison utilities loaded successfully!")
    print("Use these functions in your training scripts to compare performance.")