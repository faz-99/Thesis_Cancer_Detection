#!/usr/bin/env python3
"""
Cross-dataset evaluation script
Implements train-on-one, test-on-other evaluation for BreakHis and BACH datasets
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
import os
import sys

from src.efficientnet import EfficientNetB0Classifier
from src.data_utils import create_metadata, create_train_val_test_split, create_class_mappings, create_data_loaders
from src.bach_data_utils import create_bach_metadata, create_bach_data_loaders
from src.train import train_model, evaluate_model

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cross_dataset_evaluation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def train_on_breakhis_test_on_bach():
    """Train on BreakHis, test on BACH"""
    logger = logging.getLogger(__name__)
    logger.info("=== Training on BreakHis, Testing on BACH ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BreakHis for training
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    breakhis_metadata = create_metadata(breakhis_root)
    
    # Map BreakHis classes to BACH-compatible classes
    breakhis_to_bach_mapping = {
        'adenosis': 'Benign',
        'fibroadenoma': 'Benign', 
        'phyllodes_tumor': 'Benign',
        'tubular_adenoma': 'Benign',
        'ductal_carcinoma': 'Invasive',
        'lobular_carcinoma': 'Invasive',
        'mucinous_carcinoma': 'Invasive',
        'papillary_carcinoma': 'Invasive'
    }
    
    breakhis_metadata['bach_class'] = breakhis_metadata['subclass'].map(breakhis_to_bach_mapping)
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_split(
        breakhis_metadata, test_size=0.15, val_size=0.15, random_state=42
    )
    
    # Create class mappings for BACH classes
    bach_classes = ['Benign', 'Invasive']
    class_to_idx = {cls: idx for idx, cls in enumerate(bach_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Add class indices
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df['bach_class'].map(class_to_idx)
    
    # Calculate class weights
    from collections import Counter
    class_counts = Counter(train_df['bach_class'])
    total = sum(class_counts.values())
    class_weights = np.array([total / class_counts[cls] for cls in bach_classes], dtype=np.float32)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        train_df, val_df, test_df, class_weights_tensor, batch_size=32
    )
    
    # Train model
    model = EfficientNetB0Classifier(num_classes=len(bach_classes), pretrained=True)
    model = model.to(device)
    
    logger.info("Training model on BreakHis data...")
    model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=10, lr=1e-4, device=device
    )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/breakhis_to_bach_model.pth")
    
    # Test on BACH
    logger.info("Testing on BACH dataset...")
    bach_root = "data/bach"
    if os.path.exists(bach_root):
        bach_metadata = create_bach_metadata(bach_root)
        
        # Map BACH classes to our binary classification
        bach_to_binary = {
            'Normal': 'Benign',
            'Benign': 'Benign',
            'InSitu': 'Invasive',
            'Invasive': 'Invasive'
        }
        bach_metadata['binary_class'] = bach_metadata['class'].map(bach_to_binary)
        bach_metadata['class_idx'] = bach_metadata['binary_class'].map(class_to_idx)
        
        # Create BACH test loader
        dummy_weights = torch.ones(len(bach_classes))
        _, _, bach_test_loader = create_bach_data_loaders(
            bach_metadata[:1], bach_metadata[:1], bach_metadata, dummy_weights, 32
        )
        
        # Evaluate
        class_names = {v: k for k, v in class_to_idx.items()}
        results = evaluate_model(model, bach_test_loader, device, class_names)
        
        logger.info(f"Cross-dataset accuracy (BreakHis→BACH): {results['accuracy']:.4f}")
        logger.info("Classification Report:")
        logger.info(f"\n{results['classification_report']}")
        
        return results
    else:
        logger.warning("BACH dataset not found")
        return None

def train_on_bach_test_on_breakhis():
    """Train on BACH, test on BreakHis"""
    logger = logging.getLogger(__name__)
    logger.info("=== Training on BACH, Testing on BreakHis ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BACH for training
    bach_root = "data/bach"
    if not os.path.exists(bach_root):
        logger.warning("BACH dataset not found")
        return None
        
    bach_metadata = create_bach_metadata(bach_root)
    
    # Map BACH to binary classes
    bach_to_binary = {
        'Normal': 'Benign',
        'Benign': 'Benign', 
        'InSitu': 'Invasive',
        'Invasive': 'Invasive'
    }
    bach_metadata['binary_class'] = bach_metadata['class'].map(bach_to_binary)
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_split(
        bach_metadata, test_size=0.15, val_size=0.15, random_state=42
    )
    
    # Create class mappings
    binary_classes = ['Benign', 'Invasive']
    class_to_idx = {cls: idx for idx, cls in enumerate(binary_classes)}
    
    # Add class indices
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df['binary_class'].map(class_to_idx)
    
    # Calculate class weights
    from collections import Counter
    class_counts = Counter(train_df['binary_class'])
    total = sum(class_counts.values())
    class_weights = np.array([total / class_counts[cls] for cls in binary_classes], dtype=np.float32)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    # Create data loaders
    train_loader, val_loader, _ = create_bach_data_loaders(
        train_df, val_df, test_df, class_weights_tensor, 32
    )
    
    # Train model
    model = EfficientNetB0Classifier(num_classes=len(binary_classes), pretrained=True)
    model = model.to(device)
    
    logger.info("Training model on BACH data...")
    model, history = train_model(
        model, train_loader, val_loader,
        num_epochs=10, lr=1e-4, device=device
    )
    
    # Save model
    torch.save(model.state_dict(), "models/bach_to_breakhis_model.pth")
    
    # Test on BreakHis
    logger.info("Testing on BreakHis dataset...")
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    breakhis_metadata = create_metadata(breakhis_root)
    
    # Map BreakHis to binary classes
    breakhis_to_binary = {
        'adenosis': 'Benign',
        'fibroadenoma': 'Benign',
        'phyllodes_tumor': 'Benign', 
        'tubular_adenoma': 'Benign',
        'ductal_carcinoma': 'Invasive',
        'lobular_carcinoma': 'Invasive',
        'mucinous_carcinoma': 'Invasive',
        'papillary_carcinoma': 'Invasive'
    }
    breakhis_metadata['binary_class'] = breakhis_metadata['subclass'].map(breakhis_to_binary)
    breakhis_metadata['class_idx'] = breakhis_metadata['binary_class'].map(class_to_idx)
    
    # Create BreakHis test loader
    dummy_weights = torch.ones(len(binary_classes))
    _, _, breakhis_test_loader = create_data_loaders(
        breakhis_metadata[:1], breakhis_metadata[:1], breakhis_metadata, dummy_weights, 32
    )
    
    # Evaluate
    class_names = {v: k for k, v in class_to_idx.items()}
    results = evaluate_model(model, breakhis_test_loader, device, class_names)
    
    logger.info(f"Cross-dataset accuracy (BACH→BreakHis): {results['accuracy']:.4f}")
    logger.info("Classification Report:")
    logger.info(f"\n{results['classification_report']}")
    
    return results

def plot_cross_dataset_results(results_dict, save_path="cross_dataset_results.png"):
    """Plot cross-dataset evaluation results"""
    logger = logging.getLogger(__name__)
    logger.info("Creating cross-dataset results plot...")
    
    # Extract data for plotting
    scenarios = []
    accuracies = []
    
    for scenario, results in results_dict.items():
        if results is not None:
            scenarios.append(scenario)
            accuracies.append(results['accuracy'])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, accuracies, color=['skyblue', 'lightcoral'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Cross-Dataset Generalization Performance', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Training → Testing Scenario', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Cross-dataset results plot saved to: {save_path}")

def main():
    logger = setup_logging()
    logger.info("Starting cross-dataset evaluation...")
    
    results = {}
    
    # Train on BreakHis, test on BACH
    results['BreakHis → BACH'] = train_on_breakhis_test_on_bach()
    
    # Train on BACH, test on BreakHis  
    results['BACH → BreakHis'] = train_on_bach_test_on_breakhis()
    
    # Plot results
    plot_cross_dataset_results(results)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("CROSS-DATASET EVALUATION SUMMARY")
    logger.info("="*50)
    
    for scenario, result in results.items():
        if result is not None:
            logger.info(f"{scenario}: {result['accuracy']:.4f}")
        else:
            logger.info(f"{scenario}: Dataset not available")
    
    logger.info("Cross-dataset evaluation completed!")

if __name__ == "__main__":
    main()