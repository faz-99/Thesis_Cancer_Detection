#!/usr/bin/env python3
"""
Advanced Cross-Dataset Training for 90%+ Performance

Combines multiple state-of-the-art techniques:
1. Domain Adversarial Neural Networks (DANN)
2. Enhanced preprocessing with stain normalization
3. Progressive domain transfer
4. Test-time augmentation
5. Ensemble methods
6. Uncertainty-guided training
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from src.efficientnet import EfficientNetB0Classifier
from src.data_utils import create_metadata, create_train_val_test_split, create_data_loaders
from src.bach_data_utils import create_bach_metadata, create_bach_data_loaders
from src.advanced_domain_adaptation import (
    train_with_domain_adaptation, 
    evaluate_domain_adaptation_model,
    AdvancedDomainAdaptationModel
)
from src.enhanced_preprocessing import get_enhanced_transforms, get_test_time_augmentation_transforms

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('advanced_cross_dataset.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class TestTimeAugmentation:
    """Test-time augmentation for improved accuracy"""
    
    def __init__(self, model, transforms, device):
        self.model = model
        self.transforms = transforms
        self.device = device
    
    def predict(self, image):
        """Predict with test-time augmentation"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for transform in self.transforms:
                # Apply transform
                augmented = transform(image).unsqueeze(0).to(self.device)
                
                # Get prediction
                if hasattr(self.model, 'forward'):
                    output = self.model(augmented)
                    if isinstance(output, tuple):
                        output = output[0]  # Get class logits
                else:
                    output = self.model(augmented)
                
                # Convert to probabilities
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction

class EnsembleModel:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, models, device):
        self.models = models
        self.device = device
    
    def predict(self, dataloader):
        """Predict using ensemble of models"""
        all_predictions = []
        all_labels = []
        
        for images, labels, _ in tqdm(dataloader, desc="Ensemble prediction"):
            images = images.to(self.device)
            batch_predictions = []
            
            # Get predictions from each model
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    output = model(images)
                    if isinstance(output, tuple):
                        output = output[0]  # Get class logits
                    
                    probs = torch.softmax(output, dim=1)
                    batch_predictions.append(probs.cpu().numpy())
            
            # Average ensemble predictions
            ensemble_pred = np.mean(batch_predictions, axis=0)
            all_predictions.extend(ensemble_pred)
            all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels)

def create_enhanced_data_loaders(metadata, dataset_type, batch_size=32):
    """Create data loaders with enhanced preprocessing"""
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_split(
        metadata, test_size=0.15, val_size=0.15, random_state=42
    )
    
    # Get enhanced transforms
    train_transform = get_enhanced_transforms(dataset_type, is_training=True)
    val_transform = get_enhanced_transforms(dataset_type, is_training=False)
    
    # Create datasets with enhanced transforms
    from torch.utils.data import Dataset
    from PIL import Image
    
    class EnhancedDataset(Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe.reset_index(drop=True)
            self.transform = transform
        
        def __len__(self):
            return len(self.dataframe)
        
        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            
            # Load image
            image = Image.open(row['image_path']).convert('RGB')
            
            # Apply transform
            if self.transform:
                image = self.transform(image)
            
            return image, row['class_idx'], row['image_path']
    
    # Create datasets
    train_dataset = EnhancedDataset(train_df, train_transform)
    val_dataset = EnhancedDataset(val_df, val_transform)
    test_dataset = EnhancedDataset(test_df, val_transform)
    
    # Calculate class weights
    from collections import Counter
    class_counts = Counter(train_df['class_idx'])
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weights = torch.FloatTensor([
        total / class_counts[i] for i in range(num_classes)
    ])
    
    # Create weighted sampler
    sample_weights = [class_weights[label] for label in train_df['class_idx']]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights

def train_advanced_breakhis_to_bach():
    """Train on BreakHis with advanced techniques, test on BACH"""
    logger = logging.getLogger(__name__)
    logger.info("=== Advanced BreakHis → BACH Training ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load BreakHis data
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    breakhis_metadata = create_metadata(breakhis_root)
    
    # Map to binary classes
    breakhis_to_binary = {
        'adenosis': 0, 'fibroadenoma': 0, 'phyllodes_tumor': 0, 'tubular_adenoma': 0,
        'ductal_carcinoma': 1, 'lobular_carcinoma': 1, 'mucinous_carcinoma': 1, 'papillary_carcinoma': 1
    }
    breakhis_metadata['class_idx'] = breakhis_metadata['subclass'].map(breakhis_to_binary)
    
    # Create enhanced data loaders
    train_loader, val_loader, _, class_weights = create_enhanced_data_loaders(
        breakhis_metadata, 'breakhis', batch_size=32
    )
    
    # Load BACH data for target domain
    bach_root = "data/bach"
    if not os.path.exists(bach_root):
        logger.error("BACH dataset not found!")
        return None
    
    bach_metadata = create_bach_metadata(bach_root)
    bach_to_binary = {'Normal': 0, 'Benign': 0, 'InSitu': 1, 'Invasive': 1}
    bach_metadata['class_idx'] = bach_metadata['class'].map(bach_to_binary)
    
    # Create BACH loaders (for domain adaptation)
    bach_train_loader, bach_val_loader, bach_test_loader, _ = create_enhanced_data_loaders(
        bach_metadata, 'bach', batch_size=32
    )
    
    # Train with domain adaptation
    logger.info("Training with advanced domain adaptation...")
    model, history = train_with_domain_adaptation(
        source_loader=train_loader,
        target_loader=bach_train_loader,
        val_loader=bach_val_loader,
        num_classes=2,
        num_epochs=25,
        device=device
    )
    
    # Evaluate with test-time augmentation
    logger.info("Evaluating with test-time augmentation...")
    tta_transforms = get_test_time_augmentation_transforms()
    
    # Test on BACH
    model.eval()
    correct = 0
    total = 0
    
    for images, labels, image_paths in tqdm(bach_test_loader, desc="TTA Evaluation"):
        batch_predictions = []
        
        for i in range(images.size(0)):
            # Get single image
            single_image_path = image_paths[i]
            single_image = Image.open(single_image_path).convert('RGB')
            
            # Apply TTA
            tta = TestTimeAugmentation(model, tta_transforms, device)
            pred_probs = tta.predict(single_image)
            predicted_class = np.argmax(pred_probs)
            
            batch_predictions.append(predicted_class)
        
        # Compare with true labels
        labels_np = labels.numpy()
        correct += np.sum(np.array(batch_predictions) == labels_np)
        total += len(labels_np)
    
    accuracy = correct / total
    logger.info(f"Advanced BreakHis → BACH Accuracy: {accuracy:.4f}")
    
    return accuracy, model, history

def train_advanced_bach_to_breakhis():
    """Train on BACH with advanced techniques, test on BreakHis"""
    logger = logging.getLogger(__name__)
    logger.info("=== Advanced BACH → BreakHis Training ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BACH data
    bach_root = "data/bach"
    if not os.path.exists(bach_root):
        logger.error("BACH dataset not found!")
        return None
    
    bach_metadata = create_bach_metadata(bach_root)
    bach_to_binary = {'Normal': 0, 'Benign': 0, 'InSitu': 1, 'Invasive': 1}
    bach_metadata['class_idx'] = bach_metadata['class'].map(bach_to_binary)
    
    # Create enhanced data loaders
    train_loader, val_loader, _, class_weights = create_enhanced_data_loaders(
        bach_metadata, 'bach', batch_size=32
    )
    
    # Load BreakHis data for target domain
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    breakhis_metadata = create_metadata(breakhis_root)
    breakhis_to_binary = {
        'adenosis': 0, 'fibroadenoma': 0, 'phyllodes_tumor': 0, 'tubular_adenoma': 0,
        'ductal_carcinoma': 1, 'lobular_carcinoma': 1, 'mucinous_carcinoma': 1, 'papillary_carcinoma': 1
    }
    breakhis_metadata['class_idx'] = breakhis_metadata['subclass'].map(breakhis_to_binary)
    
    # Create BreakHis loaders
    breakhis_train_loader, breakhis_val_loader, breakhis_test_loader, _ = create_enhanced_data_loaders(
        breakhis_metadata, 'breakhis', batch_size=32
    )
    
    # Train with domain adaptation
    logger.info("Training with advanced domain adaptation...")
    model, history = train_with_domain_adaptation(
        source_loader=train_loader,
        target_loader=breakhis_train_loader,
        val_loader=breakhis_val_loader,
        num_classes=2,
        num_epochs=25,
        device=device
    )
    
    # Evaluate with test-time augmentation
    logger.info("Evaluating with test-time augmentation...")
    tta_transforms = get_test_time_augmentation_transforms()
    
    # Test on BreakHis
    model.eval()
    correct = 0
    total = 0
    
    for images, labels, image_paths in tqdm(breakhis_test_loader, desc="TTA Evaluation"):
        batch_predictions = []
        
        for i in range(images.size(0)):
            # Get single image
            single_image_path = image_paths[i]
            single_image = Image.open(single_image_path).convert('RGB')
            
            # Apply TTA
            tta = TestTimeAugmentation(model, tta_transforms, device)
            pred_probs = tta.predict(single_image)
            predicted_class = np.argmax(pred_probs)
            
            batch_predictions.append(predicted_class)
        
        # Compare with true labels
        labels_np = labels.numpy()
        correct += np.sum(np.array(batch_predictions) == labels_np)
        total += len(labels_np)
    
    accuracy = correct / total
    logger.info(f"Advanced BACH → BreakHis Accuracy: {accuracy:.4f}")
    
    return accuracy, model, history

def train_ensemble_models():
    """Train ensemble of models for maximum performance"""
    logger = logging.getLogger(__name__)
    logger.info("=== Training Ensemble Models ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train multiple models with different configurations
    models = []
    
    # Model 1: BreakHis → BACH
    logger.info("Training Model 1: BreakHis → BACH")
    acc1, model1, _ = train_advanced_breakhis_to_bach()
    if model1:
        models.append(model1)
    
    # Model 2: BACH → BreakHis  
    logger.info("Training Model 2: BACH → BreakHis")
    acc2, model2, _ = train_advanced_bach_to_breakhis()
    if model2:
        models.append(model2)
    
    # Save ensemble
    if models:
        os.makedirs("models", exist_ok=True)
        for i, model in enumerate(models):
            torch.save(model.state_dict(), f"models/ensemble_model_{i+1}.pth")
        
        logger.info(f"Ensemble of {len(models)} models saved")
        logger.info(f"Individual accuracies: {acc1:.4f}, {acc2:.4f}")
    
    return models

def plot_advanced_results(results, save_path="advanced_cross_dataset_results.png"):
    """Plot advanced cross-dataset results"""
    scenarios = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    bars = plt.bar(scenarios, accuracies, 
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add 90% target line
    plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(0.5, 0.91, '90% Target', ha='center', color='red', fontweight='bold')
    
    plt.title('Advanced Cross-Dataset Performance\n(Domain Adaptation + Enhanced Preprocessing + TTA)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Training → Testing Scenario', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run advanced cross-dataset training"""
    logger = setup_logging()
    logger.info("Starting Advanced Cross-Dataset Training for 90%+ Performance")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    results = {}
    
    try:
        # Train BreakHis → BACH
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Advanced BreakHis → BACH")
        logger.info("="*60)
        
        acc1, model1, history1 = train_advanced_breakhis_to_bach()
        if acc1:
            results['Advanced BreakHis → BACH'] = acc1
        
        # Train BACH → BreakHis
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Advanced BACH → BreakHis")
        logger.info("="*60)
        
        acc2, model2, history2 = train_advanced_bach_to_breakhis()
        if acc2:
            results['Advanced BACH → BreakHis'] = acc2
        
        # Plot results
        if results:
            plot_advanced_results(results)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("ADVANCED CROSS-DATASET TRAINING SUMMARY")
        logger.info("="*60)
        
        for scenario, accuracy in results.items():
            status = "✅ TARGET ACHIEVED" if accuracy >= 0.9 else "❌ Below Target"
            logger.info(f"{scenario}: {accuracy:.1%} - {status}")
        
        if results:
            avg_accuracy = np.mean(list(results.values()))
            logger.info(f"\nAverage Cross-Dataset Accuracy: {avg_accuracy:.1%}")
            
            if avg_accuracy >= 0.9:
                logger.info("🎉 90%+ TARGET ACHIEVED! 🎉")
            else:
                logger.info(f"Target not reached. Need {0.9 - avg_accuracy:.1%} more accuracy.")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    logger.info("Advanced cross-dataset training completed!")

if __name__ == "__main__":
    main()