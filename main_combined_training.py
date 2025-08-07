#!/usr/bin/env python3
"""
Combined training script for BreakHis + BACH datasets
Implements multi-dataset training for improved generalization
"""

import torch
import pandas as pd
import logging
import sys
import os
from src.bach_data_utils import create_combined_metadata, create_bach_data_loaders
from src.data_utils import create_train_val_test_split, create_class_mappings
from src.efficientnet import EfficientNetB0Classifier
from src.train import train_model, evaluate_model

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('combined_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    BREAKHIS_ROOT = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    BACH_ROOT = "data/bach"  # Update this path to your BACH dataset location
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {DEVICE}")
    
    # Check if datasets exist
    if not os.path.exists(BREAKHIS_ROOT):
        logger.error(f"BreakHis dataset not found at: {BREAKHIS_ROOT}")
        return
    
    if not os.path.exists(BACH_ROOT):
        logger.warning(f"BACH dataset not found at: {BACH_ROOT}")
        logger.info("Training with BreakHis only...")
        use_bach = False
    else:
        use_bach = True
        logger.info("Training with combined BreakHis + BACH datasets")
    
    # 1. Create combined metadata
    if use_bach:
        logger.info("Creating combined metadata...")
        metadata = create_combined_metadata(BREAKHIS_ROOT, BACH_ROOT)
        class_column = 'unified_class'
    else:
        from src.data_utils import create_metadata
        logger.info("Creating BreakHis metadata...")
        metadata = create_metadata(BREAKHIS_ROOT)
        metadata['unified_class'] = metadata['subclass']
        class_column = 'unified_class'
    
    logger.info(f"Total images: {len(metadata)}")
    
    # 2. Create splits
    logger.info("Creating train/val/test splits...")
    # For combined dataset, we'll use image-level splitting since BACH doesn't have patient IDs
    train_df, val_df, test_df = create_train_val_test_split(
        metadata, test_size=0.15, val_size=0.15, random_state=42
    )
    
    # 3. Create class mappings based on unified classes
    class_to_idx, idx_to_class, class_weights_tensor = create_class_mappings_unified(
        train_df, class_column
    )
    
    # Add class indices to dataframes
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df[class_column].map(class_to_idx)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Number of classes: {len(class_to_idx)}")
    logger.info(f"Classes: {list(class_to_idx.keys())}")
    
    # 4. Create data loaders
    logger.info("Creating data loaders...")
    if use_bach:
        train_loader, val_loader, test_loader = create_bach_data_loaders(
            train_df, val_df, test_df, class_weights_tensor, BATCH_SIZE
        )
    else:
        from src.data_utils import create_data_loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df, class_weights_tensor, BATCH_SIZE
        )
    
    # 5. Initialize model
    logger.info("Initializing model...")
    model = EfficientNetB0Classifier(num_classes=len(class_to_idx), pretrained=True)
    model = model.to(DEVICE)
    
    # 6. Train model
    logger.info("Starting training...")
    model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )
    
    # 7. Save model
    model_name = "combined_efficientnet_b0_best.pth" if use_bach else "breakhis_efficientnet_b0_best.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}")
    logger.info(f"Model saved to models/{model_name}")
    
    # 8. Evaluate on test set
    logger.info("Evaluating on test set...")
    class_names = {v: k for k, v in class_to_idx.items()}
    test_results = evaluate_model(model, test_loader, DEVICE, class_names)
    
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info("Classification Report:")
    logger.info(f"\n{test_results['classification_report']}")
    
    # 9. Dataset-specific evaluation if using combined data
    if use_bach:
        logger.info("Evaluating by dataset...")
        evaluate_by_dataset(model, test_df, class_to_idx, DEVICE)
    
    logger.info("Training completed successfully!")

def create_class_mappings_unified(train_df, class_column):
    """Create class mappings for unified classes"""
    from collections import Counter
    import numpy as np
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating unified class mappings from column: {class_column}")
    
    class_counts = Counter(train_df[class_column])
    classes = sorted(class_counts.keys())
    
    logger.info(f"Found {len(classes)} unified classes: {classes}")
    logger.info(f"Class counts: {dict(class_counts)}")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Calculate class weights
    total = sum(class_counts.values())
    class_weights = np.array([total / class_counts[cls] for cls in classes], dtype=np.float32)
    class_weights = class_weights / class_weights.sum() * len(classes)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info(f"Class weights: {dict(zip(classes, class_weights))}")
    
    return class_to_idx, idx_to_class, class_weights_tensor

def evaluate_by_dataset(model, test_df, class_to_idx, device):
    """Evaluate model performance separately for each dataset"""
    logger = logging.getLogger(__name__)
    
    # Split test data by dataset
    breakhis_test = test_df[test_df['dataset'] == 'BreakHis']
    bach_test = test_df[test_df['dataset'] == 'BACH']
    
    logger.info(f"BreakHis test samples: {len(breakhis_test)}")
    logger.info(f"BACH test samples: {len(bach_test)}")
    
    # Create separate loaders for each dataset
    from src.bach_data_utils import create_bach_data_loaders
    
    if len(breakhis_test) > 0:
        # Create dummy class weights for evaluation
        dummy_weights = torch.ones(len(class_to_idx))
        _, _, breakhis_loader = create_bach_data_loaders(
            breakhis_test, breakhis_test[:1], breakhis_test, dummy_weights, 32
        )
        
        class_names = {v: k for k, v in class_to_idx.items()}
        breakhis_results = evaluate_model(model, breakhis_loader, device, class_names)
        logger.info(f"BreakHis Test Accuracy: {breakhis_results['accuracy']:.4f}")
    
    if len(bach_test) > 0:
        dummy_weights = torch.ones(len(class_to_idx))
        _, _, bach_loader = create_bach_data_loaders(
            bach_test, bach_test[:1], bach_test, dummy_weights, 32
        )
        
        class_names = {v: k for k, v in class_to_idx.items()}
        bach_results = evaluate_model(model, bach_loader, device, class_names)
        logger.info(f"BACH Test Accuracy: {bach_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()