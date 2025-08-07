#!/usr/bin/env python3
"""
Main training script for Breast Cancer Detection Thesis
Implements EfficientNetB0 baseline with minimal code
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from src.data_utils import (
    create_metadata, create_train_val_test_split, 
    create_class_mappings, create_data_loaders
)
from src.efficientnet import EfficientNetB0Classifier
from src.train import train_model, evaluate_model

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    DATASET_ROOT = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {DEVICE}")
    
    # 1. Create metadata
    logger.info("Creating metadata...")
    metadata = create_metadata(DATASET_ROOT)
    logger.info(f"Found {len(metadata)} images")
    
    # 2. Create splits
    logger.info("Creating train/val/test splits...")
    train_df, val_df, test_df = create_train_val_test_split(metadata)
    
    # 3. Create class mappings
    class_to_idx, idx_to_class, class_weights_tensor = create_class_mappings(train_df)
    
    # Add class indices to dataframes
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df["subclass"].map(class_to_idx)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Number of classes: {len(class_to_idx)}")
    
    # 4. Create data loaders
    logger.info("Creating data loaders...")
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
    torch.save(model.state_dict(), "models/efficientnet_b0_best.pth")
    logger.info("Model saved to models/efficientnet_b0_best.pth")
    
    # 8. Evaluate on test set
    logger.info("Evaluating on test set...")
    class_names = {v: k for k, v in class_to_idx.items()}
    test_results = evaluate_model(model, test_loader, DEVICE, class_names)
    
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info("Classification Report:")
    logger.info(f"\n{test_results['classification_report']}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()