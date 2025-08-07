#!/usr/bin/env python3
"""
Main training script for Breast Cancer Detection Thesis
Implements EfficientNetB0 baseline with minimal code
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import (
    create_metadata, create_train_val_test_split, 
    create_class_mappings, create_data_loaders
)
from src.efficientnet import EfficientNetB0Classifier
from src.train import train_model, evaluate_model

def main():
    # Configuration
    DATASET_ROOT = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # 1. Create metadata
    print("Creating metadata...")
    metadata = create_metadata(DATASET_ROOT)
    print(f"Found {len(metadata)} images")
    
    # 2. Create splits
    print("Creating train/val/test splits...")
    train_df, val_df, test_df = create_train_val_test_split(metadata)
    
    # 3. Create class mappings
    class_to_idx, idx_to_class, class_weights_tensor = create_class_mappings(train_df)
    
    # Add class indices to dataframes
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df["subclass"].map(class_to_idx)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Number of classes: {len(class_to_idx)}")
    
    # 4. Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, class_weights_tensor, BATCH_SIZE
    )
    
    # 5. Initialize model
    print("Initializing model...")
    model = EfficientNetB0Classifier(num_classes=len(class_to_idx), pretrained=True)
    model = model.to(DEVICE)
    
    # 6. Train model
    print("Starting training...")
    model, history = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )
    
    # 7. Save model
    torch.save(model.state_dict(), "models/efficientnet_b0_best.pth")
    print("Model saved!")
    
    # 8. Evaluate on test set
    print("Evaluating on test set...")
    class_names = {v: k for k, v in class_to_idx.items()}
    test_results = evaluate_model(model, test_loader, DEVICE, class_names)
    
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(test_results['classification_report'])
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()