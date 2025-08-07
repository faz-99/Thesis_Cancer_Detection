#!/usr/bin/env python3
"""
Data utilities for Breast Cancer Detection
Handles BreakHis dataset loading, preprocessing, and data loaders

This module provides comprehensive data handling for the BreakHis dataset including:
- Metadata extraction from directory structure
- Patient-wise stratified splitting to prevent data leakage
- Class mapping and weight calculation for imbalanced dataset
- Custom dataset class with augmentation
- Data loader creation with weighted sampling
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

# Setup module logger
logger = logging.getLogger(__name__)

def create_metadata(dataset_root):
    """
    Create metadata DataFrame from BreakHis dataset structure
    
    Traverses the BreakHis dataset directory and extracts metadata for each image.
    Expected structure: dataset_root/benign|malignant/tumor_class/magnification/images
    
    Args:
        dataset_root (str): Path to BreakHis dataset root directory
        
    Returns:
        pd.DataFrame: Metadata with path, label_type, subclass, magnification, filename
    """
    logger.info(f"Creating metadata from dataset root: {dataset_root}")
    image_paths = []
    
    # Walk through directory structure to find all PNG images
    logger.info("Scanning directory structure for images...")
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    logger.info(f"Found {len(image_paths)} PNG images")
    
    data = []
    processed_count = 0
    
    # Extract metadata from each image path
    for path in image_paths:
        parts = path.split(os.sep)
        try:
            # Parse directory structure to extract labels
            # Expected: .../benign|malignant/tumor_class/magnification/image.png
            label_type = parts[-6]      # 'malignant' or 'benign'
            subclass = parts[-4]        # e.g. 'mucinous_carcinoma', 'adenosis'
            magnification = parts[-2]   # e.g. '100X', '200X'
            filename = os.path.basename(path)
            
            data.append({
                "path": path,
                "label_type": label_type,
                "subclass": subclass,
                "magnification": magnification,
                "filename": filename
            })
            processed_count += 1
            
        except IndexError:
            logger.warning(f"Could not parse path structure for: {path}")
            continue
    
    df = pd.DataFrame(data)
    logger.info(f"Successfully processed {processed_count} images")
    logger.info(f"Label distribution: {df['label_type'].value_counts().to_dict()}")
    logger.info(f"Subclass distribution: {df['subclass'].value_counts().to_dict()}")
    logger.info(f"Magnification distribution: {df['magnification'].value_counts().to_dict()}")
    
    return df

def extract_patient_id(path):
    """
    Extract patient ID from BreakHis filename
    
    BreakHis filenames follow format: SOB_[type]_[patient_id]_[magnification]_[seq].png
    This function extracts the patient_id component to enable patient-wise splitting.
    
    Args:
        path (str): Full path to image file
        
    Returns:
        str: Patient ID extracted from filename
    """
    filename = os.path.basename(path)
    try:
        # Split filename and extract patient ID (3rd component after splitting by '_')
        patient_id = filename.split("_")[2]
        return patient_id
    except IndexError:
        logger.warning(f"Could not extract patient ID from filename: {filename}")
        return "unknown"

def create_train_val_test_split(metadata, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create patient-wise stratified train/validation/test splits
    
    Ensures no data leakage by keeping all images from the same patient in the same split.
    This is critical for medical imaging to get realistic performance estimates.
    
    Process:
    1. Extract patient IDs from filenames
    2. Group patients by subclass for stratification
    3. Split patients (not images) into train/val/test
    4. Assign all images from each patient to corresponding split
    
    Args:
        metadata (pd.DataFrame): Image metadata with paths and labels
        test_size (float): Proportion for test set (default: 0.15)
        val_size (float): Proportion for validation set (default: 0.15)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df) DataFrames for each split
    """
    logger.info(f"Creating patient-wise splits: test={test_size}, val={val_size}")
    
    metadata = metadata.copy()
    # Extract patient IDs from all image paths
    metadata["patient_id"] = metadata["path"].apply(extract_patient_id)
    
    # Get unique patients with their subclass labels for stratification
    unique_patients = metadata[["patient_id", "subclass"]].drop_duplicates()
    logger.info(f"Total unique patients: {len(unique_patients)}")
    logger.info(f"Patient subclass distribution: {unique_patients['subclass'].value_counts().to_dict()}")
    
    # First split: separate test patients from train+val patients
    logger.info("Performing train+val vs test split...")
    train_ids, test_ids = train_test_split(
        unique_patients,
        test_size=test_size,
        stratify=unique_patients["subclass"],
        random_state=random_state
    )
    
    # Second split: separate train patients from validation patients
    logger.info("Performing train vs val split...")
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_size / (1 - test_size),  # Adjust for remaining data
        stratify=train_ids["subclass"],
        random_state=random_state
    )
    
    # Map patient splits back to full image metadata
    train_df = metadata[metadata["patient_id"].isin(train_ids["patient_id"])]
    val_df = metadata[metadata["patient_id"].isin(val_ids["patient_id"])]
    test_df = metadata[metadata["patient_id"].isin(test_ids["patient_id"])]
    
    # Log split statistics
    logger.info(f"Split results:")
    logger.info(f"  Train: {len(train_df)} images from {len(train_ids)} patients")
    logger.info(f"  Val: {len(val_df)} images from {len(val_ids)} patients")
    logger.info(f"  Test: {len(test_df)} images from {len(test_ids)} patients")
    
    return train_df, val_df, test_df

def create_class_mappings(train_df):
    """
    Create class to index mappings and calculate class weights for imbalanced dataset
    
    Creates mappings between class names and indices, and calculates inverse frequency
    weights to handle class imbalance during training.
    
    Args:
        train_df (pd.DataFrame): Training data with 'subclass' column
        
    Returns:
        tuple: (class_to_idx, idx_to_class, class_weights_tensor)
            - class_to_idx: dict mapping class names to indices
            - idx_to_class: dict mapping indices to class names  
            - class_weights_tensor: torch.Tensor with inverse frequency weights
    """
    logger.info("Creating class mappings and calculating weights...")
    
    # Count samples per class in training set
    class_counts = Counter(train_df["subclass"])
    classes = sorted(class_counts.keys())
    
    logger.info(f"Found {len(classes)} classes: {classes}")
    logger.info(f"Class counts: {dict(class_counts)}")
    
    # Create bidirectional mappings between class names and indices
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Calculate inverse frequency weights for handling class imbalance
    total = sum(class_counts.values())
    class_weights = np.array([total / class_counts[cls] for cls in classes], dtype=np.float32)
    
    # Normalize weights to sum to number of classes
    class_weights = class_weights / class_weights.sum() * len(classes)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info(f"Class weights: {dict(zip(classes, class_weights))}")
    
    return class_to_idx, idx_to_class, class_weights_tensor

class BreakHisDataset(Dataset):
    """
    Custom PyTorch Dataset for BreakHis histopathological images
    
    This dataset class handles loading and preprocessing of BreakHis images.
    It applies transforms for data augmentation and normalization.
    
    Args:
        dataframe (pd.DataFrame): DataFrame with image metadata including 'path' and 'class_idx'
        transform (torchvision.transforms): Image transformations to apply
    """
    
    def __init__(self, dataframe, transform=None):
        """
        Initialize dataset with image metadata and transforms
        
        Args:
            dataframe (pd.DataFrame): Must contain 'path' and 'class_idx' columns
            transform (callable, optional): Transform to apply to images
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        logger.info(f"Created BreakHisDataset with {len(self.df)} samples")

    def __len__(self):
        """Return total number of samples in dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load and return a single sample
        
        Args:
            idx (int): Index of sample to load
            
        Returns:
            tuple: (image, label, img_path)
                - image: Preprocessed image tensor
                - label: Class index as integer
                - img_path: Original image path for debugging
        """
        # Get image path and label from dataframe
        img_path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "class_idx"]
        
        try:
            # Load image and convert to RGB (handles grayscale and RGBA)
            image = Image.open(img_path).convert("RGB")
            
            # Apply transforms if provided (augmentation, normalization, etc.)
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                fallback_image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                fallback_image = Image.new('RGB', (224, 224), (0, 0, 0))
            return fallback_image, label, img_path

def get_transforms():
    """
    Get image transforms for training and validation/testing
    
    Creates different transform pipelines for training vs validation/testing:
    - Training: Includes data augmentation (flips, rotations) for better generalization
    - Validation/Test: Only basic preprocessing without augmentation for consistent evaluation
    
    Both use ImageNet normalization statistics since we use pretrained models.
    
    Returns:
        tuple: (train_transform, test_transform)
            - train_transform: Augmented transforms for training
            - test_transform: Basic transforms for validation/testing
    """
    logger.info("Creating image transforms...")
    
    # ImageNet normalization statistics (required for pretrained models)
    imagenet_mean = [0.485, 0.456, 0.406]  # RGB channel means
    imagenet_std = [0.229, 0.224, 0.225]   # RGB channel standard deviations
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),           # Resize to EfficientNet input size
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance horizontal flip
        transforms.RandomRotation(10),           # Random rotation ±10 degrees
        transforms.ColorJitter(                  # Random color adjustments
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),                   # Convert PIL to tensor [0,1]
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize
    ])
    
    # Validation/test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),           # Resize to model input size
        transforms.ToTensor(),                   # Convert PIL to tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize
    ])
    
    logger.info("Created training transforms with augmentation")
    logger.info("Created test transforms without augmentation")
    
    return train_transform, test_transform

def create_data_loaders(train_df, val_df, test_df, class_weights_tensor, batch_size=32):
    """
    Create PyTorch DataLoaders with weighted sampling for class imbalance
    
    Creates three data loaders:
    - Training: Uses weighted random sampling to handle class imbalance
    - Validation: Sequential sampling for consistent evaluation
    - Test: Sequential sampling for final evaluation
    
    Args:
        train_df (pd.DataFrame): Training data with 'class_idx' column
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        class_weights_tensor (torch.Tensor): Weights for each class
        batch_size (int): Batch size for all loaders (default: 32)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating data loaders with batch_size={batch_size}")
    
    # Get appropriate transforms for each split
    train_transform, test_transform = get_transforms()
    
    # Create dataset objects
    train_dataset = BreakHisDataset(train_df, transform=train_transform)
    val_dataset = BreakHisDataset(val_df, transform=test_transform)
    test_dataset = BreakHisDataset(test_df, transform=test_transform)
    
    logger.info(f"Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create weighted sampler for training to handle class imbalance
    # Each sample gets weight based on inverse class frequency
    logger.info("Creating weighted sampler for training...")
    sample_weights = train_df["class_idx"].map(lambda x: float(class_weights_tensor[x])).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True  # Allow sampling with replacement
    )
    
    # Create data loaders
    # Training: Use weighted sampler, no shuffle (sampler handles randomization)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0,      # Set to 0 to avoid multiprocessing issues
        pin_memory=True,    # Faster GPU transfer
        drop_last=True      # Drop incomplete batches for consistent training
    )
    
    # Validation: Sequential sampling for reproducible evaluation
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )
    
    # Test: Sequential sampling for final evaluation
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_loader)} batches with weighted sampling")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader