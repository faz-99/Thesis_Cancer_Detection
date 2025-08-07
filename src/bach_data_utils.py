#!/usr/bin/env python3
"""
BACH Dataset utilities for Breast Cancer Detection
Handles BACH dataset loading, preprocessing, and integration with BreakHis

BACH Dataset: Breast Cancer Histology Challenge
- 4 classes: Normal, Benign, In Situ Carcinoma, Invasive Carcinoma
- High-resolution histopathology images
- Can be combined with BreakHis for multi-dataset training
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

logger = logging.getLogger(__name__)

def create_bach_metadata(dataset_root):
    """
    Create metadata DataFrame from BACH dataset structure
    
    Expected BACH structure:
    dataset_root/
    ├── Normal/
    ├── Benign/
    ├── InSitu/
    └── Invasive/
    
    Args:
        dataset_root (str): Path to BACH dataset root directory
        
    Returns:
        pd.DataFrame: Metadata with path, class, filename
    """
    logger.info(f"Creating BACH metadata from: {dataset_root}")
    
    data = []
    class_folders = ['Normal', 'Benign', 'InSitu', 'Invasive']
    
    for class_name in class_folders:
        class_path = os.path.join(dataset_root, class_name)
        if not os.path.exists(class_path):
            logger.warning(f"Class folder not found: {class_path}")
            continue
            
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                full_path = os.path.join(class_path, filename)
                data.append({
                    "path": full_path,
                    "class": class_name,
                    "filename": filename,
                    "dataset": "BACH"
                })
    
    df = pd.DataFrame(data)
    logger.info(f"BACH dataset: {len(df)} images")
    logger.info(f"Class distribution: {df['class'].value_counts().to_dict()}")
    
    return df

def create_combined_metadata(breakhis_root, bach_root):
    """
    Create combined metadata from both BreakHis and BACH datasets
    
    Maps classes to unified categories:
    - BreakHis benign classes → Benign
    - BreakHis malignant classes → Malignant  
    - BACH Normal → Normal
    - BACH Benign → Benign
    - BACH InSitu → InSitu
    - BACH Invasive → Invasive
    
    Args:
        breakhis_root (str): Path to BreakHis dataset
        bach_root (str): Path to BACH dataset
        
    Returns:
        pd.DataFrame: Combined metadata with unified class labels
    """
    from .data_utils import create_metadata
    
    logger.info("Creating combined BreakHis + BACH metadata...")
    
    # Load BreakHis metadata
    breakhis_df = create_metadata(breakhis_root)
    breakhis_df['dataset'] = 'BreakHis'
    
    # Map BreakHis subclasses to unified categories
    breakhis_df['unified_class'] = breakhis_df['label_type'].map({
        'benign': 'Benign',
        'malignant': 'Malignant'
    })
    
    # Load BACH metadata
    bach_df = create_bach_metadata(bach_root)
    
    # Map BACH classes to unified categories
    bach_class_mapping = {
        'Normal': 'Normal',
        'Benign': 'Benign', 
        'InSitu': 'InSitu',
        'Invasive': 'Invasive'
    }
    bach_df['unified_class'] = bach_df['class'].map(bach_class_mapping)
    
    # Standardize columns for combination
    breakhis_cols = ['path', 'dataset', 'unified_class', 'filename']
    bach_cols = ['path', 'dataset', 'unified_class', 'filename']
    
    combined_df = pd.concat([
        breakhis_df[breakhis_cols],
        bach_df[bach_cols]
    ], ignore_index=True)
    
    logger.info(f"Combined dataset: {len(combined_df)} images")
    logger.info(f"Dataset distribution: {combined_df['dataset'].value_counts().to_dict()}")
    logger.info(f"Unified class distribution: {combined_df['unified_class'].value_counts().to_dict()}")
    
    return combined_df

class BACHDataset(Dataset):
    """
    Custom PyTorch Dataset for BACH histopathology images
    
    Args:
        dataframe (pd.DataFrame): DataFrame with image metadata
        transform (torchvision.transforms): Image transformations
    """
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        logger.info(f"Created BACHDataset with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "class_idx"]
        
        try:
            # Load and convert to RGB
            image = Image.open(img_path).convert("RGB")
            
            # BACH images are high-resolution, resize appropriately
            if image.size[0] > 1024 or image.size[1] > 1024:
                # First resize to reasonable size before final transform
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
            
        except Exception as e:
            logger.error(f"Error loading BACH image {img_path}: {e}")
            # Return black fallback image
            fallback_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                fallback_image = self.transform(fallback_image)
            return fallback_image, label, img_path

def get_bach_transforms():
    """
    Get image transforms optimized for BACH dataset
    
    BACH images are high-resolution, so we include additional preprocessing
    
    Returns:
        tuple: (train_transform, test_transform)
    """
    logger.info("Creating BACH-optimized transforms...")
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Training transforms with stronger augmentation for high-res images
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),           # Initial resize
        transforms.RandomCrop(224),              # Random crop for augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),    # Vertical flip for histology
        transforms.RandomRotation(15),           # Slightly more rotation
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return train_transform, test_transform

def create_bach_data_loaders(train_df, val_df, test_df, class_weights_tensor, batch_size=32):
    """
    Create DataLoaders for BACH dataset
    
    Args:
        train_df, val_df, test_df: DataFrames with class_idx column
        class_weights_tensor: Weights for handling class imbalance
        batch_size: Batch size for loaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info(f"Creating BACH data loaders with batch_size={batch_size}")
    
    train_transform, test_transform = get_bach_transforms()
    
    # Create datasets
    train_dataset = BACHDataset(train_df, transform=train_transform)
    val_dataset = BACHDataset(val_df, transform=test_transform)
    test_dataset = BACHDataset(test_df, transform=test_transform)
    
    # Weighted sampler for training
    sample_weights = train_df["class_idx"].map(lambda x: float(class_weights_tensor[x])).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )
    
    logger.info(f"BACH loaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    return train_loader, val_loader, test_loader