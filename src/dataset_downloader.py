#!/usr/bin/env python3
"""
Dataset downloader for BACH dataset
Provides utilities to download and setup the BACH dataset
"""

import os
import requests
import zipfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def download_bach_dataset(data_dir="data"):
    """
    Download and extract BACH dataset
    
    The BACH dataset is available from the ICIAR 2018 challenge.
    This function provides instructions and utilities for setup.
    
    Args:
        data_dir (str): Directory to store the dataset
    """
    logger.info("Setting up BACH dataset...")
    
    bach_dir = os.path.join(data_dir, "bach")
    os.makedirs(bach_dir, exist_ok=True)
    
    # BACH dataset information
    info_text = """
    BACH Dataset Setup Instructions:
    
    The BACH (BreAst Cancer Histology) dataset is from the ICIAR 2018 challenge.
    
    To download the dataset:
    1. Visit: https://iciar2018-challenge.grand-challenge.org/Dataset/
    2. Register and download the training data
    3. Extract to: {bach_dir}
    
    Expected structure after extraction:
    {bach_dir}/
    ├── Normal/
    ├── Benign/
    ├── InSitu/
    └── Invasive/
    
    Alternative sources:
    - Kaggle: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
    - Papers with Code: https://paperswithcode.com/dataset/bach
    
    The dataset contains:
    - 400 microscopy images (100 per class)
    - High resolution (2048 x 1536 pixels)
    - 4 classes: Normal, Benign, In Situ Carcinoma, Invasive Carcinoma
    """.format(bach_dir=bach_dir)
    
    print(info_text)
    
    # Create placeholder structure
    classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
    for class_name in classes:
        class_dir = os.path.join(bach_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create README in each class directory
        readme_path = os.path.join(class_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(f"Place {class_name} class images in this directory\n")
            f.write("Images should be in PNG, JPG, or TIFF format\n")
    
    logger.info(f"BACH dataset structure created at: {bach_dir}")
    logger.info("Please download the actual images following the instructions above")
    
    return bach_dir

def verify_bach_dataset(bach_dir):
    """
    Verify BACH dataset is properly set up
    
    Args:
        bach_dir (str): Path to BACH dataset directory
        
    Returns:
        bool: True if dataset is properly set up
    """
    logger.info(f"Verifying BACH dataset at: {bach_dir}")
    
    if not os.path.exists(bach_dir):
        logger.error(f"BACH directory not found: {bach_dir}")
        return False
    
    classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
    total_images = 0
    
    for class_name in classes:
        class_dir = os.path.join(bach_dir, class_name)
        if not os.path.exists(class_dir):
            logger.error(f"Class directory not found: {class_dir}")
            return False
        
        # Count images in class directory
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        logger.info(f"{class_name}: {len(image_files)} images")
        total_images += len(image_files)
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {class_name} directory")
    
    logger.info(f"Total BACH images: {total_images}")
    
    if total_images == 0:
        logger.error("No images found in BACH dataset")
        return False
    
    logger.info("BACH dataset verification completed")
    return True

def create_sample_bach_data(bach_dir, num_samples_per_class=5):
    """
    Create sample BACH data for testing (creates dummy images)
    
    Args:
        bach_dir (str): Path to BACH dataset directory
        num_samples_per_class (int): Number of sample images per class
    """
    from PIL import Image
    import numpy as np
    
    logger.info(f"Creating sample BACH data with {num_samples_per_class} images per class")
    
    classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
    
    for class_name in classes:
        class_dir = os.path.join(bach_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_samples_per_class):
            # Create a random colored image
            img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # Add some class-specific patterns
            if class_name == 'Normal':
                img_array[:, :, 1] = np.minimum(img_array[:, :, 1] + 50, 255)  # More green
            elif class_name == 'Benign':
                img_array[:, :, 0] = np.minimum(img_array[:, :, 0] + 30, 255)  # More red
            elif class_name == 'InSitu':
                img_array[:, :, 2] = np.minimum(img_array[:, :, 2] + 40, 255)  # More blue
            else:  # Invasive
                img_array = np.minimum(img_array + 20, 255)  # Brighter overall
            
            img = Image.fromarray(img_array)
            img_path = os.path.join(class_dir, f"{class_name.lower()}_{i+1:03d}.png")
            img.save(img_path)
    
    logger.info(f"Sample BACH data created at: {bach_dir}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Download/setup BACH dataset
    bach_dir = download_bach_dataset()
    
    # Create sample data for testing
    create_sample_bach_data(bach_dir)
    
    # Verify setup
    verify_bach_dataset(bach_dir)