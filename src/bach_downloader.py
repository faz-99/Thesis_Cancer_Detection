#!/usr/bin/env python3
"""
BACH Dataset Downloader
Downloads and sets up the BACH (Breast Cancer Histology) dataset
"""

import os
import requests
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_bach_dataset(data_dir="../data"):
    """
    Download BACH dataset from Kaggle
    
    Note: This requires Kaggle API credentials
    Install: pip install kaggle
    Setup: Place kaggle.json in ~/.kaggle/
    """
    
    data_path = Path(data_dir)
    bach_path = data_path / "bach"
    
    # Create directories
    bach_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading BACH dataset from Kaggle...")
    
    try:
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'paultimothymooney/breast-histopathology-images',
            path=str(bach_path),
            unzip=True
        )
        
        logger.info(f"BACH dataset downloaded to: {bach_path}")
        
        # Check structure
        expected_folders = ['Normal', 'Benign', 'InSitu', 'Invasive']
        for folder in expected_folders:
            folder_path = bach_path / folder
            if folder_path.exists():
                count = len(list(folder_path.glob('*.png')))
                logger.info(f"{folder}: {count} images")
            else:
                logger.warning(f"Expected folder not found: {folder}")
        
        return True
        
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False

def setup_bach_manually():
    """
    Print instructions for manual BACH dataset setup
    """
    print("Manual BACH Dataset Setup Instructions:")
    print("=" * 50)
    print("1. Download BACH dataset from one of these sources:")
    print("   - Official: https://iciar2018-challenge.grand-challenge.org/Dataset/")
    print("   - Kaggle: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images")
    print()
    print("2. Extract and organize in this structure:")
    print("   data/bach/")
    print("   ├── Normal/")
    print("   ├── Benign/")
    print("   ├── InSitu/")
    print("   └── Invasive/")
    print()
    print("3. Each folder should contain ~100 high-resolution histopathology images")
    print()
    print("4. Run the EDA notebook: notebooks/bach_eda.ipynb")

if __name__ == "__main__":
    # Try automatic download first
    if not download_bach_dataset():
        # Fall back to manual instructions
        setup_bach_manually()