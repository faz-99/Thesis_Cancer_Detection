#!/usr/bin/env python3
"""
Test script for BACH dataset integration
Verifies that BACH dataset utilities work correctly
"""

import logging
import sys
import os
from src.dataset_downloader import download_bach_dataset, verify_bach_dataset, create_sample_bach_data
from src.bach_data_utils import create_bach_metadata, create_combined_metadata

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Testing BACH dataset integration...")
    
    # 1. Setup BACH dataset structure
    logger.info("Step 1: Setting up BACH dataset structure...")
    bach_dir = download_bach_dataset("data")
    
    # 2. Create sample data for testing
    logger.info("Step 2: Creating sample BACH data...")
    create_sample_bach_data(bach_dir, num_samples_per_class=10)
    
    # 3. Verify dataset
    logger.info("Step 3: Verifying BACH dataset...")
    if not verify_bach_dataset(bach_dir):
        logger.error("BACH dataset verification failed")
        return
    
    # 4. Test BACH metadata creation
    logger.info("Step 4: Testing BACH metadata creation...")
    try:
        bach_metadata = create_bach_metadata(bach_dir)
        logger.info(f"BACH metadata created successfully: {len(bach_metadata)} samples")
        logger.info(f"Classes: {bach_metadata['class'].unique()}")
    except Exception as e:
        logger.error(f"BACH metadata creation failed: {e}")
        return
    
    # 5. Test combined metadata (if BreakHis exists)
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    if os.path.exists(breakhis_root):
        logger.info("Step 5: Testing combined metadata creation...")
        try:
            combined_metadata = create_combined_metadata(breakhis_root, bach_dir)
            logger.info(f"Combined metadata created: {len(combined_metadata)} samples")
            logger.info(f"Datasets: {combined_metadata['dataset'].value_counts().to_dict()}")
            logger.info(f"Unified classes: {combined_metadata['unified_class'].unique()}")
        except Exception as e:
            logger.error(f"Combined metadata creation failed: {e}")
            return
    else:
        logger.info("Step 5: Skipped (BreakHis dataset not found)")
    
    # 6. Test data loading
    logger.info("Step 6: Testing data loading...")
    try:
        from src.bach_data_utils import BACHDataset, get_bach_transforms
        
        # Add class indices for testing
        bach_metadata['class_idx'] = bach_metadata['class'].map({
            'Normal': 0, 'Benign': 1, 'InSitu': 2, 'Invasive': 3
        })
        
        train_transform, test_transform = get_bach_transforms()
        dataset = BACHDataset(bach_metadata, transform=test_transform)
        
        # Test loading a sample
        sample_image, sample_label, sample_path = dataset[0]
        logger.info(f"Sample loaded: shape={sample_image.shape}, label={sample_label}")
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return
    
    logger.info("✅ All BACH integration tests passed!")
    logger.info("\nNext steps:")
    logger.info("1. Download real BACH dataset from official sources")
    logger.info("2. Run: python main_combined_training.py")
    logger.info("3. Compare results with BreakHis-only training")

if __name__ == "__main__":
    main()