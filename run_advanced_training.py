#!/usr/bin/env python3
"""
Quick setup and run script for advanced cross-dataset training

This script:
1. Checks data availability
2. Installs missing dependencies
3. Runs advanced cross-dataset training
4. Provides performance summary
"""

import os
import sys
import subprocess
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_data_availability():
    """Check if required datasets are available"""
    logger = logging.getLogger(__name__)
    
    breakhis_path = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    bach_path = "data/bach"
    
    datasets_available = {
        'BreakHis': os.path.exists(breakhis_path),
        'BACH': os.path.exists(bach_path)
    }
    
    logger.info("Dataset Availability Check:")
    for dataset, available in datasets_available.items():
        status = "✅ Available" if available else "❌ Missing"
        logger.info(f"  {dataset}: {status}")
    
    if not all(datasets_available.values()):
        logger.warning("Some datasets are missing. Please ensure both datasets are available.")
        logger.info("BreakHis should be at: data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast")
        logger.info("BACH should be at: data/bach/")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Installing advanced training dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_advanced.txt"
        ])
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def run_advanced_training():
    """Run the advanced cross-dataset training"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting advanced cross-dataset training...")
        
        # Import and run the training
        from advanced_cross_dataset_training import main as training_main
        training_main()
        
        logger.info("Advanced training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Main execution function"""
    logger = setup_logging()
    
    logger.info("🚀 Advanced Cross-Dataset Training Setup")
    logger.info("="*50)
    
    # Step 1: Check data availability
    logger.info("Step 1: Checking data availability...")
    if not check_data_availability():
        logger.error("Please ensure both BreakHis and BACH datasets are available before proceeding.")
        return
    
    # Step 2: Install dependencies
    logger.info("\nStep 2: Installing dependencies...")
    if not install_dependencies():
        logger.error("Failed to install dependencies. Please install manually.")
        return
    
    # Step 3: Run training
    logger.info("\nStep 3: Running advanced cross-dataset training...")
    if run_advanced_training():
        logger.info("\n🎉 Advanced cross-dataset training completed successfully!")
        logger.info("Check the logs and results for performance metrics.")
    else:
        logger.error("Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()