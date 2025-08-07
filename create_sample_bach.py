#!/usr/bin/env python3
"""
Create sample BACH data for testing EDA
"""

import os
import sys
sys.path.append('src')

from src.dataset_downloader import create_sample_bach_data, verify_bach_dataset

def main():
    bach_dir = "data/bach"
    
    print("Creating sample BACH data for EDA testing...")
    create_sample_bach_data(bach_dir, num_samples_per_class=10)
    
    print("\nVerifying BACH dataset...")
    if verify_bach_dataset(bach_dir):
        print("✓ Sample BACH dataset created successfully!")
        print(f"✓ Location: {bach_dir}")
        print("✓ You can now run the BACH EDA notebook: notebooks/bach_eda.ipynb")
    else:
        print("✗ Error creating sample BACH dataset")

if __name__ == "__main__":
    main()