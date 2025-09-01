#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for Breast Cancer Detection
Converted from Jupyter notebook for direct execution
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from PIL import Image
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
DATASET_ROOT = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
RANDOM_STATE = 42

print("📊 Starting Comprehensive EDA for Breast Cancer Detection")
print(f"📁 Dataset Path: {DATASET_ROOT}")

def create_comprehensive_metadata(dataset_root):
    """
    Create comprehensive metadata from BreakHis dataset
    """
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    
    # Get all image paths
    image_paths = glob(os.path.join(dataset_root, "*", "*", "*", "*", "*", "*.png"))
    print(f"🔍 Found {len(image_paths)} images")
    
    data = []
    
    for path in image_paths:
        parts = path.split(os.sep)
        try:
            # Extract metadata from path structure
            label_type = parts[-6]         # 'malignant' or 'benign'
            subclass = parts[-4]           # e.g. 'ductal_carcinoma'
            magnification = parts[-2]      # e.g. '100X'
            filename = os.path.basename(path)
            
            # Extract patient ID from filename
            # Format: SOB_B_A-14-22549AB-40-001.png
            filename_parts = filename.split('-')
            if len(filename_parts) >= 3:
                patient_id = filename_parts[2]
            else:
                patient_id = "unknown"
            
            # Extract slide number
            slide_num = filename.split('-')[-1].replace('.png', '') if '-' in filename else "001"
            
            data.append({
                "path": path,
                "filename": filename,
                "label_type": label_type,
                "subclass": subclass,
                "magnification": magnification,
                "patient_id": patient_id,
                "slide_number": slide_num
            })
            
        except IndexError as e:
            print(f"⚠️ Skipping malformed path: {path}")
            continue
    
    metadata_df = pd.DataFrame(data)
    
    # Add derived features
    metadata_df['magnification_numeric'] = metadata_df['magnification'].str.replace('X', '').astype(int)
    metadata_df['is_malignant'] = (metadata_df['label_type'] == 'malignant').astype(int)
    
    return metadata_df

# Create metadata
try:
    metadata = create_comprehensive_metadata(DATASET_ROOT)
    print(f"✅ Created metadata for {len(metadata)} images")
    print(f"📋 Columns: {list(metadata.columns)}")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please ensure the BreakHis dataset is downloaded and placed in the correct directory.")
    exit(1)

# Display basic information
print("\n📊 DATASET OVERVIEW")
print("=" * 50)
print(f"Total Images: {len(metadata):,}")
print(f"Unique Patients: {metadata['patient_id'].nunique():,}")
print(f"Unique Subclasses: {metadata['subclass'].nunique()}")
print(f"Magnification Levels: {sorted(metadata['magnification'].unique())}")
print(f"Label Types: {metadata['label_type'].unique()}")

print("\n📋 Sample Data:")
print(metadata.head())

# Class Distribution Analysis
print("\n" + "="*60)
print("📊 CLASS DISTRIBUTION ANALYSIS")
print("="*60)

# Overall class distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Benign vs Malignant
label_counts = metadata['label_type'].value_counts()
axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Benign vs Malignant Distribution', fontsize=14, fontweight='bold')

# 2. Subclass distribution
subclass_counts = metadata['subclass'].value_counts()
axes[0, 1].barh(range(len(subclass_counts)), subclass_counts.values)
axes[0, 1].set_yticks(range(len(subclass_counts)))
axes[0, 1].set_yticklabels([label.replace('_', ' ').title() for label in subclass_counts.index])
axes[0, 1].set_xlabel('Number of Images')
axes[0, 1].set_title('Subclass Distribution', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, v in enumerate(subclass_counts.values):
    axes[0, 1].text(v + 50, i, str(v), va='center', fontweight='bold')

# 3. Magnification distribution
mag_counts = metadata['magnification'].value_counts().sort_index()
axes[1, 0].bar(mag_counts.index, mag_counts.values, color='skyblue', edgecolor='navy')
axes[1, 0].set_xlabel('Magnification Level')
axes[1, 0].set_ylabel('Number of Images')
axes[1, 0].set_title('Magnification Level Distribution', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, v in enumerate(mag_counts.values):
    axes[1, 0].text(i, v + 30, str(v), ha='center', fontweight='bold')

# 4. Images per patient distribution
patient_counts = metadata['patient_id'].value_counts()
axes[1, 1].hist(patient_counts.values, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
axes[1, 1].set_xlabel('Images per Patient')
axes[1, 1].set_ylabel('Number of Patients')
axes[1, 1].set_title('Images per Patient Distribution', fontsize=14, fontweight='bold')
axes[1, 1].axvline(patient_counts.mean(), color='red', linestyle='--', 
                   label=f'Mean: {patient_counts.mean():.1f}')
axes[1, 1].legend()

plt.tight_layout()
# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)
plt.savefig('results/eda_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics
print("\n📊 DETAILED CLASS STATISTICS")
print("=" * 50)
for label_type in metadata['label_type'].unique():
    subset = metadata[metadata['label_type'] == label_type]
    print(f"\n{label_type.upper()}:")
    print(f"  Total Images: {len(subset):,}")
    print(f"  Unique Patients: {subset['patient_id'].nunique()}")
    print(f"  Subclasses: {subset['subclass'].nunique()}")
    print(f"  Subclass breakdown:")
    for subclass, count in subset['subclass'].value_counts().items():
        print(f"    - {subclass.replace('_', ' ').title()}: {count:,} images")

# Patient-wise Analysis
print("\n" + "="*60)
print("👥 PATIENT-WISE ANALYSIS")
print("="*60)

# Patient-wise analysis
patient_analysis = metadata.groupby('patient_id').agg({
    'path': 'count',
    'label_type': lambda x: x.iloc[0],  # Assuming all images from same patient have same label
    'subclass': lambda x: x.iloc[0],
    'magnification': lambda x: list(x.unique())
}).rename(columns={'path': 'image_count'})

patient_analysis['magnification_count'] = patient_analysis['magnification'].apply(len)

# Print patient statistics
print(f"Total Unique Patients: {len(patient_analysis)}")
print(f"Benign Patients: {len(patient_analysis[patient_analysis['label_type'] == 'benign'])}")
print(f"Malignant Patients: {len(patient_analysis[patient_analysis['label_type'] == 'malignant'])}")
print(f"\nImages per Patient Statistics:")
print(f"  Mean: {patient_analysis['image_count'].mean():.2f}")
print(f"  Median: {patient_analysis['image_count'].median():.2f}")
print(f"  Min: {patient_analysis['image_count'].min()}")
print(f"  Max: {patient_analysis['image_count'].max()}")
print(f"  Std: {patient_analysis['image_count'].std():.2f}")

# Data Imbalance Analysis
print("\n" + "="*60)
print("⚖️ DATA IMBALANCE ANALYSIS")
print("="*60)

def calculate_imbalance_metrics(metadata):
    """
    Calculate various imbalance metrics
    """
    # Class distribution
    class_counts = metadata['subclass'].value_counts()
    
    # Calculate imbalance ratio
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    
    # Calculate class weights (inverse frequency)
    total_samples = len(metadata)
    n_classes = len(class_counts)
    class_weights = {}
    
    for class_name, count in class_counts.items():
        weight = total_samples / (n_classes * count)
        class_weights[class_name] = weight
    
    return {
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio,
        'class_weights': class_weights,
        'total_samples': total_samples,
        'n_classes': n_classes
    }

# Calculate imbalance metrics
imbalance_metrics = calculate_imbalance_metrics(metadata)
class_counts = imbalance_metrics['class_counts']
class_weights = imbalance_metrics['class_weights']

print(f"Imbalance Ratio (Max/Min): {imbalance_metrics['imbalance_ratio']:.2f}")
print(f"Most Common Class: {class_counts.index[0]} ({class_counts.iloc[0]:,} images)")
print(f"Least Common Class: {class_counts.index[-1]} ({class_counts.iloc[-1]:,} images)")
print(f"\nClass Distribution:")
for class_name, count in class_counts.items():
    percentage = (count / imbalance_metrics['total_samples']) * 100
    print(f"  {class_name.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")

print(f"\nRecommended Class Weights:")
for class_name, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name.replace('_', ' ').title()}: {weight:.3f}")

# Train/Validation/Test Split Analysis
print("\n" + "="*60)
print("📊 TRAIN/VALIDATION/TEST SPLIT ANALYSIS")
print("="*60)

from sklearn.model_selection import train_test_split

def create_patient_wise_splits(metadata, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create patient-wise stratified splits to avoid data leakage
    """
    # Get unique patients with their labels
    patient_labels = metadata.groupby('patient_id')['subclass'].first().reset_index()
    
    # First split: train+val vs test
    train_val_patients, test_patients = train_test_split(
        patient_labels, test_size=test_size, 
        stratify=patient_labels['subclass'], random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_size_adjusted,
        stratify=train_val_patients['subclass'], random_state=random_state
    )
    
    # Map back to full metadata
    train_data = metadata[metadata['patient_id'].isin(train_patients['patient_id'])]
    val_data = metadata[metadata['patient_id'].isin(val_patients['patient_id'])]
    test_data = metadata[metadata['patient_id'].isin(test_patients['patient_id'])]
    
    return train_data, val_data, test_data

# Create splits
train_data, val_data, test_data = create_patient_wise_splits(metadata)

# Analyze splits
splits_info = {
    'Train': train_data,
    'Validation': val_data,
    'Test': test_data
}

for split_name, data in splits_info.items():
    print(f"\n{split_name.upper()} SET:")
    print(f"  Images: {len(data):,}")
    print(f"  Patients: {data['patient_id'].nunique()}")
    print(f"  Percentage: {(len(data)/len(metadata)*100):.1f}%")
    print(f"  Class distribution:")
    for subclass, count in data['subclass'].value_counts().items():
        percentage = (count / len(data)) * 100
        print(f"    - {subclass.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

# Generate comprehensive summary and recommendations
print("\n" + "="*80)
print("📋 COMPREHENSIVE EDA SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\n🔍 DATASET OVERVIEW:")
print(f"  • Total Images: {len(metadata):,}")
print(f"  • Unique Patients: {metadata['patient_id'].nunique()}")
print(f"  • Classes: {metadata['subclass'].nunique()} subclasses (4 benign + 4 malignant)")
print(f"  • Magnifications: {len(metadata['magnification'].unique())} levels (40X, 100X, 200X, 400X)")

print("\n⚖️ DATA IMBALANCE INSIGHTS:")
print(f"  • Imbalance Ratio: {imbalance_metrics['imbalance_ratio']:.2f}:1")
print(f"  • Most Common: {class_counts.index[0].replace('_', ' ').title()} ({class_counts.iloc[0]:,} images)")
print(f"  • Least Common: {class_counts.index[-1].replace('_', ' ').title()} ({class_counts.iloc[-1]:,} images)")
print(f"  • Recommendation: Use weighted sampling or class weights during training")

print("\n👥 PATIENT-WISE ANALYSIS:")
print(f"  • Images per Patient: {patient_analysis['image_count'].mean():.1f} ± {patient_analysis['image_count'].std():.1f}")
print(f"  • Patient Distribution: {len(patient_analysis[patient_analysis['label_type'] == 'benign'])} benign, {len(patient_analysis[patient_analysis['label_type'] == 'malignant'])} malignant")
print(f"  • Recommendation: Use patient-wise splits to avoid data leakage")

print("\n📊 TRAINING RECOMMENDATIONS:")
print("  1. DATA PREPROCESSING:")
print("     • Resize images to 224x224 pixels")
print("     • Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
print("     • Apply data augmentation (rotation, flip, color jitter)")

print("\n  2. DATA SPLITTING:")
print("     • Use patient-wise stratified splits (70% train, 15% val, 15% test)")
print("     • Ensure no patient appears in multiple splits")
print("     • Maintain class distribution across splits")

print("\n  3. CLASS IMBALANCE HANDLING:")
print("     • Use WeightedRandomSampler during training")
print("     • Apply class weights in loss function")
print("     • Consider focal loss for severe imbalance")

print("\n  4. MODEL ARCHITECTURE:")
print("     • Start with EfficientNetB0 (good balance of accuracy and efficiency)")
print("     • Use transfer learning with ImageNet pretrained weights")
print("     • Fine-tune the entire network with lower learning rate")

print("\n  5. TRAINING STRATEGY:")
print("     • Batch size: 32-64 (depending on GPU memory)")
print("     • Learning rate: 1e-4 with cosine annealing")
print("     • Early stopping based on validation accuracy")
print("     • Save best model based on validation performance")

print("\n  6. EVALUATION METRICS:")
print("     • Accuracy, Precision, Recall, F1-score (per class and macro/micro avg)")
print("     • Confusion matrix analysis")
print("     • ROC-AUC curves for each class")
print("     • Per-magnification performance analysis")

print("\n🚀 NEXT STEPS:")
print("  1. Implement the recommended preprocessing pipeline")
print("  2. Create patient-wise stratified splits")
print("  3. Train EfficientNetB0 baseline with class balancing")
print("  4. Evaluate performance and analyze failure cases")
print("  5. Consider advanced techniques (ensemble, multi-scale, etc.)")

# Save key results for later use
import pickle

# Create results directory
os.makedirs('results/eda', exist_ok=True)

# Save metadata
metadata.to_csv('results/eda/dataset_metadata.csv', index=False)
print("\n✅ Saved dataset metadata to results/eda/dataset_metadata.csv")

# Save splits
train_data.to_csv('results/eda/train_split.csv', index=False)
val_data.to_csv('results/eda/val_split.csv', index=False)
test_data.to_csv('results/eda/test_split.csv', index=False)
print("✅ Saved train/val/test splits to results/eda/")

# Save class mappings and weights
class_info = {
    'class_counts': dict(class_counts),
    'class_weights': class_weights,
    'imbalance_ratio': imbalance_metrics['imbalance_ratio']
}

with open('results/eda/class_info.pkl', 'wb') as f:
    pickle.dump(class_info, f)
print("✅ Saved class information to results/eda/class_info.pkl")

print("\n" + "="*80)
print("✅ EDA COMPLETED - Ready for model development!")
print("="*80)
print("🎉 EDA results exported successfully!")
print("📁 Check the results/eda/ directory for all saved files.")