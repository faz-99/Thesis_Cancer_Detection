#!/usr/bin/env python3
"""
BACH Dataset - Exploratory Data Analysis
Converted from Jupyter notebook for direct execution
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Constants
BACH_ROOT = 'data/bach'
FIGSIZE = (12, 8)

print("📊 Starting BACH Dataset EDA")
print(f"📁 Dataset Path: {BACH_ROOT}")

def create_bach_metadata(bach_root):
    """Create metadata for BACH dataset"""
    if not os.path.exists(bach_root):
        return None
    
    data = []
    classes = ['Normal', 'Benign', 'InSitu', 'Invasive']
    
    for class_name in classes:
        class_dir = os.path.join(bach_root, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    data.append({
                        'path': os.path.join(class_dir, filename),
                        'filename': filename,
                        'class': class_name
                    })
    
    return pd.DataFrame(data)

# Check if BACH data exists
if not os.path.exists(BACH_ROOT):
    print("BACH dataset not found. Please download it first.")
    print("\\nDownload options:")
    print("1. Official: https://iciar2018-challenge.grand-challenge.org/Dataset/")
    print("2. Kaggle: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images")
    print("\\nExpected structure:")
    print("data/bach/")
    print("├── Normal/")
    print("├── Benign/")
    print("├── InSitu/")
    print("└── Invasive/")
    exit(1)
else:
    print(f"BACH dataset found at: {BACH_ROOT}")
    
    # List directory structure
    for root, dirs, files in os.walk(BACH_ROOT):
        level = root.replace(BACH_ROOT, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files)-3} more files")

# Load BACH metadata
bach_df = create_bach_metadata(BACH_ROOT)
if bach_df is None or len(bach_df) == 0:
    print("No BACH data found or unable to create metadata.")
    exit(1)

print(f"Total images: {len(bach_df)}")
print(f"\\nDataset info:")
print(bach_df.info())
print(f"\\nFirst few rows:")
print(bach_df.head())

print("\\n" + "="*60)
print("📊 CLASS DISTRIBUTION ANALYSIS")
print("="*60)

# Class distribution
class_counts = bach_df['class'].value_counts()
print("Class Distribution:")
print(class_counts)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar plot
class_counts.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('BACH Dataset - Class Distribution')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Number of Images')
axes[0].tick_params(axis='x', rotation=45)

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('BACH Dataset - Class Proportions')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/bach_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Class balance analysis
print(f"\\nClass Balance Analysis:")
print(f"Most common class: {class_counts.index[0]} ({class_counts.iloc[0]} images)")
print(f"Least common class: {class_counts.index[-1]} ({class_counts.iloc[-1]} images)")
print(f"Imbalance ratio: {class_counts.iloc[0] / class_counts.iloc[-1]:.2f}:1")

print("\\n" + "="*60)
print("🖼️ IMAGE PROPERTIES ANALYSIS")
print("="*60)

def analyze_image_properties(df, sample_size=50):
    """Analyze image dimensions, file sizes, and formats"""
    properties = []
    
    # Sample images for analysis
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        try:
            img_path = row['path']
            img = Image.open(img_path)
            
            properties.append({
                'class': row['class'],
                'width': img.width,
                'height': img.height,
                'aspect_ratio': img.width / img.height,
                'file_size_mb': os.path.getsize(img_path) / (1024 * 1024),
                'format': img.format,
                'mode': img.mode
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return pd.DataFrame(properties)

# Analyze properties
props_df = analyze_image_properties(bach_df)

print("Image Properties Summary:")
print(props_df.describe())

# Visualize dimensions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Width distribution
props_df['width'].hist(bins=20, ax=axes[0,0], alpha=0.7)
axes[0,0].set_title('Image Width Distribution')
axes[0,0].set_xlabel('Width (pixels)')

# Height distribution
props_df['height'].hist(bins=20, ax=axes[0,1], alpha=0.7)
axes[0,1].set_title('Image Height Distribution')
axes[0,1].set_xlabel('Height (pixels)')

# Aspect ratio by class
sns.boxplot(data=props_df, x='class', y='aspect_ratio', ax=axes[1,0])
axes[1,0].set_title('Aspect Ratio by Class')
axes[1,0].tick_params(axis='x', rotation=45)

# File size distribution
props_df['file_size_mb'].hist(bins=20, ax=axes[1,1], alpha=0.7)
axes[1,1].set_title('File Size Distribution')
axes[1,1].set_xlabel('File Size (MB)')

plt.tight_layout()
plt.savefig('results/bach_image_properties.png', dpi=300, bbox_inches='tight')
plt.show()

# Format and mode analysis
print(f"\\nImage Formats: {props_df['format'].value_counts().to_dict()}")
print(f"Color Modes: {props_df['mode'].value_counts().to_dict()}")

print("\\n" + "="*60)
print("🎨 COLOR ANALYSIS")
print("="*60)

def analyze_color_properties(df, sample_size=20):
    """Analyze color properties of images"""
    color_stats = []
    
    # Sample images for analysis
    sample_df = df.groupby('class').apply(
        lambda x: x.sample(min(sample_size//len(df['class'].unique()), len(x)), random_state=42)
    ).reset_index(drop=True)
    
    for _, row in sample_df.iterrows():
        try:
            img = Image.open(row['path']).convert('RGB')
            # Resize for faster processing
            img_small = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img_small)
            
            # Calculate color statistics
            color_stats.append({
                'class': row['class'],
                'mean_r': np.mean(img_array[:,:,0]),
                'mean_g': np.mean(img_array[:,:,1]),
                'mean_b': np.mean(img_array[:,:,2]),
                'std_r': np.std(img_array[:,:,0]),
                'std_g': np.std(img_array[:,:,1]),
                'std_b': np.std(img_array[:,:,2]),
                'brightness': np.mean(img_array),
                'contrast': np.std(img_array)
            })
        except Exception as e:
            print(f"Error processing {row['path']}: {e}")
    
    return pd.DataFrame(color_stats)

# Analyze colors
color_df = analyze_color_properties(bach_df)

if not color_df.empty:
    # Visualize color properties
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RGB means by class
    rgb_means = color_df.groupby('class')[['mean_r', 'mean_g', 'mean_b']].mean()
    rgb_means.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Average RGB Values by Class')
    axes[0,0].set_ylabel('Mean Pixel Value')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(['Red', 'Green', 'Blue'])
    
    # Brightness distribution
    sns.boxplot(data=color_df, x='class', y='brightness', ax=axes[0,1])
    axes[0,1].set_title('Brightness Distribution by Class')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Contrast distribution
    sns.boxplot(data=color_df, x='class', y='contrast', ax=axes[1,0])
    axes[1,0].set_title('Contrast Distribution by Class')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Color variance
    color_variance = color_df.groupby('class')[['std_r', 'std_g', 'std_b']].mean()
    color_variance.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Color Variance by Class')
    axes[1,1].set_ylabel('Standard Deviation')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend(['Red Std', 'Green Std', 'Blue Std'])
    
    plt.tight_layout()
    plt.savefig('results/bach_color_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\\nColor Statistics Summary:")
    print(color_df.groupby('class')[['brightness', 'contrast']].describe())

print("\\n" + "="*60)
print("🔍 DATA QUALITY ASSESSMENT")
print("="*60)

def assess_data_quality(df):
    """Assess data quality issues"""
    issues = []
    
    print("Data Quality Assessment:")
    print("=" * 50)
    
    # Check for missing files
    missing_files = 0
    corrupted_files = 0
    
    for _, row in df.iterrows():
        img_path = row['path']
        
        # Check if file exists
        if not os.path.exists(img_path):
            missing_files += 1
            issues.append(f"Missing file: {img_path}")
            continue
        
        # Check if file can be opened
        try:
            img = Image.open(img_path)
            img.verify()  # Verify image integrity
        except Exception as e:
            corrupted_files += 1
            issues.append(f"Corrupted file: {img_path} - {e}")
    
    print(f"Total images: {len(df)}")
    print(f"Missing files: {missing_files}")
    print(f"Corrupted files: {corrupted_files}")
    print(f"Valid images: {len(df) - missing_files - corrupted_files}")
    
    # Check class balance
    class_counts = df['class'].value_counts()
    min_class_size = class_counts.min()
    max_class_size = class_counts.max()
    imbalance_ratio = max_class_size / min_class_size
    
    print(f"\\nClass Balance:")
    print(f"Most common class: {max_class_size} images")
    print(f"Least common class: {min_class_size} images")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2.0:
        issues.append(f"Significant class imbalance detected: {imbalance_ratio:.2f}:1")
    
    # Check for duplicate filenames
    duplicate_names = df['filename'].duplicated().sum()
    if duplicate_names > 0:
        issues.append(f"Found {duplicate_names} duplicate filenames")
    
    print(f"\\nData Quality Issues Found: {len(issues)}")
    for issue in issues[:10]:  # Show first 10 issues
        print(f"- {issue}")
    
    if len(issues) > 10:
        print(f"... and {len(issues) - 10} more issues")
    
    return issues

# Assess data quality
quality_issues = assess_data_quality(bach_df)

print("\\n" + "="*80)
print("📋 BACH DATASET SUMMARY & RECOMMENDATIONS")
print("="*80)

# Dataset characteristics
total_images = len(bach_df)
n_classes = bach_df['class'].nunique()
class_counts = bach_df['class'].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()

print(f"Dataset Size: {total_images} images")
print(f"Number of Classes: {n_classes}")
print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")

print("\\nRecommendations:")
print("-" * 30)

# Data augmentation
if total_images < 1000:
    print("✓ Use aggressive data augmentation (rotation, flipping, color jittering)")
    print("✓ Consider GAN-based augmentation for synthetic data generation")

# Class imbalance
if imbalance_ratio > 1.5:
    print("✓ Use weighted sampling or class weights to handle imbalance")
    print("✓ Consider focal loss for better handling of hard examples")

# Model architecture
print("✓ Use transfer learning with ImageNet pretrained models")
print("✓ EfficientNet or ResNet architectures recommended for histopathology")

# Training strategy
print("✓ Use stratified splits to maintain class distribution")
print("✓ Implement early stopping and learning rate scheduling")
print("✓ Use cross-validation for robust performance estimation")

# High-resolution considerations
print("✓ BACH images are high-resolution - consider multi-scale training")
print("✓ Use progressive resizing: start with smaller images, increase size")

# Evaluation
print("✓ Use multiple metrics: Accuracy, F1-score, AUC-ROC")
print("✓ Analyze per-class performance and confusion matrices")

# Combined training
if os.path.exists('data/breakhis'):
    print("\\n✓ Consider combining with BreakHis dataset for improved generalization")
    print("✓ Use domain adaptation techniques for multi-dataset training")

print("\\nSuggested Train/Val/Test Split:")
print(f"- Training: {int(total_images * 0.7)} images (70%)")
print(f"- Validation: {int(total_images * 0.15)} images (15%)")
print(f"- Testing: {int(total_images * 0.15)} images (15%)")

# Save results
os.makedirs('results/eda', exist_ok=True)
bach_df.to_csv('results/eda/bach_metadata.csv', index=False)
print("\\n✅ Saved BACH metadata to results/eda/bach_metadata.csv")

print("\\n" + "="*80)
print("✅ BACH EDA COMPLETED!")
print("="*80)