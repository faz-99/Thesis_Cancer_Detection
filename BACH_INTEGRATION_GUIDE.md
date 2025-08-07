# BACH Dataset Integration Guide

This guide explains how to integrate and use the BACH dataset alongside BreakHis for improved breast cancer detection.

## Overview

The BACH (BreAst Cancer Histology) dataset from ICIAR 2018 challenge provides:
- **4 classes**: Normal, Benign, In Situ Carcinoma, Invasive Carcinoma
- **400 high-resolution images** (100 per class)
- **2048 x 1536 pixel resolution**
- **Complementary to BreakHis** for better generalization

## Quick Setup

### 1. Download BACH Dataset

```bash
# Run the dataset downloader
python src/dataset_downloader.py

# This creates the directory structure and provides download instructions
```

### 2. Manual Download Options

**Official Source:**
- Visit: https://iciar2018-challenge.grand-challenge.org/Dataset/
- Register and download training data
- Extract to `data/bach/`

**Alternative Sources:**
- Kaggle: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
- Papers with Code: https://paperswithcode.com/dataset/bach

### 3. Verify Setup

```bash
# Test the integration
python test_bach_integration.py
```

## Training Options

### Option 1: BreakHis Only (Original)
```bash
python main_training.py
```
- Uses 8 BreakHis subclasses
- Patient-wise splitting
- ~7,900 images

### Option 2: Combined Training (Recommended)
```bash
python main_combined_training.py
```
- Uses unified class mapping
- Combines both datasets
- ~8,300 total images
- Better generalization

## Class Mapping Strategy

### BreakHis Classes → Unified Classes
- `benign` subclasses → **Benign**
- `malignant` subclasses → **Malignant**

### BACH Classes → Unified Classes
- `Normal` → **Normal**
- `Benign` → **Benign**
- `InSitu` → **InSitu**
- `Invasive` → **Invasive**

### Final Unified Classes
1. **Normal** (BACH only)
2. **Benign** (Both datasets)
3. **InSitu** (BACH only)
4. **Invasive** (BACH only)
5. **Malignant** (BreakHis only)

## Code Examples

### Load BACH Metadata
```python
from src.bach_data_utils import create_bach_metadata

bach_metadata = create_bach_metadata("data/bach")
print(f"BACH images: {len(bach_metadata)}")
```

### Combined Dataset Training
```python
from src.bach_data_utils import create_combined_metadata
from src.efficientnet import EfficientNetB0Classifier

# Create combined metadata
metadata = create_combined_metadata(
    breakhis_root="data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
    bach_root="data/bach"
)

# Train model with 5 unified classes
model = EfficientNetB0Classifier(num_classes=5)
```

### Performance Comparison
```python
from src.dataset_comparison import compare_dataset_performance

# Compare single vs multi-dataset training
results = compare_dataset_performance(
    breakhis_model_path="models/breakhis_efficientnet_b0_best.pth",
    combined_model_path="models/combined_efficientnet_b0_best.pth",
    test_loaders_dict=test_loaders,
    class_mappings_dict=class_mappings,
    device=device
)
```

## Expected Benefits

### 1. Improved Generalization
- Training on diverse imaging conditions
- Better performance on unseen data
- Reduced overfitting to single dataset characteristics

### 2. Enhanced Class Coverage
- Normal tissue samples (missing in BreakHis)
- More detailed malignancy staging
- Broader histological patterns

### 3. Robustness Testing
- Cross-dataset evaluation
- Domain adaptation analysis
- Real-world performance assessment

## File Structure After Setup

```
data/
├── breakhis/
│   └── BreaKHis_v1/
│       └── BreaKHis_v1/
│           └── histology_slides/
│               └── breast/
│                   ├── benign/
│                   └── malignant/
└── bach/
    ├── Normal/
    ├── Benign/
    ├── InSitu/
    └── Invasive/
```

## Troubleshooting

### Common Issues

1. **BACH dataset not found**
   ```bash
   # Verify directory structure
   ls -la data/bach/
   
   # Run verification
   python -c "from src.dataset_downloader import verify_bach_dataset; verify_bach_dataset('data/bach')"
   ```

2. **Memory issues with high-resolution images**
   - BACH images are resized to 512x512 before final 224x224 transform
   - Reduce batch size if needed
   - Use `num_workers=0` in DataLoader

3. **Class imbalance warnings**
   - Normal for combined dataset due to different dataset sizes
   - Weighted sampling handles this automatically

### Performance Tips

1. **Use combined training for better results**
2. **Increase epochs (15-20) for combined dataset**
3. **Monitor both dataset-specific and overall metrics**
4. **Use cross-dataset evaluation for robustness testing**

## Research Applications

This integration enables several research directions:

1. **Domain Adaptation Studies**
   - How well do models transfer between datasets?
   - What features are dataset-specific vs universal?

2. **Multi-Dataset Learning**
   - Optimal strategies for combining heterogeneous datasets
   - Class balancing across different data sources

3. **Generalization Analysis**
   - Performance degradation across domains
   - Robustness to imaging condition variations

4. **Clinical Validation**
   - More comprehensive evaluation
   - Better real-world performance estimates

## Citation

If you use this integration in your research, please cite both datasets:

```bibtex
@article{spanhol2016dataset,
  title={A dataset for breast cancer histopathological image classification},
  author={Spanhol, Fabio A and Oliveira, Luiz S and Petitjean, Caroline and Heutte, Laurent},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={63},
  number={7},
  pages={1455--1462},
  year={2016}
}

@inproceedings{aresta2019bach,
  title={BACH: Grand challenge on breast cancer histology images},
  author={Aresta, Guilherme and Ara{\'u}jo, Teresa and Kwok, Scotty and Chennamsetty, Sai Saketh and Safwan, Mohammed and Alex, Varghese and Marami, Bahram and Prastawa, Marcel and Chan, Monica and Donovan, Michael and others},
  booktitle={Medical Image Analysis},
  volume={56},
  pages={122--139},
  year={2019}
}
```