# Breast Cancer Detection Thesis

## Overview
This project implements a comprehensive breast cancer detection system using deep learning techniques on the BreakHis and BACH datasets.

## Project Structure
```
├── data/                   # Dataset storage
│   ├── breakhis/          # BreakHis dataset
│   └── bach/              # BACH dataset
├── models/                 # Saved model weights
├── src/                    # Source code
│   ├── data_utils.py      # BreakHis data utilities
│   ├── bach_data_utils.py # BACH data utilities
│   ├── dataset_downloader.py # Dataset download utilities
│   ├── efficientnet.py    # EfficientNetB0 model
│   ├── train.py           # Training utilities
│   └── inference/         # Inference code
├── notebooks/             # Jupyter notebooks
├── api/                   # FastAPI backend
├── frontend/              # Vue.js frontend
├── main_training.py       # BreakHis training script
├── main_combined_training.py # Combined datasets training
└── requirements.txt       # Dependencies
```

## Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

#### BreakHis Dataset
Ensure your BreakHis dataset is in the following structure:
```
data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
├── benign/
└── malignant/
```

#### BACH Dataset
Download and setup the BACH dataset:
```bash
python src/dataset_downloader.py
```

Or manually create the structure:
```
data/bach/
├── Normal/
├── Benign/
├── InSitu/
└── Invasive/
```

Download BACH dataset from:
- Official: https://iciar2018-challenge.grand-challenge.org/Dataset/
- Kaggle: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

### 3. Train Model

#### BreakHis Only
```bash
python main_training.py
```

#### Combined BreakHis + BACH
```bash
python main_combined_training.py
```

### 4. Run API Server
```bash
cd api
uvicorn main:app --reload
```

## Implementation Steps Completed

✅ **Step 1: Data Pipeline**
- BreakHis dataset loading
- Patient-wise stratified splits
- Metadata extraction

✅ **Step 2: Preprocessing**
- Image resizing (224x224)
- Normalization with ImageNet stats
- Data augmentation

✅ **Step 3: Class Imbalance Handling**
- Weighted random sampling
- Class weight calculation

✅ **Step 4: Model Architecture**
- EfficientNetB0 baseline implementation
- Transfer learning from ImageNet

✅ **Step 5: Training Pipeline**
- Training and validation loops
- Model checkpointing
- Metrics tracking

## Next Steps for Full Thesis

### Remaining Components:
1. **GAN-based Data Augmentation** (Step 3)
2. **SupConViT Implementation** (Step 4)
3. **Multimodal Learning** (Step 5)
4. **Magnification Robustness** (Step 6)
5. **RAG-based Interpretability** (Step 7)
6. **Frontend Development** (Step 8)

### Usage Examples

#### BreakHis Training
```python
from src.efficientnet import EfficientNetB0Classifier
from src.train import train_model

model = EfficientNetB0Classifier(num_classes=8)
trained_model, history = train_model(model, train_loader, val_loader)
```

#### Combined Dataset Training
```python
from src.bach_data_utils import create_combined_metadata
from src.efficientnet import EfficientNetB0Classifier

# Load combined metadata
metadata = create_combined_metadata(breakhis_root, bach_root)

# Train with unified classes
model = EfficientNetB0Classifier(num_classes=5)  # Unified classes
trained_model, history = train_model(model, train_loader, val_loader)
```

#### Inference
```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    predictions = torch.max(outputs, 1)[1]
```

## Dataset Information

### BreakHis Dataset
- **Classes**: 8 (4 benign + 4 malignant)
- **Magnifications**: 40X, 100X, 200X, 400X
- **Total Images**: ~7,900
- **Split**: 70% train, 15% val, 15% test (patient-wise)

### BACH Dataset
- **Classes**: 4 (Normal, Benign, In Situ Carcinoma, Invasive Carcinoma)
- **Resolution**: High-resolution (2048 x 1536 pixels)
- **Total Images**: 400 (100 per class)
- **Split**: 70% train, 15% val, 15% test

### Combined Dataset
- **Unified Classes**: Normal, Benign, InSitu, Invasive, Malignant
- **Total Images**: ~8,300
- **Benefits**: Improved generalization across different imaging conditions

## Performance Metrics
- Accuracy, Precision, Recall, F1-score
- Per-class and per-magnification analysis
- Confusion matrices
- ROC curves

## Citation
```bibtex
@misc{thesis2024,
  title={Breast Cancer Detection using Deep Learning},
  author={Your Name},
  year={2024}
}
```