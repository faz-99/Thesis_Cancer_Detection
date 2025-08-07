# Breast Cancer Detection Thesis

## Overview
This project implements a comprehensive breast cancer detection system using deep learning techniques on the BreakHis dataset.

## Project Structure
```
├── data/                   # Dataset storage
├── models/                 # Saved model weights
├── src/                    # Source code
│   ├── data_utils.py      # Data processing utilities
│   ├── efficientnet.py    # EfficientNetB0 model
│   ├── train.py           # Training utilities
│   └── inference/         # Inference code
├── notebooks/             # Jupyter notebooks
├── api/                   # FastAPI backend
├── frontend/              # Vue.js frontend
├── main_training.py       # Main training script
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
Ensure your BreakHis dataset is in the following structure:
```
data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
├── benign/
└── malignant/
```

### 3. Train Model
```bash
python main_training.py
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

#### Training
```python
from src.efficientnet import EfficientNetB0Classifier
from src.train import train_model

model = EfficientNetB0Classifier(num_classes=8)
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
- **Classes**: 8 (4 benign + 4 malignant)
- **Magnifications**: 40X, 100X, 200X, 400X
- **Total Images**: ~7,900
- **Split**: 70% train, 15% val, 15% test (patient-wise)

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