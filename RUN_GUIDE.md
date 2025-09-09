# 🚀 Complete Setup & Run Guide

## Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements_explainability.txt

# Install Node.js dependencies (for frontend)
cd frontend
npm install
cd ..
```

## 🔧 Setup Steps

### 1. Prepare Data (if needed)
```bash
# Download BreakHis dataset (if not already done)
python src/dataset_downloader.py
```

### 2. Train Model (optional - use existing model)
```bash
# Quick training
python main_training.py

# Or use existing model at models/efficientnet_b0_best.pth
```

## 🎯 Run Explainability System

### Option A: Complete System (Recommended)
```bash
# Terminal 1: Start Backend API
python run_explainability_server.py

# Terminal 2: Start Frontend
cd frontend
npm run dev

# Access: http://localhost:3000/explainability
```

### Option B: Backend Only (API Testing)
```bash
# Start API server
python run_explainability_server.py

# Test with curl
curl -X POST "http://localhost:8000/analyze" -F "file=@path/to/image.png"

# API docs: http://localhost:8000/docs
```

### Option C: Demo Script (No Frontend)
```bash
# Run standalone demo
python demo_explainability.py

# Check outputs in: outputs/demo_explainability/
```

## 📁 File Structure Check
```
Thesis_Cancer_Detection/
├── src/
│   ├── explainability_pipeline.py ✓
│   ├── efficientnet.py ✓
│   └── nuclei_segmentation.py ✓
├── api/
│   └── explainability_api.py ✓
├── frontend/
│   ├── components/
│   │   └── ExplainabilityAnalysis.vue ✓
│   └── pages/
│       └── explainability.vue ✓
├── models/
│   └── efficientnet_b0_best.pth (or similar)
└── data/breakhis/ (dataset)
```

## 🔍 Testing
```bash
# Test pipeline components
python test_explainability.py

# Test with sample image
python -c "
from src.explainability_pipeline import ExplainabilityPipeline
from src.efficientnet import EfficientNetB0Classifier
import os

model = EfficientNetB0Classifier(num_classes=8)
pipeline = ExplainabilityPipeline(model)
print('✅ Pipeline ready')
"
```

## 🌐 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000/explainability | Main UI |
| API | http://localhost:8000 | Backend API |
| API Docs | http://localhost:8000/docs | Interactive API docs |

## 🎮 Usage Flow

1. **Upload Image**: Drag & drop or click to select histopathology image
2. **Analyze**: Click "Analyze Image" button
3. **View Results**:
   - Prediction class & confidence
   - Human-readable explanation
   - Grad-CAM heatmap overlay
   - Nuclei segmentation
   - Quantitative metrics

## 🛠️ Troubleshooting

### Common Issues

**Backend won't start:**
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python run_explainability_server.py
```

**Frontend won't start:**
```bash
cd frontend
npm install  # Reinstall dependencies
npm run dev
```

**Model not found:**
```bash
# Use demo model (creates random weights)
python demo_explainability.py
```

**CORS errors:**
- Ensure backend is running on port 8000
- Check browser console for specific errors

### Quick Fixes
```bash
# Reset everything
rm -rf outputs/ uploads/
mkdir -p outputs uploads
python run_explainability_server.py
```

## 📊 Expected Output

### API Response Format:
```json
{
  "prediction": {
    "class": "Ductal_carcinoma",
    "confidence": 0.8745
  },
  "explanation": {
    "text": "Prediction: Ductal_carcinoma...",
    "phrases": ["activated region largely overlaps nuclei clusters"],
    "nuclei_count": 45,
    "overlap_ratio": 0.73,
    "faithfulness": true,
    "confidence_drop": 0.23
  },
  "images": {
    "heatmap": "base64_image_data",
    "overlay": "base64_image_data",
    "activation_mask": "base64_image_data",
    "nuclei_mask": "base64_image_data"
  }
}
```

## 🎯 Success Indicators

✅ Backend starts without errors  
✅ Frontend loads at localhost:3000  
✅ Image upload works  
✅ Analysis completes successfully  
✅ Visualizations display correctly  
✅ Explanations are generated  

## 📞 Support

If issues persist:
1. Check all dependencies are installed
2. Verify file paths in configuration
3. Ensure ports 3000 and 8000 are available
4. Check console logs for specific errors