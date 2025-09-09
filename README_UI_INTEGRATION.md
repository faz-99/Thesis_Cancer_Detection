# UI Integration for Explainability Pipeline

## Overview
Complete integration of the explainability pipeline with Vue.js frontend, providing:
- Image upload interface
- Real-time classification
- Grad-CAM heatmap visualization  
- Human-readable explanations
- Nuclei segmentation results

## Architecture

```
Frontend (Vue/Nuxt)     Backend (FastAPI)        AI Pipeline
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ExplainabilityAnalysis │ ←→ │ /analyze endpoint │ ←→ │ ExplainabilityPipeline │
│ - Image upload  │    │ - File processing │    │ - EfficientNet  │
│ - Results display│    │ - Base64 encoding │    │ - Grad-CAM      │
│ - Visualizations│    │ - CORS handling   │    │ - Nuclei seg    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Start Backend Server
```bash
python run_explainability_server.py
```

### 2. Start Frontend (in separate terminal)
```bash
cd frontend
npm run dev
```

### 3. Access Application
- Frontend: http://localhost:3000/explainability
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Features

### 🔬 **Image Analysis**
- Drag & drop or click to upload
- Supports PNG, JPG, JPEG formats
- Real-time processing feedback

### 🎯 **Classification Results**
- Predicted class with confidence score
- 8-class BreakHis classification
- Visual confidence indicators

### 🧠 **Explainable AI**
- Human-readable explanations
- Key findings extraction
- Clinical relevance mapping

### 📊 **Visual Analysis**
- **Grad-CAM Overlay**: Shows model attention on original image
- **Activation Heatmap**: Pure heatmap visualization
- **Activation Mask**: Binary mask of important regions
- **Nuclei Segmentation**: Detected cellular structures

### 📈 **Quantitative Metrics**
- **Nuclei Count**: Number of detected nuclei
- **Overlap Ratio**: Percentage of nuclei in activated regions
- **Confidence Drop**: Faithfulness validation metric
- **Faithful**: Whether explanation is reliable

## API Endpoints

### POST `/analyze`
Complete explainability analysis
```json
{
  "prediction": {
    "class": "Ductal_carcinoma",
    "confidence": 0.8745
  },
  "explanation": {
    "text": "Prediction: Ductal_carcinoma (confidence: 0.87)...",
    "phrases": ["activated region largely overlaps nuclei clusters"],
    "nuclei_count": 45,
    "overlap_ratio": 0.73,
    "faithfulness": true,
    "confidence_drop": 0.23
  },
  "images": {
    "heatmap": "base64_encoded_image",
    "overlay": "base64_encoded_image",
    "activation_mask": "base64_encoded_image", 
    "nuclei_mask": "base64_encoded_image"
  }
}
```

### POST `/predict`
Simple classification only
```json
{
  "class": "Ductal_carcinoma",
  "confidence": 0.8745
}
```

## Vue Component Usage

```vue
<template>
  <div>
    <ExplainabilityAnalysis />
  </div>
</template>
```

The component handles:
- File upload with drag & drop
- API communication
- Results visualization
- Error handling
- Loading states

## Customization

### Adding New Explanation Rules
Extend `generate_explanation_rules()` in `explainability_pipeline.py`:

```python
def generate_explanation_rules(self, overlap_data, texture_data, activation_mask):
    phrases = []
    
    # Add custom rules
    if custom_condition:
        phrases.append("custom medical finding")
    
    return phrases
```

### Styling Modifications
Update component styles in `ExplainabilityAnalysis.vue`:

```vue
<style scoped>
.custom-class {
  /* Your custom styles */
}
</style>
```

### API Response Format
Modify response structure in `explainability_api.py`:

```python
response = {
    "prediction": {...},
    "explanation": {...},
    "images": {...},
    "custom_field": custom_data
}
```

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure backend CORS is configured for frontend URL
   - Check browser console for specific errors

2. **Image Upload Fails**
   - Verify file size limits
   - Check supported formats (PNG, JPG, JPEG)
   - Ensure backend uploads directory exists

3. **Model Loading Errors**
   - Check model path in `explainability_api.py`
   - Verify model file exists and is accessible
   - Check PyTorch version compatibility

4. **Visualization Issues**
   - Ensure base64 encoding is working
   - Check image generation in pipeline
   - Verify output directory permissions

### Performance Optimization

1. **Model Loading**
   - Load model once on startup
   - Use GPU if available
   - Consider model quantization

2. **Image Processing**
   - Resize large images before processing
   - Use efficient image formats
   - Implement caching for repeated analyses

3. **Frontend**
   - Lazy load components
   - Optimize image display
   - Add progress indicators

## Development

### Adding New Features

1. **Backend**: Extend `ExplainabilityPipeline` class
2. **API**: Add new endpoints in `explainability_api.py`
3. **Frontend**: Update Vue component with new UI elements

### Testing

```bash
# Test backend
python test_explainability.py

# Test API
curl -X POST "http://localhost:8000/analyze" -F "file=@test_image.png"

# Test frontend
npm run test
```

## Deployment

### Production Setup

1. **Backend**
   ```bash
   gunicorn explainability_api:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Frontend**
   ```bash
   npm run build
   npm run start
   ```

3. **Docker** (optional)
   ```dockerfile
   FROM python:3.9
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "api.explainability_api:app", "--host", "0.0.0.0"]
   ```

## Security Considerations

- Validate uploaded file types and sizes
- Sanitize file names
- Implement rate limiting
- Use HTTPS in production
- Secure API endpoints with authentication if needed

## Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Export results to PDF reports
- [ ] Integration with medical databases
- [ ] Real-time collaboration features
- [ ] Mobile-responsive design improvements