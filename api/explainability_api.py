from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch
import json
import base64
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

# Add src to path
sys.path.append('../src')

from explainability_pipeline import ExplainabilityPipeline
from efficientnet import EfficientNetB0Classifier

app = FastAPI(title="Breast Cancer Explainability API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline = None
model = None

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs/api"
STATIC_DIR = "static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

def load_model():
    """Load the trained model"""
    global model, pipeline
    
    model_path = "../models/efficientnet_breakhis_best.pth"
    model = EfficientNetB0Classifier(num_classes=8)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Loaded trained model")
    else:
        print("⚠️ Using untrained model for demo")
    
    pipeline = ExplainabilityPipeline(model)
    return True

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check"""
    return {"message": "Breast Cancer Explainability API", "status": "running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze uploaded image and generate explanation"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image content
        content = await file.read()
        
        # Save uploaded file with timestamp
        import time
        timestamp = str(int(time.time()))
        file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process image
        metrics = pipeline.process_image(file_path, OUTPUT_DIR)
        
        # Prepare response with base64 encoded images
        base_name = os.path.splitext(f"{timestamp}_{file.filename}")[0]
        
        def image_to_base64(image_path):
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            return None
        
        # Read explanation text
        explanation_text = ""
        explanation_path = os.path.join(OUTPUT_DIR, f"{base_name}_explanation.txt")
        if os.path.exists(explanation_path):
            with open(explanation_path, 'r') as f:
                explanation_text = f.read()
        
        response = {
            "prediction": {
                "class": metrics['prediction']['class_name'],
                "confidence": round(metrics['prediction']['confidence'], 4)
            },
            "explanation": {
                "text": explanation_text,
                "phrases": metrics['explanation_phrases'],
                "nuclei_count": metrics['overlap_analysis']['n_total'],
                "overlap_ratio": round(metrics['overlap_analysis']['ratio'], 3),
                "faithfulness": metrics['faithfulness']['is_faithful'],
                "confidence_drop": round(metrics['faithfulness']['confidence_drop'], 3)
            },
            "images": {
                "heatmap": image_to_base64(os.path.join(OUTPUT_DIR, f"{base_name}_heatmap.png")),
                "overlay": image_to_base64(os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png")),
                "activation_mask": image_to_base64(os.path.join(OUTPUT_DIR, f"{base_name}_activation_mask.png")),
                "nuclei_mask": image_to_base64(os.path.join(OUTPUT_DIR, f"{base_name}_nuclei_mask.png"))
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/predict")
async def predict_only(file: UploadFile = File(...)):
    """Simple prediction without explainability"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        content = await file.read()
        image = Image.open(BytesIO(content))
        
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Get prediction using pipeline
        tensor_img, _ = pipeline.load_and_preprocess_image_from_array(img_array)
        pred_class, confidence, _ = pipeline.get_prediction(tensor_img)
        
        return JSONResponse(content={
            "class": pipeline.class_names.get(pred_class, f"Class_{pred_class}"),
            "confidence": round(confidence, 4)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Add method to pipeline for array input
def add_array_method():
    """Add method to process numpy arrays"""
    def load_and_preprocess_image_from_array(self, image_array):
        """Load from numpy array instead of file"""
        original_img = image_array
        pil_img = Image.fromarray(original_img)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor_img, original_img
    
    ExplainabilityPipeline.load_and_preprocess_image_from_array = load_and_preprocess_image_from_array

if __name__ == "__main__":
    add_array_method()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)