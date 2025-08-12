from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
from torchvision import transforms

# Import your models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.efficientnet import EfficientNetB0Classifier
from src.enhanced_efficientnet import EnhancedEfficientNetB0
from src.rag_explainer import RAGExplainer, explain_prediction

app = FastAPI(title="Breast Cancer Detection API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
rag_explainer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mappings
class_names = {
    0: "adenosis",
    1: "fibroadenoma", 
    2: "phyllodes_tumor",
    3: "tubular_adenoma",
    4: "ductal_carcinoma",
    5: "lobular_carcinoma",
    6: "mucinous_carcinoma",
    7: "papillary_carcinoma"
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global model, rag_explainer
    
    try:
        # Try to load lobular-enhanced model first
        try:
            model = EnhancedEfficientNetB0(num_classes=8, pretrained=False)
            model.load_state_dict(torch.load("models/enhanced_efficientnet_lobular_v1.pth", map_location=device))
            print("✅ Loaded lobular-enhanced model v1")
        except FileNotFoundError:
            # Fallback to regular model
            try:
                model = EfficientNetB0Classifier(num_classes=8, pretrained=False)
                model.load_state_dict(torch.load("models/efficientnet_b0_best.pth", map_location=device))
                print("✅ Loaded original trained model")
            except FileNotFoundError:
                print("⚠️ No saved weights found, using pretrained model")
                model = EfficientNetB0Classifier(num_classes=8, pretrained=True)
        
        model = model.to(device)
        model.eval()
        
        # Initialize RAG explainer
        rag_explainer = RAGExplainer()
        print("✅ Models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Breast Cancer Detection API is running", "status": "healthy"}

@app.post("/api/predict")
async def predict_image(image: UploadFile = File(...)):
    """Predict breast cancer type from histopathological image"""
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(pil_image)
        
        # Make prediction
        with torch.no_grad():
            if isinstance(model, EnhancedEfficientNetB0):
                main_output, lobular_output = model(input_tensor.unsqueeze(0).to(device))
                output = main_output
            else:
                output = model(input_tensor.unsqueeze(0).to(device))
            
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        predicted_class = class_names[prediction.item()]
        confidence_score = confidence.item()
        
        # Generate explanation using RAG
        explanation_result = rag_explainer.generate_explanation(
            prediction.item(), 
            confidence_score
        )
        
        # Prepare response
        result = {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "probabilities": {class_names[i]: float(probabilities[0][i]) for i in range(8)},
            "textual_explanation": explanation_result,
            "risk_level": get_risk_level(predicted_class)
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/explain")
async def explain_image(image: UploadFile = File(...)):
    """Generate comprehensive explanation including visual analysis"""
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(pil_image)
        
        # Generate comprehensive explanation
        explanation = explain_prediction(model, input_tensor, rag_explainer, device)
        
        # Convert numpy arrays to lists for JSON serialization
        if 'visual_explanation' in explanation:
            explanation['visual_explanation'] = explanation['visual_explanation'].tolist()
        
        return JSONResponse(content=explanation)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/api/classes")
async def get_classes():
    """Get all available cancer types"""
    return {
        "classes": class_names,
        "benign": ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"],
        "malignant": ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
    }

def get_risk_level(prediction):
    """Determine risk level based on prediction"""
    malignant_types = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
    return "high" if prediction in malignant_types else "low"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)