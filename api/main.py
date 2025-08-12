from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import cv2
import base64
from torch.nn import functional as F

# Import your models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.efficientnet import EfficientNetB0Classifier
try:
    from src.enhanced_efficientnet import EnhancedEfficientNetB0
except ImportError:
    EnhancedEfficientNetB0 = None
try:
    from src.rag_explainer import RAGExplainer, explain_prediction
except ImportError:
    RAGExplainer = None
    explain_prediction = None

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
        # Load regular model
        try:
            model = EfficientNetB0Classifier(num_classes=8, pretrained=False)
            model.load_state_dict(torch.load("models/efficientnet_b0_best.pth", map_location=device))
            print("✅ Loaded original trained model")
        except FileNotFoundError:
            print("⚠️ No saved weights found, using pretrained model")
            model = EfficientNetB0Classifier(num_classes=8, pretrained=True)
        
        model = model.to(device)
        model.eval()
        
        # Initialize RAG explainer with error handling
        if RAGExplainer:
            try:
                rag_explainer = RAGExplainer()
                print("✅ RAG explainer loaded successfully")
            except Exception as e:
                print(f"⚠️ RAG explainer failed to load: {e}")
                rag_explainer = None
        else:
            rag_explainer = None
        
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
            output = model(input_tensor.unsqueeze(0).to(device))
            
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        predicted_class = class_names[prediction.item()]
        confidence_score = confidence.item()
        
        # Generate explanation using RAG
        if rag_explainer:
            try:
                explanation_result = rag_explainer.generate_explanation(
                    prediction.item(), 
                    confidence_score
                )
            except Exception as e:
                print(f"RAG explanation failed: {e}")
                explanation_result = {
                    'prediction': predicted_class,
                    'confidence': confidence_score,
                    'explanation': f"Predicted as {predicted_class} with {confidence_score:.2%} confidence.",
                    'relevant_facts': []
                }
        else:
            explanation_result = {
                'prediction': predicted_class,
                'confidence': confidence_score,
                'explanation': f"Predicted as {predicted_class} with {confidence_score:.2%} confidence.",
                'relevant_facts': []
            }
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = {
            class_names[idx.item()].replace('_', ' ').title(): float(prob)
            for idx, prob in zip(top3_indices, top3_probs)
        }
        print(f"Top 3 predictions: {top3_predictions}")  # Debug print
        
        # Generate Grad-CAM visualization
        try:
            heatmap = generate_gradcam(model, input_tensor, prediction.item())
            original_img, heatmap_img, overlay_img = create_side_by_side_visualization(pil_image, heatmap)
            visualization = {
                "original": original_img,
                "heatmap": heatmap_img,
                "overlay": overlay_img
            }
            print("✅ Visualization generated successfully")
        except Exception as e:
            print(f"❌ Grad-CAM failed: {e}")
            import traceback
            traceback.print_exc()
            visualization = None
        
        # Prepare response
        result = {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "top3": top3_predictions,
            "probabilities": top3_predictions,  # For backward compatibility
            "textual_explanation": explanation_result,
            "risk_level": get_risk_level(predicted_class),
            "visualization": visualization
        }
        
        print(f"Full API response: {result}")
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
        
        # Make basic prediction first
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0).to(device))
            
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        predicted_class = class_names[prediction.item()]
        confidence_score = confidence.item()
        
        # Try to generate comprehensive explanation
        try:
            explanation = explain_prediction(model, input_tensor, rag_explainer, device)
            if 'visual_explanation' in explanation:
                explanation['visual_explanation'] = explanation['visual_explanation'].tolist()
        except Exception as e:
            print(f"Visual explanation failed: {e}")
            # Fallback to basic explanation
            explanation = {
                'prediction': prediction.item(),
                'confidence': confidence_score,
                'textual_explanation': rag_explainer.generate_explanation(
                    prediction.item(), confidence_score
                ) if rag_explainer else {
                    'prediction': predicted_class,
                    'confidence': confidence_score,
                    'explanation': f"Predicted as {predicted_class} with {confidence_score:.2%} confidence.",
                    'relevant_facts': []
                },
                'visual_explanation': None
            }
        
        return JSONResponse(content=explanation)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/classes")
async def get_classes():
    """Get all available cancer types"""
    return {
        "classes": class_names,
        "benign": ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"],
        "malignant": ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
    }

def generate_gradcam(model, input_tensor, target_class):
    """Generate Grad-CAM heatmap"""
    model.eval()
    
    # Hook for gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the last convolutional layer
    target_layer = model.backbone.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    input_tensor.requires_grad_()
    output = model(input_tensor.unsqueeze(0))
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Generate heatmap
    gradients_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(gradients_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations_val[i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam

def create_side_by_side_visualization(image, heatmap):
    """Create side-by-side visualization with original, heatmap, and overlay"""
    # Resize original image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    
    # Create high-contrast heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlayed = 0.5 * img_array + 0.5 * heatmap_colored
    overlayed = np.uint8(overlayed)
    
    # Convert to base64
    def to_base64(img_array):
        pil_img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    
    return (
        to_base64(img_array),           # Original
        to_base64(heatmap_colored),     # Heatmap only
        to_base64(overlayed)            # Overlay
    )

def overlay_heatmap(image, heatmap):
    """Overlay heatmap on original image"""
    # Convert PIL to numpy
    img_array = np.array(image.resize((224, 224)))
    
    # Create heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = 0.6 * img_array + 0.4 * heatmap_colored
    overlayed = np.uint8(overlayed)
    
    # Convert to base64
    pil_img = Image.fromarray(overlayed)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def get_risk_level(prediction):
    """Determine risk level based on prediction"""
    malignant_types = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
    return "high" if prediction in malignant_types else "low"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)