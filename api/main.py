# api/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io
import torch
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from src.inference.explainer import generate_gradcam

app = FastAPI()

# ---- Load Model on Startup ----
num_classes = 8  # 🔁 Update this based on your dataset

weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("models/efficientnet_focal.pth", map_location="cpu"))
model.eval()

# ---- Transform for Inference ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/explain")
async def explain(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    output_image_path = "cam_output.jpg"
    generate_gradcam(model, input_tensor, target_class=None, save_path=output_image_path)

    return FileResponse(output_image_path, media_type="image/jpeg")
