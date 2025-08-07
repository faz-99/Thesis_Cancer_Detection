import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(768, 256)
        
    def forward(self, text_inputs):
        with torch.no_grad():
            outputs = self.model(**text_inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.projection(embeddings)

class ImageEncoder(nn.Module):
    def __init__(self, backbone_model, feature_dim=256):
        super().__init__()
        self.backbone = backbone_model
        # Remove final classification layer
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        self.projection = nn.Linear(in_features, feature_dim)
        
    def forward(self, images):
        features = self.backbone(images)
        return self.projection(features)

class MultimodalFusion(nn.Module):
    def __init__(self, image_dim=256, text_dim=256, hidden_dim=512, num_classes=8):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention mechanism
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, image_features, text_features):
        # Project to same dimension
        img_proj = self.image_proj(image_features).unsqueeze(1)  # (B, 1, hidden_dim)
        text_proj = self.text_proj(text_features).unsqueeze(1)   # (B, 1, hidden_dim)
        
        # Cross-attention
        img_attended, _ = self.cross_attn(img_proj, text_proj, text_proj)
        text_attended, _ = self.cross_attn(text_proj, img_proj, img_proj)
        
        # Concatenate and fuse
        fused = torch.cat([img_attended.squeeze(1), text_attended.squeeze(1)], dim=1)
        output = self.fusion(fused)
        
        return output

class MultimodalModel(nn.Module):
    def __init__(self, image_backbone, num_classes=8):
        super().__init__()
        self.image_encoder = ImageEncoder(image_backbone)
        self.text_encoder = TextEncoder()
        self.fusion = MultimodalFusion(num_classes=num_classes)
        
    def forward(self, images, text_inputs=None):
        image_features = self.image_encoder(images)
        
        if text_inputs is not None:
            text_features = self.text_encoder(text_inputs)
            return self.fusion(image_features, text_features)
        else:
            # Image-only inference
            return self.fusion.fusion(torch.cat([image_features, torch.zeros_like(image_features)], dim=1))

def create_expert_cues():
    """Generate expert medical cues for each class"""
    expert_cues = {
        'adenosis': 'Benign breast condition with enlarged lobules and increased glandular tissue',
        'fibroadenoma': 'Benign solid tumor with well-defined borders and uniform cellular pattern',
        'phyllodes_tumor': 'Rare breast tumor with leaf-like architecture and stromal proliferation',
        'tubular_adenoma': 'Benign tumor composed of uniform tubular structures',
        'ductal_carcinoma': 'Malignant tumor arising from milk ducts with invasive growth pattern',
        'lobular_carcinoma': 'Malignant tumor from lobules with single-file growth pattern',
        'mucinous_carcinoma': 'Malignant tumor with abundant mucin production and favorable prognosis',
        'papillary_carcinoma': 'Malignant tumor with papillary architecture and central fibrovascular cores'
    }
    return expert_cues

def prepare_text_inputs(labels, tokenizer, expert_cues, device='cpu'):
    """Prepare text inputs from expert cues"""
    class_names = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
                   'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
    
    texts = [expert_cues[class_names[label.item()]] for label in labels]
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    return {k: v.to(device) for k, v in inputs.items()}

def train_multimodal(model, dataloader, num_epochs=50, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    expert_cues = create_expert_cues()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Prepare text inputs
            text_inputs = prepare_text_inputs(labels, model.text_encoder.tokenizer, expert_cues, device)
            
            optimizer.zero_grad()
            outputs = model(images, text_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model