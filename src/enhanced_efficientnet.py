import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from lobular_enhancement import LobularAttentionModule

class EnhancedEfficientNetB0(nn.Module):
    """EfficientNetB0 with lobular carcinoma enhancement"""
    
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        
        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b0')
        
        # Get feature dimension
        feature_dim = self.backbone._fc.in_features
        
        # Remove original classifier
        self.backbone._fc = nn.Identity()
        
        # Add lobular attention
        self.lobular_attention = LobularAttentionModule(feature_dim)
        
        # Enhanced classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Lobular-specific branch
        self.lobular_branch = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # lobular vs non-lobular
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attended_features = self.lobular_attention(features.unsqueeze(-1).unsqueeze(-1))
        attended_features = attended_features.squeeze(-1).squeeze(-1)
        
        # Main classification
        main_output = self.classifier(attended_features)
        
        # Lobular-specific classification
        lobular_output = self.lobular_branch(attended_features)
        
        return main_output, lobular_output
    
    def get_features(self, x):
        """Extract features for analysis"""
        return self.backbone(x)