import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(EfficientNetB0Classifier, self).__init__()
        
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Replace classifier
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before final classification"""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features