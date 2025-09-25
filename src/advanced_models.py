#!/usr/bin/env python3
"""
Advanced Model Architectures for Breast Cancer Detection
Implements Vision Transformers and enhanced CNN architectures beyond EfficientNet baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import logging

logger = logging.getLogger(__name__)

class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer for breast cancer classification
    Provides hierarchical vision transformer with shifted windows
    """
    
    def __init__(self, num_classes=8, pretrained=True, model_size='tiny'):
        super().__init__()
        self.num_classes = num_classes
        
        # Load Swin Transformer variants
        model_names = {
            'tiny': 'swin_tiny_patch4_window7_224',
            'small': 'swin_small_patch4_window7_224',
            'base': 'swin_base_patch4_window7_224'
        }
        
        logger.info(f"Loading Swin Transformer {model_size}")
        self.backbone = timm.create_model(
            model_names[model_size], 
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features before classification"""
        return self.backbone.forward_features(x)





class EnsembleModel(nn.Module):
    """
    Ensemble of CNN and Transformer models
    Combines EfficientNet, Swin Transformer, and ViT predictions
    """
    
    def __init__(self, num_classes=8):
        super().__init__()
        
        # Individual models
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier[1] = nn.Linear(1280, num_classes)
        
        self.swin = SwinTransformerClassifier(num_classes, model_size='tiny')
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
        
    def forward(self, x):
        # Get predictions from each model
        eff_out = self.efficientnet(x)
        swin_out = self.swin(x)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = (weights[0] * eff_out + 
                       weights[1] * swin_out)
        
        return ensemble_out, (eff_out, swin_out)

def create_advanced_model(model_type='swin', num_classes=8, **kwargs):
    """
    Factory function to create advanced models
    
    Args:
        model_type: 'swin', 'ensemble'
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    """
    
    if model_type == 'swin':
        return SwinTransformerClassifier(
            num_classes=num_classes,
            model_size=kwargs.get('size', 'tiny')
        )
    
    
    elif model_type == 'ensemble':
        return EnsembleModel(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")