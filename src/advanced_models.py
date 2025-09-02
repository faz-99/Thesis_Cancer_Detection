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

class ViTClassifier(nn.Module):
    """
    Vision Transformer for breast cancer classification
    Standard ViT implementation with patch-based attention
    """
    
    def __init__(self, num_classes=8, pretrained=True, model_size='base'):
        super().__init__()
        self.num_classes = num_classes
        
        model_names = {
            'base': 'vit_base_patch16_224',
            'small': 'vit_small_patch16_224',
            'large': 'vit_large_patch16_224'
        }
        
        logger.info(f"Loading Vision Transformer {model_size}")
        self.backbone = timm.create_model(
            model_names[model_size],
            pretrained=pretrained,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        return self.backbone.forward_features(x)

class MultiScaleAttentionCNN(nn.Module):
    """
    Multi-scale CNN with attention for magnification generalization
    Handles different magnification levels through multi-scale processing
    """
    
    def __init__(self, num_classes=8, backbone='efficientnet_b0'):
        super().__init__()
        
        # Load backbone
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = 1280
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        
        # Remove final classifier
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        # Multi-scale attention
        self.scale_attention = nn.MultiheadAttention(feature_dim, num_heads=8)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention (treating batch as sequence)
        features = features.unsqueeze(1)  # [B, 1, D]
        attended_features, _ = self.scale_attention(features, features, features)
        attended_features = attended_features.squeeze(1)  # [B, D]
        
        return self.classifier(attended_features)

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
        self.vit = ViTClassifier(num_classes, model_size='small')
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        # Get predictions from each model
        eff_out = self.efficientnet(x)
        swin_out = self.swin(x)
        vit_out = self.vit(x)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = (weights[0] * eff_out + 
                       weights[1] * swin_out + 
                       weights[2] * vit_out)
        
        return ensemble_out, (eff_out, swin_out, vit_out)

def create_advanced_model(model_type='swin', num_classes=8, **kwargs):
    """
    Factory function to create advanced models
    
    Args:
        model_type: 'swin', 'vit', 'multiscale', 'ensemble'
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
    
    elif model_type == 'vit':
        return ViTClassifier(
            num_classes=num_classes,
            model_size=kwargs.get('size', 'base')
        )
    
    elif model_type == 'multiscale':
        return MultiScaleAttentionCNN(
            num_classes=num_classes,
            backbone=kwargs.get('backbone', 'efficientnet_b0')
        )
    
    elif model_type == 'ensemble':
        return EnsembleModel(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")