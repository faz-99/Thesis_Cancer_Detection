#!/usr/bin/env python3
"""
EfficientNetB0 model implementation for breast cancer classification

This module implements a transfer learning approach using EfficientNetB0 as the backbone.
EfficientNetB0 is chosen for its excellent balance of accuracy and efficiency, making it
suitable for medical image classification tasks.

Key features:
- Transfer learning from ImageNet pretrained weights
- Custom classifier head for 8-class breast cancer classification
- Feature extraction capability for interpretability
- Efficient architecture suitable for deployment
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import logging

# Setup module logger
logger = logging.getLogger(__name__)

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNetB0-based classifier for breast cancer histopathology images
    
    This model uses EfficientNetB0 as a feature extractor with a custom classification head.
    EfficientNetB0 provides an optimal balance between accuracy and computational efficiency,
    making it suitable for medical imaging applications.
    
    Architecture:
    - Backbone: EfficientNetB0 (pretrained on ImageNet)
    - Input: 224x224x3 RGB images
    - Output: num_classes logits for classification
    - Parameters: ~5.3M (efficient for deployment)
    
    Args:
        num_classes (int): Number of output classes (default: 8 for BreakHis)
        pretrained (bool): Whether to use ImageNet pretrained weights (default: True)
    """
    
    def __init__(self, num_classes=8, pretrained=True):
        """
        Initialize EfficientNetB0 classifier
        
        Args:
            num_classes (int): Number of classes for final classification
            pretrained (bool): Use ImageNet pretrained weights for transfer learning
        """
        super(EfficientNetB0Classifier, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        logger.info(f"Initializing EfficientNetB0 classifier:")
        logger.info(f"  - Number of classes: {num_classes}")
        logger.info(f"  - Pretrained: {pretrained}")
        
        # Load EfficientNetB0 backbone
        if pretrained:
            logger.info("Loading ImageNet pretrained weights...")
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            logger.info("Initializing model with random weights...")
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get original classifier input features
        original_features = self.backbone.classifier[1].in_features
        logger.info(f"Original classifier input features: {original_features}")
        
        # Replace the final classification layer for our specific task
        # EfficientNetB0 classifier is a Sequential with Dropout + Linear
        self.backbone.classifier[1] = nn.Linear(original_features, num_classes)
        
        logger.info(f"Replaced final layer: {original_features} -> {num_classes}")
        
        # Log model statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input batch of images [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Logits for each class [batch_size, num_classes]
        """
        # Pass through EfficientNetB0 backbone with custom classifier
        return self.backbone(x)
    
    def get_features(self, x):
        """
        Extract feature representations before final classification
        
        This method is useful for:
        - Feature visualization and analysis
        - Transfer learning to other tasks
        - Similarity analysis between images
        - Interpretability studies
        
        Args:
            x (torch.Tensor): Input batch of images [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Feature vectors [batch_size, 1280] for EfficientNetB0
        """
        # Extract features from backbone (before classifier)
        features = self.backbone.features(x)  # [batch_size, 1280, 7, 7]
        
        # Global average pooling to get fixed-size feature vectors
        features = self.backbone.avgpool(features)  # [batch_size, 1280, 1, 1]
        
        # Flatten to feature vectors
        features = torch.flatten(features, 1)  # [batch_size, 1280]
        
        return features
    
    def get_model_info(self):
        """
        Get detailed information about the model architecture
        
        Returns:
            dict: Model information including parameters, architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'EfficientNetB0Classifier',
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 224, 224),
            'feature_dim': 1280
        }
    
    def freeze_backbone(self):
        """
        Freeze backbone parameters for fine-tuning only the classifier
        
        This is useful when you want to:
        - Fine-tune only the final layers
        - Reduce training time and memory usage
        - Prevent overfitting with limited data
        """
        logger.info("Freezing backbone parameters...")
        
        # Freeze all backbone parameters except classifier
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        # Count trainable parameters after freezing
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after freezing: {trainable_params:,}")
    
    def unfreeze_backbone(self):
        """
        Unfreeze all backbone parameters for full fine-tuning
        """
        logger.info("Unfreezing all backbone parameters...")
        
        # Unfreeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # Count trainable parameters after unfreezing
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after unfreezing: {trainable_params:,}")