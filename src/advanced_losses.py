#!/usr/bin/env python3
"""
Advanced Loss Functions for Breast Cancer Classification
Implements specialized losses for handling class imbalance and rare subclasses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses learning on hard examples by down-weighting easy examples
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss using effective number of samples
    Addresses class imbalance by reweighting based on effective sample numbers
    """
    
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, loss_type='focal'):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # Calculate effective numbers
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        
        self.register_buffer('weights', torch.FloatTensor(weights))
        
    def forward(self, inputs, targets):
        if self.loss_type == 'focal':
            cb_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
            pt = torch.exp(-cb_loss)
            focal_loss = (1 - pt) ** self.gamma * cb_loss
            return focal_loss.mean()
        else:
            return F.cross_entropy(inputs, targets, weight=self.weights)

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling rare subclasses
    Combines Tversky index with focal mechanism for better rare class handling
    """
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # Convert to one-hot
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Calculate Tversky index for each class
        tp = (inputs_soft * targets_one_hot).sum(dim=0)
        fp = (inputs_soft * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - inputs_soft) * targets_one_hot).sum(dim=0)
        
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + 1e-8)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky.mean()

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Pulls together samples from same class, pushes apart different classes
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal elements
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss

class CombinedLoss(nn.Module):
    """
    Combined loss function using multiple loss components
    Combines classification loss with contrastive learning
    """
    
    def __init__(self, samples_per_class, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for focal loss
        self.gamma = gamma  # Weight for contrastive loss
        
        # Initialize individual losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=2.0)
        self.cb_loss = ClassBalancedLoss(samples_per_class)
        self.supcon_loss = SupConLoss()
        
    def forward(self, logits, features, targets):
        # Classification losses
        ce = self.ce_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        cb = self.cb_loss(logits, targets)
        
        # Contrastive loss (if features provided)
        if features is not None:
            contrastive = self.supcon_loss(features, targets)
        else:
            contrastive = 0
        
        # Combined loss
        total_loss = (self.alpha * ce + 
                     self.beta * focal + 
                     self.beta * cb + 
                     self.gamma * contrastive)
        
        return total_loss, {
            'ce_loss': ce.item(),
            'focal_loss': focal.item(),
            'cb_loss': cb.item(),
            'contrastive_loss': contrastive.item() if isinstance(contrastive, torch.Tensor) else contrastive
        }

def create_loss_function(loss_type, samples_per_class=None, **kwargs):
    """
    Factory function to create loss functions
    
    Args:
        loss_type: 'focal', 'class_balanced', 'focal_tversky', 'supcon', 'combined'
        samples_per_class: List of sample counts per class
        **kwargs: Additional loss-specific parameters
    
    Returns:
        Loss function
    """
    
    if loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 1),
            gamma=kwargs.get('gamma', 2)
        )
    
    elif loss_type == 'class_balanced':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for class_balanced loss")
        return ClassBalancedLoss(
            samples_per_class=samples_per_class,
            beta=kwargs.get('beta', 0.9999),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    elif loss_type == 'focal_tversky':
        return FocalTverskyLoss(
            alpha=kwargs.get('alpha', 0.7),
            beta=kwargs.get('beta', 0.3),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    elif loss_type == 'supcon':
        return SupConLoss(
            temperature=kwargs.get('temperature', 0.07)
        )
    
    elif loss_type == 'combined':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for combined loss")
        return CombinedLoss(
            samples_per_class=samples_per_class,
            alpha=kwargs.get('alpha', 0.5),
            beta=kwargs.get('beta', 0.3),
            gamma=kwargs.get('gamma', 0.2)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")