import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class LobularFocusedLoss(nn.Module):
    """Focal loss with extra emphasis on lobular carcinoma"""
    def __init__(self, alpha=None, gamma=2.0, lobular_boost=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lobular_boost = lobular_boost
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Boost lobular carcinoma (class 5)
        lobular_mask = (targets == 5)
        focal_loss[lobular_mask] *= self.lobular_boost
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

class LobularSpecificAugmentation:
    """Augmentations specifically designed for lobular carcinoma patterns"""
    
    @staticmethod
    def get_lobular_transforms():
        return transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        ])

class LobularAttentionModule(nn.Module):
    """Attention module to focus on lobular carcinoma features"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        att = self.attention(x)
        return x * att

def create_lobular_weighted_sampler(dataset, boost_factor=3.0):
    """Create weighted sampler with extra emphasis on lobular carcinoma"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    
    # Calculate weights
    weights = 1.0 / class_counts
    
    # Boost lobular carcinoma (class 5)
    weights[5] *= boost_factor
    
    sample_weights = [weights[label] for label in labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def get_lobular_class_weights():
    """Get class weights with emphasis on lobular carcinoma"""
    # Based on your results, lobular needs more weight
    weights = torch.tensor([
        1.0,  # adenosis
        1.0,  # fibroadenoma  
        1.2,  # phyllodes_tumor
        1.0,  # tubular_adenoma
        1.0,  # ductal_carcinoma
        2.5,  # lobular_carcinoma - increased weight
        1.0,  # mucinous_carcinoma
        1.0   # papillary_carcinoma
    ])
    return weights