#!/usr/bin/env python3
"""
Enhanced Preprocessing for Cross-Dataset Robustness

Advanced preprocessing techniques to improve cross-dataset generalization:
1. Stain Normalization (Macenko, Reinhard)
2. Multi-scale Augmentation
3. Domain-specific Normalization
4. Adaptive Histogram Equalization
5. Color Constancy
"""

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class StainNormalization:
    """Stain normalization using Macenko method"""
    
    def __init__(self, target_stain_matrix=None):
        # Default target stain matrix (H&E)
        if target_stain_matrix is None:
            self.target_stain_matrix = np.array([
                [0.5626, 0.2159],
                [0.7201, 0.8012],
                [0.4062, 0.5581]
            ])
        else:
            self.target_stain_matrix = target_stain_matrix
    
    def normalize_stain(self, image):
        """Apply Macenko stain normalization"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to OD space
        od = self.rgb_to_od(image)
        
        # Estimate stain matrix
        stain_matrix = self.estimate_stain_matrix(od)
        
        if stain_matrix is None:
            return Image.fromarray(image)
        
        # Normalize
        normalized_od = self.normalize_od(od, stain_matrix, self.target_stain_matrix)
        
        # Convert back to RGB
        normalized_rgb = self.od_to_rgb(normalized_od)
        
        return Image.fromarray(normalized_rgb)
    
    def rgb_to_od(self, rgb):
        """Convert RGB to Optical Density"""
        rgb = rgb.astype(np.float64)
        rgb = np.maximum(rgb, 1)  # Avoid log(0)
        od = -np.log(rgb / 255.0)
        return od
    
    def od_to_rgb(self, od):
        """Convert Optical Density to RGB"""
        rgb = np.exp(-od) * 255
        rgb = np.clip(rgb, 0, 255)
        return rgb.astype(np.uint8)
    
    def estimate_stain_matrix(self, od):
        """Estimate stain matrix using PCA"""
        od_flat = od.reshape(-1, 3)
        
        # Remove background pixels
        mask = np.sum(od_flat, axis=1) > 0.15
        od_tissue = od_flat[mask]
        
        if len(od_tissue) < 100:
            return None
        
        # PCA to find stain directions
        try:
            pca = PCA(n_components=2)
            pca.fit(od_tissue)
            stain_matrix = pca.components_.T
            
            # Ensure positive stain vectors
            if stain_matrix[0, 0] < 0:
                stain_matrix[:, 0] *= -1
            if stain_matrix[0, 1] < 0:
                stain_matrix[:, 1] *= -1
                
            return stain_matrix
        except:
            return None
    
    def normalize_od(self, od, source_stain, target_stain):
        """Normalize OD using stain matrices"""
        od_flat = od.reshape(-1, 3)
        
        # Project to stain space
        try:
            stain_concentrations = np.linalg.lstsq(source_stain, od_flat.T, rcond=None)[0]
            
            # Transform to target stain space
            normalized_od_flat = target_stain @ stain_concentrations
            
            normalized_od = normalized_od_flat.T.reshape(od.shape)
            return normalized_od
        except:
            return od

class AdaptiveHistogramEqualization:
    """Adaptive histogram equalization for contrast enhancement"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)

class ColorConstancy:
    """Color constancy using Gray World assumption"""
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Calculate channel means
        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        
        # Gray world assumption
        gray_mean = (r_mean + g_mean + b_mean) / 3
        
        # Correction factors
        r_factor = gray_mean / r_mean if r_mean > 0 else 1
        g_factor = gray_mean / g_mean if g_mean > 0 else 1
        b_factor = gray_mean / b_mean if b_mean > 0 else 1
        
        # Apply correction
        corrected = image.copy().astype(np.float32)
        corrected[:, :, 0] *= r_factor
        corrected[:, :, 1] *= g_factor
        corrected[:, :, 2] *= b_factor
        
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return Image.fromarray(corrected)

class MultiScaleAugmentation:
    """Multi-scale augmentation for robustness"""
    
    def __init__(self, scales=[0.8, 1.0, 1.2], prob=0.5):
        self.scales = scales
        self.prob = prob
    
    def __call__(self, image):
        if np.random.random() < self.prob:
            scale = np.random.choice(self.scales)
            
            # Get original size
            w, h = image.size
            
            # Scale
            new_w, new_h = int(w * scale), int(h * scale)
            scaled = image.resize((new_w, new_h), Image.LANCZOS)
            
            # Crop or pad to original size
            if scale > 1.0:
                # Crop center
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                scaled = scaled.crop((left, top, left + w, top + h))
            elif scale < 1.0:
                # Pad
                pad_w = (w - new_w) // 2
                pad_h = (h - new_h) // 2
                
                # Create new image with padding
                padded = Image.new('RGB', (w, h), (255, 255, 255))
                padded.paste(scaled, (pad_w, pad_h))
                scaled = padded
            
            return scaled
        
        return image

class EnhancedColorJitter:
    """Enhanced color jittering for domain robustness"""
    
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image):
        # Random brightness
        if self.brightness > 0:
            factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        # Random contrast
        if self.contrast > 0:
            factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        # Random saturation
        if self.saturation > 0:
            factor = np.random.uniform(1 - self.saturation, 1 + self.saturation)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        
        return image

def get_enhanced_transforms(dataset_type='breakhis', is_training=True):
    """Get enhanced transforms for cross-dataset robustness"""
    
    # Common preprocessing
    stain_norm = StainNormalization()
    clahe = AdaptiveHistogramEqualization()
    color_constancy = ColorConstancy()
    
    if is_training:
        if dataset_type == 'breakhis':
            transform = transforms.Compose([
                # Stain normalization
                transforms.Lambda(lambda x: stain_norm.normalize_stain(x)),
                
                # Adaptive histogram equalization
                clahe,
                
                # Color constancy
                color_constancy,
                
                # Multi-scale augmentation
                MultiScaleAugmentation(scales=[0.8, 1.0, 1.2], prob=0.3),
                
                # Enhanced color jittering
                EnhancedColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                
                # Geometric augmentations
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                
                # Advanced augmentations
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.2),
                
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
                ], p=0.3),
                
                # Convert to tensor
                transforms.ToTensor(),
                
                # Normalization (ImageNet stats)
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Random erasing for robustness
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ])
        
        else:  # BACH dataset
            transform = transforms.Compose([
                # Stain normalization
                transforms.Lambda(lambda x: stain_norm.normalize_stain(x)),
                
                # Adaptive histogram equalization
                clahe,
                
                # Color constancy
                color_constancy,
                
                # Multi-scale augmentation
                MultiScaleAugmentation(scales=[0.9, 1.0, 1.1], prob=0.3),
                
                # Enhanced color jittering (less aggressive for BACH)
                EnhancedColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
                
                # Geometric augmentations
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                
                # Convert to tensor
                transforms.ToTensor(),
                
                # Normalization
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Random erasing
                transforms.RandomErasing(p=0.05, scale=(0.02, 0.08))
            ])
    
    else:  # Validation/Test transforms
        transform = transforms.Compose([
            # Stain normalization
            transforms.Lambda(lambda x: stain_norm.normalize_stain(x)),
            
            # Adaptive histogram equalization
            clahe,
            
            # Color constancy
            color_constancy,
            
            # Resize and center crop
            transforms.Resize(256),
            transforms.CenterCrop(224),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform

def get_test_time_augmentation_transforms():
    """Get transforms for test-time augmentation"""
    
    stain_norm = StainNormalization()
    clahe = AdaptiveHistogramEqualization()
    color_constancy = ColorConstancy()
    
    # Multiple test-time augmentations
    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Lambda(lambda x: stain_norm.normalize_stain(x)),
            clahe,
            color_constancy,
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        
        # Horizontal flip
        transforms.Compose([
            transforms.Lambda(lambda x: stain_norm.normalize_stain(x)),
            clahe,
            color_constancy,
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        
        # Slight rotation
        transforms.Compose([
            transforms.Lambda(lambda x: stain_norm.normalize_stain(x)),
            clahe,
            color_constancy,
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        
        # Different crop
        transforms.Compose([
            transforms.Lambda(lambda x: stain_norm.normalize_stain(x)),
            clahe,
            color_constancy,
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]
    
    return tta_transforms