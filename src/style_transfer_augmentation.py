#!/usr/bin/env python3
"""
Style Transfer Augmentation for Breast Cancer Classification
Implements stain normalization and CycleGAN-based augmentation to address staining variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class StainNormalizer:
    """
    Stain normalization using Macenko method
    Addresses H&E staining variations between different labs
    """
    
    def __init__(self, target_concentrations=None):
        # Default target stain matrix (H&E)
        if target_concentrations is None:
            self.target_he = np.array([[0.5626, 0.2159],
                                      [0.7201, 0.8012],
                                      [0.4062, 0.5581]])
        else:
            self.target_he = target_concentrations
            
        self.target_maxC = np.array([1.9705, 1.0308])
        
    def normalize_stain(self, image):
        """
        Normalize H&E staining of input image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Stain normalized image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to OD space
        od = self._rgb_to_od(image)
        
        # Estimate stain matrix
        he_matrix = self._estimate_stain_matrix(od)
        
        # Calculate concentrations
        concentrations = self._calculate_concentrations(od, he_matrix)
        
        # Normalize concentrations
        maxC = np.percentile(concentrations, 99, axis=0)
        normalized_concentrations = concentrations * (self.target_maxC / maxC)
        
        # Reconstruct image
        normalized_od = normalized_concentrations @ self.target_he.T
        normalized_rgb = self._od_to_rgb(normalized_od)
        
        return normalized_rgb.astype(np.uint8)
    
    def _rgb_to_od(self, rgb):
        """Convert RGB to Optical Density"""
        rgb = rgb.astype(np.float64)
        rgb = np.maximum(rgb, 1)  # Avoid log(0)
        od = -np.log(rgb / 255.0)
        return od
    
    def _od_to_rgb(self, od):
        """Convert Optical Density to RGB"""
        rgb = 255 * np.exp(-od)
        return np.clip(rgb, 0, 255)
    
    def _estimate_stain_matrix(self, od):
        """Estimate H&E stain matrix using SVD"""
        # Remove background pixels
        od_flat = od.reshape(-1, 3)
        od_flat = od_flat[np.sum(od_flat, axis=1) > 0.15]
        
        # SVD
        U, s, Vt = np.linalg.svd(od_flat, full_matrices=False)
        
        # Extract H&E vectors
        he_matrix = Vt[:2, :].T
        
        # Ensure correct orientation
        if he_matrix[0, 0] < 0:
            he_matrix[:, 0] *= -1
        if he_matrix[0, 1] < 0:
            he_matrix[:, 1] *= -1
            
        return he_matrix
    
    def _calculate_concentrations(self, od, he_matrix):
        """Calculate stain concentrations"""
        od_flat = od.reshape(-1, 3)
        concentrations = np.linalg.lstsq(he_matrix, od_flat.T, rcond=None)[0].T
        concentrations = np.maximum(concentrations, 0)
        return concentrations.reshape(od.shape[:2] + (2,))

class CycleGANGenerator(nn.Module):
    """
    CycleGAN Generator for style transfer between different staining styles
    """
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(ngf * 4) for _ in range(9)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            nn.Conv2d(ngf, output_nc, 7, 1, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        residual = self.residual_blocks(encoded)
        decoded = self.decoder(residual)
        return decoded

class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator"""
    
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

class CycleGANDiscriminator(nn.Module):
    """
    CycleGAN Discriminator (PatchGAN)
    """
    
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class StyleTransferAugmenter:
    """
    Complete style transfer augmentation system
    Combines stain normalization with CycleGAN-based style transfer
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.stain_normalizer = StainNormalizer()
        
        # Initialize CycleGAN models
        self.generator_AB = CycleGANGenerator().to(device)
        self.generator_BA = CycleGANGenerator().to(device)
        self.discriminator_A = CycleGANDiscriminator().to(device)
        self.discriminator_B = CycleGANDiscriminator().to(device)
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
    def train_cyclegan(self, dataloader_A, dataloader_B, num_epochs=100, lr=0.0002):
        """
        Train CycleGAN for style transfer between two domains
        
        Args:
            dataloader_A: DataLoader for domain A (e.g., lab 1 staining)
            dataloader_B: DataLoader for domain B (e.g., lab 2 staining)
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        logger.info("Training CycleGAN for style transfer...")
        
        # Optimizers
        optimizer_G = torch.optim.Adam(
            list(self.generator_AB.parameters()) + list(self.generator_BA.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        optimizer_D_A = torch.optim.Adam(self.discriminator_A.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(self.discriminator_B.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Training loop
        for epoch in range(num_epochs):
            for i, (batch_A, batch_B) in enumerate(zip(dataloader_A, dataloader_B)):
                real_A = batch_A[0].to(self.device)
                real_B = batch_B[0].to(self.device)
                
                # Train Generators
                optimizer_G.zero_grad()
                
                # Identity loss
                loss_identity_A = self.criterion_identity(self.generator_BA(real_A), real_A) * 5.0
                loss_identity_B = self.criterion_identity(self.generator_AB(real_B), real_B) * 5.0
                
                # GAN loss
                fake_B = self.generator_AB(real_A)
                pred_fake = self.discriminator_B(fake_B)
                loss_GAN_AB = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                
                fake_A = self.generator_BA(real_B)
                pred_fake = self.discriminator_A(fake_A)
                loss_GAN_BA = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                
                # Cycle loss
                recovered_A = self.generator_BA(fake_B)
                loss_cycle_A = self.criterion_cycle(recovered_A, real_A) * 10.0
                
                recovered_B = self.generator_AB(fake_A)
                loss_cycle_B = self.criterion_cycle(recovered_B, real_B) * 10.0
                
                # Total generator loss
                loss_G = (loss_identity_A + loss_identity_B + 
                         loss_GAN_AB + loss_GAN_BA + 
                         loss_cycle_A + loss_cycle_B)
                
                loss_G.backward()
                optimizer_G.step()
                
                # Train Discriminator A
                optimizer_D_A.zero_grad()
                
                pred_real = self.discriminator_A(real_A)
                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
                
                pred_fake = self.discriminator_A(fake_A.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                loss_D_A.backward()
                optimizer_D_A.step()
                
                # Train Discriminator B
                optimizer_D_B.zero_grad()
                
                pred_real = self.discriminator_B(real_B)
                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
                
                pred_fake = self.discriminator_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                loss_D_B.backward()
                optimizer_D_B.step()
                
            if epoch % 10 == 0:
                logger.info(f'Epoch [{epoch}/{num_epochs}] '
                           f'Loss_G: {loss_G.item():.4f} '
                           f'Loss_D_A: {loss_D_A.item():.4f} '
                           f'Loss_D_B: {loss_D_B.item():.4f}')
        
        logger.info("CycleGAN training completed")
    
    def augment_image(self, image, augmentation_type='stain_normalize'):
        """
        Apply style transfer augmentation to image
        
        Args:
            image: Input image (PIL Image or numpy array)
            augmentation_type: 'stain_normalize', 'style_transfer', 'both'
            
        Returns:
            Augmented image
        """
        if augmentation_type == 'stain_normalize':
            return self.stain_normalizer.normalize_stain(image)
        
        elif augmentation_type == 'style_transfer':
            return self._apply_style_transfer(image)
        
        elif augmentation_type == 'both':
            # First normalize stain
            normalized = self.stain_normalizer.normalize_stain(image)
            # Then apply style transfer
            return self._apply_style_transfer(normalized)
        
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    def _apply_style_transfer(self, image):
        """Apply CycleGAN style transfer"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        tensor_image = transform(image).unsqueeze(0).to(self.device)
        
        # Apply style transfer
        with torch.no_grad():
            fake_image = self.generator_AB(tensor_image)
        
        # Convert back to PIL
        fake_image = fake_image.squeeze(0).cpu()
        fake_image = (fake_image + 1) / 2  # Denormalize
        fake_image = transforms.ToPILImage()(fake_image)
        
        return np.array(fake_image)
    
    def create_augmented_dataset(self, original_dataloader, augmentation_ratio=0.5):
        """
        Create augmented dataset with style transfer
        
        Args:
            original_dataloader: Original dataset loader
            augmentation_ratio: Ratio of augmented samples to add
            
        Returns:
            Combined dataset with original and augmented samples
        """
        logger.info(f"Creating augmented dataset with ratio {augmentation_ratio}")
        
        augmented_samples = []
        
        for batch_idx, (images, labels, paths) in enumerate(original_dataloader):
            # Randomly select samples for augmentation
            n_augment = int(len(images) * augmentation_ratio)
            if n_augment == 0:
                continue
                
            indices = np.random.choice(len(images), n_augment, replace=False)
            
            for idx in indices:
                original_image = transforms.ToPILImage()(images[idx])
                
                # Apply random augmentation
                aug_type = np.random.choice(['stain_normalize', 'style_transfer', 'both'])
                augmented_image = self.augment_image(original_image, aug_type)
                
                # Convert back to tensor
                augmented_tensor = transforms.ToTensor()(Image.fromarray(augmented_image))
                
                augmented_samples.append({
                    'image': augmented_tensor,
                    'label': labels[idx],
                    'path': f"augmented_{aug_type}_{paths[idx]}",
                    'augmentation_type': aug_type
                })
        
        logger.info(f"Created {len(augmented_samples)} augmented samples")
        return augmented_samples
    
    def save_models(self, save_dir):
        """Save trained CycleGAN models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.generator_AB.state_dict(), f"{save_dir}/generator_AB.pth")
        torch.save(self.generator_BA.state_dict(), f"{save_dir}/generator_BA.pth")
        torch.save(self.discriminator_A.state_dict(), f"{save_dir}/discriminator_A.pth")
        torch.save(self.discriminator_B.state_dict(), f"{save_dir}/discriminator_B.pth")
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir):
        """Load trained CycleGAN models"""
        self.generator_AB.load_state_dict(torch.load(f"{save_dir}/generator_AB.pth"))
        self.generator_BA.load_state_dict(torch.load(f"{save_dir}/generator_BA.pth"))
        self.discriminator_A.load_state_dict(torch.load(f"{save_dir}/discriminator_A.pth"))
        self.discriminator_B.load_state_dict(torch.load(f"{save_dir}/discriminator_B.pth"))
        
        logger.info(f"Models loaded from {save_dir}")

def create_style_transfer_augmenter(device='cpu'):
    """
    Factory function to create style transfer augmenter
    
    Args:
        device: Computing device
        
    Returns:
        StyleTransferAugmenter instance
    """
    return StyleTransferAugmenter(device)