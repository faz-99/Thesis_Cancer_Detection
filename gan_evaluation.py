#!/usr/bin/env python3
"""
GAN-based augmentation evaluation with FID scores and performance comparison
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import logging
from scipy.linalg import sqrtm
from sklearn.metrics import classification_report

from src.gan_augmentation import Generator, train_gan, generate_synthetic_images
from src.efficientnet import EfficientNetB0Classifier
from src.data_utils import create_metadata, create_train_val_test_split, create_class_mappings, create_data_loaders
from src.train import train_model, evaluate_model

class InceptionV3Features(nn.Module):
    """Extract features from InceptionV3 for FID calculation"""
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3
        inception = inception_v3(pretrained=True)
        self.features = nn.Sequential(*list(inception.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)

def calculate_fid(real_features, fake_features):
    """Calculate Fréchet Inception Distance"""
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def extract_features(dataloader, feature_extractor, device, max_samples=1000):
    """Extract features from images using InceptionV3"""
    feature_extractor.eval()
    features = []
    
    with torch.no_grad():
        for i, (images, _, _) in enumerate(dataloader):
            if i * images.size(0) >= max_samples:
                break
                
            images = images.to(device)
            # Resize to 299x299 for InceptionV3
            images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear')
            batch_features = feature_extractor(images)
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def train_gan_and_evaluate():
    """Train GAN and evaluate with FID scores"""
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    metadata = create_metadata(breakhis_root)
    
    train_df, val_df, test_df = create_train_val_test_split(
        metadata, test_size=0.15, val_size=0.15, random_state=42
    )
    
    class_to_idx, idx_to_class, class_weights_tensor = create_class_mappings(train_df)
    
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df["subclass"].map(class_to_idx)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, class_weights_tensor, batch_size=32
    )
    
    # Train GAN
    logger.info("Training GAN...")
    generator = train_gan(train_loader, num_epochs=50, device=device)
    
    # Save generator
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), "models/generator.pth")
    
    # Generate synthetic images
    logger.info("Generating synthetic images...")
    num_synthetic = 1000
    synthetic_images = generate_synthetic_images(generator, num_synthetic, device)
    
    # Calculate FID score
    logger.info("Calculating FID score...")
    feature_extractor = InceptionV3Features().to(device)
    
    # Extract features from real images
    real_features = extract_features(train_loader, feature_extractor, device)
    
    # Create synthetic dataloader
    class SyntheticDataset(Dataset):
        def __init__(self, images):
            self.images = images
            
        def __len__(self):
            return len(self.images)
            
        def __getitem__(self, idx):
            return self.images[idx], 0, ""
    
    synthetic_dataset = SyntheticDataset(synthetic_images)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=False)
    
    # Extract features from synthetic images
    fake_features = extract_features(synthetic_loader, feature_extractor, device)
    
    # Calculate FID
    fid_score = calculate_fid(real_features, fake_features)
    logger.info(f"FID Score: {fid_score:.2f}")
    
    return generator, fid_score, synthetic_images

def compare_with_without_gan():
    """Compare model performance with and without GAN augmentation"""
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    metadata = create_metadata(breakhis_root)
    
    train_df, val_df, test_df = create_train_val_test_split(
        metadata, test_size=0.15, val_size=0.15, random_state=42
    )
    
    class_to_idx, idx_to_class, class_weights_tensor = create_class_mappings(train_df)
    
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df["subclass"].map(class_to_idx)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, class_weights_tensor, batch_size=32
    )
    
    results = {}
    
    # 1. Train without GAN augmentation
    logger.info("Training model WITHOUT GAN augmentation...")
    model_no_gan = EfficientNetB0Classifier(num_classes=len(class_to_idx), pretrained=True)
    model_no_gan = model_no_gan.to(device)
    
    model_no_gan, history_no_gan = train_model(
        model_no_gan, train_loader, val_loader,
        num_epochs=10, lr=1e-4, device=device
    )
    
    # Evaluate without GAN
    class_names = {v: k for k, v in class_to_idx.items()}
    results_no_gan = evaluate_model(model_no_gan, test_loader, device, class_names)
    results['without_gan'] = results_no_gan
    
    logger.info(f"Accuracy WITHOUT GAN: {results_no_gan['accuracy']:.4f}")
    
    # 2. Train with GAN augmentation
    logger.info("Training model WITH GAN augmentation...")
    
    # Load or train generator
    generator_path = "models/generator.pth"
    if os.path.exists(generator_path):
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(generator_path))
    else:
        generator = train_gan(train_loader, num_epochs=50, device=device)
        torch.save(generator.state_dict(), generator_path)
    
    # Create augmented dataset
    class AugmentedDataset(Dataset):
        def __init__(self, original_dataset, generator, device, augment_ratio=0.5):
            self.original_dataset = original_dataset
            self.generator = generator
            self.device = device
            self.augment_ratio = augment_ratio
            
        def __len__(self):
            return int(len(self.original_dataset) * (1 + self.augment_ratio))
            
        def __getitem__(self, idx):
            if idx < len(self.original_dataset):
                return self.original_dataset[idx]
            else:
                # Generate synthetic sample
                with torch.no_grad():
                    noise = torch.randn(1, 100, 1, 1, device=self.device)
                    synthetic_image = self.generator(noise).squeeze(0)
                    # Random class assignment (in practice, use class-conditional GAN)
                    random_class = np.random.randint(0, len(class_to_idx))
                    return synthetic_image.cpu(), random_class, "synthetic"
    
    # Create augmented train loader
    augmented_dataset = AugmentedDataset(train_loader.dataset, generator, device)
    augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    
    # Train with augmentation
    model_with_gan = EfficientNetB0Classifier(num_classes=len(class_to_idx), pretrained=True)
    model_with_gan = model_with_gan.to(device)
    
    model_with_gan, history_with_gan = train_model(
        model_with_gan, augmented_loader, val_loader,
        num_epochs=10, lr=1e-4, device=device
    )
    
    # Evaluate with GAN
    results_with_gan = evaluate_model(model_with_gan, test_loader, device, class_names)
    results['with_gan'] = results_with_gan
    
    logger.info(f"Accuracy WITH GAN: {results_with_gan['accuracy']:.4f}")
    
    return results

def visualize_synthetic_samples(generator, device, num_samples=16, save_path="synthetic_samples.png"):
    """Visualize synthetic samples generated by GAN"""
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        synthetic_images = generator(noise)
        
        # Denormalize images
        synthetic_images = (synthetic_images + 1) / 2  # From [-1,1] to [0,1]
        
        # Create grid
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                img = synthetic_images[i].cpu().permute(1, 2, 0)
                ax.imshow(img)
                ax.set_title(f'Synthetic {i+1}')
            ax.axis('off')
        
        plt.suptitle('GAN Generated Synthetic Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def plot_gan_comparison(results, fid_score, save_path="gan_comparison.png"):
    """Plot comparison of model performance with/without GAN"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    methods = ['Without GAN', 'With GAN']
    accuracies = [results['without_gan']['accuracy'], results['with_gan']['accuracy']]
    
    bars = ax1.bar(methods, accuracies, color=['lightcoral', 'skyblue'])
    ax1.set_title('Model Performance Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # FID score
    ax2.bar(['FID Score'], [fid_score], color='orange')
    ax2.set_title('GAN Quality (FID Score)')
    ax2.set_ylabel('FID Score (lower is better)')
    ax2.text(0, fid_score + fid_score*0.05, f'{fid_score:.1f}', 
             ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GAN evaluation...")
    
    # Train GAN and calculate FID
    generator, fid_score, synthetic_images = train_gan_and_evaluate()
    
    # Visualize synthetic samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_synthetic_samples(generator, device)
    
    # Compare performance with/without GAN
    results = compare_with_without_gan()
    
    # Plot results
    plot_gan_comparison(results, fid_score)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("GAN EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"FID Score: {fid_score:.2f}")
    logger.info(f"Accuracy without GAN: {results['without_gan']['accuracy']:.4f}")
    logger.info(f"Accuracy with GAN: {results['with_gan']['accuracy']:.4f}")
    improvement = results['with_gan']['accuracy'] - results['without_gan']['accuracy']
    logger.info(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    logger.info("GAN evaluation completed!")

if __name__ == "__main__":
    main()