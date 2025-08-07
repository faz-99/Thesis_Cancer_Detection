#!/usr/bin/env python3
"""
SupConViT implementation and evaluation with t-SNE visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import seaborn as sns
import logging
import os

from src.supconvit import SupConViT, SupConLoss, train_supcon_vit
from src.efficientnet import EfficientNetB0Classifier
from src.data_utils import create_metadata, create_train_val_test_split, create_class_mappings, create_data_loaders
from src.train import train_model, evaluate_model

def extract_features_supconvit(model, dataloader, device):
    """Extract features from SupConViT model"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels, _ in dataloader:
            images = images.to(device)
            batch_features = model(images, return_features=True)
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def plot_tsne_embeddings(features, labels, class_names, save_path="tsne_embeddings.png"):
    """Create t-SNE visualization of learned embeddings"""
    logger = logging.getLogger(__name__)
    logger.info("Creating t-SNE visualization...")
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names.values()):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=class_name, alpha=0.7, s=50)
    
    plt.title('t-SNE Visualization of SupConViT Learned Embeddings', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"t-SNE plot saved to: {save_path}")

def compare_supconvit_vs_efficientnet():
    """Compare SupConViT vs EfficientNet performance"""
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
    
    # 1. Train SupConViT
    logger.info("Training SupConViT...")
    supconvit_model = SupConViT(num_classes=len(class_to_idx)).to(device)
    
    # Phase 1: Contrastive pretraining
    logger.info("Phase 1: Contrastive pretraining...")
    supconvit_model = train_supcon_vit(supconvit_model, train_loader, num_epochs=50, device=device)
    
    # Phase 2: Fine-tuning for classification
    logger.info("Phase 2: Fine-tuning for classification...")
    optimizer = torch.optim.Adam(supconvit_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    supconvit_model.train()
    for epoch in range(20):
        total_loss = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = supconvit_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            logger.info(f'Fine-tuning Epoch [{epoch}/20], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluate SupConViT
    class_names = {v: k for k, v in class_to_idx.items()}
    supconvit_results = evaluate_model(supconvit_model, test_loader, device, class_names)
    results['SupConViT'] = supconvit_results
    
    logger.info(f"SupConViT Test Accuracy: {supconvit_results['accuracy']:.4f}")
    
    # Extract features for t-SNE
    features, labels = extract_features_supconvit(supconvit_model, test_loader, device)
    plot_tsne_embeddings(features, labels, class_names, "supconvit_tsne.png")
    
    # 2. Train EfficientNet for comparison
    logger.info("Training EfficientNet for comparison...")
    efficientnet_model = EfficientNetB0Classifier(num_classes=len(class_to_idx), pretrained=True)
    efficientnet_model = efficientnet_model.to(device)
    
    efficientnet_model, history = train_model(
        efficientnet_model, train_loader, val_loader,
        num_epochs=20, lr=1e-4, device=device
    )
    
    # Evaluate EfficientNet
    efficientnet_results = evaluate_model(efficientnet_model, test_loader, device, class_names)
    results['EfficientNet'] = efficientnet_results
    
    logger.info(f"EfficientNet Test Accuracy: {efficientnet_results['accuracy']:.4f}")
    
    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(supconvit_model.state_dict(), "models/supconvit_model.pth")
    torch.save(efficientnet_model.state_dict(), "models/efficientnet_comparison.pth")
    
    return results, features, labels

def analyze_embedding_quality(features, labels, class_names):
    """Analyze quality of learned embeddings"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing embedding quality...")
    
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    
    # Calculate silhouette score
    silhouette = silhouette_score(features, labels)
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    
    # Calculate intra-class and inter-class distances
    intra_distances = []
    inter_distances = []
    
    for class_idx in range(len(class_names)):
        class_mask = labels == class_idx
        class_features = features[class_mask]
        
        if len(class_features) > 1:
            # Intra-class distances
            nbrs = NearestNeighbors(n_neighbors=min(5, len(class_features))).fit(class_features)
            distances, _ = nbrs.kneighbors(class_features)
            intra_distances.extend(distances[:, 1:].flatten())  # Exclude self
            
            # Inter-class distances
            other_features = features[~class_mask]
            if len(other_features) > 0:
                nbrs_inter = NearestNeighbors(n_neighbors=min(5, len(other_features))).fit(other_features)
                distances_inter, _ = nbrs_inter.kneighbors(class_features)
                inter_distances.extend(distances_inter.flatten())
    
    avg_intra = np.mean(intra_distances)
    avg_inter = np.mean(inter_distances)
    separation_ratio = avg_inter / avg_intra if avg_intra > 0 else 0
    
    logger.info(f"Average intra-class distance: {avg_intra:.4f}")
    logger.info(f"Average inter-class distance: {avg_inter:.4f}")
    logger.info(f"Separation ratio: {separation_ratio:.4f}")
    
    return {
        'silhouette_score': silhouette,
        'avg_intra_distance': avg_intra,
        'avg_inter_distance': avg_inter,
        'separation_ratio': separation_ratio
    }

def plot_model_comparison(results, embedding_metrics, save_path="supconvit_comparison.png"):
    """Plot comparison between SupConViT and EfficientNet"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    bars = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Embedding quality metrics
    metrics = ['Silhouette Score', 'Separation Ratio']
    values = [embedding_metrics['silhouette_score'], embedding_metrics['separation_ratio']]
    
    ax2.bar(metrics, values, color=['orange', 'green'])
    ax2.set_title('SupConViT Embedding Quality')
    ax2.set_ylabel('Score')
    
    for i, (metric, value) in enumerate(zip(metrics, values)):
        ax2.text(i, value + value*0.05, f'{value:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Per-class accuracy comparison (if available)
    if 'per_class_accuracy' in results['SupConViT']:
        supcon_per_class = results['SupConViT']['per_class_accuracy']
        efficient_per_class = results['EfficientNet']['per_class_accuracy']
        
        classes = list(supcon_per_class.keys())
        supcon_accs = [supcon_per_class[cls] for cls in classes]
        efficient_accs = [efficient_per_class[cls] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax3.bar(x - width/2, supcon_accs, width, label='SupConViT', color='skyblue')
        ax3.bar(x + width/2, efficient_accs, width, label='EfficientNet', color='lightcoral')
        
        ax3.set_title('Per-Class Accuracy Comparison')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Classes')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45)
        ax3.legend()
    
    # Distance analysis
    distances = ['Intra-class', 'Inter-class']
    dist_values = [embedding_metrics['avg_intra_distance'], embedding_metrics['avg_inter_distance']]
    
    ax4.bar(distances, dist_values, color=['red', 'blue'])
    ax4.set_title('SupConViT Distance Analysis')
    ax4.set_ylabel('Average Distance')
    
    for i, (dist, value) in enumerate(zip(distances, dist_values)):
        ax4.text(i, value + value*0.05, f'{value:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SupConViT evaluation...")
    
    # Compare models and get embeddings
    results, features, labels = compare_supconvit_vs_efficientnet()
    
    # Analyze embedding quality
    class_names = {0: 'adenosis', 1: 'ductal_carcinoma', 2: 'fibroadenoma', 
                   3: 'lobular_carcinoma', 4: 'mucinous_carcinoma', 
                   5: 'papillary_carcinoma', 6: 'phyllodes_tumor', 7: 'tubular_adenoma'}
    
    embedding_metrics = analyze_embedding_quality(features, labels, class_names)
    
    # Plot comprehensive comparison
    plot_model_comparison(results, embedding_metrics)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SUPCONVIT EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"SupConViT Accuracy: {results['SupConViT']['accuracy']:.4f}")
    logger.info(f"EfficientNet Accuracy: {results['EfficientNet']['accuracy']:.4f}")
    improvement = results['SupConViT']['accuracy'] - results['EfficientNet']['accuracy']
    logger.info(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    logger.info(f"Silhouette Score: {embedding_metrics['silhouette_score']:.4f}")
    logger.info(f"Separation Ratio: {embedding_metrics['separation_ratio']:.4f}")
    
    logger.info("SupConViT evaluation completed!")

if __name__ == "__main__":
    main()