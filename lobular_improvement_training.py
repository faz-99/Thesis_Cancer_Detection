import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.enhanced_efficientnet import EnhancedEfficientNetB0
from src.lobular_enhancement import (
    LobularFocusedLoss, 
    LobularSpecificAugmentation,
    create_lobular_weighted_sampler,
    get_lobular_class_weights
)
from src.data_utils import BreakHisDataset, get_patient_splits

def train_lobular_enhanced_model():
    """Train model with lobular carcinoma enhancements"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data with lobular-specific augmentations
    lobular_transforms = LobularSpecificAugmentation.get_lobular_transforms()
    
    # Create datasets (assuming you have your data loading setup)
    train_dataset = BreakHisDataset(
        root_dir="data/breakhis",
        split="train",
        transform=lobular_transforms
    )
    
    val_dataset = BreakHisDataset(
        root_dir="data/breakhis", 
        split="val"
    )
    
    # Create lobular-weighted sampler
    train_sampler = create_lobular_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize enhanced model
    model = EnhancedEfficientNetB0(num_classes=8).to(device)
    
    # Multi-task loss
    main_criterion = LobularFocusedLoss(
        alpha=get_lobular_class_weights().to(device),
        gamma=2.0,
        lobular_boost=2.5
    )
    
    lobular_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    best_lobular_f1 = 0.0
    
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            main_outputs, lobular_outputs = model(images)
            
            # Create lobular binary labels (1 if lobular, 0 otherwise)
            lobular_labels = (labels == 5).long()
            
            # Combined loss
            main_loss = main_criterion(main_outputs, labels)
            lobular_loss = lobular_criterion(lobular_outputs, lobular_labels)
            
            total_loss = main_loss + 0.3 * lobular_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                main_outputs, _ = model(images)
                predictions = torch.max(main_outputs, 1)[1]
                
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate lobular-specific metrics
        lobular_mask = np.array(val_labels) == 5
        if lobular_mask.sum() > 0:
            lobular_predictions = np.array(val_predictions)[lobular_mask]
            lobular_true = np.array(val_labels)[lobular_mask]
            
            from sklearn.metrics import f1_score
            lobular_f1 = f1_score(lobular_true, lobular_predictions, average='macro')
            
            print(f"Epoch {epoch+1}: Lobular F1: {lobular_f1:.4f}")
            
            if lobular_f1 > best_lobular_f1:
                best_lobular_f1 = lobular_f1
                torch.save(model.state_dict(), "models/enhanced_efficientnet_lobular_v1.pth")
                print(f"✅ New best lobular F1: {best_lobular_f1:.4f}")
        
        scheduler.step()
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    report = classification_report(
        val_labels, 
        val_predictions,
        target_names=[
            "adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma",
            "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"
        ],
        digits=4
    )
    print(report)
    
    return model

if __name__ == "__main__":
    model = train_lobular_enhanced_model()