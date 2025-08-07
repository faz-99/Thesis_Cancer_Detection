import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def create_metadata(dataset_root):
    """Create metadata DataFrame from BreakHis dataset"""
    image_paths = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    data = []
    for path in image_paths:
        parts = path.split(os.sep)
        try:
            label_type = parts[-6]  # 'malignant' or 'benign'
            subclass = parts[-4]    # e.g. 'mucinous_carcinoma'
            magnification = parts[-2]  # e.g. '100X'
            filename = os.path.basename(path)
            
            data.append({
                "path": path,
                "label_type": label_type,
                "subclass": subclass,
                "magnification": magnification,
                "filename": filename
            })
        except IndexError:
            continue
    
    return pd.DataFrame(data)

def extract_patient_id(path):
    """Extract patient ID from filename"""
    filename = os.path.basename(path)
    return filename.split("_")[2]

def create_train_val_test_split(metadata, test_size=0.15, val_size=0.15, random_state=42):
    """Create patient-wise stratified splits"""
    metadata = metadata.copy()
    metadata["patient_id"] = metadata["path"].apply(extract_patient_id)
    
    # Get unique patients per subclass
    unique_patients = metadata[["patient_id", "subclass"]].drop_duplicates()
    
    # Train-test split
    train_ids, test_ids = train_test_split(
        unique_patients,
        test_size=test_size,
        stratify=unique_patients["subclass"],
        random_state=random_state
    )
    
    # Train-val split
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_size / (1 - test_size),
        stratify=train_ids["subclass"],
        random_state=random_state
    )
    
    # Map to full metadata
    train_df = metadata[metadata["patient_id"].isin(train_ids["patient_id"])]
    val_df = metadata[metadata["patient_id"].isin(val_ids["patient_id"])]
    test_df = metadata[metadata["patient_id"].isin(test_ids["patient_id"])]
    
    return train_df, val_df, test_df

def create_class_mappings(train_df):
    """Create class to index mappings and weights"""
    class_counts = Counter(train_df["subclass"])
    classes = sorted(class_counts.keys())
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Calculate class weights
    total = sum(class_counts.values())
    class_weights = np.array([total / class_counts[cls] for cls in classes], dtype=np.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    return class_to_idx, idx_to_class, class_weights_tensor

class BreakHisDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "class_idx"]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

def get_transforms():
    """Get training and validation transforms"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return train_transform, test_transform

def create_data_loaders(train_df, val_df, test_df, class_weights_tensor, batch_size=32):
    """Create data loaders with weighted sampling"""
    train_transform, test_transform = get_transforms()
    
    # Create datasets
    train_dataset = BreakHisDataset(train_df, transform=train_transform)
    val_dataset = BreakHisDataset(val_df, transform=test_transform)
    test_dataset = BreakHisDataset(test_df, transform=test_transform)
    
    # Create weighted sampler for training
    sample_weights = train_df["class_idx"].map(lambda x: float(class_weights_tensor[x])).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader