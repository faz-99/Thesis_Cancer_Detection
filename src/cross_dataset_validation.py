#!/usr/bin/env python3
"""
Cross-Dataset Validation for Breast Cancer Classification
Implements training on one dataset and testing on another for generalization assessment
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
from .data_utils import create_metadata, create_train_val_test_split, create_class_mappings
from .bach_data_utils import create_bach_metadata, create_combined_metadata
from .train import evaluate_model

logger = logging.getLogger(__name__)

class CrossDatasetValidator:
    """
    Handles cross-dataset validation experiments
    Train on one dataset, test on another to assess generalization
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        
    def train_breakhis_test_bach(self, model, breakhis_root, bach_root, batch_size=32):
        """
        Train on BreakHis, test on BACH
        Maps classes appropriately between datasets
        """
        logger.info("Cross-dataset validation: BreakHis → BACH")
        
        # Load BreakHis for training
        breakhis_metadata = create_metadata(breakhis_root)
        train_df, val_df, _ = create_train_val_test_split(breakhis_metadata)
        
        # Create class mappings for BreakHis
        class_to_idx, idx_to_class, class_weights = create_class_mappings(train_df)
        
        # Add class indices
        for df in [train_df, val_df]:
            df["class_idx"] = df["subclass"].map(class_to_idx)
        
        # Create data loaders for training
        from .data_utils import create_data_loaders
        train_loader, val_loader, _ = create_data_loaders(
            train_df, val_df, val_df, class_weights, batch_size
        )
        
        # Train model on BreakHis
        from .train import train_model
        trained_model, history = train_model(
            model, train_loader, val_loader, 
            num_epochs=20, device=self.device
        )
        
        # Load BACH for testing
        bach_metadata = create_bach_metadata(bach_root)
        
        # Map BACH classes to BreakHis classes
        bach_mapped = self._map_bach_to_breakhis(bach_metadata, class_to_idx)
        
        if len(bach_mapped) == 0:
            logger.warning("No BACH samples could be mapped to BreakHis classes")
            return None
        
        # Create BACH test loader
        from .bach_data_utils import create_bach_data_loaders
        dummy_weights = torch.ones(len(class_to_idx))
        _, _, bach_test_loader = create_bach_data_loaders(
            bach_mapped, bach_mapped[:1], bach_mapped, dummy_weights, batch_size
        )
        
        # Evaluate on BACH
        class_names = {v: k for k, v in class_to_idx.items()}
        bach_results = evaluate_model(trained_model, bach_test_loader, self.device, class_names)
        
        self.results['breakhis_to_bach'] = {
            'training_history': history,
            'bach_test_results': bach_results,
            'class_mapping': class_to_idx,
            'mapped_samples': len(bach_mapped)
        }
        
        logger.info(f"BreakHis→BACH Cross-validation Accuracy: {bach_results['accuracy']:.4f}")
        return bach_results
    
    def train_bach_test_breakhis(self, model, bach_root, breakhis_root, batch_size=32):
        """
        Train on BACH, test on BreakHis
        """
        logger.info("Cross-dataset validation: BACH → BreakHis")
        
        # Load BACH for training
        bach_metadata = create_bach_metadata(bach_root)
        
        # Create simplified class mapping for BACH
        bach_class_mapping = {
            'Normal': 0,
            'Benign': 1, 
            'InSitu': 2,
            'Invasive': 3
        }
        
        bach_metadata['class_idx'] = bach_metadata['class'].map(bach_class_mapping)
        
        # Split BACH data
        train_df, val_df, _ = train_test_split(
            bach_metadata, test_size=0.2, stratify=bach_metadata['class'], random_state=42
        )
        
        # Calculate class weights for BACH
        from collections import Counter
        class_counts = Counter(train_df['class'])
        total = sum(class_counts.values())
        class_weights = torch.FloatTensor([
            total / class_counts[cls] for cls in sorted(class_counts.keys())
        ])
        
        # Create BACH data loaders
        from .bach_data_utils import create_bach_data_loaders
        train_loader, val_loader, _ = create_bach_data_loaders(
            train_df, val_df, val_df, class_weights, batch_size
        )
        
        # Train model on BACH
        from .train import train_model
        trained_model, history = train_model(
            model, train_loader, val_loader,
            num_epochs=20, device=self.device
        )
        
        # Load BreakHis for testing
        breakhis_metadata = create_metadata(breakhis_root)
        
        # Map BreakHis classes to BACH classes
        breakhis_mapped = self._map_breakhis_to_bach(breakhis_metadata, bach_class_mapping)
        
        if len(breakhis_mapped) == 0:
            logger.warning("No BreakHis samples could be mapped to BACH classes")
            return None
        
        # Create BreakHis test loader
        from .data_utils import create_data_loaders
        dummy_weights = torch.ones(len(bach_class_mapping))
        _, _, breakhis_test_loader = create_data_loaders(
            breakhis_mapped, breakhis_mapped[:1], breakhis_mapped, dummy_weights, batch_size
        )
        
        # Evaluate on BreakHis
        class_names = {v: k for k, v in bach_class_mapping.items()}
        breakhis_results = evaluate_model(trained_model, breakhis_test_loader, self.device, class_names)
        
        self.results['bach_to_breakhis'] = {
            'training_history': history,
            'breakhis_test_results': breakhis_results,
            'class_mapping': bach_class_mapping,
            'mapped_samples': len(breakhis_mapped)
        }
        
        logger.info(f"BACH→BreakHis Cross-validation Accuracy: {breakhis_results['accuracy']:.4f}")
        return breakhis_results
    
    def _map_bach_to_breakhis(self, bach_metadata, breakhis_class_mapping):
        """
        Map BACH classes to BreakHis classes
        BACH: Normal, Benign, InSitu, Invasive
        BreakHis: 8 specific subclasses
        """
        # Simple mapping strategy
        bach_to_breakhis_map = {
            'Benign': 'adenosis',  # Map to most common benign
            'Invasive': 'ductal_carcinoma'  # Map to most common malignant
        }
        
        # Filter BACH data to mappable classes
        mappable_classes = list(bach_to_breakhis_map.keys())
        bach_filtered = bach_metadata[bach_metadata['class'].isin(mappable_classes)].copy()
        
        # Map to BreakHis class indices
        bach_filtered['mapped_class'] = bach_filtered['class'].map(bach_to_breakhis_map)
        bach_filtered['class_idx'] = bach_filtered['mapped_class'].map(breakhis_class_mapping)
        
        # Remove unmappable samples
        bach_filtered = bach_filtered.dropna(subset=['class_idx'])
        bach_filtered['class_idx'] = bach_filtered['class_idx'].astype(int)
        
        logger.info(f"Mapped {len(bach_filtered)} BACH samples to BreakHis classes")
        return bach_filtered
    
    def _map_breakhis_to_bach(self, breakhis_metadata, bach_class_mapping):
        """
        Map BreakHis classes to BACH classes
        """
        # Map BreakHis subclasses to BACH categories
        breakhis_to_bach_map = {
            # Benign classes → Benign
            'adenosis': 'Benign',
            'fibroadenoma': 'Benign',
            'phyllodes_tumor': 'Benign',
            'tubular_adenoma': 'Benign',
            # Malignant classes → Invasive
            'ductal_carcinoma': 'Invasive',
            'lobular_carcinoma': 'Invasive',
            'mucinous_carcinoma': 'Invasive',
            'papillary_carcinoma': 'Invasive'
        }
        
        # Map BreakHis subclasses
        breakhis_metadata['mapped_class'] = breakhis_metadata['subclass'].map(breakhis_to_bach_map)
        breakhis_metadata['class_idx'] = breakhis_metadata['mapped_class'].map(bach_class_mapping)
        
        # Remove unmappable samples
        breakhis_filtered = breakhis_metadata.dropna(subset=['class_idx']).copy()
        breakhis_filtered['class_idx'] = breakhis_filtered['class_idx'].astype(int)
        
        logger.info(f"Mapped {len(breakhis_filtered)} BreakHis samples to BACH classes")
        return breakhis_filtered
    
    def domain_adaptation_experiment(self, model, source_root, target_root, adaptation_method='gradual'):
        """
        Perform domain adaptation experiment
        
        Args:
            model: Model to adapt
            source_root: Source dataset path
            target_root: Target dataset path
            adaptation_method: 'gradual', 'adversarial', or 'fine_tuning'
        """
        logger.info(f"Domain adaptation experiment: {adaptation_method}")
        
        if adaptation_method == 'gradual':
            return self._gradual_domain_adaptation(model, source_root, target_root)
        elif adaptation_method == 'fine_tuning':
            return self._fine_tuning_adaptation(model, source_root, target_root)
        else:
            logger.warning(f"Adaptation method {adaptation_method} not implemented")
            return None
    
    def _gradual_domain_adaptation(self, model, source_root, target_root):
        """
        Gradual domain adaptation by mixing datasets
        """
        # Load both datasets
        source_metadata = create_metadata(source_root)
        target_metadata = create_bach_metadata(target_root)
        
        # Create unified metadata
        combined_metadata = create_combined_metadata(source_root, target_root)
        
        # Progressive training with increasing target data
        adaptation_results = []
        target_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for ratio in target_ratios:
            logger.info(f"Training with {ratio:.1%} target data")
            
            # Sample target data according to ratio
            n_target_samples = int(len(target_metadata) * ratio)
            target_sample = target_metadata.sample(n_target_samples, random_state=42)
            
            # Combine with source data
            mixed_metadata = pd.concat([source_metadata, target_sample], ignore_index=True)
            
            # Train and evaluate
            # Implementation would continue here...
            
        return adaptation_results
    
    def _fine_tuning_adaptation(self, model, source_root, target_root):
        """
        Fine-tuning based domain adaptation
        """
        # Pre-train on source domain
        logger.info("Pre-training on source domain...")
        
        # Fine-tune on target domain
        logger.info("Fine-tuning on target domain...")
        
        # Implementation would continue here...
        pass
    
    def generate_cross_dataset_report(self):
        """
        Generate comprehensive cross-dataset validation report
        """
        if not self.results:
            logger.warning("No cross-dataset results available")
            return None
        
        report = {
            'summary': {},
            'detailed_results': self.results
        }
        
        # Calculate summary statistics
        if 'breakhis_to_bach' in self.results:
            report['summary']['breakhis_to_bach_accuracy'] = self.results['breakhis_to_bach']['bach_test_results']['accuracy']
        
        if 'bach_to_breakhis' in self.results:
            report['summary']['bach_to_breakhis_accuracy'] = self.results['bach_to_breakhis']['breakhis_test_results']['accuracy']
        
        # Calculate generalization gap
        if len(report['summary']) == 2:
            accuracies = list(report['summary'].values())
            report['summary']['mean_cross_accuracy'] = np.mean(accuracies)
            report['summary']['generalization_gap'] = np.std(accuracies)
        
        logger.info("Cross-dataset validation report generated")
        return report

def run_cross_dataset_validation(model_class, breakhis_root, bach_root, device='cpu'):
    """
    Run complete cross-dataset validation experiment
    
    Args:
        model_class: Model class to instantiate
        breakhis_root: Path to BreakHis dataset
        bach_root: Path to BACH dataset
        device: Computing device
    
    Returns:
        Validation results
    """
    validator = CrossDatasetValidator(device)
    
    # Experiment 1: BreakHis → BACH
    model1 = model_class(num_classes=8)  # BreakHis has 8 classes
    model1 = model1.to(device)
    
    breakhis_to_bach = validator.train_breakhis_test_bach(
        model1, breakhis_root, bach_root
    )
    
    # Experiment 2: BACH → BreakHis
    model2 = model_class(num_classes=4)  # BACH has 4 classes
    model2 = model2.to(device)
    
    bach_to_breakhis = validator.train_bach_test_breakhis(
        model2, bach_root, breakhis_root
    )
    
    # Generate report
    report = validator.generate_cross_dataset_report()
    
    return report