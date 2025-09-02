#!/usr/bin/env python3
"""
Advanced Thesis Training Script
Implements all advanced components for comprehensive breast cancer detection thesis
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import numpy as np
from datetime import datetime

# Import all advanced components
from src.advanced_models import create_advanced_model
from src.advanced_losses import create_loss_function
from src.advanced_explainability import create_explainer
from src.cross_dataset_validation import run_cross_dataset_validation
from src.style_transfer_augmentation import create_style_transfer_augmenter
from src.data_utils import create_metadata, create_train_val_test_split, create_class_mappings, create_data_loaders
from src.bach_data_utils import create_combined_metadata
from src.train import train_model, evaluate_model

def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"advanced_thesis_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("🚀 Starting Advanced Thesis Training Pipeline")
    
    # Configuration
    config = {
        'breakhis_root': "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
        'bach_root': "data/bach",
        'batch_size': 32,
        'num_epochs': 25,
        'learning_rate': 1e-4,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'models_to_test': ['efficientnet', 'swin', 'vit', 'ensemble'],
        'loss_functions': ['focal', 'class_balanced', 'combined'],
        'explainability_methods': ['shap', 'integrated_gradients', 'comprehensive']
    }
    
    logger.info(f"Device: {config['device']}")
    logger.info(f"Models to test: {config['models_to_test']}")
    
    # ============ STEP 1: DATA PREPARATION ============
    logger.info("📊 Step 1: Advanced Data Preparation")
    
    # Load and combine datasets
    if os.path.exists(config['bach_root']):
        logger.info("Loading combined BreakHis + BACH datasets")
        metadata = create_combined_metadata(config['breakhis_root'], config['bach_root'])
        use_combined = True
    else:
        logger.info("Loading BreakHis dataset only")
        metadata = create_metadata(config['breakhis_root'])
        metadata['unified_class'] = metadata['subclass']
        use_combined = False
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_split(metadata)
    
    # Create class mappings
    if use_combined:
        class_column = 'unified_class'
    else:
        class_column = 'subclass'
    
    from collections import Counter
    class_counts = Counter(train_df[class_column])
    classes = sorted(class_counts.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Add class indices
    for df in [train_df, val_df, test_df]:
        df["class_idx"] = df[class_column].map(class_to_idx)
    
    num_classes = len(classes)
    samples_per_class = [class_counts[cls] for cls in classes]
    
    logger.info(f"Dataset: {len(metadata)} images, {num_classes} classes")
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    # ============ STEP 2: STYLE TRANSFER AUGMENTATION ============
    logger.info("🎨 Step 2: Style Transfer Augmentation")
    
    try:
        style_augmenter = create_style_transfer_augmenter(config['device'])
        logger.info("Style transfer augmenter initialized")
        # Note: Full CycleGAN training would require separate domain datasets
        # For thesis, demonstrate capability with stain normalization
    except Exception as e:
        logger.warning(f"Style transfer augmentation failed: {e}")
        style_augmenter = None
    
    # ============ STEP 3: ADVANCED MODEL TRAINING ============
    logger.info("🧠 Step 3: Advanced Model Training")
    
    results = {}
    
    for model_type in config['models_to_test']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type.upper()} Model")
        logger.info(f"{'='*50}")
        
        try:
            # Create model
            if model_type == 'efficientnet':
                from src.efficientnet import EfficientNetB0Classifier
                model = EfficientNetB0Classifier(num_classes=num_classes)
            else:
                model = create_advanced_model(model_type, num_classes=num_classes)
            
            model = model.to(config['device'])
            
            # Test different loss functions
            for loss_type in config['loss_functions']:
                logger.info(f"\n--- Training with {loss_type} loss ---")
                
                try:
                    # Create loss function
                    if loss_type == 'standard':
                        criterion = nn.CrossEntropyLoss()
                    else:
                        criterion = create_loss_function(
                            loss_type, 
                            samples_per_class=samples_per_class
                        )
                    
                    # Create data loaders
                    if use_combined:
                        from src.bach_data_utils import create_bach_data_loaders
                        class_weights = torch.ones(num_classes)  # Simplified for demo
                        train_loader, val_loader, test_loader = create_bach_data_loaders(
                            train_df, val_df, test_df, class_weights, config['batch_size']
                        )
                    else:
                        _, _, class_weights = create_class_mappings(train_df)
                        train_loader, val_loader, test_loader = create_data_loaders(
                            train_df, val_df, test_df, class_weights, config['batch_size']
                        )
                    
                    # Train model
                    trained_model, history = train_model(
                        model, train_loader, val_loader,
                        num_epochs=config['num_epochs'],
                        lr=config['learning_rate'],
                        device=config['device']
                    )
                    
                    # Evaluate model
                    class_names = {v: k for k, v in class_to_idx.items()}
                    test_results = evaluate_model(
                        trained_model, test_loader, config['device'], class_names
                    )
                    
                    # Store results
                    key = f"{model_type}_{loss_type}"
                    results[key] = {
                        'model_type': model_type,
                        'loss_type': loss_type,
                        'test_accuracy': test_results['accuracy'],
                        'training_history': history,
                        'test_results': test_results
                    }
                    
                    logger.info(f"✅ {key}: Test Accuracy = {test_results['accuracy']:.4f}")
                    
                    # Save model
                    model_path = f"models/advanced_{key}_best.pth"
                    os.makedirs("models", exist_ok=True)
                    torch.save(trained_model.state_dict(), model_path)
                    
                except Exception as e:
                    logger.error(f"❌ Training failed for {model_type} with {loss_type}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"❌ Model {model_type} failed: {e}")
            continue
    
    # ============ STEP 4: ADVANCED EXPLAINABILITY ============
    logger.info("🔍 Step 4: Advanced Explainability Analysis")
    
    # Get best performing model
    if results:
        best_key = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_result = results[best_key]
        logger.info(f"Best model: {best_key} with accuracy {best_result['test_accuracy']:.4f}")
        
        try:
            # Load best model for explainability
            if 'efficientnet' in best_key:
                from src.efficientnet import EfficientNetB0Classifier
                best_model = EfficientNetB0Classifier(num_classes=num_classes)
            else:
                model_type = best_key.split('_')[0]
                best_model = create_advanced_model(model_type, num_classes=num_classes)
            
            best_model.load_state_dict(torch.load(f"models/advanced_{best_key}_best.pth"))
            best_model = best_model.to(config['device'])
            best_model.eval()
            
            # Create background data for SHAP
            sample_images = []
            for i, (images, _, _) in enumerate(train_loader):
                sample_images.append(images)
                if i >= 2:  # Use first few batches as background
                    break
            background_data = torch.cat(sample_images, dim=0)[:50]  # 50 background samples
            
            # Test explainability methods
            for method in config['explainability_methods']:
                logger.info(f"Testing {method} explainability...")
                
                try:
                    explainer = create_explainer(
                        method, best_model, background_data, device=config['device']
                    )
                    
                    # Test on a sample image
                    test_image = next(iter(test_loader))[0][:1]  # First test image
                    
                    if method == 'comprehensive':
                        explanations = explainer.explain_comprehensive(test_image)
                        logger.info(f"✅ {method}: Generated comprehensive explanations")
                    else:
                        explanation = explainer.explain(test_image)
                        logger.info(f"✅ {method}: Generated explanation")
                    
                except Exception as e:
                    logger.warning(f"⚠️ {method} explainability failed: {e}")
        
        except Exception as e:
            logger.error(f"❌ Explainability analysis failed: {e}")
    
    # ============ STEP 5: CROSS-DATASET VALIDATION ============
    logger.info("🔄 Step 5: Cross-Dataset Validation")
    
    if use_combined and os.path.exists(config['bach_root']):
        try:
            from src.efficientnet import EfficientNetB0Classifier
            cross_validation_results = run_cross_dataset_validation(
                EfficientNetB0Classifier,
                config['breakhis_root'],
                config['bach_root'],
                config['device']
            )
            
            if cross_validation_results:
                logger.info("✅ Cross-dataset validation completed")
                logger.info(f"Results: {cross_validation_results['summary']}")
            
        except Exception as e:
            logger.error(f"❌ Cross-dataset validation failed: {e}")
    else:
        logger.info("⚠️ Skipping cross-dataset validation (BACH not available)")
    
    # ============ STEP 6: RESULTS SUMMARY ============
    logger.info("📋 Step 6: Results Summary")
    
    logger.info(f"\n{'='*60}")
    logger.info("ADVANCED THESIS TRAINING RESULTS")
    logger.info(f"{'='*60}")
    
    if results:
        # Sort results by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        logger.info("\nModel Performance Ranking:")
        for i, (key, result) in enumerate(sorted_results, 1):
            logger.info(f"{i}. {key}: {result['test_accuracy']:.4f}")
        
        # Best model details
        best_key, best_result = sorted_results[0]
        logger.info(f"\n🏆 BEST MODEL: {best_key}")
        logger.info(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
        logger.info(f"   Model Type: {best_result['model_type']}")
        logger.info(f"   Loss Function: {best_result['loss_type']}")
        
        # Save comprehensive results
        import json
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {k: str(v) for k, v in config.items()},
            'dataset_info': {
                'total_images': len(metadata),
                'num_classes': num_classes,
                'classes': classes,
                'samples_per_class': dict(zip(classes, samples_per_class))
            },
            'model_results': {k: {
                'test_accuracy': v['test_accuracy'],
                'model_type': v['model_type'],
                'loss_type': v['loss_type']
            } for k, v in results.items()},
            'best_model': best_key
        }
        
        with open('advanced_thesis_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info("📁 Results saved to advanced_thesis_results.json")
    
    else:
        logger.error("❌ No successful training results")
    
    logger.info(f"\n{'='*60}")
    logger.info("🎉 ADVANCED THESIS TRAINING COMPLETED")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()