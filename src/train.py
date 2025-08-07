#!/usr/bin/env python3
"""
Training utilities for breast cancer classification model

This module provides comprehensive training and evaluation functions for the
breast cancer classification task. It includes:

- Model training with validation monitoring
- Early stopping based on validation accuracy
- Comprehensive evaluation with metrics
- Progress tracking and logging
- Model checkpointing

Key features:
- Cross-entropy loss with class weighting support
- Adam optimizer with configurable learning rate
- Validation-based model selection
- Detailed metrics calculation (accuracy, precision, recall, F1)
- Confusion matrix generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging
import time

# Setup module logger
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, device='cpu', 
                class_weights=None, patience=5):
    """
    Train the breast cancer classification model with comprehensive monitoring
    
    This function implements a complete training loop with:
    - Training and validation phases for each epoch
    - Loss and accuracy tracking
    - Model checkpointing based on best validation accuracy
    - Early stopping to prevent overfitting
    - Detailed logging of training progress
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Maximum number of training epochs (default: 10)
        lr (float): Learning rate for optimizer (default: 1e-4)
        device (str): Device to run training on ('cpu' or 'cuda')
        class_weights (torch.Tensor, optional): Weights for handling class imbalance
        patience (int): Early stopping patience (default: 5)
        
    Returns:
        tuple: (trained_model, training_history)
            - trained_model: Model with best validation weights loaded
            - training_history: Dict with losses, accuracies, and metrics
    """
    logger.info(f"Starting model training:")
    logger.info(f"  - Epochs: {num_epochs}")
    logger.info(f"  - Learning rate: {lr}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Training batches: {len(train_loader)}")
    logger.info(f"  - Validation batches: {len(val_loader)}")
    
    # Setup loss function with optional class weights
    if class_weights is not None:
        logger.info("Using weighted CrossEntropyLoss for class imbalance")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        logger.info("Using standard CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    logger.info(f"Using Adam optimizer with weight_decay=1e-5")
    
    # Initialize tracking variables
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    
    # Training history storage
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    # Record training start time
    training_start_time = time.time()
    
    # Main training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # ============ TRAINING PHASE ============
        logger.info("Starting training phase...")
        model.train()  # Set model to training mode
        
        # Initialize training metrics
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (images, labels, _) in enumerate(train_pbar):
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update training statistics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += batch_size
            
            # Update progress bar
            current_acc = correct_predictions / total_samples
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        # Calculate epoch training metrics
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = correct_predictions / total_samples
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        logger.info(f"Training - Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.4f}")
        
        # ============ VALIDATION PHASE ============
        logger.info("Starting validation phase...")
        model.eval()  # Set model to evaluation mode
        
        # Initialize validation metrics
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        
        # Validation loop (no gradient computation)
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for images, labels, _ in val_pbar:
                # Move data to device
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass only
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Update validation statistics
                batch_size = images.size(0)
                val_running_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_samples += batch_size
                
                # Update progress bar
                current_val_acc = val_correct_predictions / val_total_samples
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_val_acc:.4f}'
                })
        
        # Calculate epoch validation metrics
        epoch_val_loss = val_running_loss / val_total_samples
        epoch_val_acc = val_correct_predictions / val_total_samples
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        logger.info(f"Validation - Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.4f}")
        
        # ============ MODEL CHECKPOINTING ============
        # Save model if validation accuracy improved
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            logger.info(f"✅ New best model! Validation accuracy: {best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # ============ TRAINING COMPLETION ============
    total_training_time = time.time() - training_start_time
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    logger.info(f"\n{'='*50}")
    logger.info(f"TRAINING COMPLETED")
    logger.info(f"{'='*50}")
    logger.info(f"Total training time: {total_training_time:.2f}s")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Final training accuracy: {train_accuracies[-1]:.4f}")
    logger.info(f"Model weights restored to best checkpoint")
    
    # Prepare training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'total_epochs': len(train_losses),
        'training_time': total_training_time
    }
    
    return model, training_history

def evaluate_model(model, test_loader, device='cpu', class_names=None):
    """
    Comprehensive evaluation of the trained model on test set
    
    This function performs thorough evaluation including:
    - Overall accuracy calculation
    - Per-class precision, recall, and F1-score
    - Confusion matrix generation
    - Detailed classification report
    
    Args:
        model (nn.Module): Trained model to evaluate
        test_loader (DataLoader): Test data loader
        device (str): Device to run evaluation on
        class_names (dict, optional): Mapping from indices to class names
        
    Returns:
        dict: Comprehensive evaluation results including:
            - accuracy: Overall accuracy
            - classification_report: Detailed per-class metrics
            - confusion_matrix: Confusion matrix
            - predictions: All model predictions
            - labels: All true labels
    """
    logger.info(f"Starting model evaluation on test set...")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Storage for predictions and labels
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    # Evaluation loop
    eval_start_time = time.time()
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating")
        for batch_idx, (images, labels, image_paths) in enumerate(test_pbar):
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress
            current_acc = np.mean(np.array(all_predictions) == np.array(all_true_labels))
            test_pbar.set_postfix({'Accuracy': f'{current_acc:.4f}'})
    
    eval_time = time.time() - eval_start_time
    
    # Convert to numpy arrays for analysis
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate overall accuracy
    accuracy = np.mean(all_predictions == all_true_labels)
    
    logger.info(f"Evaluation completed in {eval_time:.2f}s")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info(f"Total test samples: {len(all_true_labels)}")
    
    # Prepare results dictionary
    results = {
        'accuracy': accuracy,
        'predictions': all_predictions.tolist(),
        'labels': all_true_labels.tolist(),
        'probabilities': all_probabilities.tolist(),
        'num_samples': len(all_true_labels),
        'evaluation_time': eval_time
    }
    
    # Generate detailed metrics if class names provided
    if class_names:
        logger.info("Generating detailed classification metrics...")
        
        # Create target names list in correct order
        target_names = [class_names[i] for i in sorted(class_names.keys())]
        
        # Generate classification report
        classification_rep = classification_report(
            all_true_labels, 
            all_predictions, 
            target_names=target_names,
            digits=4
        )
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(all_true_labels, all_predictions)
        
        # Add detailed metrics to results
        results.update({
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': target_names
        })
        
        # Log per-class accuracy
        logger.info("Per-class accuracy:")
        for i, class_name in enumerate(target_names):
            class_mask = all_true_labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(all_predictions[class_mask] == all_true_labels[class_mask])
                logger.info(f"  {class_name}: {class_acc:.4f} ({np.sum(class_mask)} samples)")
    
    logger.info("Model evaluation completed successfully")
    return results

def save_training_history(history, filepath):
    """
    Save training history to file for later analysis
    
    Args:
        history (dict): Training history from train_model
        filepath (str): Path to save the history
    """
    import json
    
    logger.info(f"Saving training history to {filepath}")
    
    # Convert any numpy arrays to lists for JSON serialization
    serializable_history = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            serializable_history[key] = value.tolist()
        else:
            serializable_history[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    logger.info("Training history saved successfully")