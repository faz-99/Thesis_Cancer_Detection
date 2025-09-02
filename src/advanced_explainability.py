#!/usr/bin/env python3
"""
Advanced Explainability Methods for Breast Cancer Classification
Implements SHAP, Integrated Gradients, and other interpretability techniques beyond GradCAM
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import shap
from captum.attr import IntegratedGradients, GradientShap, Occlusion, LayerGradCam
import logging

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretability
    Provides pixel-level feature attribution using game theory
    """
    
    def __init__(self, model, background_data, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Create SHAP explainer
        self.explainer = shap.DeepExplainer(model, background_data.to(device))
        
    def explain(self, input_tensor, target_class=None):
        """
        Generate SHAP explanations for input
        
        Args:
            input_tensor: Input image tensor [1, 3, 224, 224]
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            SHAP values and visualization
        """
        with torch.no_grad():
            # Get prediction if target not specified
            if target_class is None:
                outputs = self.model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(input_tensor)
            
            # Convert to numpy for visualization
            if isinstance(shap_values, list):
                shap_values = shap_values[target_class]
            
            shap_values = shap_values[0]  # Remove batch dimension
            
            return {
                'shap_values': shap_values,
                'target_class': target_class,
                'attribution_map': self._create_attribution_map(shap_values)
            }
    
    def _create_attribution_map(self, shap_values):
        """Create heatmap from SHAP values"""
        # Sum across channels for visualization
        attribution = np.sum(np.abs(shap_values), axis=0)
        
        # Normalize to [0, 1]
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        
        return attribution

class IntegratedGradientsExplainer:
    """
    Integrated Gradients for attribution analysis
    Computes gradients along path from baseline to input
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(model)
        
    def explain(self, input_tensor, target_class=None, n_steps=50):
        """
        Generate Integrated Gradients explanations
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            n_steps: Number of integration steps
        
        Returns:
            Attribution map and metadata
        """
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
        
        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor)
        
        # Calculate attributions
        attributions = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=n_steps
        )
        
        # Convert to numpy
        attributions_np = attributions.squeeze().cpu().numpy()
        
        return {
            'attributions': attributions_np,
            'target_class': target_class,
            'attribution_map': self._create_attribution_map(attributions_np)
        }
    
    def _create_attribution_map(self, attributions):
        """Create visualization from attributions"""
        # Sum across channels
        attribution_map = np.sum(np.abs(attributions), axis=0)
        
        # Normalize
        attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min())
        
        return attribution_map

class GradientSHAPExplainer:
    """
    Gradient SHAP combining gradients with SHAP sampling
    Provides more stable attributions than pure gradients
    """
    
    def __init__(self, model, background_data, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.gradient_shap = GradientShap(model)
        self.background_data = background_data.to(device)
        
    def explain(self, input_tensor, target_class=None, n_samples=50):
        """Generate Gradient SHAP explanations"""
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
        
        # Calculate attributions
        attributions = self.gradient_shap.attribute(
            input_tensor,
            baselines=self.background_data,
            target=target_class,
            n_samples=n_samples
        )
        
        attributions_np = attributions.squeeze().cpu().numpy()
        
        return {
            'attributions': attributions_np,
            'target_class': target_class,
            'attribution_map': self._create_attribution_map(attributions_np)
        }
    
    def _create_attribution_map(self, attributions):
        attribution_map = np.sum(np.abs(attributions), axis=0)
        attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min())
        return attribution_map

class OcclusionExplainer:
    """
    Occlusion-based explanations
    Tests importance of image regions by occluding them
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.occlusion = Occlusion(model)
        
    def explain(self, input_tensor, target_class=None, sliding_window_shapes=(3, 15, 15)):
        """Generate occlusion-based explanations"""
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
        
        # Calculate occlusion attributions
        attributions = self.occlusion.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            target=target_class,
            show_progress=False
        )
        
        attributions_np = attributions.squeeze().cpu().numpy()
        
        return {
            'attributions': attributions_np,
            'target_class': target_class,
            'attribution_map': self._create_attribution_map(attributions_np)
        }
    
    def _create_attribution_map(self, attributions):
        attribution_map = np.sum(attributions, axis=0)
        attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min())
        return attribution_map

class LayerGradCAMExplainer:
    """
    Layer-wise GradCAM for different network depths
    Provides multi-scale explanations
    """
    
    def __init__(self, model, target_layers, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Create GradCAM for each target layer
        self.gradcams = {}
        for layer_name, layer in target_layers.items():
            self.gradcams[layer_name] = LayerGradCam(model, layer)
    
    def explain(self, input_tensor, target_class=None):
        """Generate multi-layer GradCAM explanations"""
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
        
        explanations = {}
        
        for layer_name, gradcam in self.gradcams.items():
            # Calculate GradCAM for this layer
            attributions = gradcam.attribute(
                input_tensor,
                target=target_class
            )
            
            # Upsample to input size
            attributions_upsampled = F.interpolate(
                attributions.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            attributions_np = attributions_upsampled.cpu().numpy()
            
            explanations[layer_name] = {
                'attributions': attributions_np,
                'attribution_map': self._create_attribution_map(attributions_np)
            }
        
        return {
            'layer_explanations': explanations,
            'target_class': target_class
        }
    
    def _create_attribution_map(self, attributions):
        if len(attributions.shape) > 2:
            attribution_map = np.mean(attributions, axis=0)
        else:
            attribution_map = attributions
        
        attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min())
        return attribution_map

class ComprehensiveExplainer:
    """
    Combines multiple explanation methods for comprehensive analysis
    """
    
    def __init__(self, model, background_data, target_layers=None, device='cpu'):
        self.model = model
        self.device = device
        
        # Initialize all explainers
        self.shap_explainer = SHAPExplainer(model, background_data, device)
        self.ig_explainer = IntegratedGradientsExplainer(model, device)
        self.gradient_shap_explainer = GradientSHAPExplainer(model, background_data, device)
        self.occlusion_explainer = OcclusionExplainer(model, device)
        
        if target_layers:
            self.layer_gradcam_explainer = LayerGradCAMExplainer(model, target_layers, device)
        else:
            self.layer_gradcam_explainer = None
    
    def explain_comprehensive(self, input_tensor, target_class=None):
        """
        Generate comprehensive explanations using all methods
        
        Returns:
            Dictionary with all explanation results
        """
        logger.info("Generating comprehensive explanations...")
        
        results = {}
        
        try:
            # SHAP explanations
            logger.info("Computing SHAP explanations...")
            results['shap'] = self.shap_explainer.explain(input_tensor, target_class)
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            results['shap'] = None
        
        try:
            # Integrated Gradients
            logger.info("Computing Integrated Gradients...")
            results['integrated_gradients'] = self.ig_explainer.explain(input_tensor, target_class)
        except Exception as e:
            logger.warning(f"Integrated Gradients failed: {e}")
            results['integrated_gradients'] = None
        
        try:
            # Gradient SHAP
            logger.info("Computing Gradient SHAP...")
            results['gradient_shap'] = self.gradient_shap_explainer.explain(input_tensor, target_class)
        except Exception as e:
            logger.warning(f"Gradient SHAP failed: {e}")
            results['gradient_shap'] = None
        
        try:
            # Occlusion
            logger.info("Computing Occlusion explanations...")
            results['occlusion'] = self.occlusion_explainer.explain(input_tensor, target_class)
        except Exception as e:
            logger.warning(f"Occlusion explanation failed: {e}")
            results['occlusion'] = None
        
        if self.layer_gradcam_explainer:
            try:
                # Layer GradCAM
                logger.info("Computing Layer GradCAM...")
                results['layer_gradcam'] = self.layer_gradcam_explainer.explain(input_tensor, target_class)
            except Exception as e:
                logger.warning(f"Layer GradCAM failed: {e}")
                results['layer_gradcam'] = None
        
        logger.info("Comprehensive explanations completed")
        return results
    
    def create_visualization_grid(self, explanations, original_image):
        """
        Create a grid visualization of all explanations
        
        Args:
            explanations: Results from explain_comprehensive
            original_image: Original input image as PIL Image
        
        Returns:
            Combined visualization grid
        """
        import matplotlib.pyplot as plt
        
        # Prepare subplots
        methods = [k for k, v in explanations.items() if v is not None]
        n_methods = len(methods)
        
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        # Plot original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot each explanation
        for i, method in enumerate(methods[1:], 1):
            if i < len(axes):
                explanation = explanations[method]
                if 'attribution_map' in explanation:
                    axes[i].imshow(explanation['attribution_map'], cmap='hot')
                    axes[i].set_title(f'{method.replace("_", " ").title()}')
                    axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(methods), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig

def create_explainer(explainer_type, model, background_data=None, target_layers=None, device='cpu'):
    """
    Factory function to create explainers
    
    Args:
        explainer_type: 'shap', 'integrated_gradients', 'gradient_shap', 'occlusion', 'layer_gradcam', 'comprehensive'
        model: PyTorch model
        background_data: Background data for SHAP-based methods
        target_layers: Dictionary of layer names and layers for GradCAM
        device: Computing device
    
    Returns:
        Explainer instance
    """
    
    if explainer_type == 'shap':
        return SHAPExplainer(model, background_data, device)
    
    elif explainer_type == 'integrated_gradients':
        return IntegratedGradientsExplainer(model, device)
    
    elif explainer_type == 'gradient_shap':
        return GradientSHAPExplainer(model, background_data, device)
    
    elif explainer_type == 'occlusion':
        return OcclusionExplainer(model, device)
    
    elif explainer_type == 'layer_gradcam':
        return LayerGradCAMExplainer(model, target_layers, device)
    
    elif explainer_type == 'comprehensive':
        return ComprehensiveExplainer(model, background_data, target_layers, device)
    
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")