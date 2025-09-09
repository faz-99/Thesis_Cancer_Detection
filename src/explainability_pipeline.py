import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import json
import os
from skimage import measure, filters
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ExplainabilityPipeline:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names mapping
        self.class_names = {
            0: 'Adenosis', 1: 'Fibroadenoma', 2: 'Phyllodes_tumor', 3: 'Tubular_adenoma',
            4: 'Ductal_carcinoma', 5: 'Lobular_carcinoma', 6: 'Mucinous_carcinoma', 7: 'Papillary_carcinoma'
        }
    
    def load_and_preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Step 1: Load model and prepare input"""
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        pil_img = Image.fromarray(original_img)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        return tensor_img, original_img
    
    def get_prediction(self, tensor_img: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """Get model prediction and probabilities"""
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities
    
    def generate_gradcam(self, tensor_img: torch.Tensor, target_class: int) -> np.ndarray:
        """Step 2: Generate Grad-CAM heatmap"""
        # Hook for gradients
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        # Register hooks on last conv layer
        target_layer = None
        for name, module in self.model.named_modules():
            if 'features' in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            # Fallback for different architectures
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        
        handle_backward = target_layer.register_backward_hook(backward_hook)
        handle_forward = target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        outputs = self.model(tensor_img)
        
        # Backward pass
        self.model.zero_grad()
        outputs[0, target_class].backward()
        
        # Generate Grad-CAM
        grads = gradients[0][0]  # [C, H, W]
        acts = activations[0][0]  # [C, H, W]
        
        weights = torch.mean(grads, dim=(1, 2))  # [C]
        cam = torch.sum(weights.unsqueeze(1).unsqueeze(2) * acts, dim=0)  # [H, W]
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Remove hooks
        handle_backward.remove()
        handle_forward.remove()
        
        return cam
    
    def create_activation_mask(self, heatmap: np.ndarray, threshold_method='percentile') -> np.ndarray:
        """Step 3: Create activation mask from heatmap"""
        if threshold_method == 'percentile':
            threshold = np.percentile(heatmap, 80)  # Top 20%
        else:  # Otsu's method
            threshold = filters.threshold_otsu(heatmap)
        
        binary_mask = (heatmap >= threshold).astype(np.uint8)
        return binary_mask
    
    def segment_nuclei(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Step 4: Perform nuclei segmentation using simple watershed"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Find peaks (nuclei centers)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Label connected components
        num_labels, labels = cv2.connectedComponents(sure_fg)
        
        # Extract region properties
        props = measure.regionprops(labels)
        nuclei_data = []
        
        for prop in props:
            if prop.area > 10:  # Filter small regions
                nuclei_data.append({
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'bbox': prop.bbox
                })
        
        return labels, nuclei_data
    
    def compute_overlap_analysis(self, activation_mask: np.ndarray, nuclei_data: List[Dict]) -> Dict:
        """Step 5: Compute overlap between activation and nuclei"""
        # Resize activation mask to match image size
        h, w = activation_mask.shape
        
        nuclei_inside = []
        nuclei_outside = []
        
        for nucleus in nuclei_data:
            y, x = nucleus['centroid']
            # Scale coordinates to mask size
            mask_y = int(y * h / 224)  # Assuming original was resized to 224
            mask_x = int(x * w / 224)
            
            # Check bounds
            if 0 <= mask_y < h and 0 <= mask_x < w:
                if activation_mask[mask_y, mask_x] > 0:
                    nuclei_inside.append(nucleus)
                else:
                    nuclei_outside.append(nucleus)
        
        n_total = len(nuclei_data)
        n_inside = len(nuclei_inside)
        n_outside = len(nuclei_outside)
        ratio = n_inside / n_total if n_total > 0 else 0
        
        # Compute statistics
        def get_stats(nuclei_list):
            if not nuclei_list:
                return {'area': 0, 'eccentricity': 0, 'solidity': 0}
            areas = [n['area'] for n in nuclei_list]
            eccentricities = [n['eccentricity'] for n in nuclei_list]
            solidities = [n['solidity'] for n in nuclei_list]
            return {
                'area': np.mean(areas),
                'eccentricity': np.mean(eccentricities),
                'solidity': np.mean(solidities)
            }
        
        stats_inside = get_stats(nuclei_inside)
        stats_outside = get_stats(nuclei_outside)
        
        return {
            'n_total': n_total,
            'n_inside': n_inside,
            'n_outside': n_outside,
            'ratio': ratio,
            'stats_inside': stats_inside,
            'stats_outside': stats_outside
        }
    
    def compute_texture_features(self, image: np.ndarray, activation_mask: np.ndarray) -> Dict:
        """Step 6: Compute texture features in activated region"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize mask to match image
        mask_resized = cv2.resize(activation_mask.astype(np.uint8), 
                                (gray.shape[1], gray.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Extract pixels inside and outside activation
        inside_pixels = gray[mask_resized > 0]
        outside_pixels = gray[mask_resized == 0]
        
        def compute_glcm_features(pixels):
            if len(pixels) < 4:
                return {'contrast': 0, 'homogeneity': 0, 'entropy': 0}
            
            # Reshape for GLCM
            pixels_2d = pixels.reshape(-1, 1)
            if pixels_2d.shape[0] < 2:
                return {'contrast': 0, 'homogeneity': 0, 'entropy': 0}
            
            # Compute GLCM
            glcm = greycomatrix(pixels_2d, [1], [0], levels=256, symmetric=True, normed=True)
            
            contrast = greycoprops(glcm, 'contrast')[0, 0]
            homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
            
            # Compute entropy manually
            glcm_norm = glcm / (glcm.sum() + 1e-8)
            entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-8))
            
            return {
                'contrast': float(contrast),
                'homogeneity': float(homogeneity),
                'entropy': float(entropy)
            }
        
        features_inside = compute_glcm_features(inside_pixels)
        features_outside = compute_glcm_features(outside_pixels)
        
        return {
            'inside': features_inside,
            'outside': features_outside
        }
    
    def generate_explanation_rules(self, overlap_data: Dict, texture_data: Dict, 
                                 activation_mask: np.ndarray) -> List[str]:
        """Step 7: Rule-based mapping to human-readable phrases"""
        phrases = []
        
        # Nuclei overlap rules
        if overlap_data['ratio'] > 0.6:
            phrases.append("activated region largely overlaps nuclei clusters")
        
        # Nuclear enlargement
        if (overlap_data['stats_outside']['area'] > 0 and 
            overlap_data['stats_inside']['area'] >= 1.5 * overlap_data['stats_outside']['area']):
            phrases.append("nuclei inside activation are enlarged (nuclear enlargement)")
        
        # Nuclear shape irregularity
        if (overlap_data['stats_outside']['eccentricity'] > 0 and 
            overlap_data['stats_inside']['eccentricity'] >= 1.2 * overlap_data['stats_outside']['eccentricity']):
            phrases.append("activated region shows irregular/elongated nuclear shapes")
        
        # Nuclear compactness
        if (overlap_data['stats_outside']['solidity'] > 0 and 
            overlap_data['stats_inside']['solidity'] <= 0.85 * overlap_data['stats_outside']['solidity']):
            phrases.append("activated nuclei are less compact, indicating irregular borders")
        
        # Tissue heterogeneity
        if (texture_data['outside']['entropy'] > 0 and 
            texture_data['inside']['entropy'] >= 1.2 * texture_data['outside']['entropy']):
            phrases.append("activated region shows high tissue heterogeneity")
        
        # Spatial location analysis
        h, w = activation_mask.shape
        center_y, center_x = h // 2, w // 2
        
        # Find centroid of activation
        y_coords, x_coords = np.where(activation_mask > 0)
        if len(y_coords) > 0:
            activation_center_y = np.mean(y_coords)
            activation_center_x = np.mean(x_coords)
            
            # Check if near center
            if (abs(activation_center_y - center_y) < h * 0.2 and 
                abs(activation_center_x - center_x) < w * 0.2):
                phrases.append("model focused on central region")
            
            # Check if near edge
            if (activation_center_y < h * 0.1 or activation_center_y > h * 0.9 or
                activation_center_x < w * 0.1 or activation_center_x > w * 0.9):
                phrases.append("model focused on glandular boundary")
        
        return phrases
    
    def construct_final_explanation(self, predicted_class: int, confidence: float, 
                                  phrases: List[str]) -> str:
        """Step 8: Construct final explanation"""
        class_name = self.class_names.get(predicted_class, f"Class_{predicted_class}")
        
        explanation = f"Prediction: {class_name} (confidence: {confidence:.2f})\n"
        explanation += f"Explanation: The model focused on regions with specific histopathological patterns.\n"
        
        if phrases:
            explanation += f"Key findings: {', '.join(phrases)}."
        else:
            explanation += "Key findings: Standard tissue patterns detected."
        
        return explanation
    
    def validate_faithfulness(self, tensor_img: torch.Tensor, activation_mask: np.ndarray, 
                            original_confidence: float) -> Dict:
        """Step 9: Faithfulness validation"""
        # Create masked image (zero out activation region)
        masked_img = tensor_img.clone()
        
        # Resize mask to match tensor dimensions
        mask_resized = cv2.resize(activation_mask.astype(np.float32), (224, 224))
        mask_tensor = torch.from_numpy(mask_resized).to(self.device)
        
        # Apply mask to all channels
        for c in range(3):
            masked_img[0, c] = masked_img[0, c] * (1 - mask_tensor)
        
        # Get prediction on masked image
        with torch.no_grad():
            masked_outputs = self.model(masked_img)
            masked_probs = F.softmax(masked_outputs, dim=1)
            masked_confidence = torch.max(masked_probs).item()
        
        confidence_drop = original_confidence - masked_confidence
        is_faithful = confidence_drop > 0.1  # Threshold for faithfulness
        
        return {
            'original_confidence': original_confidence,
            'masked_confidence': masked_confidence,
            'confidence_drop': confidence_drop,
            'is_faithful': is_faithful
        }
    
    def save_outputs(self, output_dir: str, image_path: str, predicted_class: int, 
                    confidence: float, heatmap: np.ndarray, original_img: np.ndarray,
                    activation_mask: np.ndarray, nuclei_mask: np.ndarray, 
                    explanation: str, metrics: Dict):
        """Save all outputs"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save prediction
        with open(os.path.join(output_dir, f"{base_name}_prediction_label.txt"), 'w') as f:
            f.write(f"{self.class_names.get(predicted_class, f'Class_{predicted_class}')}\n")
            f.write(f"Confidence: {confidence:.4f}")
        
        # Save heatmap
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        heatmap_img = (heatmap_colored * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_heatmap.png"), 
                   cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        
        # Save overlay
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        overlay = original_img * 0.6 + (plt.cm.jet(heatmap_resized)[:, :, :3] * 255) * 0.4
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), 
                   cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Save activation mask
        mask_img = (activation_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_activation_mask.png"), mask_img)
        
        # Save nuclei mask
        nuclei_colored = (nuclei_mask / nuclei_mask.max() * 255).astype(np.uint8) if nuclei_mask.max() > 0 else nuclei_mask
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_nuclei_mask.png"), nuclei_colored)
        
        # Save explanation
        with open(os.path.join(output_dir, f"{base_name}_explanation.txt"), 'w') as f:
            f.write(explanation)
        
        # Save metrics
        with open(os.path.join(output_dir, f"{base_name}_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def process_image(self, image_path: str, output_dir: str) -> Dict:
        """Complete pipeline for single image"""
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Step 1: Load and preprocess
        tensor_img, original_img = self.load_and_preprocess_image(image_path)
        
        # Get prediction
        predicted_class, confidence, _ = self.get_prediction(tensor_img)
        
        # Step 2: Generate Grad-CAM
        heatmap = self.generate_gradcam(tensor_img, predicted_class)
        
        # Step 3: Create activation mask
        activation_mask = self.create_activation_mask(heatmap)
        
        # Step 4: Nuclei segmentation
        nuclei_mask, nuclei_data = self.segment_nuclei(original_img)
        
        # Step 5: Overlap analysis
        overlap_data = self.compute_overlap_analysis(activation_mask, nuclei_data)
        
        # Step 6: Texture features
        texture_data = self.compute_texture_features(original_img, activation_mask)
        
        # Step 7: Generate explanation phrases
        phrases = self.generate_explanation_rules(overlap_data, texture_data, activation_mask)
        
        # Step 8: Construct final explanation
        explanation = self.construct_final_explanation(predicted_class, confidence, phrases)
        
        # Step 9: Validate faithfulness
        faithfulness = self.validate_faithfulness(tensor_img, activation_mask, confidence)
        
        # Compile metrics
        metrics = {
            'prediction': {
                'class': predicted_class,
                'class_name': self.class_names.get(predicted_class, f'Class_{predicted_class}'),
                'confidence': confidence
            },
            'overlap_analysis': overlap_data,
            'texture_features': texture_data,
            'faithfulness': faithfulness,
            'explanation_phrases': phrases
        }
        
        # Save outputs
        self.save_outputs(output_dir, image_path, predicted_class, confidence, 
                         heatmap, original_img, activation_mask, nuclei_mask, 
                         explanation, metrics)
        
        print(f"✅ Completed: {os.path.basename(image_path)}")
        return metrics