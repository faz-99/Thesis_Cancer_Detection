#!/usr/bin/env python3
"""
Demo script for breast cancer explainability pipeline
Processes a single image and shows all outputs
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.append('src')

from explainability_pipeline import ExplainabilityPipeline
from efficientnet import EfficientNetB0Classifier

def create_demo_model():
    """Create a demo model for testing (since we might not have trained weights)"""
    model = EfficientNetB0Classifier(num_classes=8)
    # Initialize with random weights for demo
    return model

def visualize_results(output_dir: str, base_name: str):
    """Create a comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Load images
    try:
        # Original overlay
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        if os.path.exists(overlay_path):
            overlay = cv2.imread(overlay_path)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            axes[0, 0].imshow(overlay)
            axes[0, 0].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
        
        # Heatmap
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
        if os.path.exists(heatmap_path):
            heatmap = cv2.imread(heatmap_path)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            axes[0, 1].imshow(heatmap)
            axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
        
        # Activation mask
        mask_path = os.path.join(output_dir, f"{base_name}_activation_mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            axes[0, 2].imshow(mask, cmap='gray')
            axes[0, 2].set_title('Activation Mask', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
        
        # Nuclei segmentation
        nuclei_path = os.path.join(output_dir, f"{base_name}_nuclei_mask.png")
        if os.path.exists(nuclei_path):
            nuclei = cv2.imread(nuclei_path, cv2.IMREAD_GRAYSCALE)
            axes[1, 0].imshow(nuclei, cmap='nipy_spectral')
            axes[1, 0].set_title('Nuclei Segmentation', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
        
        # Load and display explanation
        explanation_path = os.path.join(output_dir, f"{base_name}_explanation.txt")
        if os.path.exists(explanation_path):
            with open(explanation_path, 'r') as f:
                explanation = f.read()
            
            axes[1, 1].text(0.05, 0.95, explanation, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', wrap=True,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 1].set_title('Generated Explanation', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
        
        # Load and display metrics
        import json
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Create metrics summary
            metrics_text = f"""PREDICTION METRICS
Class: {metrics['prediction']['class_name']}
Confidence: {metrics['prediction']['confidence']:.3f}

NUCLEI ANALYSIS
Total nuclei: {metrics['overlap_analysis']['n_total']}
Inside activation: {metrics['overlap_analysis']['n_inside']}
Overlap ratio: {metrics['overlap_analysis']['ratio']:.3f}

FAITHFULNESS
Confidence drop: {metrics['faithfulness']['confidence_drop']:.3f}
Is faithful: {metrics['faithfulness']['is_faithful']}

EXPLANATION PHRASES
{chr(10).join(f"• {phrase}" for phrase in metrics['explanation_phrases'])}"""
            
            axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            axes[1, 2].set_title('Quantitative Metrics', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
        
    except Exception as e:
        print(f"⚠️ Error in visualization: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_complete_analysis.png"), 
                dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    OUTPUT_DIR = "outputs/demo_explainability"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create demo model
    print("🔄 Creating demo model...")
    model = create_demo_model()
    
    # Initialize pipeline
    print("🔄 Initializing explainability pipeline...")
    pipeline = ExplainabilityPipeline(model)
    
    # Find a sample image
    sample_image_paths = [
        "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/adenosis/SOB_B_A-14-22549AB/40X/SOB_B_A-14-22549AB-40-001.png",
        "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant/ductal_carcinoma/SOB_M_DC-14-10926/40X/SOB_M_DC-14-10926-40-001.png"
    ]
    
    # Find first available image
    sample_image = None
    for path in sample_image_paths:
        if os.path.exists(path):
            sample_image = path
            break
    
    if sample_image is None:
        # Create a synthetic test image
        print("📝 Creating synthetic test image...")
        synthetic_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some structure to make it more realistic
        for i in range(10):
            center = (np.random.randint(20, 204), np.random.randint(20, 204))
            radius = np.random.randint(5, 15)
            color = (np.random.randint(100, 255), np.random.randint(50, 150), np.random.randint(100, 200))
            cv2.circle(synthetic_img, center, radius, color, -1)
        
        sample_image = os.path.join(OUTPUT_DIR, "synthetic_test.png")
        cv2.imwrite(sample_image, cv2.cvtColor(synthetic_img, cv2.COLOR_RGB2BGR))
        print(f"✅ Created synthetic test image: {sample_image}")
    
    print(f"🖼️ Processing image: {os.path.basename(sample_image)}")
    
    try:
        # Process the image
        metrics = pipeline.process_image(sample_image, OUTPUT_DIR)
        
        print(f"\n📊 PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"Predicted class: {metrics['prediction']['class_name']}")
        print(f"Confidence: {metrics['prediction']['confidence']:.3f}")
        print(f"Nuclei detected: {metrics['overlap_analysis']['n_total']}")
        print(f"Overlap ratio: {metrics['overlap_analysis']['ratio']:.3f}")
        print(f"Faithfulness: {metrics['faithfulness']['is_faithful']}")
        print(f"Explanation phrases: {len(metrics['explanation_phrases'])}")
        
        if metrics['explanation_phrases']:
            print(f"\nKey findings:")
            for phrase in metrics['explanation_phrases']:
                print(f"  • {phrase}")
        
        # Create comprehensive visualization
        print(f"\n🎨 Creating visualization...")
        base_name = os.path.splitext(os.path.basename(sample_image))[0]
        visualize_results(OUTPUT_DIR, base_name)
        
        print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")
        print(f"📁 Generated files:")
        for file in os.listdir(OUTPUT_DIR):
            if base_name in file:
                print(f"   - {file}")
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()