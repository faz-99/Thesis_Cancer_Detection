#!/usr/bin/env python3
"""
Quick test script for explainability pipeline
Tests all components without requiring real data
"""

import sys
import os
import numpy as np
import torch
from PIL import Image
import cv2

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if all required modules can be imported"""
    print("🔄 Testing imports...")
    
    try:
        from explainability_pipeline import ExplainabilityPipeline
        print("✅ ExplainabilityPipeline imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ExplainabilityPipeline: {e}")
        return False
    
    try:
        from nuclei_segmentation import NucleiSegmenter
        print("✅ NucleiSegmenter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NucleiSegmenter: {e}")
        return False
    
    try:
        from efficientnet import EfficientNetB0Classifier
        print("✅ EfficientNetB0Classifier imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import EfficientNetB0Classifier: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n🔄 Testing model creation...")
    
    try:
        from efficientnet import EfficientNetB0Classifier
        model = EfficientNetB0Classifier(num_classes=8)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None

def test_nuclei_segmentation():
    """Test nuclei segmentation"""
    print("\n🔄 Testing nuclei segmentation...")
    
    try:
        from nuclei_segmentation import NucleiSegmenter
        
        # Create synthetic image
        synthetic_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add some circular structures (mock nuclei)
        for i in range(5):
            center = (np.random.randint(50, 174), np.random.randint(50, 174))
            radius = np.random.randint(8, 20)
            color = (np.random.randint(50, 150), np.random.randint(100, 200), np.random.randint(150, 255))
            cv2.circle(synthetic_img, center, radius, color, -1)
        
        segmenter = NucleiSegmenter(method='watershed')
        labels, nuclei_data = segmenter.segment(synthetic_img)
        
        print(f"✅ Segmentation completed: found {len(nuclei_data)} nuclei")
        print(f"   Label map shape: {labels.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Nuclei segmentation failed: {e}")
        return False

def test_pipeline_components():
    """Test individual pipeline components"""
    print("\n🔄 Testing pipeline components...")
    
    try:
        from explainability_pipeline import ExplainabilityPipeline
        from efficientnet import EfficientNetB0Classifier
        
        # Create model and pipeline
        model = EfficientNetB0Classifier(num_classes=8)
        pipeline = ExplainabilityPipeline(model)
        
        # Create synthetic image
        synthetic_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test image preprocessing
        tensor_img, original_img = pipeline.load_and_preprocess_image_from_array(synthetic_img)
        print(f"✅ Image preprocessing: tensor shape {tensor_img.shape}")
        
        # Test prediction
        pred_class, confidence, probs = pipeline.get_prediction(tensor_img)
        print(f"✅ Prediction: class {pred_class}, confidence {confidence:.3f}")
        
        # Test Grad-CAM
        heatmap = pipeline.generate_gradcam(tensor_img, pred_class)
        print(f"✅ Grad-CAM: heatmap shape {heatmap.shape}")
        
        # Test activation mask
        activation_mask = pipeline.create_activation_mask(heatmap)
        print(f"✅ Activation mask: shape {activation_mask.shape}, activated pixels {np.sum(activation_mask)}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_synthetic_test_image():
    """Create a synthetic test image for testing"""
    print("\n🔄 Creating synthetic test image...")
    
    # Create base image
    img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    
    # Add some tissue-like structures
    for i in range(15):
        center = (np.random.randint(20, 204), np.random.randint(20, 204))
        radius = np.random.randint(5, 15)
        color = (
            np.random.randint(80, 180),
            np.random.randint(100, 200), 
            np.random.randint(120, 220)
        )
        cv2.circle(img, center, radius, color, -1)
    
    # Add some irregular shapes
    for i in range(8):
        pts = np.random.randint(0, 224, (6, 2))
        pts = pts.reshape((-1, 1, 2))
        color = (
            np.random.randint(60, 160),
            np.random.randint(80, 180),
            np.random.randint(100, 200)
        )
        cv2.fillPoly(img, [pts], color)
    
    # Save test image
    os.makedirs("outputs/test", exist_ok=True)
    test_path = "outputs/test/synthetic_test.png"
    cv2.imwrite(test_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Synthetic test image saved: {test_path}")
    return test_path

def main():
    print("🧪 EXPLAINABILITY PIPELINE TEST SUITE")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import test failed - check dependencies")
        return
    
    # Test 2: Model creation
    model = test_model_creation()
    if model is None:
        print("❌ Model creation failed")
        return
    
    # Test 3: Nuclei segmentation
    if not test_nuclei_segmentation():
        print("❌ Nuclei segmentation test failed")
        return
    
    # Test 4: Pipeline components
    if not test_pipeline_components():
        print("❌ Pipeline component test failed")
        return
    
    # Test 5: Create synthetic test image
    test_image_path = create_synthetic_test_image()
    
    print("\n✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("🎯 The explainability pipeline is ready to use!")
    print(f"📁 Test image created: {test_image_path}")
    print("\nNext steps:")
    print("1. Run: python demo_explainability.py")
    print("2. Or run: python run_explainability.py")

# Add method to pipeline for testing
def add_test_method():
    """Add test method to pipeline class"""
    from explainability_pipeline import ExplainabilityPipeline
    
    def load_and_preprocess_image_from_array(self, image_array):
        """Load from numpy array instead of file"""
        original_img = image_array
        
        # Preprocess for model
        pil_img = Image.fromarray(original_img)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        return tensor_img, original_img
    
    ExplainabilityPipeline.load_and_preprocess_image_from_array = load_and_preprocess_image_from_array

if __name__ == "__main__":
    add_test_method()
    main()