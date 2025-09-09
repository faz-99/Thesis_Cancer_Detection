#!/usr/bin/env python3
"""
Breast Cancer Explainability Pipeline Runner
Implements Path A: Explainability Without Annotations
"""

import os
import torch
from glob import glob
from src.explainability_pipeline import ExplainabilityPipeline
from src.efficientnet import EfficientNetB0Classifier

def load_trained_model(model_path: str, num_classes: int = 8):
    """Load the trained EfficientNet model"""
    model = EfficientNetB0Classifier(num_classes=num_classes)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️ Model not found at {model_path}, using untrained model")
    
    return model

def main():
    # Configuration
    MODEL_PATH = "models/efficientnet_breakhis_best.pth"
    TEST_IMAGES_DIR = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    OUTPUT_DIR = "outputs/explainability"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("🔄 Loading trained model...")
    model = load_trained_model(MODEL_PATH)
    
    # Initialize pipeline
    print("🔄 Initializing explainability pipeline...")
    pipeline = ExplainabilityPipeline(model)
    
    # Get test images (sample from each class)
    print("🔍 Finding test images...")
    test_images = []
    
    # Sample images from different classes and magnifications
    for class_type in ['benign', 'malignant']:
        class_dir = os.path.join(TEST_IMAGES_DIR, class_type)
        if os.path.exists(class_dir):
            # Get all subclasses
            for subclass in os.listdir(class_dir):
                subclass_dir = os.path.join(class_dir, subclass)
                if os.path.isdir(subclass_dir):
                    # Get different magnifications
                    for mag in ['40X', '100X', '200X', '400X']:
                        mag_dir = os.path.join(subclass_dir, mag)
                        if os.path.exists(mag_dir):
                            # Get first image from this category
                            images = glob(os.path.join(mag_dir, "*.png"))
                            if images:
                                test_images.append(images[0])
                                if len(test_images) >= 16:  # Limit for demo
                                    break
                    if len(test_images) >= 16:
                        break
            if len(test_images) >= 16:
                break
    
    print(f"📊 Found {len(test_images)} test images")
    
    # Process each image
    results = []
    for i, image_path in enumerate(test_images):
        print(f"\n🔄 Processing image {i+1}/{len(test_images)}")
        try:
            metrics = pipeline.process_image(image_path, OUTPUT_DIR)
            results.append({
                'image_path': image_path,
                'metrics': metrics
            })
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            continue
    
    # Summary statistics
    print(f"\n📊 EXPLAINABILITY PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {len(results)}")
    
    if results:
        # Faithfulness statistics
        faithful_count = sum(1 for r in results if r['metrics']['faithfulness']['is_faithful'])
        print(f"Faithful explanations: {faithful_count}/{len(results)} ({faithful_count/len(results)*100:.1f}%)")
        
        # Average confidence drop
        avg_drop = sum(r['metrics']['faithfulness']['confidence_drop'] for r in results) / len(results)
        print(f"Average confidence drop: {avg_drop:.3f}")
        
        # Class distribution
        class_counts = {}
        for r in results:
            class_name = r['metrics']['prediction']['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        # Common explanation phrases
        all_phrases = []
        for r in results:
            all_phrases.extend(r['metrics']['explanation_phrases'])
        
        if all_phrases:
            from collections import Counter
            phrase_counts = Counter(all_phrases)
            print(f"\nMost common explanation patterns:")
            for phrase, count in phrase_counts.most_common(5):
                print(f"  {phrase}: {count} times")
    
    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")
    print(f"📁 Check the following files for each image:")
    print(f"   - *_prediction_label.txt")
    print(f"   - *_heatmap.png")
    print(f"   - *_overlay.png") 
    print(f"   - *_activation_mask.png")
    print(f"   - *_nuclei_mask.png")
    print(f"   - *_explanation.txt")
    print(f"   - *_metrics.json")

if __name__ == "__main__":
    main()