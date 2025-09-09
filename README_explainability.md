# Breast Cancer Explainability Pipeline

## Overview
This implements **Path A: Explainability Without Annotations** - a comprehensive pipeline that generates human-readable explanations for breast cancer histopathology image classifications without requiring expert-annotated text reports.

## Pipeline Steps

### 🔹 Step 1 — Load Model and Prepare Input
- Loads fine-tuned EfficientNet model
- Preprocesses input image (resize, normalize)
- Generates class prediction and confidence scores

### 🔹 Step 2 — Generate Grad-CAM Heatmap
- Targets last convolutional layer of EfficientNet
- Creates activation heatmap for predicted class
- Normalizes and resizes to match input dimensions

### 🔹 Step 3 — Create Activation Mask
- Applies thresholding to Grad-CAM heatmap
- Uses top 20% percentile or Otsu's method
- Creates binary mask of activated regions

### 🔹 Step 4 — Perform Nuclei Segmentation
- Multiple segmentation methods available:
  - **Watershed**: Classical computer vision approach
  - **Cellpose**: Deep learning-based segmentation (optional)
  - **Hybrid**: Combines multiple methods
- Extracts morphological properties of each nucleus

### 🔹 Step 5 — Compute Overlap Analysis
- Determines which nuclei fall within activated regions
- Computes statistics for nuclei inside vs outside activation
- Analyzes morphological differences

### 🔹 Step 6 — Compute Texture Features
- Extracts GLCM (Gray-Level Co-occurrence Matrix) features
- Compares texture properties inside vs outside activation
- Measures tissue heterogeneity and patterns

### 🔹 Step 7 — Rule-Based Mapping
- Translates numerical features into human-readable phrases
- Applies clinical knowledge rules:
  - Nuclear enlargement detection
  - Shape irregularity assessment
  - Tissue heterogeneity analysis
  - Spatial pattern recognition

### 🔹 Step 8 — Construct Final Explanation
- Combines all findings into coherent explanation
- Provides prediction confidence and key findings
- Generates natural language description

### 🔹 Step 9 — Faithfulness Validation
- Tests explanation reliability by masking activated regions
- Measures confidence drop to validate explanation quality
- Ensures explanations reflect actual model reasoning

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_explainability.txt
```

### 2. Run Demo
```bash
python demo_explainability.py
```

### 3. Process Multiple Images
```bash
python run_explainability.py
```

## Output Files

For each processed image, the pipeline generates:

- **`*_prediction_label.txt`** - Predicted class and confidence
- **`*_heatmap.png`** - Grad-CAM heatmap visualization
- **`*_overlay.png`** - Heatmap overlaid on original image
- **`*_activation_mask.png`** - Binary activation mask
- **`*_nuclei_mask.png`** - Nuclei segmentation result
- **`*_explanation.txt`** - Human-readable explanation
- **`*_metrics.json`** - Quantitative analysis metrics
- **`*_complete_analysis.png`** - Comprehensive visualization

## Example Output

```
Prediction: Ductal_carcinoma (confidence: 0.87)
Explanation: The model focused on regions with specific histopathological patterns.
Key findings: activated region largely overlaps nuclei clusters, nuclei inside activation are enlarged (nuclear enlargement), activated region shows high tissue heterogeneity.
```

## Architecture

```
src/
├── explainability_pipeline.py    # Main pipeline implementation
├── nuclei_segmentation.py        # Advanced nuclei segmentation
└── efficientnet.py              # Model architecture

demo_explainability.py            # Single image demo
run_explainability.py            # Batch processing script
```

## Key Features

### 🎯 **No Annotation Required**
- Works without expert-labeled text reports
- Uses only image data and model predictions
- Generates explanations from visual patterns

### 🔬 **Clinical Relevance**
- Focuses on histopathologically relevant features
- Maps to known cancer indicators (nuclear enlargement, irregularity)
- Provides spatial context (central vs boundary regions)

### 🔍 **Multi-Scale Analysis**
- Combines pixel-level (Grad-CAM) and object-level (nuclei) features
- Analyzes both local texture and global patterns
- Integrates morphological and spatial information

### ✅ **Validation Built-in**
- Faithfulness testing through masking experiments
- Confidence drop measurement
- Quantitative reliability assessment

## Customization

### Adding New Rules
Extend the `generate_explanation_rules()` method in `ExplainabilityPipeline`:

```python
def generate_explanation_rules(self, overlap_data, texture_data, activation_mask):
    phrases = []
    
    # Add your custom rules here
    if custom_condition:
        phrases.append("custom explanation phrase")
    
    return phrases
```

### Different Segmentation Methods
Configure nuclei segmentation in the pipeline:

```python
# Use Cellpose (if available)
pipeline = ExplainabilityPipeline(model, segmentation_method='cellpose')

# Use watershed (default)
pipeline = ExplainabilityPipeline(model, segmentation_method='watershed')

# Use hybrid approach
pipeline = ExplainabilityPipeline(model, segmentation_method='hybrid')
```

## Performance Metrics

The pipeline tracks several validation metrics:

- **Faithfulness Rate**: Percentage of explanations that cause significant confidence drop when masked
- **Coverage**: Proportion of image area covered by explanations
- **Consistency**: Similarity of explanations for similar images
- **Clinical Relevance**: Alignment with known pathological indicators

## Limitations

1. **Segmentation Quality**: Nuclei segmentation accuracy affects explanation quality
2. **Rule Calibration**: Explanation rules may need adjustment for different datasets
3. **Model Dependency**: Explanations are only as good as the underlying model
4. **Validation Scope**: Limited validation on expert-annotated explanations

## Future Enhancements

- [ ] Integration with pathologist feedback for rule refinement
- [ ] Multi-magnification explanation consistency
- [ ] Uncertainty quantification in explanations
- [ ] Interactive explanation refinement interface
- [ ] Integration with medical knowledge graphs

## Citation

```bibtex
@misc{explainability2024,
  title={Explainable Breast Cancer Detection without Annotations},
  author={Your Name},
  year={2024},
  note={Grad-CAM + Nuclei Segmentation + Rule-based Explanation Pipeline}
}
```