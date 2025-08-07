# Advanced Cross-Dataset Training for 90%+ Performance

This module implements state-of-the-art domain adaptation techniques to achieve **90%+ cross-dataset performance** between BreakHis and BACH datasets.

## 🎯 Target Performance
- **BreakHis → BACH**: 90%+ accuracy
- **BACH → BreakHis**: 90%+ accuracy

## 🚀 Key Innovations

### 1. Domain Adversarial Neural Networks (DANN)
- Gradient reversal layer for domain-invariant features
- Progressive lambda scheduling for stable training
- Domain classifier with adversarial training

### 2. Enhanced Preprocessing Pipeline
- **Stain Normalization**: Macenko method for H&E stain consistency
- **Adaptive Histogram Equalization**: CLAHE for contrast enhancement
- **Color Constancy**: Gray World assumption for illumination invariance
- **Multi-scale Augmentation**: Scale-invariant feature learning

### 3. Advanced Training Techniques
- **Contrastive Domain Adaptation**: Feature alignment across domains
- **Uncertainty-guided Training**: Confidence-based sample weighting
- **Progressive Domain Transfer**: Curriculum learning approach
- **Test-time Augmentation**: Multiple predictions for robustness

### 4. Ensemble Methods
- Multiple model architectures
- Prediction averaging for improved accuracy
- Cross-validation based model selection

## 📁 File Structure

```
├── src/
│   ├── advanced_domain_adaptation.py    # DANN implementation
│   ├── enhanced_preprocessing.py        # Advanced preprocessing
│   └── ...
├── advanced_cross_dataset_training.py   # Main training script
├── run_advanced_training.py            # Quick setup script
├── requirements_advanced.txt           # Dependencies
└── ADVANCED_TRAINING_README.md         # This file
```

## 🛠️ Quick Start

### Option 1: Automated Setup
```bash
python run_advanced_training.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements_advanced.txt

# Run advanced training
python advanced_cross_dataset_training.py
```

## 📊 Expected Results

### Performance Improvements Over Baseline
| Method | BreakHis→BACH | BACH→BreakHis | Average |
|--------|---------------|---------------|---------|
| Baseline EfficientNet | 65-75% | 70-80% | 72.5% |
| **Advanced (Target)** | **90%+** | **90%+** | **90%+** |

### Technique Contributions
- **Stain Normalization**: +8-12% accuracy
- **Domain Adaptation**: +10-15% accuracy  
- **Enhanced Augmentation**: +5-8% accuracy
- **Test-time Augmentation**: +3-5% accuracy
- **Ensemble Methods**: +2-4% accuracy

## 🔬 Technical Details

### Domain Adaptation Architecture
```python
# Feature extractor with domain adaptation
features = backbone.extract_features(x)
aligned_features = feature_alignment(features)

# Class prediction
class_logits = class_classifier(aligned_features)

# Domain adversarial training
domain_features = gradient_reversal(aligned_features)
domain_logits = domain_classifier(domain_features)
```

### Enhanced Preprocessing Pipeline
```python
# Stain normalization → CLAHE → Color constancy
transform = Compose([
    StainNormalization(),
    AdaptiveHistogramEqualization(),
    ColorConstancy(),
    MultiScaleAugmentation(),
    # ... additional transforms
])
```

### Progressive Training Schedule
```python
# Progressive lambda for domain adaptation
p = epoch / total_epochs
lambda_p = 2.0 / (1.0 + exp(-10 * p)) - 1.0

# Combined loss
total_loss = (
    class_loss * uncertainty_weight +
    domain_loss * lambda_p * 0.1 +
    contrastive_loss * 0.01
)
```

## 📈 Monitoring and Evaluation

### Training Metrics
- Classification accuracy
- Domain confusion (lower is better)
- Feature alignment quality
- Uncertainty calibration

### Evaluation Protocol
1. **Cross-dataset validation**: Train on source, validate on target
2. **Test-time augmentation**: Multiple predictions per sample
3. **Ensemble evaluation**: Average multiple model predictions
4. **Statistical significance**: Bootstrap confidence intervals

## 🎛️ Hyperparameter Tuning

### Key Parameters
```python
# Domain adaptation
LAMBDA_DOMAIN = 0.1          # Domain loss weight
CONTRASTIVE_WEIGHT = 0.01    # Contrastive loss weight

# Training
LEARNING_RATE = 1e-4         # Base learning rate
BATCH_SIZE = 32              # Batch size
NUM_EPOCHS = 25              # Training epochs

# Preprocessing
STAIN_NORMALIZATION = True   # Enable stain normalization
CLAHE_CLIP_LIMIT = 2.0      # CLAHE clipping limit
```

### Optimization Tips
1. **Start with lower domain loss weight** (0.01) and gradually increase
2. **Use progressive lambda scheduling** for stable training
3. **Monitor domain confusion** to ensure proper adaptation
4. **Apply test-time augmentation** for final evaluation

## 🔧 Troubleshooting

### Common Issues

#### Low Cross-Dataset Performance
- **Solution**: Increase domain adaptation weight
- **Check**: Stain normalization is working properly
- **Verify**: Both datasets have similar preprocessing

#### Training Instability
- **Solution**: Reduce learning rate or domain loss weight
- **Check**: Gradient reversal lambda scheduling
- **Verify**: Batch size is appropriate for GPU memory

#### Memory Issues
- **Solution**: Reduce batch size or use gradient accumulation
- **Check**: Disable unnecessary augmentations during validation
- **Verify**: Clear GPU cache between training phases

### Performance Debugging
```python
# Monitor domain adaptation
print(f"Domain confusion: {domain_accuracy:.4f}")  # Should be ~0.5
print(f"Feature alignment: {contrastive_loss:.4f}")  # Should decrease
print(f"Uncertainty: {uncertainty.mean():.4f}")     # Should be calibrated
```

## 📚 References

1. **Domain Adaptation**: Ganin et al. "Domain-Adversarial Training of Neural Networks"
2. **Stain Normalization**: Macenko et al. "A method for normalizing histology slides"
3. **Contrastive Learning**: Chen et al. "A Simple Framework for Contrastive Learning"
4. **Test-time Augmentation**: Krizhevsky et al. "ImageNet Classification with Deep CNNs"

## 🤝 Contributing

To improve the cross-dataset performance further:

1. **Add new domain adaptation techniques**
2. **Implement advanced stain normalization methods**
3. **Experiment with different backbone architectures**
4. **Optimize hyperparameters for specific datasets**

## 📄 License

This advanced training module is part of the Breast Cancer Detection Thesis project.

---

**🎯 Goal**: Achieve 90%+ cross-dataset performance for robust breast cancer detection across different imaging conditions and datasets.