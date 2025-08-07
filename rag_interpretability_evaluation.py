#!/usr/bin/env python3
"""
RAG-based interpretability evaluation with retrieval index and clinical explanations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import logging
import os
from PIL import Image
import pandas as pd

from src.rag_explainer import RAGExplainer, explain_prediction, GradCAM
from src.efficientnet import EfficientNetB0Classifier
from src.data_utils import create_metadata, create_train_val_test_split, create_class_mappings
from torchvision import transforms

class EnhancedRAGExplainer(RAGExplainer):
    """Enhanced RAG explainer with comprehensive medical knowledge base"""
    
    def _create_knowledge_base(self):
        """Create comprehensive medical knowledge base"""
        knowledge = {
            'adenosis': [
                'Adenosis is a benign breast condition characterized by enlarged lobules with increased glandular tissue',
                'Histologically shows preserved lobular architecture with increased acinar structures',
                'May present as palpable mass or mammographic density, often bilateral',
                'Sclerosing adenosis can mimic invasive carcinoma but maintains basement membrane integrity',
                'Associated with increased breast density and may cause diagnostic challenges on imaging',
                'Treatment is typically observation unless symptomatic or causing diagnostic uncertainty'
            ],
            'fibroadenoma': [
                'Fibroadenoma is the most common benign breast tumor, especially in young women aged 15-35',
                'Characterized by well-circumscribed borders and biphasic pattern of epithelial and stromal components',
                'Typically presents as mobile, painless, rubbery mass on physical examination',
                'Mammographically appears as well-defined, oval or round mass with possible coarse calcifications',
                'Ultrasound shows hypoechoic mass with well-defined margins and possible posterior acoustic enhancement',
                'Management includes observation for small lesions or surgical excision for large or growing masses'
            ],
            'phyllodes_tumor': [
                'Phyllodes tumor is a rare fibroepithelial tumor with leaf-like architecture',
                'Can be benign, borderline, or malignant based on stromal cellularity and mitotic activity',
                'Typically presents as rapidly growing, large, painless breast mass',
                'Histologically shows stromal hypercellularity with leaf-like projections into cystic spaces',
                'Requires wide local excision due to potential for local recurrence',
                'Malignant phyllodes can metastasize hematogenously, typically to lungs and bones'
            ],
            'tubular_adenoma': [
                'Tubular adenoma is a rare benign breast tumor composed of closely packed tubular structures',
                'Typically occurs in young women and may be associated with pregnancy or lactation',
                'Histologically shows uniform tubular structures lined by epithelial and myoepithelial cells',
                'May be difficult to distinguish from tubular carcinoma, requiring careful histologic evaluation',
                'Usually presents as well-circumscribed, mobile mass on clinical examination',
                'Treatment is typically surgical excision for definitive diagnosis and symptom relief'
            ],
            'ductal_carcinoma': [
                'Invasive ductal carcinoma (IDC) is the most common type of breast cancer, accounting for 70-80% of cases',
                'Arises from ductal epithelium and invades surrounding breast tissue and stroma',
                'Histologically shows irregular nests and cords of malignant cells with desmoplastic stroma',
                'Often presents as irregular, spiculated mass on mammography with possible microcalcifications',
                'May show skin retraction, nipple inversion, or peau d\'orange appearance in advanced cases',
                'Requires multidisciplinary treatment including surgery, chemotherapy, and/or radiation therapy'
            ],
            'lobular_carcinoma': [
                'Invasive lobular carcinoma (ILC) accounts for 10-15% of invasive breast cancers',
                'Characterized by single-file growth pattern of malignant cells through breast tissue',
                'Often difficult to detect clinically and radiographically due to growth pattern',
                'May present as subtle architectural distortion rather than discrete mass on imaging',
                'Higher propensity for multifocal, multicentric, and bilateral disease compared to IDC',
                'Treatment approach similar to IDC but may require more extensive surgical planning'
            ],
            'mucinous_carcinoma': [
                'Mucinous carcinoma is a special type of invasive breast cancer with abundant extracellular mucin',
                'Accounts for 1-4% of invasive breast cancers, typically occurs in older women',
                'Histologically shows clusters of malignant cells floating in pools of mucin',
                'Generally has better prognosis than invasive ductal carcinoma of similar size',
                'May present as well-circumscribed mass that can be mistaken for benign lesion',
                'Treatment follows standard breast cancer protocols but often has favorable outcomes'
            ],
            'papillary_carcinoma': [
                'Papillary carcinoma is a rare form of breast cancer with papillary architecture',
                'Can be invasive or in-situ, with invasive form having better prognosis than typical IDC',
                'Histologically shows papillary fronds with fibrovascular cores lined by malignant cells',
                'May present with bloody nipple discharge, especially in central location',
                'Often occurs in older women and may be associated with DCIS',
                'Treatment includes surgery with consideration for sentinel lymph node biopsy'
            ]
        }
        
        # Add diagnostic features and clinical correlations
        diagnostic_features = {
            'benign_features': [
                'Well-defined borders and smooth contours suggest benign pathology',
                'Preserved cellular architecture indicates non-invasive process',
                'Absence of nuclear pleomorphism and mitotic activity favors benign diagnosis',
                'Intact basement membrane is key feature distinguishing benign from malignant lesions'
            ],
            'malignant_features': [
                'Irregular borders and spiculated margins are suspicious for malignancy',
                'Nuclear pleomorphism and increased mitotic activity indicate malignant transformation',
                'Loss of normal tissue architecture suggests invasive process',
                'Desmoplastic stromal reaction is characteristic of invasive carcinoma'
            ],
            'imaging_correlation': [
                'Mammographic findings should be correlated with histologic appearance',
                'Ultrasound characteristics help differentiate solid from cystic lesions',
                'MRI enhancement patterns provide additional diagnostic information',
                'Histologic diagnosis remains gold standard for definitive classification'
            ]
        }
        
        # Flatten all knowledge
        all_facts = []
        fact_labels = []
        
        for label, facts in knowledge.items():
            all_facts.extend(facts)
            fact_labels.extend([label] * len(facts))
        
        for category, facts in diagnostic_features.items():
            all_facts.extend(facts)
            fact_labels.extend([category] * len(facts))
        
        return {'facts': all_facts, 'labels': fact_labels}
    
    def generate_clinical_report(self, prediction, confidence, image_features=None, patient_info=None):
        """Generate comprehensive clinical report"""
        class_names = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
                      'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
        
        predicted_class = class_names[prediction]
        is_malignant = prediction >= 4  # Classes 4-7 are malignant
        
        # Retrieve relevant facts
        query = f"breast {predicted_class} diagnosis histology clinical features"
        relevant_facts = self.retrieve_relevant_facts(query, k=5)
        
        # Generate structured report
        report = {
            'diagnosis': {
                'primary': predicted_class.replace('_', ' ').title(),
                'confidence': f"{confidence:.1%}",
                'category': 'Malignant' if is_malignant else 'Benign'
            },
            'histologic_features': [],
            'clinical_significance': [],
            'recommendations': [],
            'differential_diagnosis': []
        }
        
        # Extract features based on prediction
        for fact in relevant_facts:
            if 'histolog' in fact['fact'].lower():
                report['histologic_features'].append(fact['fact'])
            elif 'treatment' in fact['fact'].lower() or 'management' in fact['fact'].lower():
                report['recommendations'].append(fact['fact'])
            else:
                report['clinical_significance'].append(fact['fact'])
        
        # Add recommendations based on diagnosis
        if is_malignant:
            report['recommendations'].extend([
                'Immediate oncology referral for staging and treatment planning',
                'Consider sentinel lymph node biopsy if clinically indicated',
                'Multidisciplinary team discussion for optimal treatment strategy',
                'Patient counseling regarding diagnosis and treatment options'
            ])
        else:
            report['recommendations'].extend([
                'Clinical correlation with imaging findings recommended',
                'Consider follow-up imaging in 6-12 months if clinically indicated',
                'Patient reassurance regarding benign nature of lesion',
                'Routine breast cancer screening as per guidelines'
            ])
        
        return report

def create_retrieval_index_demo():
    """Demonstrate retrieval index functionality"""
    logger = logging.getLogger(__name__)
    logger.info("Creating retrieval index demonstration...")
    
    rag_explainer = EnhancedRAGExplainer()
    
    # Test queries
    test_queries = [
        "invasive ductal carcinoma features",
        "benign breast lesion characteristics", 
        "fibroadenoma diagnosis imaging",
        "malignant breast cancer treatment"
    ]
    
    results = {}
    for query in test_queries:
        relevant_facts = rag_explainer.retrieve_relevant_facts(query, k=3)
        results[query] = relevant_facts
        
        logger.info(f"\nQuery: {query}")
        logger.info("Retrieved facts:")
        for i, fact in enumerate(relevant_facts, 1):
            logger.info(f"  {i}. {fact['fact']} (Label: {fact['label']})")
    
    return results

def evaluate_explanations_on_test_set():
    """Evaluate RAG explanations on test set"""
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = "models/efficientnet_b0_best.pth"
    if not os.path.exists(model_path):
        logger.warning("Model not found. Please train a model first.")
        return None
    
    model = EfficientNetB0Classifier(num_classes=8, pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test data
    breakhis_root = "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
    metadata = create_metadata(breakhis_root)
    
    train_df, val_df, test_df = create_train_val_test_split(
        metadata, test_size=0.15, val_size=0.15, random_state=42
    )
    
    class_to_idx, idx_to_class, _ = create_class_mappings(train_df)
    
    # Initialize RAG explainer
    rag_explainer = EnhancedRAGExplainer()
    
    # Transform for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluate on sample of test images
    sample_size = min(10, len(test_df))
    sample_df = test_df.sample(n=sample_size, random_state=42)
    
    explanations = []
    
    for idx, row in sample_df.iterrows():
        try:
            # Load and preprocess image
            image_path = row['path']
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image)
            
            # Get prediction
            with torch.no_grad():
                output = model(image_tensor.unsqueeze(0).to(device))
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            # Generate explanation
            explanation = explain_prediction(model, image_tensor, rag_explainer, device)
            
            # Generate clinical report
            clinical_report = rag_explainer.generate_clinical_report(
                prediction.item(), confidence.item()
            )
            
            explanations.append({
                'image_path': image_path,
                'true_class': row['subclass'],
                'predicted_class': idx_to_class[prediction.item()],
                'confidence': confidence.item(),
                'explanation': explanation,
                'clinical_report': clinical_report
            })
            
            logger.info(f"Processed {len(explanations)}/{sample_size} images")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            continue
    
    return explanations

def visualize_explanation_example(explanation_data, save_path="explanation_example.png"):
    """Visualize explanation example with GradCAM and text"""
    if not explanation_data:
        return
    
    example = explanation_data[0]  # Take first example
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    image = Image.open(example['image_path'])
    ax1.imshow(image)
    ax1.set_title(f"Original Image\nTrue: {example['true_class']}")
    ax1.axis('off')
    
    # GradCAM visualization
    gradcam = example['explanation']['visual_explanation']
    ax2.imshow(image, alpha=0.7)
    ax2.imshow(gradcam, cmap='jet', alpha=0.5)
    ax2.set_title(f"GradCAM Visualization\nPredicted: {example['predicted_class']}")
    ax2.axis('off')
    
    # Prediction confidence
    confidence = example['confidence']
    ax3.bar(['Confidence'], [confidence], color='skyblue')
    ax3.set_title('Prediction Confidence')
    ax3.set_ylim(0, 1)
    ax3.text(0, confidence + 0.05, f'{confidence:.3f}', ha='center', fontweight='bold')
    
    # Clinical report summary
    clinical_report = example['clinical_report']
    report_text = f"Diagnosis: {clinical_report['diagnosis']['primary']}\n"
    report_text += f"Category: {clinical_report['diagnosis']['category']}\n"
    report_text += f"Confidence: {clinical_report['diagnosis']['confidence']}\n\n"
    
    if clinical_report['histologic_features']:
        report_text += "Key Features:\n"
        for feature in clinical_report['histologic_features'][:2]:
            report_text += f"• {feature[:60]}...\n"
    
    ax4.text(0.05, 0.95, report_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.set_title('Clinical Report Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_clinician_style_reports(explanations, save_path="clinical_reports.json"):
    """Generate clinician-style interpretation reports"""
    logger = logging.getLogger(__name__)
    logger.info("Generating clinician-style reports...")
    
    clinical_reports = []
    
    for exp in explanations:
        report = {
            'case_id': os.path.basename(exp['image_path']),
            'clinical_impression': exp['clinical_report']['diagnosis'],
            'histologic_findings': exp['clinical_report']['histologic_features'],
            'clinical_significance': exp['clinical_report']['clinical_significance'],
            'recommendations': exp['clinical_report']['recommendations'],
            'ai_confidence': f"{exp['confidence']:.1%}",
            'concordance': exp['true_class'] == exp['predicted_class']
        }
        clinical_reports.append(report)
    
    # Save reports
    with open(save_path, 'w') as f:
        json.dump(clinical_reports, f, indent=2)
    
    logger.info(f"Clinical reports saved to: {save_path}")
    
    # Generate summary statistics
    total_cases = len(clinical_reports)
    concordant_cases = sum(1 for r in clinical_reports if r['concordance'])
    accuracy = concordant_cases / total_cases if total_cases > 0 else 0
    
    logger.info(f"Generated {total_cases} clinical reports")
    logger.info(f"AI-Pathologist concordance: {accuracy:.1%}")
    
    return clinical_reports

def plot_rag_performance_metrics(explanations, save_path="rag_performance.png"):
    """Plot RAG system performance metrics"""
    if not explanations:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confidence distribution
    confidences = [exp['confidence'] for exp in explanations]
    ax1.hist(confidences, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax1.legend()
    
    # Accuracy by class
    true_classes = [exp['true_class'] for exp in explanations]
    pred_classes = [exp['predicted_class'] for exp in explanations]
    
    unique_classes = list(set(true_classes))
    class_accuracies = []
    
    for cls in unique_classes:
        cls_indices = [i for i, tc in enumerate(true_classes) if tc == cls]
        if cls_indices:
            cls_correct = sum(1 for i in cls_indices if true_classes[i] == pred_classes[i])
            cls_accuracy = cls_correct / len(cls_indices)
            class_accuracies.append(cls_accuracy)
        else:
            class_accuracies.append(0)
    
    ax2.bar(range(len(unique_classes)), class_accuracies, color='lightcoral')
    ax2.set_title('Per-Class Accuracy')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(range(len(unique_classes)))
    ax2.set_xticklabels(unique_classes, rotation=45)
    
    # Explanation length distribution
    explanation_lengths = []
    for exp in explanations:
        text_exp = exp['explanation']['textual_explanation']['explanation']
        explanation_lengths.append(len(text_exp.split()))
    
    ax3.hist(explanation_lengths, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_title('Explanation Length Distribution')
    ax3.set_xlabel('Number of Words')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(explanation_lengths), color='red', linestyle='--',
                label=f'Mean: {np.mean(explanation_lengths):.1f}')
    ax3.legend()
    
    # Confidence vs Accuracy
    correct_predictions = [tc == pc for tc, pc in zip(true_classes, pred_classes)]
    
    # Bin confidences and calculate accuracy for each bin
    conf_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(conf_bins)-1):
        bin_mask = [(conf_bins[i] <= c < conf_bins[i+1]) for c in confidences]
        if any(bin_mask):
            bin_acc = np.mean([correct_predictions[j] for j, mask in enumerate(bin_mask) if mask])
            bin_accuracies.append(bin_acc)
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
    
    ax4.plot(bin_centers, bin_accuracies, 'o-', color='purple', linewidth=2, markersize=8)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax4.set_title('Confidence vs Accuracy (Calibration)')
    ax4.set_xlabel('Confidence')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RAG interpretability evaluation...")
    
    # 1. Demonstrate retrieval index
    logger.info("=== Retrieval Index Demonstration ===")
    retrieval_results = create_retrieval_index_demo()
    
    # 2. Evaluate explanations on test set
    logger.info("=== Evaluating Explanations on Test Set ===")
    explanations = evaluate_explanations_on_test_set()
    
    if explanations:
        # 3. Visualize explanation example
        logger.info("=== Creating Explanation Visualization ===")
        visualize_explanation_example(explanations)
        
        # 4. Generate clinical reports
        logger.info("=== Generating Clinical Reports ===")
        clinical_reports = generate_clinician_style_reports(explanations)
        
        # 5. Plot performance metrics
        logger.info("=== Creating Performance Metrics ===")
        plot_rag_performance_metrics(explanations)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("RAG INTERPRETABILITY EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Processed {len(explanations)} test cases")
        logger.info(f"Generated {len(clinical_reports)} clinical reports")
        
        if explanations:
            avg_confidence = np.mean([exp['confidence'] for exp in explanations])
            accuracy = np.mean([exp['true_class'] == exp['predicted_class'] for exp in explanations])
            logger.info(f"Average confidence: {avg_confidence:.3f}")
            logger.info(f"Accuracy: {accuracy:.3f}")
    
    logger.info("RAG interpretability evaluation completed!")

if __name__ == "__main__":
    main()