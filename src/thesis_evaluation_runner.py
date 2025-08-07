"""
Main runner for thesis evaluation with statistical rigor
Demonstrates proper baseline management and validation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from evaluation_framework import BaselineManager, ValidationSampleTracker, StatisticalEvaluator
from efficientnet import EfficientNetB0Classifier
import json
from datetime import datetime

class ThesisEvaluationRunner:
    """Main class to run comprehensive thesis evaluation"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.baseline_manager = BaselineManager()
        self.sample_tracker = ValidationSampleTracker()
        self.results = {}
    
    def setup_models(self):
        """Setup all models with clear baseline relationships"""
        
        # 1. Independent Baselines
        efficientnet_base = EfficientNetB0Classifier(num_classes=8)
        self.baseline_manager.register_baseline(
            name="efficientnet_baseline",
            model=efficientnet_base,
            dataset="breakhis",
            description="EfficientNetB0, ImageNet pretrained, standard augmentation only",
            sample_size=1200
        )
        
        # 2. Incremental Improvements (GAN augmentation)
        efficientnet_gan = EfficientNetB0Classifier(num_classes=8)
        self.baseline_manager.register_experiment(
            name="efficientnet_gan",
            model=efficientnet_gan,
            base_model="efficientnet_baseline",
            modifications=["GAN-based data augmentation"],
            dataset="breakhis_augmented",
            description="EfficientNetB0 + GAN augmentation (incremental to baseline)",
            sample_size=1200
        )
        
        # 3. Independent Architecture Change
        # Note: SupConViT would be implemented separately
        supconvit_base = EfficientNetB0Classifier(num_classes=8)  # Placeholder
        self.baseline_manager.register_experiment(
            name="supconvit_baseline",
            model=supconvit_base,
            base_model="efficientnet_baseline",
            modifications=["SupConViT architecture"],
            dataset="breakhis",
            description="SupConViT without augmentation (independent architecture change)",
            sample_size=1200
        )
        
        # 4. Combined Improvements
        supconvit_gan = EfficientNetB0Classifier(num_classes=8)  # Placeholder
        self.baseline_manager.register_experiment(
            name="supconvit_gan",
            model=supconvit_gan,
            base_model="supconvit_baseline",
            modifications=["GAN-based data augmentation"],
            dataset="breakhis_augmented",
            description="SupConViT + GAN augmentation (incremental to SupConViT baseline)",
            sample_size=1200
        )
        
        # 5. Multimodal Extension
        multimodal_model = EfficientNetB0Classifier(num_classes=8)  # Placeholder
        self.baseline_manager.register_experiment(
            name="multimodal_supconvit",
            model=multimodal_model,
            base_model="supconvit_gan",
            modifications=["Multimodal learning (clinical + histology)"],
            dataset="breakhis_bach_combined",
            description="Multimodal SupConViT (incremental to SupConViT+GAN)",
            sample_size=1500
        )
    
    def setup_validation_studies(self):
        """Setup validation studies with proper sample sizes"""
        
        # Pathologist validation for interpretability
        self.sample_tracker.register_pathologist_validation(
            study_name="rag_interpretability",
            pathologist_count=3,
            cases_per_pathologist=100,
            total_cases=100,
            agreement_metric="fleiss_kappa"
        )
        
        # RAG system validation
        self.sample_tracker.register_rag_validation(
            study_name="explanation_retrieval",
            query_count=500,
            documents_retrieved=2500,
            expert_evaluations=500,
            relevance_threshold=0.8
        )
        
        # Clinical validation
        self.sample_tracker.register_clinical_validation(
            study_name="prospective_validation",
            patient_count=200,
            image_count=800,
            ground_truth_source="Consensus of 2 pathologists",
            validation_period="6 months"
        )
        
        # Magnification robustness validation
        self.sample_tracker.register_clinical_validation(
            study_name="magnification_robustness",
            patient_count=100,
            image_count=400,
            ground_truth_source="Same cases at different magnifications",
            validation_period="Cross-sectional"
        )
    
    def run_statistical_comparisons(self, test_loader: DataLoader):
        """Run all statistical comparisons with proper significance testing"""
        
        comparisons = [
            # Incremental improvements
            ("efficientnet_baseline", "efficientnet_gan"),
            ("supconvit_baseline", "supconvit_gan"),
            
            # Architecture comparisons
            ("efficientnet_baseline", "supconvit_baseline"),
            ("efficientnet_gan", "supconvit_gan"),
            
            # Final comparison
            ("efficientnet_baseline", "multimodal_supconvit"),
            ("supconvit_gan", "multimodal_supconvit")
        ]
        
        results = {}
        for model1, model2 in comparisons:
            print(f"Comparing {model1} vs {model2}...")
            comparison = self.baseline_manager.compare_models(
                model1, model2, test_loader, self.device
            )
            results[f"{model1}_vs_{model2}"] = comparison
        
        self.results['statistical_comparisons'] = results
        return results
    
    def generate_thesis_results_table(self):
        """Generate publication-ready results table"""
        
        table_data = []
        
        # Define the evaluation order (baseline -> incremental improvements)
        model_order = [
            "efficientnet_baseline",
            "efficientnet_gan", 
            "supconvit_baseline",
            "supconvit_gan",
            "multimodal_supconvit"
        ]
        
        for model_name in model_order:
            if model_name in self.baseline_manager.baselines:
                model_info = self.baseline_manager.baselines[model_name]
            else:
                model_info = self.baseline_manager.experiments[model_name]
            
            results = model_info.get('results')
            if results:
                acc = results['accuracy']
                f1 = results['f1_score']
                
                table_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Base Model': model_info.get('base_model', 'N/A'),
                    'Modifications': ', '.join(model_info.get('modifications', ['None'])),
                    'Sample Size': results['sample_size'],
                    'Accuracy': f"{acc['value']:.3f} ± {acc['stats']['std']:.3f}",
                    'Accuracy CI': f"[{acc['stats']['ci_lower']:.3f}, {acc['stats']['ci_upper']:.3f}]",
                    'F1 Score': f"{f1['value']:.3f} ± {f1['stats']['std']:.3f}",
                    'F1 CI': f"[{f1['stats']['ci_lower']:.3f}, {f1['stats']['ci_upper']:.3f}]"
                })
        
        return table_data
    
    def generate_significance_table(self):
        """Generate statistical significance comparison table"""
        
        if 'statistical_comparisons' not in self.results:
            return []
        
        sig_data = []
        for comparison_name, comparison in self.results['statistical_comparisons'].items():
            mcnemar = comparison['mcnemar_test']
            
            sig_data.append({
                'Comparison': comparison_name.replace('_vs_', ' vs ').replace('_', ' ').title(),
                'Sample Size': comparison['sample_size'],
                'Accuracy Difference': f"{comparison['accuracy_difference']:.4f}",
                'McNemar χ²': f"{mcnemar['statistic']:.4f}",
                'p-value': f"{mcnemar['p_value']:.4f}",
                'Significant (α=0.05)': 'Yes' if mcnemar['significant'] else 'No',
                'Effect Size': self._interpret_effect_size(comparison['accuracy_difference'])
            })
        
        return sig_data
    
    def _interpret_effect_size(self, diff: float) -> str:
        """Interpret effect size for accuracy difference"""
        abs_diff = abs(diff)
        if abs_diff < 0.01:
            return "Negligible"
        elif abs_diff < 0.03:
            return "Small"
        elif abs_diff < 0.05:
            return "Medium"
        else:
            return "Large"
    
    def save_comprehensive_report(self, filepath: str):
        """Save comprehensive evaluation report"""
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'evaluation_framework_version': '1.0',
                'statistical_alpha': 0.05,
                'bootstrap_iterations': 1000
            },
            'baseline_clarity': {
                'baselines': {name: {
                    'description': info['description'],
                    'dataset': info['dataset'],
                    'sample_size': info['sample_size']
                } for name, info in self.baseline_manager.baselines.items()},
                'experiments': {name: {
                    'base_model': info['base_model'],
                    'modifications': info['modifications'],
                    'description': info['description'],
                    'dataset': info['dataset'],
                    'sample_size': info['sample_size']
                } for name, info in self.baseline_manager.experiments.items()}
            },
            'results_table': self.generate_thesis_results_table(),
            'significance_testing': self.generate_significance_table(),
            'validation_samples': {name: details for name, details in self.sample_tracker.validation_samples.items()},
            'statistical_comparisons': self.results.get('statistical_comparisons', {})
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive evaluation report saved to {filepath}")

def main():
    """Main execution function"""
    
    # Initialize evaluation runner
    runner = ThesisEvaluationRunner()
    
    # Setup models and validation studies
    runner.setup_models()
    runner.setup_validation_studies()
    
    print("Thesis Evaluation Framework Setup Complete")
    print("\nBaseline Clarity:")
    print("- Independent baselines clearly defined")
    print("- Incremental improvements tracked")
    print("- Architecture changes separated from augmentation effects")
    
    print("\nStatistical Rigor:")
    print("- Bootstrap confidence intervals for all metrics")
    print("- McNemar's test for model comparisons")
    print("- Effect size interpretation")
    
    print("\nValidation Sample Sizes:")
    print("- Pathologist validation: 3 pathologists, 100 cases each")
    print("- RAG validation: 500 queries, 500 expert evaluations")
    print("- Clinical validation: 200 patients, 800 images")
    
    # Generate validation summary
    validation_summary = runner.sample_tracker.get_validation_summary()
    print("\n" + validation_summary)
    
    # Note: Actual model evaluation would require trained models and test data
    # runner.run_statistical_comparisons(test_loader)
    # runner.save_comprehensive_report('thesis_evaluation_report.json')

if __name__ == "__main__":
    main()