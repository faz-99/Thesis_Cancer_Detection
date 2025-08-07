"""
Comprehensive evaluation framework with statistical rigor for thesis
Addresses: baseline clarity, statistical significance, proper validation
"""

import numpy as np
import torch
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class StatisticalEvaluator:
    """Handles statistical significance testing and confidence intervals"""
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
    
    def bootstrap_metric(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        metric_func, n_bootstrap: Optional[int] = None) -> Dict:
        """Bootstrap confidence intervals for any metric"""
        n_bootstrap = n_bootstrap or self.n_bootstrap
        n_samples = len(y_true)
        
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = resample(range(n_samples), n_samples=n_samples)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        ci_lower = np.percentile(bootstrap_scores, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - self.alpha/2) * 100)
        
        return {
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower
        }
    
    def mcnemar_test(self, y_true: np.ndarray, y_pred1: np.ndarray, 
                    y_pred2: np.ndarray) -> Dict:
        """McNemar's test for comparing two models"""
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # Contingency table
        both_correct = np.sum(correct1 & correct2)
        only_1_correct = np.sum(correct1 & ~correct2)
        only_2_correct = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # McNemar's test statistic
        if only_1_correct + only_2_correct == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            statistic = (abs(only_1_correct - only_2_correct) - 1)**2 / (only_1_correct + only_2_correct)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'contingency': {
                'both_correct': both_correct,
                'only_model1_correct': only_1_correct,
                'only_model2_correct': only_2_correct,
                'both_wrong': both_wrong
            }
        }

class BaselineManager:
    """Manages baseline comparisons with clear incremental vs independent gains"""
    
    def __init__(self):
        self.baselines = {}
        self.experiments = {}
        self.evaluation_chain = []
    
    def register_baseline(self, name: str, model, dataset: str, 
                         description: str, sample_size: int):
        """Register a baseline model"""
        self.baselines[name] = {
            'model': model,
            'dataset': dataset,
            'description': description,
            'sample_size': sample_size,
            'timestamp': datetime.now(),
            'results': None
        }
    
    def register_experiment(self, name: str, model, base_model: str,
                          modifications: List[str], dataset: str,
                          description: str, sample_size: int):
        """Register an experimental model with clear baseline reference"""
        self.experiments[name] = {
            'model': model,
            'base_model': base_model,
            'modifications': modifications,
            'dataset': dataset,
            'description': description,
            'sample_size': sample_size,
            'timestamp': datetime.now(),
            'results': None
        }
    
    def evaluate_model(self, name: str, test_loader, device: str = 'cuda'):
        """Evaluate a registered model"""
        if name in self.baselines:
            model_info = self.baselines[name]
        elif name in self.experiments:
            model_info = self.experiments[name]
        else:
            raise ValueError(f"Model {name} not registered")
        
        model = model_info['model']
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        results = self._compute_metrics(np.array(all_labels), 
                                      np.array(all_preds), 
                                      np.array(all_probs))
        
        model_info['results'] = results
        return results
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_probs: np.ndarray) -> Dict:
        """Compute comprehensive metrics with confidence intervals"""
        evaluator = StatisticalEvaluator()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Bootstrap confidence intervals
        acc_stats = evaluator.bootstrap_metric(y_true, y_pred, accuracy_score)
        f1_stats = evaluator.bootstrap_metric(y_true, y_pred, 
                                            lambda yt, yp: precision_recall_fscore_support(yt, yp, average='weighted')[2])
        
        # AUC for multiclass (one-vs-rest)
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
            auc_stats = evaluator.bootstrap_metric(y_true, y_probs,
                                                 lambda yt, yp: roc_auc_score(yt, yp, multi_class='ovr', average='weighted'))
        except:
            auc = None
            auc_stats = None
        
        return {
            'accuracy': {
                'value': accuracy,
                'stats': acc_stats
            },
            'f1_score': {
                'value': f1,
                'stats': f1_stats
            },
            'auc': {
                'value': auc,
                'stats': auc_stats
            } if auc is not None else None,
            'precision': precision,
            'recall': recall,
            'sample_size': len(y_true)
        }
    
    def compare_models(self, model1_name: str, model2_name: str, 
                      test_loader, device: str = 'cuda') -> Dict:
        """Statistical comparison between two models"""
        # Get predictions for both models
        results1 = self.evaluate_model(model1_name, test_loader, device)
        results2 = self.evaluate_model(model2_name, test_loader, device)
        
        # Get raw predictions for statistical tests
        model1 = (self.baselines.get(model1_name) or self.experiments.get(model1_name))['model']
        model2 = (self.baselines.get(model2_name) or self.experiments.get(model2_name))['model']
        
        all_labels = []
        preds1 = []
        preds2 = []
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                pred1 = model1(data).argmax(dim=1)
                pred2 = model2(data).argmax(dim=1)
                
                all_labels.extend(target.cpu().numpy())
                preds1.extend(pred1.cpu().numpy())
                preds2.extend(pred2.cpu().numpy())
        
        # Statistical comparison
        evaluator = StatisticalEvaluator()
        mcnemar_result = evaluator.mcnemar_test(np.array(all_labels), 
                                              np.array(preds1), 
                                              np.array(preds2))
        
        # Effect size (difference in accuracy)
        acc_diff = results2['accuracy']['value'] - results1['accuracy']['value']
        
        return {
            'model1': model1_name,
            'model2': model2_name,
            'model1_results': results1,
            'model2_results': results2,
            'accuracy_difference': acc_diff,
            'mcnemar_test': mcnemar_result,
            'sample_size': len(all_labels)
        }
    
    def generate_comparison_report(self, comparisons: List[Tuple[str, str]], 
                                 test_loader, device: str = 'cuda') -> str:
        """Generate comprehensive comparison report"""
        report = []
        report.append("# Statistical Model Comparison Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for model1, model2 in comparisons:
            comparison = self.compare_models(model1, model2, test_loader, device)
            
            report.append(f"## {model1} vs {model2}")
            report.append("")
            
            # Sample size
            report.append(f"**Sample Size**: {comparison['sample_size']}")
            report.append("")
            
            # Results table
            report.append("| Metric | Model 1 | Model 2 | Difference | CI Lower | CI Upper |")
            report.append("|--------|---------|---------|------------|----------|----------|")
            
            acc1 = comparison['model1_results']['accuracy']
            acc2 = comparison['model2_results']['accuracy']
            acc_diff = comparison['accuracy_difference']
            
            report.append(f"| Accuracy | {acc1['value']:.4f} ± {acc1['stats']['std']:.4f} | "
                         f"{acc2['value']:.4f} ± {acc2['stats']['std']:.4f} | "
                         f"{acc_diff:.4f} | {acc1['stats']['ci_lower']:.4f} | {acc1['stats']['ci_upper']:.4f} |")
            
            # Statistical significance
            mcnemar = comparison['mcnemar_test']
            report.append("")
            report.append(f"**McNemar's Test**: χ² = {mcnemar['statistic']:.4f}, p = {mcnemar['p_value']:.4f}")
            report.append(f"**Statistically Significant**: {'Yes' if mcnemar['significant'] else 'No'}")
            report.append("")
        
        return "\n".join(report)

class ValidationSampleTracker:
    """Tracks sample sizes for all validation components"""
    
    def __init__(self):
        self.validation_samples = {}
    
    def register_pathologist_validation(self, study_name: str, 
                                      pathologist_count: int,
                                      cases_per_pathologist: int,
                                      total_cases: int,
                                      agreement_metric: str = "kappa"):
        """Register pathologist validation study"""
        self.validation_samples[f"pathologist_{study_name}"] = {
            'type': 'pathologist_validation',
            'pathologist_count': pathologist_count,
            'cases_per_pathologist': cases_per_pathologist,
            'total_cases': total_cases,
            'total_annotations': pathologist_count * cases_per_pathologist,
            'agreement_metric': agreement_metric,
            'timestamp': datetime.now()
        }
    
    def register_rag_validation(self, study_name: str, 
                               query_count: int,
                               documents_retrieved: int,
                               expert_evaluations: int,
                               relevance_threshold: float = 0.8):
        """Register RAG system validation"""
        self.validation_samples[f"rag_{study_name}"] = {
            'type': 'rag_validation',
            'query_count': query_count,
            'documents_retrieved': documents_retrieved,
            'expert_evaluations': expert_evaluations,
            'relevance_threshold': relevance_threshold,
            'timestamp': datetime.now()
        }
    
    def register_clinical_validation(self, study_name: str,
                                   patient_count: int,
                                   image_count: int,
                                   ground_truth_source: str,
                                   validation_period: str):
        """Register clinical validation study"""
        self.validation_samples[f"clinical_{study_name}"] = {
            'type': 'clinical_validation',
            'patient_count': patient_count,
            'image_count': image_count,
            'ground_truth_source': ground_truth_source,
            'validation_period': validation_period,
            'timestamp': datetime.now()
        }
    
    def get_validation_summary(self) -> str:
        """Generate validation sample size summary"""
        report = []
        report.append("# Validation Sample Size Summary")
        report.append("")
        
        for study_name, details in self.validation_samples.items():
            report.append(f"## {study_name}")
            report.append(f"**Type**: {details['type']}")
            
            if details['type'] == 'pathologist_validation':
                report.append(f"**Pathologists**: {details['pathologist_count']}")
                report.append(f"**Cases per Pathologist**: {details['cases_per_pathologist']}")
                report.append(f"**Total Cases**: {details['total_cases']}")
                report.append(f"**Total Annotations**: {details['total_annotations']}")
                
            elif details['type'] == 'rag_validation':
                report.append(f"**Queries Tested**: {details['query_count']}")
                report.append(f"**Documents Retrieved**: {details['documents_retrieved']}")
                report.append(f"**Expert Evaluations**: {details['expert_evaluations']}")
                
            elif details['type'] == 'clinical_validation':
                report.append(f"**Patients**: {details['patient_count']}")
                report.append(f"**Images**: {details['image_count']}")
                report.append(f"**Ground Truth**: {details['ground_truth_source']}")
            
            report.append("")
        
        return "\n".join(report)

# Example usage functions
def setup_thesis_evaluation():
    """Setup complete evaluation framework for thesis"""
    baseline_manager = BaselineManager()
    sample_tracker = ValidationSampleTracker()
    
    return baseline_manager, sample_tracker

def example_baseline_setup(baseline_manager: BaselineManager):
    """Example of how to set up baselines clearly"""
    
    # Base models (independent)
    baseline_manager.register_baseline(
        name="efficientnet_baseline",
        model=None,  # Your EfficientNet model
        dataset="breakhis",
        description="EfficientNetB0 with ImageNet pretraining, no augmentation",
        sample_size=1200
    )
    
    # Incremental improvements
    baseline_manager.register_experiment(
        name="efficientnet_gan_aug",
        model=None,  # Your model with GAN augmentation
        base_model="efficientnet_baseline",
        modifications=["GAN-based data augmentation"],
        dataset="breakhis",
        description="EfficientNetB0 + GAN augmentation (applied to baseline)",
        sample_size=1200
    )
    
    baseline_manager.register_experiment(
        name="supconvit_baseline",
        model=None,  # Your SupConViT model
        base_model="efficientnet_baseline",
        modifications=["SupConViT architecture"],
        dataset="breakhis",
        description="SupConViT without augmentation (independent from GAN)",
        sample_size=1200
    )
    
    baseline_manager.register_experiment(
        name="supconvit_gan_aug",
        model=None,  # Your SupConViT + GAN model
        base_model="supconvit_baseline",
        modifications=["GAN-based data augmentation"],
        dataset="breakhis",
        description="SupConViT + GAN augmentation (incremental to SupConViT)",
        sample_size=1200
    )

def example_validation_setup(sample_tracker: ValidationSampleTracker):
    """Example of proper validation sample tracking"""
    
    # Pathologist validation
    sample_tracker.register_pathologist_validation(
        study_name="interpretability_study",
        pathologist_count=3,
        cases_per_pathologist=100,
        total_cases=100,  # Same cases evaluated by all pathologists
        agreement_metric="fleiss_kappa"
    )
    
    # RAG validation
    sample_tracker.register_rag_validation(
        study_name="explanation_quality",
        query_count=500,
        documents_retrieved=2500,  # 5 per query
        expert_evaluations=500,
        relevance_threshold=0.8
    )
    
    # Clinical validation
    sample_tracker.register_clinical_validation(
        study_name="prospective_validation",
        patient_count=150,
        image_count=600,
        ground_truth_source="Histopathological diagnosis",
        validation_period="6 months"
    )