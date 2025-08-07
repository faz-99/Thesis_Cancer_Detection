#!/usr/bin/env python3
"""
Master evaluation script to run all thesis components
Generates comprehensive results for all missing evaluation components
"""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"complete_evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__), log_filename

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {description}...")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully in {duration:.1f}s")
            if result.stdout:
                logger.info("Output:")
                logger.info(result.stdout)
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                logger.error("Error output:")
                logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {str(e)}")
        return False
    
    logger.info("="*60)
    return True

def check_prerequisites():
    """Check if required files and directories exist"""
    logger = logging.getLogger(__name__)
    logger.info("Checking prerequisites...")
    
    required_dirs = [
        "data/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast",
        "src",
        "models"
    ]
    
    optional_dirs = [
        "data/bach"
    ]
    
    missing_required = []
    missing_optional = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_required.append(dir_path)
    
    for dir_path in optional_dirs:
        if not os.path.exists(dir_path):
            missing_optional.append(dir_path)
    
    if missing_required:
        logger.error("Missing required directories:")
        for dir_path in missing_required:
            logger.error(f"  - {dir_path}")
        return False
    
    if missing_optional:
        logger.warning("Missing optional directories (some evaluations may be skipped):")
        for dir_path in missing_optional:
            logger.warning(f"  - {dir_path}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    logger.info("Prerequisites check completed")
    return True

def create_results_summary():
    """Create a summary of all generated results"""
    logger = logging.getLogger(__name__)
    logger.info("Creating results summary...")
    
    # List of expected output files
    expected_files = [
        "cross_dataset_results.png",
        "synthetic_samples.png", 
        "gan_comparison.png",
        "supconvit_tsne.png",
        "supconvit_comparison.png",
        "explanation_example.png",
        "rag_performance.png",
        "clinical_reports.json"
    ]
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'generated_files': [],
        'missing_files': [],
        'models_created': []
    }
    
    # Check generated files
    for filename in expected_files:
        if os.path.exists(filename):
            summary['generated_files'].append(filename)
            logger.info(f"✅ Generated: {filename}")
        else:
            summary['missing_files'].append(filename)
            logger.warning(f"❌ Missing: {filename}")
    
    # Check created models
    model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
    summary['models_created'] = model_files
    
    for model_file in model_files:
        logger.info(f"📁 Model created: models/{model_file}")
    
    # Save summary
    import json
    with open("evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Results summary saved to evaluation_summary.json")
    return summary

def main():
    """Main evaluation pipeline"""
    logger, log_filename = setup_logging()
    
    logger.info("🚀 Starting Complete Thesis Evaluation Pipeline")
    logger.info(f"📝 Logging to: {log_filename}")
    logger.info("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Please ensure all required data is available.")
        return 1
    
    # Define evaluation components
    evaluations = [
        {
            'script': 'cross_dataset_evaluation.py',
            'description': 'Cross-Dataset Evaluation (Train-on-one, Test-on-other)',
            'required': True
        },
        {
            'script': 'gan_evaluation.py', 
            'description': 'GAN-based Augmentation with FID Scores',
            'required': True
        },
        {
            'script': 'supconvit_evaluation.py',
            'description': 'SupConViT Implementation with t-SNE Visualization',
            'required': True
        },
        {
            'script': 'rag_interpretability_evaluation.py',
            'description': 'RAG-based Interpretability with Clinical Explanations',
            'required': True
        }
    ]
    
    # Track results
    results = {
        'total_evaluations': len(evaluations),
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Run each evaluation
    for i, evaluation in enumerate(evaluations, 1):
        logger.info(f"\n📊 Evaluation {i}/{len(evaluations)}: {evaluation['description']}")
        
        if not os.path.exists(evaluation['script']):
            logger.error(f"Script not found: {evaluation['script']}")
            if evaluation['required']:
                results['failed'] += 1
            else:
                results['skipped'] += 1
            continue
        
        success = run_script(evaluation['script'], evaluation['description'])
        
        if success:
            results['successful'] += 1
        else:
            results['failed'] += 1
            if evaluation['required']:
                logger.error(f"Required evaluation failed: {evaluation['description']}")
    
    # Create results summary
    logger.info("\n📋 Creating Results Summary...")
    summary = create_results_summary()
    
    # Final report
    logger.info("\n" + "="*80)
    logger.info("🎯 COMPLETE THESIS EVALUATION SUMMARY")
    logger.info("="*80)
    
    logger.info(f"📊 Evaluation Results:")
    logger.info(f"  ✅ Successful: {results['successful']}/{results['total_evaluations']}")
    logger.info(f"  ❌ Failed: {results['failed']}/{results['total_evaluations']}")
    logger.info(f"  ⏭️  Skipped: {results['skipped']}/{results['total_evaluations']}")
    
    logger.info(f"\n📁 Generated Files: {len(summary['generated_files'])}")
    for filename in summary['generated_files']:
        logger.info(f"  📄 {filename}")
    
    logger.info(f"\n🤖 Models Created: {len(summary['models_created'])}")
    for model_file in summary['models_created']:
        logger.info(f"  🧠 {model_file}")
    
    if summary['missing_files']:
        logger.warning(f"\n⚠️  Missing Files: {len(summary['missing_files'])}")
        for filename in summary['missing_files']:
            logger.warning(f"  ❌ {filename}")
    
    # Success criteria
    success_rate = results['successful'] / results['total_evaluations']
    
    if success_rate >= 0.75:
        logger.info("\n🎉 Evaluation pipeline completed successfully!")
        logger.info("✅ All major thesis components have been evaluated.")
        return_code = 0
    elif success_rate >= 0.5:
        logger.warning("\n⚠️  Evaluation pipeline completed with some issues.")
        logger.warning("🔍 Please review failed evaluations and missing files.")
        return_code = 1
    else:
        logger.error("\n❌ Evaluation pipeline failed.")
        logger.error("🚨 Multiple critical evaluations failed. Please check logs.")
        return_code = 2
    
    logger.info(f"\n📝 Complete log available at: {log_filename}")
    logger.info("="*80)
    
    return return_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)