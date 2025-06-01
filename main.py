"""
Main entry point for the SVM Analysis Project
============================================

This script runs the complete SVM analysis pipeline including:
- Data loading and preprocessing
- SVM implementation and training
- Model comparison
- Visualization generation
- Results reporting
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.data_loader import DataLoader
from src.utils.visualization import SVMVisualizer
from src.svm.kernel_svm import KernelSVM
from src.models.baseline_models import BaselineModels
from src.utils.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('svm_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main execution function for the SVM analysis pipeline
    """
    logger.info("Starting SVM Analysis Pipeline")
    
    try:
        # Initialize components
        data_loader = DataLoader()
        visualizer = SVMVisualizer()
        evaluator = ModelEvaluator()
        
        # 1. Classification Analysis
        logger.info("=" * 50)
        logger.info("CLASSIFICATION ANALYSIS")
        logger.info("=" * 50)
        
        # Load Heart Disease Dataset
        logger.info("Loading Heart Disease Dataset...")
        X_train_heart, X_test_heart, y_train_heart, y_test_heart = data_loader.load_heart_disease_data()
        
        # Train SVM models with different kernels
        logger.info("Training SVM models...")
        svm_results = {}
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        
        for kernel in kernels:
            logger.info(f"Training SVM with {kernel} kernel...")
            svm = KernelSVM(kernel=kernel, C=1.0)
            svm.fit(X_train_heart, y_train_heart)
            predictions = svm.predict(X_test_heart)
            
            # Evaluate
            metrics = evaluator.evaluate_classification(y_test_heart, predictions)
            svm_results[f'svm_{kernel}'] = metrics
            logger.info(f"SVM {kernel} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        
        # Compare with baseline models
        logger.info("Training baseline models...")
        baseline_models = BaselineModels()
        baseline_results = baseline_models.compare_classification_models(
            X_train_heart, X_test_heart, y_train_heart, y_test_heart
        )
        
        # Visualize results
        logger.info("Generating classification visualizations...")
        visualizer.plot_classification_comparison(svm_results, baseline_results)
        visualizer.plot_decision_boundaries(X_test_heart, y_test_heart, svm_results)
        
        # 2. Regression Analysis
        logger.info("=" * 50)
        logger.info("REGRESSION ANALYSIS")
        logger.info("=" * 50)
        
        # Load California Housing Dataset
        logger.info("Loading California Housing Dataset...")
        X_train_house, X_test_house, y_train_house, y_test_house = data_loader.load_california_housing_data()
        
        # Train SVR models
        logger.info("Training SVR models...")
        svr_results = {}
        
        for kernel in ['linear', 'rbf', 'poly']:
            logger.info(f"Training SVR with {kernel} kernel...")
            from src.svm.svr import SupportVectorRegressor
            svr = SupportVectorRegressor(kernel=kernel, C=1.0)
            svr.fit(X_train_house, y_train_house)
            predictions = svr.predict(X_test_house)
            
            # Evaluate
            metrics = evaluator.evaluate_regression(y_test_house, predictions)
            svr_results[f'svr_{kernel}'] = metrics
            logger.info(f"SVR {kernel} - RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
        
        # Compare with baseline regression models
        logger.info("Training baseline regression models...")
        baseline_reg_results = baseline_models.compare_regression_models(
            X_train_house, X_test_house, y_train_house, y_test_house
        )
        
        # Visualize regression results
        logger.info("Generating regression visualizations...")
        visualizer.plot_regression_comparison(svr_results, baseline_reg_results)
        
        # 3. Text Classification Analysis
        logger.info("=" * 50)
        logger.info("TEXT CLASSIFICATION ANALYSIS")
        logger.info("=" * 50)
        
        # Load BBC News Dataset
        logger.info("Loading BBC News Dataset...")
        X_train_text, X_test_text, y_train_text, y_test_text = data_loader.load_bbc_news_data()
        
        # Train SVM for text classification
        logger.info("Training SVM for text classification...")
        text_svm = KernelSVM(kernel='linear', C=1.0)
        text_svm.fit(X_train_text, y_train_text)
        text_predictions = text_svm.predict(X_test_text)
        
        # Evaluate text classification
        text_metrics = evaluator.evaluate_classification(y_test_text, text_predictions)
        logger.info(f"Text SVM - Accuracy: {text_metrics['accuracy']:.3f}, F1: {text_metrics['f1']:.3f}")
        
        # Generate comprehensive report
        logger.info("Generating comprehensive analysis report...")
        evaluator.generate_comprehensive_report(
            classification_results={'svm': svm_results, 'baseline': baseline_results},
            regression_results={'svr': svr_results, 'baseline': baseline_reg_results},
            text_results={'svm': text_metrics}
        )
        
        logger.info("=" * 50)
        logger.info("SVM Analysis Pipeline Completed Successfully!")
        logger.info("=" * 50)
        logger.info("Check the following directories for results:")
        logger.info("- visualizations/ : All generated plots and charts")
        logger.info("- results/ : Detailed analysis results")
        logger.info("- notebooks/ : Interactive analysis notebooks")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
