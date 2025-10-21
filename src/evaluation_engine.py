"""
Quantitative evaluation engine with CV, bias-variance analysis, and visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.utils import resample
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings

import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import CV_FOLDS, RANDOM_STATE, REPORTS_DIR
    from src.logger import evaluation_logger
    from src.cv_engine import create_cv_engine

    from redis_cache import cache_evaluation_metrics, get_cached_evaluation_metrics


    from src.utils.visualization_utils import (
         plot_residuals, plot_feature_importance, plot_bias_variance_tradeoff,
         plot_learning_curve, create_model_comparison_plot, plot_shap_summary,
         save_evaluation_report
     )
except ImportError as e:
    print(f"Import error in evaluation_engine.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

class EvaluationEngine:
    """
    Comprehensive evaluation engine for quantitative model assessment.
    """

    def __init__(self, cv_folds: int = CV_FOLDS):
        self.cv_folds = cv_folds
        self.random_state = RANDOM_STATE
        self.cv_engine = create_cv_engine(cv_folds=cv_folds)

    def evaluate_regression_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 y_train: np.ndarray, y_test: np.ndarray,
                                 model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation for regression models.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_name: Name for the model

        Returns:
            Dictionary with evaluation results
        """
        evaluation_logger.info(f"Starting regression evaluation for {model_name}")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Basic metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # MAPE (Mean Absolute Percentage Error) - avoid division by zero
        epsilon = 1e-10
        train_mape = np.mean(np.abs((y_train - y_train_pred) / (y_train + epsilon))) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + epsilon))) * 100

        # Comprehensive K-fold CV analysis
        cv_results = self.cv_engine.perform_comprehensive_cv(
            model, X_train, y_train, 'regression', model_name
        )

        # Extract traditional CV metrics for backward compatibility
        cv_summary = cv_results['cv_summary']
        cv_mse = cv_summary['mse']['mean']
        cv_rmse = cv_summary['rmse']['mean']

        # Bias-Variance analysis
        bias_var_results = self._analyze_bias_variance(model, X_train, y_train, X_test, y_test)

        # Comprehensive residual analysis for linear models
        residual_analysis = self._comprehensive_residual_analysis(
            model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, model_name
        )

        # Generate automatic visualizations
        visualization_paths = self._generate_regression_visualizations(
            model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred,
            model_name, bias_var_results, residual_analysis
        )

        results = {
            'model_name': model_name,
            'evaluation_type': 'regression',
            'train_metrics': {
                'mse': train_mse,
                'rmse': train_rmse,
                'mae': train_mae,
                'r2': train_r2,
                'mape': train_mape
            },
            'test_metrics': {
                'mse': test_mse,
                'rmse': test_rmse,
                'mae': test_mae,
                'r2': test_r2,
                'mape': test_mape
            },
            'cv_metrics': {
                'mse_mean': cv_mse,
                'mse_std': cv_summary['mse']['std'],
                'rmse_mean': cv_rmse,
                'rmse_std': cv_summary['rmse']['std'],
                'cv_stability_score': cv_results['stability_analysis'].get('overall_stability_score', 0),
                'cv_stability': cv_results['stability_analysis'].get('overall_stability', 'unknown')
            },
            'comprehensive_cv': cv_results,
            'bias_variance': bias_var_results,
            'residual_analysis': residual_analysis,
            'predictions': {
                'y_test_pred': y_test_pred.tolist(),
                'residuals': (y_test - y_test_pred).tolist()
            },
            'visualizations': visualization_paths,
            'evaluation_timestamp': datetime.now().isoformat()
        }

        evaluation_logger.info(f"Regression evaluation completed for {model_name}. Test R²: {test_r2:.4f}")

        return results

    def _comprehensive_residual_analysis(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                       y_train: np.ndarray, y_test: np.ndarray,
                                       y_train_pred: np.ndarray, y_test_pred: np.ndarray,
                                       model_name: str) -> Dict[str, Any]:
        """
        Comprehensive residual analysis for linear models.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            y_train_pred: Training predictions
            y_test_pred: Test predictions
            model_name: Name of the model

        Returns:
            Dictionary with comprehensive residual analysis
        """
        # Test residuals
        test_residuals = y_test - y_test_pred
        train_residuals = y_train - y_train_pred

        # Basic residual statistics
        residual_stats = {
            'test_residuals': {
                'mean': float(np.mean(test_residuals)),
                'std': float(np.std(test_residuals)),
                'skewness': float(pd.Series(test_residuals).skew()),
                'kurtosis': float(pd.Series(test_residuals).kurtosis()),
                'min': float(np.min(test_residuals)),
                'max': float(np.max(test_residuals))
            },
            'train_residuals': {
                'mean': float(np.mean(train_residuals)),
                'std': float(np.std(train_residuals)),
                'skewness': float(pd.Series(train_residuals).skew()),
                'kurtosis': float(pd.Series(train_residuals).kurtosis()),
                'min': float(np.min(train_residuals)),
                'max': float(np.max(train_residuals))
            }
        }

        # Statistical tests for linear model assumptions
        assumption_tests = self._test_linear_assumptions(
            model, X_train, X_test, y_train, y_test, test_residuals, model_name
        )

        # Outlier detection
        outlier_analysis = self._detect_residual_outliers(test_residuals, y_test_pred)

        # Influence analysis (Cook's distance, leverage)
        influence_analysis = self._analyze_influence(model, X_train, y_train, model_name)

        # Heteroscedasticity tests
        heteroscedasticity_tests = self._test_heteroscedasticity(X_test, test_residuals)

        # Autocorrelation tests (Durbin-Watson)
        autocorrelation_tests = self._test_autocorrelation(test_residuals)

        # Model effectiveness score based on residual analysis
        effectiveness_score = self._calculate_residual_effectiveness_score(
            residual_stats, assumption_tests, outlier_analysis, heteroscedasticity_tests
        )

        comprehensive_analysis = {
            'residual_statistics': residual_stats,
            'assumption_tests': assumption_tests,
            'outlier_analysis': outlier_analysis,
            'influence_analysis': influence_analysis,
            'heteroscedasticity_tests': heteroscedasticity_tests,
            'autocorrelation_tests': autocorrelation_tests,
            'model_effectiveness_score': effectiveness_score,
            'diagnostic_summary': self._generate_residual_diagnostic_summary(
                residual_stats, assumption_tests, effectiveness_score
            )
        }

        return comprehensive_analysis

    def _test_linear_assumptions(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: np.ndarray, y_test: np.ndarray, residuals: np.ndarray,
                               model_name: str) -> Dict[str, Any]:
        """
        Test linear regression assumptions.
        """
        tests = {}

        # Normality test (Shapiro-Wilk)
        try:
            from scipy import stats
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            tests['normality_shapiro'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'normal_distribution': 1 if shapiro_p > 0.05 else 0
            }
        except:
            tests['normality_shapiro'] = {'error': 'Could not perform Shapiro test'}

        # Jarque-Bera test for normality
        try:
            jb_stat, jb_p = stats.jarque_bera(residuals)
            tests['normality_jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'normal_distribution': 1 if jb_p > 0.05 else 0
            }
        except:
            tests['normality_jarque_bera'] = {'error': 'Could not perform Jarque-Bera test'}

        # Mean zero test
        t_stat, p_value = stats.ttest_1samp(residuals, 0)
        tests['mean_zero_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_is_zero': 1 if p_value > 0.05 else 0
        }

        # Homoscedasticity test (Breusch-Pagan)
        tests['homoscedasticity'] = self._test_homoscedasticity_bp(X_test, residuals)

        return tests

    def _test_homoscedasticity_bp(self, X: pd.DataFrame, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Breusch-Pagan test for homoscedasticity.
        """
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.api import OLS, add_constant

            # Regress squared residuals on features
            squared_residuals = residuals ** 2
            X_with_const = add_constant(X)
            bp_model = OLS(squared_residuals, X_with_const).fit()

            # Breusch-Pagan test
            bp_stat, bp_p, _, _ = het_breuschpagan(bp_model.resid, X_with_const)

            return {
                'bp_statistic': float(bp_stat),
                'p_value': float(bp_p),
                'homoscedastic': 1 if bp_p > 0.05 else 0,  # Fail to reject null = homoscedastic
                'test_type': 'breusch_pagan'
            }
        except ImportError:
            # Fallback: simple variance comparison
            return {
                'error': 'statsmodels not available for Breusch-Pagan test',
                'fallback_test': 'variance_ratio_test'
            }
        except:
            return {'error': 'Could not perform homoscedasticity test'}

    def _detect_residual_outliers(self, residuals: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Detect outliers in residuals using standardized residuals.
        """
        # Standardized residuals
        std_residuals = residuals / np.std(residuals)

        # Outlier thresholds
        outlier_threshold = 2.5  # |std_residual| > 2.5
        severe_outlier_threshold = 3.0  # |std_residual| > 3.0

        outliers = np.abs(std_residuals) > outlier_threshold
        severe_outliers = np.abs(std_residuals) > severe_outlier_threshold

        return {
            'outlier_threshold': outlier_threshold,
            'severe_outlier_threshold': severe_outlier_threshold,
            'n_outliers': int(np.sum(outliers)),
            'n_severe_outliers': int(np.sum(severe_outliers)),
            'outlier_percentage': float(np.mean(outliers) * 100),
            'severe_outlier_percentage': float(np.mean(severe_outliers) * 100),
            'outlier_indices': np.where(outliers)[0].tolist(),
            'standardized_residuals': std_residuals.tolist()
        }

    def _analyze_influence(self, model, X_train: pd.DataFrame, y_train: np.ndarray,
                          model_name: str) -> Dict[str, Any]:
        """
        Analyze influential observations using Cook's distance and leverage.
        """
        try:
            from statsmodels.api import OLS, add_constant
            from statsmodels.stats.outliers_influence import OLSInfluence

            # Add constant for statsmodels
            X_with_const = add_constant(X_train)

            # Fit OLS model
            sm_model = OLS(y_train, X_with_const).fit()

            # Get influence measures
            influence = OLSInfluence(sm_model)

            # Cook's distance
            cooks_d = influence.cooks_distance[0]
            cooks_d_threshold = 4 / len(X_train)  # Common threshold

            # Leverage (hat values)
            leverage = influence.hat_matrix_diag
            leverage_threshold = 2 * (X_train.shape[1] + 1) / len(X_train)

            influential_points = cooks_d > cooks_d_threshold
            high_leverage_points = leverage > leverage_threshold

            return {
                'cooks_distance_threshold': float(cooks_d_threshold),
                'leverage_threshold': float(leverage_threshold),
                'n_influential_points': int(np.sum(influential_points)),
                'n_high_leverage_points': int(np.sum(high_leverage_points)),
                'influential_indices': np.where(influential_points)[0].tolist(),
                'high_leverage_indices': np.where(high_leverage_points)[0].tolist(),
                'max_cooks_distance': float(np.max(cooks_d)),
                'max_leverage': float(np.max(leverage))
            }
        except ImportError:
            return {'error': 'statsmodels not available for influence analysis'}
        except:
            return {'error': 'Could not perform influence analysis'}

    def _test_heteroscedasticity(self, X: pd.DataFrame, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Test for heteroscedasticity using multiple methods.
        """
        tests = {}

        # Goldfeld-Quandt test
        try:
            from statsmodels.stats.diagnostic import het_goldfeldquandt
            gq_stat, gq_p = het_goldfeldquandt(residuals, X)
            tests['goldfeld_quandt'] = {
                'statistic': float(gq_stat),
                'p_value': float(gq_p),
                'homoscedastic': 1 if gq_p > 0.05 else 0
            }
        except:
            tests['goldfeld_quandt'] = {'error': 'Could not perform Goldfeld-Quandt test'}

        # White test
        try:
            from statsmodels.stats.diagnostic import het_white
            white_stat, white_p, _, _ = het_white(residuals, add_constant(X))
            tests['white_test'] = {
                'statistic': float(white_stat),
                'p_value': float(white_p),
                'homoscedastic': 1 if white_p > 0.05 else 0
            }
        except:
            tests['white_test'] = {'error': 'Could not perform White test'}

        return tests

    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Test for autocorrelation in residuals (Durbin-Watson test).
        """
        try:
            from statsmodels.stats.stattools import durbin_watson

            dw_stat = durbin_watson(residuals)

            # Interpret Durbin-Watson statistic
            # DW ≈ 2: No autocorrelation
            # DW < 2: Positive autocorrelation
            # DW > 2: Negative autocorrelation
            if dw_stat < 1.5:
                interpretation = "positive_autocorrelation"
            elif dw_stat > 2.5:
                interpretation = "negative_autocorrelation"
            else:
                interpretation = "no_autocorrelation"

            return {
                'durbin_watson_statistic': float(dw_stat),
                'interpretation': interpretation,
                'no_autocorrelation': 1 if 1.5 <= dw_stat <= 2.5 else 0
            }
        except ImportError:
            return {'error': 'statsmodels not available for Durbin-Watson test'}
        except:
            return {'error': 'Could not perform autocorrelation test'}

    def _calculate_residual_effectiveness_score(self, residual_stats: Dict, assumption_tests: Dict,
                                              outlier_analysis: Dict, heteroscedasticity_tests: Dict) -> float:
        """
        Calculate a model effectiveness score based on residual analysis.
        """
        score_components = []

        # Normality score (0-1, higher is better)
        normality_score = 0.5  # Default
        if 'normality_shapiro' in assumption_tests:
            shapiro = assumption_tests['normality_shapiro']
            if 'normal_distribution' in shapiro:
                normality_score = 1.0 if shapiro['normal_distribution'] else 0.3

        score_components.append(normality_score)

        # Mean zero score
        mean_zero_score = 0.5
        if 'mean_zero_test' in assumption_tests:
            mean_test = assumption_tests['mean_zero_test']
            if 'mean_is_zero' in mean_test:
                mean_zero_score = 1.0 if mean_test['mean_is_zero'] else 0.2

        score_components.append(mean_zero_score)

        # Homoscedasticity score
        homoscedasticity_score = 0.5
        if 'homoscedasticity' in assumption_tests:
            homo_test = assumption_tests['homoscedasticity']
            if 'homoscedastic' in homo_test:
                homoscedasticity_score = 1.0 if homo_test['homoscedastic'] else 0.4

        score_components.append(homoscedasticity_score)

        # Outlier penalty (lower outliers = higher score)
        outlier_penalty = min(1.0, outlier_analysis.get('outlier_percentage', 0) / 10)
        outlier_score = 1.0 - outlier_penalty
        score_components.append(outlier_score)

        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Weights for normality, mean_zero, homoscedasticity, outliers
        effectiveness_score = np.average(score_components, weights=weights)

        return float(effectiveness_score)

    def _generate_residual_diagnostic_summary(self, residual_stats: Dict, assumption_tests: Dict,
                                            effectiveness_score: float) -> str:
        """
        Generate a human-readable summary of residual diagnostics.
        """
        summary_lines = []

        # Overall assessment
        if effectiveness_score >= 0.8:
            summary_lines.append("EXCELLENT: Residuals show strong adherence to linear model assumptions.")
        elif effectiveness_score >= 0.6:
            summary_lines.append("GOOD: Residuals are generally well-behaved with minor issues.")
        elif effectiveness_score >= 0.4:
            summary_lines.append("FAIR: Residuals show some violations of assumptions that may affect reliability.")
        else:
            summary_lines.append("POOR: Residuals show significant violations of linear model assumptions.")

        # Specific issues
        issues = []

        # Normality
        if 'normality_shapiro' in assumption_tests:
            shapiro = assumption_tests['normality_shapiro']
            if 'normal_distribution' in shapiro and not shapiro['normal_distribution']:
                issues.append("Residuals may not be normally distributed")

        # Mean zero
        if 'mean_zero_test' in assumption_tests:
            mean_test = assumption_tests['mean_zero_test']
            if 'mean_is_zero' in mean_test and not mean_test['mean_is_zero']:
                issues.append("Residual mean is significantly different from zero")

        # Homoscedasticity
        if 'homoscedasticity' in assumption_tests:
            homo_test = assumption_tests['homoscedasticity']
            if 'homoscedastic' in homo_test and not homo_test['homoscedastic']:
                issues.append("Evidence of heteroscedasticity (non-constant variance)")

        # Outliers
        test_stats = residual_stats.get('test_residuals', {})
        if test_stats.get('skewness', 0) > 1 or test_stats.get('skewness', 0) < -1:
            issues.append("Residuals show significant skewness")

        if issues:
            summary_lines.append("\nKey Issues Identified:")
            for issue in issues:
                summary_lines.append(f"- {issue}")

        summary_lines.append(f"\nModel Effectiveness Score: {effectiveness_score:.2f}/1.00")

        return "\n".join(summary_lines)

    def evaluate_classification_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                     y_train: np.ndarray, y_test: np.ndarray,
                                     model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification models.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_name: Name for the model

        Returns:
            Dictionary with evaluation results
        """
        evaluation_logger.info(f"Starting classification evaluation for {model_name}")

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities for ROC/AUC if available
        try:
            y_train_proba = model.predict_proba(X_train)
            y_test_proba = model.predict_proba(X_test)
            has_proba = True
        except:
            has_proba = False

        # Basic metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)

        # Classification report
        class_report = classification_report(y_test, y_test_pred, output_dict=True)

        # Comprehensive K-fold CV analysis
        cv_results = self.cv_engine.perform_comprehensive_cv(
            model, X_train, y_train, 'classification', model_name
        )

        # Extract traditional CV metrics for backward compatibility
        cv_summary = cv_results['cv_summary']
        cv_accuracy_mean = cv_summary['accuracy']['mean']
        cv_accuracy_std = cv_summary['accuracy']['std']

        # ROC-AUC if binary classification
        roc_auc = None
        if has_proba and len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
            roc_auc = auc(fpr, tpr)

        # Generate automatic visualizations
        visualization_paths = self._generate_classification_visualizations(
            model, X_train, X_test, y_train, y_test, model_name, cv_results
        )

        results = {
            'model_name': model_name,
            'evaluation_type': 'classification',
            'train_metrics': {
                'accuracy': train_accuracy,
                'precision': train_precision,
                'recall': train_recall,
                'f1': train_f1
            },
            'test_metrics': {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1
            },
            'cv_metrics': {
                'accuracy_mean': cv_accuracy_mean,
                'accuracy_std': cv_accuracy_std,
                'cv_stability_score': cv_results['stability_analysis'].get('overall_stability_score', 0),
                'cv_stability': cv_results['stability_analysis'].get('overall_stability', 'unknown')
            },
            'comprehensive_cv': cv_results,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'visualizations': visualization_paths,
            'evaluation_timestamp': datetime.now().isoformat()
        }

        evaluation_logger.info(f"Classification evaluation completed for {model_name}. Test Accuracy: {test_accuracy:.4f}")

        return results

    def _analyze_bias_variance(self, model, X_train: pd.DataFrame, y_train: np.ndarray,
                             X_test: pd.DataFrame, y_test: np.ndarray,
                             n_bootstraps: int = 50) -> Dict[str, Any]:
        """
        Perform bias-variance analysis using bootstrapping.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            n_bootstraps: Number of bootstrap samples

        Returns:
            Dictionary with bias-variance results
        """
        evaluation_logger.info("Performing bias-variance analysis")

        predictions = []

        for i in range(n_bootstraps):
            # Bootstrap sample
            X_boot, y_boot = resample(X_train, y_train, random_state=self.random_state + i)

            # Train model on bootstrap sample
            model_boot = model.__class__(**model.get_params())
            model_boot.fit(X_boot, y_boot)

            # Predict on test set
            y_pred = model_boot.predict(X_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)

        # Calculate bias and variance
        pred_mean = np.mean(predictions, axis=0)
        pred_var = np.var(predictions, axis=0)

        bias = np.mean((pred_mean - y_test) ** 2)
        variance = np.mean(pred_var)
        noise = np.var(y_test) if len(y_test) > 1 else 0

        # Irreducible error (noise) estimation
        irreducible_error = noise

        results = {
            'bias': float(bias),
            'variance': float(variance),
            'irreducible_error': float(irreducible_error),
            'total_error': float(bias + variance + irreducible_error),
            'bias_variance_ratio': float(variance / bias) if bias > 0 else float('inf'),
            'n_bootstraps': n_bootstraps
        }

        evaluation_logger.info(f"Bias-variance analysis completed. Bias: {bias:.4f}, Variance: {variance:.4f}")

        return results

    def generate_learning_curves(self, model, X_train: pd.DataFrame, y_train: np.ndarray,
                                model_name: str = "model", save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate learning curves for model evaluation.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            model_name: Name for the model
            save_plots: Whether to save plots

        Returns:
            Dictionary with learning curve data
        """
        evaluation_logger.info(f"Generating learning curves for {model_name}")

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=self.cv_folds,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error' if hasattr(model, 'predict') and len(np.unique(y_train)) > 10 else 'accuracy',
            random_state=self.random_state
        )

        # Convert negative MSE back to positive
        if 'neg_mean_squared_error' in str(model):
            train_scores = -train_scores
            val_scores = -val_scores

        results = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
            'val_scores_std': np.std(val_scores, axis=1).tolist()
        }

        if save_plots:
            fig = plot_learning_curve(
                np.mean(train_scores, axis=1),
                np.mean(val_scores, axis=1),
                train_sizes,
                model_name,
                save_path=REPORTS_DIR / f"{model_name}_learning_curve.png"
            )

            # Generate residual plots for linear models
            if hasattr(model, 'coef_') or hasattr(model, 'intercept_'):
                try:
                    # This is a linear model, generate residual diagnostics
                    fig_residuals = plot_residuals(
                        y_train, model.predict(X_train), f"{model_name}_train",
                        save_path=REPORTS_DIR / f"{model_name}_residuals_train.png"
                    )
                    plt.close(fig_residuals)

                    # For test set if available
                    if 'y_test' in locals() and 'X_test' in locals():
                        fig_residuals_test = plot_residuals(
                            y_test, model.predict(X_test), f"{model_name}_test",
                            save_path=REPORTS_DIR / f"{model_name}_residuals_test.png"
                        )
                        plt.close(fig_residuals_test)
                except Exception as e:
                    evaluation_logger.warning(f"Could not generate residual plots for {model_name}: {e}")

        evaluation_logger.info(f"Learning curves generated for {model_name}")

        return results

    def generate_validation_curves(self, model_class, X_train: pd.DataFrame, y_train: np.ndarray,
                                 param_name: str, param_range: List,
                                 model_name: str = "model", save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate validation curves for hyperparameter tuning visualization.

        Args:
            model_class: Model class
            X_train: Training features
            y_train: Training targets
            param_name: Parameter name to vary
            param_range: Range of parameter values
            model_name: Name for the model
            save_plots: Whether to save plots

        Returns:
            Dictionary with validation curve data
        """
        evaluation_logger.info(f"Generating validation curves for {model_name} - {param_name}")

        train_scores, val_scores = validation_curve(
            model_class(random_state=self.random_state), X_train, y_train,
            param_name=param_name, param_range=param_range,
            cv=self.cv_folds, scoring='r2' if len(np.unique(y_train)) > 10 else 'accuracy'
        )

        results = {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
            'val_scores_std': np.std(val_scores, axis=1).tolist()
        }

        if save_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training score')
            ax.fill_between(param_range,
                          np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                          np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
            ax.plot(param_range, np.mean(val_scores, axis=1), 'o-', label='Validation score')
            ax.fill_between(param_range,
                          np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                          np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Score')
            ax.set_title(f'Validation Curve - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = REPORTS_DIR / f"{model_name}_validation_curve_{param_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            evaluation_logger.info(f"Saved validation curve plot to {save_path}")

        evaluation_logger.info(f"Validation curves generated for {model_name}")

        return results

    def _generate_regression_visualizations(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                          y_train: np.ndarray, y_test: np.ndarray,
                                          y_train_pred: np.ndarray, y_test_pred: np.ndarray,
                                          model_name: str, bias_var_results: Dict[str, Any],
                                          residual_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate comprehensive visualizations for regression model evaluation.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            y_train_pred: Training predictions
            y_test_pred: Test predictions
            model_name: Name of the model
            bias_var_results: Bias-variance analysis results
            residual_analysis: Residual analysis results

        Returns:
            Dictionary with paths to generated visualizations
        """
        visualization_paths = {}

        try:
            # Residual diagnostics plot
            fig_residuals = plot_residuals(
                y_test, y_test_pred, f"{model_name}_test",
                save_path=REPORTS_DIR / f"{model_name}_residual_diagnostics.png"
            )
            visualization_paths['residual_diagnostics'] = str(REPORTS_DIR / f"{model_name}_residual_diagnostics.png")
            plt.close(fig_residuals)

            # Learning curve with overfitting indicators
            learning_curve_data = self.generate_learning_curves(model, X_train, y_train, model_name, save_plots=True)
            visualization_paths['learning_curve'] = str(REPORTS_DIR / f"{model_name}_learning_curve.png")

            # Bias-variance tradeoff plot
            if bias_var_results:
                fig_bias_var = plot_bias_variance_tradeoff(
                    [bias_var_results['bias']], [bias_var_results['variance']],
                    [1.0], model_name,  # Using dummy complexity for now
                    save_path=REPORTS_DIR / f"{model_name}_bias_variance.png"
                )
                visualization_paths['bias_variance'] = str(REPORTS_DIR / f"{model_name}_bias_variance.png")
                plt.close(fig_bias_var)

            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                fig_importance = plot_feature_importance(
                    X_train.columns.tolist(), model.feature_importances_,
                    model_name, save_path=REPORTS_DIR / f"{model_name}_feature_importance.png"
                )
                visualization_paths['feature_importance'] = str(REPORTS_DIR / f"{model_name}_feature_importance.png")
                plt.close(fig_importance)

            # Model comparison plot (if multiple models evaluated)
            # This would be handled at a higher level

        except Exception as e:
            evaluation_logger.warning(f"Could not generate some visualizations for {model_name}: {e}")

        return visualization_paths

    def _generate_classification_visualizations(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                              y_train: np.ndarray, y_test: np.ndarray,
                                              model_name: str, cv_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate comprehensive visualizations for classification model evaluation.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_name: Name of the model
            cv_results: Cross-validation results

        Returns:
            Dictionary with paths to generated visualizations
        """
        visualization_paths = {}

        try:
            # Learning curve with overfitting indicators
            learning_curve_data = self.generate_learning_curves(model, X_train, y_train, model_name, save_plots=True)
            visualization_paths['learning_curve'] = str(REPORTS_DIR / f"{model_name}_learning_curve.png")

            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                fig_importance = plot_feature_importance(
                    X_train.columns.tolist(), model.feature_importances_,
                    model_name, save_path=REPORTS_DIR / f"{model_name}_feature_importance.png"
                )
                visualization_paths['feature_importance'] = str(REPORTS_DIR / f"{model_name}_feature_importance.png")
                plt.close(fig_importance)

            # Logistic regression specific plots
            if hasattr(model, 'predict_proba') and hasattr(model, 'coef_'):
                fig_logistic_residuals = plot_logistic_residuals(
                    model, X_test, y_test, model_name,
                    save_path=REPORTS_DIR / f"{model_name}_logistic_residuals.png"
                )
                visualization_paths['logistic_residuals'] = str(REPORTS_DIR / f"{model_name}_logistic_residuals.png")
                plt.close(fig_logistic_residuals)

        except Exception as e:
            evaluation_logger.warning(f"Could not generate some visualizations for {model_name}: {e}")

        return visualization_paths

    def create_evaluation_report(self, evaluation_results: Dict[str, Any],
                                save_path: Optional[Path] = None) -> str:
        """
        Create a comprehensive evaluation report.

        Args:
            evaluation_results: Results from model evaluation
            save_path: Optional path to save the report

        Returns:
            Report as formatted string
        """
        report_lines = []
        report_lines.append("# Model Evaluation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        for model_name, results in evaluation_results.items():
            if 'error' in results:
                continue

            report_lines.append(f"## {model_name}")
            report_lines.append("")

            eval_type = results.get('evaluation_type', 'unknown')

            if eval_type == 'regression':
                # Training metrics
                report_lines.append("### Training Metrics")
                train_metrics = results['train_metrics']
                report_lines.append(f"- MSE: {train_metrics['mse']:.4f}")
                report_lines.append(f"- RMSE: {train_metrics['rmse']:.4f}")
                report_lines.append(f"- MAE: {train_metrics['mae']:.4f}")
                report_lines.append(f"- R²: {train_metrics['r2']:.4f}")
                report_lines.append(f"- MAPE: {train_metrics['mape']:.2f}%")
                report_lines.append("")

                # Test metrics
                report_lines.append("### Test Metrics")
                test_metrics = results['test_metrics']
                report_lines.append(f"- MSE: {test_metrics['mse']:.4f}")
                report_lines.append(f"- RMSE: {test_metrics['rmse']:.4f}")
                report_lines.append(f"- MAE: {test_metrics['mae']:.4f}")
                report_lines.append(f"- R²: {test_metrics['r2']:.4f}")
                report_lines.append(f"- MAPE: {test_metrics['mape']:.2f}%")
                report_lines.append("")

                # CV metrics
                report_lines.append("### Cross-Validation Metrics")
                cv_metrics = results['cv_metrics']
                report_lines.append(f"- CV MSE: {cv_metrics['mse_mean']:.4f} ± {cv_metrics['mse_std']:.4f}")
                report_lines.append(f"- CV RMSE: {cv_metrics['rmse_mean']:.4f} ± {cv_metrics['rmse_std']:.4f}")
                report_lines.append("")

                # Bias-Variance
                report_lines.append("### Bias-Variance Analysis")
                bv = results['bias_variance']
                report_lines.append(f"- Bias: {bv['bias']:.4f}")
                report_lines.append(f"- Variance: {bv['variance']:.4f}")
                report_lines.append(f"- Irreducible Error: {bv['irreducible_error']:.4f}")
                report_lines.append(f"- Total Error: {bv['total_error']:.4f}")
                report_lines.append(f"- Bias-Variance Ratio: {bv['bias_variance_ratio']:.2f}")
                report_lines.append("")

            elif eval_type == 'classification':
                # Training metrics
                report_lines.append("### Training Metrics")
                train_metrics = results['train_metrics']
                report_lines.append(f"- Accuracy: {train_metrics['accuracy']:.4f}")
                report_lines.append(f"- Precision: {train_metrics['precision']:.4f}")
                report_lines.append(f"- Recall: {train_metrics['recall']:.4f}")
                report_lines.append(f"- F1-Score: {train_metrics['f1']:.4f}")
                report_lines.append("")

                # Test metrics
                report_lines.append("### Test Metrics")
                test_metrics = results['test_metrics']
                report_lines.append(f"- Accuracy: {test_metrics['accuracy']:.4f}")
                report_lines.append(f"- Precision: {test_metrics['precision']:.4f}")
                report_lines.append(f"- Recall: {test_metrics['recall']:.4f}")
                report_lines.append(f"- F1-Score: {test_metrics['f1']:.4f}")
                report_lines.append("")

                # CV metrics
                report_lines.append("### Cross-Validation Metrics")
                cv_metrics = results['cv_metrics']
                report_lines.append(f"- CV Accuracy: {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")
                report_lines.append("")

                if results.get('roc_auc'):
                    report_lines.append(f"- ROC-AUC: {results['roc_auc']:.4f}")
                    report_lines.append("")

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            evaluation_logger.info(f"Evaluation report saved to {save_path}")

        return report

def create_evaluation_engine(cv_folds: int = CV_FOLDS) -> EvaluationEngine:
    """
    Factory function to create EvaluationEngine instance.

    Returns:
        EvaluationEngine instance
    """
    return EvaluationEngine(cv_folds=cv_folds)