"""
Comprehensive K-Fold Cross-Validation Engine for Advanced Evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.base import clone
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
from pathlib import Path
import json
from datetime import datetime

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import RANDOM_STATE, REPORTS_DIR
    from src.logger import evaluation_logger
except ImportError as e:
    print(f"Import error in cv_engine.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise


class CrossValidationEngine:
    """
    Comprehensive CV engine supporting multiple strategies and stability analysis.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = RANDOM_STATE):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def get_cv_strategy(self, task_type: str, target_distribution: Optional[np.ndarray] = None) -> Union[KFold, StratifiedKFold]:
        """
        Select appropriate CV strategy based on data characteristics.

        Args:
            task_type: 'regression' or 'classification'
            target_distribution: Class distribution for stratification

        Returns:
            CV strategy object
        """
        if task_type == 'classification' and target_distribution is not None:
            # Check if data is imbalanced (any class < 10% of total)
            total_samples = np.sum(target_distribution)
            min_class_ratio = np.min(target_distribution) / total_samples

            if min_class_ratio < 0.1:  # Imbalanced if any class < 10%
                evaluation_logger.info(f"Using StratifiedKFold for imbalanced classification (min class ratio: {min_class_ratio:.3f})")
                return StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_state
                )
            else:
                evaluation_logger.info("Using StratifiedKFold for balanced classification")
                return StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_state
                )
        else:
            evaluation_logger.info("Using KFold for regression")
            return KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )

    def perform_comprehensive_cv(self, model, X: pd.DataFrame, y: np.ndarray,
                               task_type: str, model_name: str = "model") -> Dict[str, Any]:
        """
        Perform comprehensive CV with multiple metrics and stability analysis.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            task_type: 'regression' or 'classification'
            model_name: Name for logging

        Returns:
            Comprehensive CV results
        """
        evaluation_logger.info(f"Starting comprehensive CV for {model_name}")

        # Get appropriate CV strategy
        target_dist = np.bincount(y.astype(int)) if task_type == 'classification' else None
        cv_strategy = self.get_cv_strategy(task_type, target_dist)

        # Initialize results storage
        fold_results = []
        predictions_per_fold = []
        feature_importance_per_fold = []

        # Perform CV fold by fold for detailed analysis
        evaluation_logger.info(f"Starting CV with {self.cv_folds} folds, X shape: {X.shape}, y shape: {y.shape}")
        evaluation_logger.info(f"Task type: {task_type}, CV strategy: {type(cv_strategy).__name__}")
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y)):
            evaluation_logger.info(f"Fold {fold_idx + 1}: train_idx length: {len(train_idx)}, val_idx length: {len(val_idx)}")
            evaluation_logger.info(f"Train indices range: {train_idx[:5]}...{train_idx[-5:]}")
            evaluation_logger.info(f"Val indices range: {val_idx[:5]}...{val_idx[-5:]}")

            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Clone and train model
            model_fold = clone(model)
            model_fold.fit(X_train_fold, y_train_fold)

            # Make predictions
            y_pred_fold = model_fold.predict(X_val_fold)
            evaluation_logger.info(f"Fold {fold_idx + 1} predictions shape: {y_pred_fold.shape}, true values shape: {y_val_fold.shape}")
            evaluation_logger.info(f"Fold {fold_idx + 1} prediction sample: {y_pred_fold[:5]}, true sample: {y_val_fold[:5]}")

            # Check fold balance
            fold_balance = self._check_fold_balance(train_idx, val_idx, X.shape[0])
            evaluation_logger.info(f"Fold {fold_idx + 1} balance: {fold_balance}")

            # Calculate fold metrics
            fold_metrics = self._calculate_fold_metrics(
                y_val_fold, y_pred_fold, task_type, fold_idx
            )
            evaluation_logger.info(f"Fold {fold_idx + 1} metrics: {fold_metrics}")

            # Store predictions for stability analysis
            fold_predictions = {
                'fold': fold_idx,
                'y_true': y_val_fold.tolist(),
                'y_pred': y_pred_fold.tolist(),
                'indices': val_idx.tolist()
            }
            predictions_per_fold.append(fold_predictions)

            # Store feature importance if available
            if hasattr(model_fold, 'feature_importances_'):
                fold_importance = {
                    'fold': fold_idx,
                    'importance': model_fold.feature_importances_.tolist(),
                    'feature_names': X.columns.tolist()
                }
                feature_importance_per_fold.append(fold_importance)
            elif hasattr(model_fold, 'coef_'):
                # For linear models
                coef = model_fold.coef_
                if coef.ndim > 1:  # Multi-class
                    coef = np.mean(np.abs(coef), axis=0)
                fold_importance = {
                    'fold': fold_idx,
                    'importance': np.abs(coef).tolist(),
                    'feature_names': X.columns.tolist()
                }
                feature_importance_per_fold.append(fold_importance)

            fold_results.append(fold_metrics)

        # Validate CV results before aggregation
        validation_result = self._validate_cv_results(fold_results, task_type)
        if not validation_result['valid']:
            evaluation_logger.warning(f"CV validation failed: {validation_result['issues']}")
            # Continue but log warnings

        # Aggregate results
        evaluation_logger.info(f"Aggregating results from {len(fold_results)} folds")
        evaluation_logger.info(f"Fold results keys: {list(fold_results[0].keys()) if fold_results else 'No folds'}")
        cv_summary = self._aggregate_cv_results(fold_results, task_type)
        evaluation_logger.info(f"CV summary keys: {list(cv_summary.keys())}")
        evaluation_logger.info(f"CV summary sample: {dict(list(cv_summary.items())[:2])}")

        # Perform stability analysis
        stability_analysis = self._analyze_cv_stability(
            fold_results, predictions_per_fold, task_type
        )

        # Feature importance stability
        if feature_importance_per_fold:
            importance_stability = self._analyze_feature_importance_stability(
                feature_importance_per_fold
            )
        else:
            importance_stability = {}

        # Compile comprehensive results
        comprehensive_results = {
            'cv_summary': cv_summary,
            'stability_analysis': stability_analysis,
            'feature_importance_stability': importance_stability,
            'fold_results': fold_results,
            'predictions_per_fold': predictions_per_fold,
            'cv_strategy': str(type(cv_strategy).__name__),
            'n_folds': self.cv_folds,
            'task_type': task_type,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }

        evaluation_logger.info(f"Comprehensive CV completed for {model_name}")
        return comprehensive_results

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              task_type: str, fold_idx: int) -> Dict[str, Any]:
        """
        Calculate metrics for a single CV fold.

        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: 'regression' or 'classification'
            fold_idx: Fold index

        Returns:
            Fold metrics dictionary
        """
        metrics = {'fold': fold_idx}

        if task_type == 'regression':
            # Safe MAPE calculation to prevent division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_values = np.abs((y_true - y_pred) / y_true)
                mape_values = mape_values[np.isfinite(mape_values)]  # Remove inf and nan
                safe_mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else np.nan

            metrics.update({
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred)),
                'mape': float(safe_mape) if not np.isnan(safe_mape) else None
            })
        else:  # classification
            metrics.update({
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            })

        return metrics

    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]],
                            task_type: str) -> Dict[str, Any]:
        """
        Aggregate results across all CV folds.

        Args:
            fold_results: List of fold metrics
            task_type: 'regression' or 'classification'

        Returns:
            Aggregated CV results
        """
        # Convert to DataFrame for easy aggregation
        results_df = pd.DataFrame(fold_results)

        # Calculate mean, std, min, max for each metric
        summary = {}
        for col in results_df.columns:
            if col != 'fold':
                # Ensure values are numeric and convert to float array
                values = pd.to_numeric(results_df[col], errors='coerce').values

                # Skip if all values are NaN
                if np.all(np.isnan(values)):
                    evaluation_logger.warning(f"Column {col} contains all NaN values, skipping")
                    continue

                # Remove NaN values for calculation
                valid_values = values[~np.isnan(values)]

                if len(valid_values) == 0:
                    evaluation_logger.warning(f"Column {col} has no valid numeric values, skipping")
                    continue

                # Enhanced error handling for edge cases
                try:
                    mean_val = float(np.mean(valid_values))
                    std_val = float(np.std(valid_values))

                    # Check for invalid values
                    if not np.isfinite(mean_val) or not np.isfinite(std_val):
                        evaluation_logger.warning(f"Invalid statistics for {col}: mean={mean_val}, std={std_val}")
                        continue

                    min_val = float(np.min(valid_values))
                    max_val = float(np.max(valid_values))

                    # Coefficient of variation with safe division
                    cv_val = float(std_val / mean_val) if abs(mean_val) > 1e-10 else 0.0

                    summary[col] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'cv': cv_val
                    }
                except Exception as e:
                    evaluation_logger.error(f"Error calculating statistics for {col}: {str(e)}")
                    continue

        # Add confidence intervals (95%)
        for col in results_df.columns:
            if col != 'fold' and col in summary:
                # Ensure values are numeric
                values = pd.to_numeric(results_df[col], errors='coerce').values
                valid_values = values[~np.isnan(values)]
                
                if len(valid_values) == 0:
                    continue
                
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                n_folds = len(valid_values)

                # t-distribution critical value (approximate for 95% CI)
                t_critical = 2.776 if n_folds == 5 else 2.571 if n_folds == 10 else 2.0

                margin_error = t_critical * (std_val / np.sqrt(n_folds))
                summary[col]['ci_95_lower'] = float(mean_val - margin_error)
                summary[col]['ci_95_upper'] = float(mean_val + margin_error)

        # Validate aggregated results
        if not summary:
            evaluation_logger.error("No valid metrics could be aggregated from fold results")
            return {}

        return summary

    def _analyze_cv_stability(self, fold_results: List[Dict[str, Any]],
                            predictions_per_fold: List[Dict[str, Any]],
                            task_type: str) -> Dict[str, Any]:
        """
        Analyze stability of CV results across folds.

        Args:
            fold_results: List of fold metrics
            predictions_per_fold: List of fold predictions
            task_type: 'regression' or 'classification'

        Returns:
            Stability analysis results
        """
        results_df = pd.DataFrame(fold_results)

        stability_metrics = {}

        # Coefficient of variation for each metric
        for col in results_df.columns:
            if col != 'fold':
                values = results_df[col].values
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                stability_metrics[f'{col}_cv'] = float(cv)

                # Stability classification
                if cv < 0.1:
                    stability = 'very_stable'
                elif cv < 0.2:
                    stability = 'stable'
                elif cv < 0.3:
                    stability = 'moderately_stable'
                else:
                    stability = 'unstable'

                stability_metrics[f'{col}_stability'] = stability

        # Overall stability score (average CV across key metrics)
        key_metrics = ['r2', 'accuracy'] if task_type == 'classification' else ['r2', 'rmse']
        key_cvs = [stability_metrics[f'{metric}_cv'] for metric in key_metrics if f'{metric}_cv' in stability_metrics]

        if key_cvs:
            overall_stability_score = float(np.mean(key_cvs))
            stability_metrics['overall_stability_score'] = overall_stability_score

            if overall_stability_score < 0.15:
                stability_metrics['overall_stability'] = 'very_stable'
            elif overall_stability_score < 0.25:
                stability_metrics['overall_stability'] = 'stable'
            elif overall_stability_score < 0.35:
                stability_metrics['overall_stability'] = 'moderately_stable'
            else:
                stability_metrics['overall_stability'] = 'unstable'

        # Prediction consistency analysis
        prediction_consistency = self._analyze_prediction_consistency(predictions_per_fold, task_type)
        stability_metrics['prediction_consistency'] = prediction_consistency

        return stability_metrics

    def _analyze_prediction_consistency(self, predictions_per_fold: List[Dict[str, Any]],
                                       task_type: str) -> Dict[str, Any]:
        """
        Analyze consistency of predictions across folds.

        Args:
            predictions_per_fold: List of fold predictions
            task_type: 'regression' or 'classification'

        Returns:
            Prediction consistency analysis
        """
        consistency_metrics = {}

        if len(predictions_per_fold) < 2:
            return {'error': 'Need at least 2 folds for consistency analysis'}

        # Check if all folds have the same number of predictions
        fold_lengths = [len(fold_data['y_pred']) for fold_data in predictions_per_fold]
        if len(set(fold_lengths)) > 1:
            consistency_metrics['error'] = 'Folds have different sizes, prediction consistency analysis skipped'
            return consistency_metrics

        # For regression: analyze prediction variance
        if task_type == 'regression':
            fold_predictions = []
            for fold_data in predictions_per_fold:
                fold_predictions.append(np.array(fold_data['y_pred']))

            # Calculate prediction variance across folds for each sample
            pred_matrix = np.array(fold_predictions)
            pred_variance = np.var(pred_matrix, axis=0)

            consistency_metrics.update({
                'mean_prediction_variance': float(np.mean(pred_variance)),
                'max_prediction_variance': float(np.max(pred_variance)),
                'prediction_variance_percentiles': {
                    '25th': float(np.percentile(pred_variance, 25)),
                    '50th': float(np.percentile(pred_variance, 50)),
                    '75th': float(np.percentile(pred_variance, 75)),
                    '95th': float(np.percentile(pred_variance, 95))
                }
            })

        else:  # classification
            # For classification: analyze prediction agreement
            fold_predictions = []
            for fold_data in predictions_per_fold:
                fold_predictions.append(np.array(fold_data['y_pred']))

            pred_matrix = np.array(fold_predictions)

            # Calculate agreement rate (how often folds agree on prediction)
            n_folds, n_samples = pred_matrix.shape
            agreement_counts = []

            for sample_idx in range(n_samples):
                sample_preds = pred_matrix[:, sample_idx]
                most_common_pred = np.bincount(sample_preds.astype(int)).argmax()
                agreement = np.sum(sample_preds == most_common_pred) / n_folds
                agreement_counts.append(agreement)

            agreement_counts = np.array(agreement_counts)

            consistency_metrics.update({
                'mean_agreement_rate': float(np.mean(agreement_counts)),
                'min_agreement_rate': float(np.min(agreement_counts)),
                'agreement_rate_percentiles': {
                    '25th': float(np.percentile(agreement_counts, 25)),
                    '50th': float(np.percentile(agreement_counts, 50)),
                    '75th': float(np.percentile(agreement_counts, 75)),
                    '95th': float(np.percentile(agreement_counts, 95))
                }
            })

        return consistency_metrics

    def _analyze_feature_importance_stability(self, importance_per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze stability of feature importance across folds.

        Args:
            importance_per_fold: List of fold importance data

        Returns:
            Feature importance stability analysis
        """
        if not importance_per_fold:
            return {}

        # Extract importance arrays
    def _check_fold_balance(self, train_idx: np.ndarray, val_idx: np.ndarray, total_samples: int) -> Dict[str, Any]:
        """
        Check balance of train/validation split in a fold.

        Args:
            train_idx: Training indices
            val_idx: Validation indices
            total_samples: Total number of samples

        Returns:
            Balance analysis dictionary
        """
        train_size = len(train_idx)
        val_size = len(val_idx)
        total_size = train_size + val_size

        balance_info = {
            'train_size': train_size,
            'val_size': val_size,
            'train_ratio': train_size / total_size,
            'val_ratio': val_size / total_size,
            'total_samples': total_samples,
            'fold_total': total_size
        }

        # Check for potential issues
        issues = []
        if abs(balance_info['train_ratio'] - 0.8) > 0.1:  # Allow 10% deviation from 80/20 split
            issues.append(f"Unbalanced split: train_ratio={balance_info['train_ratio']:.3f}")

        if total_size != total_samples:
            issues.append(f"Fold size mismatch: fold_total={total_size}, expected={total_samples}")

        if train_size == 0 or val_size == 0:
            issues.append("Empty train or validation set")

        balance_info['issues'] = issues
        balance_info['balanced'] = len(issues) == 0

        return balance_info

    def _validate_cv_results(self, fold_results: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """
        Validate CV results for consistency and reliability.

        Args:
            fold_results: List of fold metrics
            task_type: 'regression' or 'classification'

        Returns:
            Validation results dictionary
        """
        validation = {'valid': True, 'issues': []}

        if not fold_results:
            validation['valid'] = False
            validation['issues'].append("No fold results provided")
            return validation

        # Check number of folds
        expected_folds = len(fold_results)
        if expected_folds < 2:
            validation['issues'].append(f"Insufficient folds: {expected_folds} (minimum 2 required)")

        # Check metric consistency across folds
        first_fold_keys = set(fold_results[0].keys()) - {'fold'}
        for i, fold in enumerate(fold_results[1:], 1):
            fold_keys = set(fold.keys()) - {'fold'}
            if fold_keys != first_fold_keys:
                validation['issues'].append(f"Fold {i} has different metrics: {fold_keys ^ first_fold_keys}")

        # Check for NaN/inf values
        for i, fold in enumerate(fold_results):
            for metric, value in fold.items():
                if metric == 'fold':
                    continue
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    validation['issues'].append(f"Fold {i} has invalid {metric}: {value}")

        # Task-specific validations
        if task_type == 'regression':
            for i, fold in enumerate(fold_results):
                if 'r2' in fold and fold['r2'] is not None:
                    if fold['r2'] < -1 or fold['r2'] > 1:
                        validation['issues'].append(f"Fold {i} R² out of range: {fold['r2']}")
        elif task_type == 'classification':
            for i, fold in enumerate(fold_results):
                if 'accuracy' in fold and fold['accuracy'] is not None:
                    if fold['accuracy'] < 0 or fold['accuracy'] > 1:
                        validation['issues'].append(f"Fold {i} accuracy out of range: {fold['accuracy']}")

        # Check for extreme variance
        results_df = pd.DataFrame(fold_results)
        for col in results_df.columns:
            if col == 'fold':
                continue
            values = pd.to_numeric(results_df[col], errors='coerce').values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 1:
                cv = np.std(valid_values) / np.mean(valid_values) if np.mean(valid_values) != 0 else 0
                if cv > 1.0:  # Coefficient of variation > 100%
                    validation['issues'].append(f"High variance in {col}: CV={cv:.3f}")

        validation['valid'] = len(validation['issues']) == 0
        return validation
        importance_matrix = []
        feature_names = importance_per_fold[0]['feature_names']

        for fold_data in importance_per_fold:
            importance_matrix.append(fold_data['importance'])

        importance_matrix = np.array(importance_matrix)

        # Calculate stability metrics for each feature
        feature_stability = {}
        for feature_idx, feature_name in enumerate(feature_names):
            feature_importance = importance_matrix[:, feature_idx]
            cv = np.std(feature_importance) / np.mean(feature_importance) if np.mean(feature_importance) != 0 else 0

            feature_stability[feature_name] = {
                'mean_importance': float(np.mean(feature_importance)),
                'std_importance': float(np.std(feature_importance)),
                'cv': float(cv),
                'stability': 'stable' if cv < 0.3 else 'unstable'
            }

        # Overall importance stability
        all_cvs = [data['cv'] for data in feature_stability.values()]
        overall_importance_stability = {
            'mean_cv': float(np.mean(all_cvs)),
            'stable_features': len([cv for cv in all_cvs if cv < 0.3]),
            'unstable_features': len([cv for cv in all_cvs if cv >= 0.3]),
            'most_stable_feature': min(feature_stability.items(), key=lambda x: x[1]['cv'])[0],
            'least_stable_feature': max(feature_stability.items(), key=lambda x: x[1]['cv'])[0]
        }

        # Validate feature importance stability results
        if not feature_stability:
            evaluation_logger.warning("No valid feature importance stability data calculated")
            return {}

        return {
            'feature_stability': feature_stability,
            'overall_stability': overall_importance_stability
        }

    def generate_cv_report(self, cv_results: Dict[str, Any], save_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive CV report.

        Args:
            cv_results: Results from comprehensive CV
            save_path: Optional path to save report

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("# Cross-Validation Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Model and CV info
        report_lines.append("## Model Information")
        report_lines.append(f"- Model: {cv_results.get('model_name', 'Unknown')}")
        report_lines.append(f"- Task Type: {cv_results.get('task_type', 'Unknown')}")
        report_lines.append(f"- CV Strategy: {cv_results.get('cv_strategy', 'Unknown')}")
        report_lines.append(f"- Number of Folds: {cv_results.get('n_folds', 'Unknown')}")
        report_lines.append("")

        # CV Summary
        if 'cv_summary' in cv_results:
            report_lines.append("## Cross-Validation Summary")
            cv_summary = cv_results['cv_summary']

            for metric, stats in cv_summary.items():
                report_lines.append(f"### {metric.upper()}")
                report_lines.append(f"- Mean: {stats['mean']:.4f}")
                report_lines.append(f"- Std: {stats['std']:.4f}")
                report_lines.append(f"- CV: {stats['cv']:.4f}")
                report_lines.append(f"- 95% CI: [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
                report_lines.append("")

        # Stability Analysis
        if 'stability_analysis' in cv_results:
            report_lines.append("## Stability Analysis")
            stability = cv_results['stability_analysis']

            if 'overall_stability' in stability:
                report_lines.append(f"- Overall Stability: {stability['overall_stability']}")
                report_lines.append(f"- Overall Stability Score: {stability['overall_stability_score']:.4f}")
                report_lines.append("")

            # Prediction consistency
            if 'prediction_consistency' in stability:
                pc = stability['prediction_consistency']
                report_lines.append("### Prediction Consistency")
                if 'mean_prediction_variance' in pc:
                    report_lines.append(f"- Mean Prediction Variance: {pc['mean_prediction_variance']:.4f}")
                    report_lines.append(f"- Max Prediction Variance: {pc['max_prediction_variance']:.4f}")
                elif 'mean_agreement_rate' in pc:
                    report_lines.append(f"- Mean Agreement Rate: {pc['mean_agreement_rate']:.4f}")
                    report_lines.append(f"- Min Agreement Rate: {pc['min_agreement_rate']:.4f}")
                report_lines.append("")

        # Feature Importance Stability
        if 'feature_importance_stability' in cv_results:
            fis = cv_results['feature_importance_stability']
            if 'overall_stability' in fis:
                report_lines.append("## Feature Importance Stability")
                os_data = fis['overall_stability']
                report_lines.append(f"- Mean CV: {os_data['mean_cv']:.4f}")
                report_lines.append(f"- Stable Features: {os_data['stable_features']}")
                report_lines.append(f"- Unstable Features: {os_data['unstable_features']}")
                report_lines.append(f"- Most Stable: {os_data['most_stable_feature']}")
                report_lines.append(f"- Least Stable: {os_data['least_stable_feature']}")
                report_lines.append("")

        report = "\n".join(report_lines)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            evaluation_logger.info(f"CV report saved to {save_path}")

        return report


def create_cv_engine(cv_folds: int = 5) -> CrossValidationEngine:
    """
    Factory function to create CrossValidationEngine instance.

    Args:
        cv_folds: Number of CV folds

    Returns:
        CrossValidationEngine instance
    """
    return CrossValidationEngine(cv_folds=cv_folds)