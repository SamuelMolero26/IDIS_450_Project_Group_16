"""
Qualitative evaluation module with SHAP interpretability, error analysis, and business alignment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
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
    from src.config import (
        TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
        BUSINESS_RULES, REPORTS_DIR, SHAP_MAX_EVALS, SHAP_SAMPLE_SIZE
    )
    from src.logger import evaluation_logger
except ImportError as e:
    print(f"Import error in qualitative_evaluator.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise


class QualitativeEvaluator:
    """
    Qualitative evaluation with interpretability, error analysis, and business alignment.
    """

    def __init__(self):
        self.shap_available = self._check_shap_availability()

    def _check_shap_availability(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap
            return True
        except ImportError:
            evaluation_logger.warning("SHAP not available. Interpretability features will be limited.")
            return False

    def perform_shap_analysis(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                             model_name: str = "model", save_plots: bool = True) -> Dict[str, Any]:
        """
        Perform SHAP analysis for model interpretability.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            model_name: Name for the model
            save_plots: Whether to save plots

        Returns:
            Dictionary with SHAP analysis results
        """
        if not self.shap_available:
            evaluation_logger.warning("SHAP not available, skipping SHAP analysis")
            return {'error': 'SHAP not available'}

        evaluation_logger.info(f"Performing SHAP analysis for {model_name}")

        try:
            import shap

            # Sample data for SHAP (to avoid computational issues with large datasets)
            X_sample = X_test.sample(min(SHAP_SAMPLE_SIZE, len(X_test)),
                                   random_state=42)

            # Choose explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X_train)
            else:
                explainer = shap.LinearExplainer(model, X_train)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values_main = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_main = shap_values

            # Calculate feature importance from SHAP
            feature_importance = np.abs(shap_values_main).mean(axis=0)
            shap_importance = dict(zip(X_sample.columns, feature_importance))

            # Calculate SHAP summary statistics
            shap_stats = {
                'mean_abs_shap': feature_importance.tolist(),
                'feature_names': X_sample.columns.tolist(),
                'shap_values_sample': shap_values_main[:100].tolist() if len(shap_values_main) > 100 else shap_values_main.tolist(),
                'X_sample': X_sample.iloc[:100].to_dict('records') if len(X_sample) > 100 else X_sample.to_dict('records')
            }

            # Generate SHAP summary plot
            if save_plots:
                try:
                    fig = plot_shap_summary(
                        shap_values_main, X_sample.columns.tolist(),
                        X_sample, model_name,
                        save_path=REPORTS_DIR / f"{model_name}_shap_summary.png"
                    )
                except Exception as e:
                    evaluation_logger.warning(f"Failed to generate SHAP plot: {e}")

            results = {
                'shap_available': True,
                'feature_importance': shap_importance,
                'shap_statistics': shap_stats,
                'analysis_timestamp': datetime.now().isoformat()
            }

            evaluation_logger.info(f"SHAP analysis completed for {model_name}")

        except Exception as e:
            evaluation_logger.error(f"SHAP analysis failed: {e}")
            results = {
                'shap_available': False,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }

        return results

    def perform_error_analysis(self, model, X_test: pd.DataFrame, y_test: np.ndarray,
                             y_pred: np.ndarray, model_name: str = "model") -> Dict[str, Any]:
        """
        Perform detailed error analysis.

        Args:
            model: Trained model
            X_test: Test features
            y_test: True test targets
            y_pred: Predicted test targets
            model_name: Name for the model

        Returns:
            Dictionary with error analysis results
        """
        evaluation_logger.info(f"Performing error analysis for {model_name}")

        # Calculate residuals/errors
        if len(y_test.shape) > 1:  # Classification
            errors = (y_test != y_pred).astype(int)
            error_magnitude = errors  # Binary: 0 or 1
        else:  # Regression
            errors = y_test - y_pred
            error_magnitude = np.abs(errors)

        # Create error analysis dataframe
        error_df = X_test.copy()
        error_df['true_value'] = y_test
        error_df['predicted_value'] = y_pred
        error_df['error'] = errors
        error_df['error_magnitude'] = error_magnitude
        error_df['is_error'] = (error_magnitude > 0).astype(int)

        # Error statistics
        error_stats = {
            'total_samples': len(error_df),
            'error_rate': error_df['is_error'].mean(),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'max_error': float(np.max(np.abs(errors))),
            'min_error': float(np.min(np.abs(errors)))
        }

        # Feature correlation with errors
        numeric_cols = error_df.select_dtypes(include=[np.number]).columns
        error_correlations = {}

        for col in numeric_cols:
            if col not in ['error', 'error_magnitude', 'is_error', 'true_value', 'predicted_value']:
                corr = error_df[col].corr(error_df['error_magnitude'])
                if not np.isnan(corr):
                    error_correlations[col] = float(corr)

        # Sort by absolute correlation
        error_correlations = dict(sorted(error_correlations.items(),
                                       key=lambda x: abs(x[1]), reverse=True))

        # Error patterns by feature ranges
        error_patterns = self._analyze_error_patterns(error_df)

        # Worst prediction analysis
        worst_predictions = self._analyze_worst_predictions(error_df)

        results = {
            'error_statistics': error_stats,
            'error_correlations': error_correlations,
            'error_patterns': error_patterns,
            'worst_predictions': worst_predictions,
            'error_samples': error_df.nlargest(10, 'error_magnitude').to_dict('records'),
            'analysis_timestamp': datetime.now().isoformat()
        }

        evaluation_logger.info(f"Error analysis completed for {model_name}. Error rate: {error_stats['error_rate']:.4f}")

        return results

    def _analyze_error_patterns(self, error_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze error patterns across different feature ranges.

        Args:
            error_df: DataFrame with errors

        Returns:
            Dictionary with error pattern analysis
        """
        patterns = {}

        # Analyze numeric features
        numeric_cols = error_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['error', 'error_magnitude', 'is_error', 'true_value', 'predicted_value']]

        for col in numeric_cols[:5]:  # Limit to top 5 features
            try:
                # Create bins
                error_df[f'{col}_bin'] = pd.qcut(error_df[col], q=4, duplicates='drop')

                # Error rate by bin
                bin_errors = error_df.groupby(f'{col}_bin')['is_error'].mean()
                patterns[col] = {
                    'error_rate_by_bin': bin_errors.to_dict(),
                    'bin_counts': error_df.groupby(f'{col}_bin').size().to_dict()
                }
            except Exception as e:
                evaluation_logger.warning(f"Could not analyze error patterns for {col}: {e}")

        return patterns

    def _analyze_worst_predictions(self, error_df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
        """
        Analyze the worst predictions.

        Args:
            error_df: DataFrame with errors
            top_n: Number of worst predictions to analyze

        Returns:
            Dictionary with worst prediction analysis
        """
        worst_predictions = error_df.nlargest(top_n, 'error_magnitude')

        analysis = {
            'worst_predictions': worst_predictions.to_dict('records'),
            'common_characteristics': {}
        }

        # Analyze categorical features in worst predictions
        cat_cols = worst_predictions.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            if col in worst_predictions.columns:
                value_counts = worst_predictions[col].value_counts()
                analysis['common_characteristics'][col] = value_counts.to_dict()

        # Analyze numeric feature ranges in worst predictions
        numeric_cols = worst_predictions.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['error', 'error_magnitude', 'is_error', 'true_value', 'predicted_value']]

        for col in numeric_cols[:3]:  # Top 3 numeric features
            analysis['common_characteristics'][f'{col}_range'] = {
                'min': worst_predictions[col].min(),
                'max': worst_predictions[col].max(),
                'mean': worst_predictions[col].mean()
            }

        return analysis

    def check_business_alignment(self, model, X_test: pd.DataFrame, y_test: np.ndarray,
                               y_pred: np.ndarray, model_name: str = "model") -> Dict[str, Any]:
        """
        Check business alignment of model predictions.

        Args:
            model: Trained model
            X_test: Test features
            y_test: True test targets
            y_pred: Predicted test targets
            model_name: Name for the model

        Returns:
            Dictionary with business alignment results
        """
        evaluation_logger.info(f"Checking business alignment for {model_name}")

        alignment_checks = {}

        # Business rule 1: Profit margin constraints
        if 'Profit_Margin' in X_test.columns:
            profit_margin_threshold = BUSINESS_RULES['profit_margin_threshold']
            low_margin_predictions = X_test['Profit_Margin'] < profit_margin_threshold

            alignment_checks['profit_margin'] = {
                'threshold': profit_margin_threshold,
                'low_margin_cases': low_margin_predictions.sum(),
                'low_margin_percentage': low_margin_predictions.mean() * 100,
                'error_rate_in_low_margin': ((y_test - y_pred) != 0)[low_margin_predictions].mean() if low_margin_predictions.any() else 0
            }

        # Business rule 2: Lead time constraints
        if 'Total_Lead_Time' in X_test.columns:
            lead_time_max = BUSINESS_RULES['lead_time_max']
            long_lead_times = X_test['Total_Lead_Time'] > lead_time_max

            alignment_checks['lead_time'] = {
                'max_threshold': lead_time_max,
                'long_lead_cases': long_lead_times.sum(),
                'long_lead_percentage': long_lead_times.mean() * 100,
                'avg_lead_time_long_cases': X_test.loc[long_lead_times, 'Total_Lead_Time'].mean() if long_lead_times.any() else 0
            }

        # Business rule 3: Discount constraints
        if 'Discount Applied' in X_test.columns:
            discount_max = BUSINESS_RULES['discount_max']
            high_discounts = X_test['Discount Applied'] > discount_max

            alignment_checks['discount'] = {
                'max_threshold': discount_max,
                'high_discount_cases': high_discounts.sum(),
                'high_discount_percentage': high_discounts.mean() * 100,
                'avg_discount_high_cases': X_test.loc[high_discounts, 'Discount Applied'].mean() if high_discounts.any() else 0
            }

        # Business rule 4: High-value order analysis
        if TARGET_COLUMN in X_test.columns or 'Total_Revenue' in X_test.columns:
            revenue_col = 'Total_Revenue' if 'Total_Revenue' in X_test.columns else TARGET_COLUMN
            high_value_threshold = BUSINESS_RULES['high_value_order_threshold']
            high_value_orders = X_test[revenue_col] > high_value_threshold

            alignment_checks['high_value_orders'] = {
                'threshold': high_value_threshold,
                'high_value_cases': high_value_orders.sum(),
                'high_value_percentage': high_value_orders.mean() * 100,
                'error_rate_high_value': ((y_test - y_pred) != 0)[high_value_orders].mean() if high_value_orders.any() else 0
            }

        # Overall business alignment score
        business_score = self._calculate_business_alignment_score(alignment_checks)

        results = {
            'business_checks': alignment_checks,
            'business_alignment_score': business_score,
            'recommendations': self._generate_business_recommendations(alignment_checks),
            'analysis_timestamp': datetime.now().isoformat()
        }

        evaluation_logger.info(f"Business alignment check completed for {model_name}. Score: {business_score:.2f}")

        return results

    def _calculate_business_alignment_score(self, alignment_checks: Dict[str, Any]) -> float:
        """
        Calculate overall business alignment score.

        Args:
            alignment_checks: Results from business rule checks

        Returns:
            Business alignment score (0-100)
        """
        scores = []

        for check_name, check_data in alignment_checks.items():
            if check_name == 'profit_margin':
                # Lower error rate in low margin cases is better
                error_rate = check_data.get('error_rate_in_low_margin', 0)
                score = max(0, 100 - (error_rate * 100))
                scores.append(score)

            elif check_name == 'high_value_orders':
                # Lower error rate in high-value orders is better
                error_rate = check_data.get('error_rate_high_value', 0)
                score = max(0, 100 - (error_rate * 100))
                scores.append(score)

            elif check_name in ['lead_time', 'discount']:
                # Lower percentage of problematic cases is better
                percentage = check_data.get(f'{check_name.split("_")[0]}_percentage', 0)
                score = max(0, 100 - percentage)
                scores.append(score)

        return np.mean(scores) if scores else 50.0

    def _generate_business_recommendations(self, alignment_checks: Dict[str, Any]) -> List[str]:
        """
        Generate business recommendations based on alignment checks.

        Args:
            alignment_checks: Results from business rule checks

        Returns:
            List of recommendations
        """
        recommendations = []

        for check_name, check_data in alignment_checks.items():
            if check_name == 'profit_margin':
                error_rate = check_data.get('error_rate_in_low_margin', 0)
                if error_rate > 0.3:
                    recommendations.append("High error rate in low-profit margin cases. Consider additional features or model adjustments for profitability analysis.")

            elif check_name == 'lead_time':
                percentage = check_data.get('long_lead_percentage', 0)
                if percentage > 20:
                    recommendations.append(f"{percentage:.1f}% of cases exceed lead time threshold. Review supply chain processes.")

            elif check_name == 'discount':
                percentage = check_data.get('high_discount_percentage', 0)
                if percentage > 15:
                    recommendations.append(f"{percentage:.1f}% of cases have high discounts. Evaluate discount strategy effectiveness.")

            elif check_name == 'high_value_orders':
                error_rate = check_data.get('error_rate_high_value', 0)
                if error_rate > 0.2:
                    recommendations.append("High error rate in valuable orders. Focus model improvements on high-value predictions.")

        if not recommendations:
            recommendations.append("Model shows good business alignment across all checked criteria.")

        return recommendations

    def generate_qualitative_report(self, qualitative_results: Dict[str, Any],
                                  save_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive qualitative evaluation report.

        Args:
            qualitative_results: Results from qualitative evaluation
            save_path: Optional path to save the report

        Returns:
            Report as formatted string
        """
        report_lines = []
        report_lines.append("# Qualitative Evaluation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        for model_name, results in qualitative_results.items():
            report_lines.append(f"## {model_name}")
            report_lines.append("")

            # SHAP Analysis
            if 'shap_available' in results and results['shap_available']:
                report_lines.append("### SHAP Interpretability")
                shap_results = results.get('shap_statistics', {})
                if 'feature_importance' in results:
                    top_features = sorted(results['feature_importance'].items(),
                                        key=lambda x: x[1], reverse=True)[:5]
                    report_lines.append("Top 5 Most Important Features (SHAP):")
                    for feature, importance in top_features:
                        report_lines.append(f"- {feature}: {importance:.4f}")
                report_lines.append("")

            # Error Analysis
            if 'error_statistics' in results:
                error_stats = results['error_statistics']
                report_lines.append("### Error Analysis")
                report_lines.append(f"- Total Samples: {error_stats['total_samples']}")
                report_lines.append(f"- Error Rate: {error_stats['error_rate']:.2f}")
                report_lines.append(f"- Mean Error: {error_stats['mean_error']:.4f}")
                report_lines.append(f"- Max Error: {error_stats['max_error']:.4f}")
                report_lines.append("")

                # Error correlations
                if 'error_correlations' in results:
                    top_correlations = list(results['error_correlations'].items())[:3]
                    if top_correlations:
                        report_lines.append("Features Most Correlated with Errors:")
                        for feature, corr in top_correlations:
                            report_lines.append(f"- {feature}: {corr:.4f}")
                        report_lines.append("")

            # Business Alignment
            if 'business_checks' in results:
                business_results = results['business_checks']
                report_lines.append("### Business Alignment")
                report_lines.append(f"- Overall Business Score: {results.get('business_alignment_score', 'N/A'):.2f}/100")
                report_lines.append("")

                for check_name, check_data in business_results.items():
                    if check_name == 'profit_margin':
                        report_lines.append(f"- Low Profit Margin Cases: {check_data['low_margin_cases']} ({check_data['low_margin_percentage']:.1f}%)")
                    elif check_name == 'lead_time':
                        report_lines.append(f"- Long Lead Time Cases: {check_data['long_lead_cases']} ({check_data['long_lead_percentage']:.1f}%)")
                    elif check_name == 'discount':
                        report_lines.append(f"- High Discount Cases: {check_data['high_discount_cases']} ({check_data['high_discount_percentage']:.1f}%)")
                    elif check_name == 'high_value_orders':
                        report_lines.append(f"- High-Value Orders: {check_data['high_value_cases']} ({check_data['high_value_percentage']:.1f}%)")
                report_lines.append("")

                # Recommendations
                if 'recommendations' in results:
                    report_lines.append("### Recommendations")
                    for rec in results['recommendations']:
                        report_lines.append(f"- {rec}")
                    report_lines.append("")

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            evaluation_logger.info(f"Qualitative report saved to {save_path}")

        return report

def create_qualitative_evaluator() -> QualitativeEvaluator:
    """
    Factory function to create QualitativeEvaluator instance.

    Returns:
        QualitativeEvaluator instance
    """
    return QualitativeEvaluator()