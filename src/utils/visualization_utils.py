"""
Visualization utilities for the advanced modeling pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import REPORTS_DIR
    from src.logger import evaluation_logger
except ImportError as e:
    print(f"Import error in visualization_utils.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                    save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot comprehensive residual diagnostics for regression model evaluation.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    residuals = y_true - y_pred
    std_residuals = residuals / np.std(residuals)

    # Check for skewness and use appropriate scales
    skewness = pd.Series(residuals).skew()
    use_log_scale = abs(skewness) > 1.5  # Use log scale for highly skewed data

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)

    # Use log scale for y-axis if residuals are skewed
    if use_log_scale and np.all(residuals > 0):
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_ylabel('Residuals (log scale)')

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')

    # Scale-Location plot (Sqrt(|Standardized Residuals|) vs Fitted)
    sqrt_std_residuals = np.sqrt(np.abs(std_residuals))
    axes[0, 2].scatter(y_pred, sqrt_std_residuals, alpha=0.6, color='green')
    axes[0, 2].set_xlabel('Fitted Values')
    axes[0, 2].set_ylabel('√|Standardized Residuals|')
    axes[0, 2].set_title('Scale-Location Plot')
    axes[0, 2].grid(True, alpha=0.3)

    # Residuals histogram with normal curve
    if use_log_scale and np.all(residuals > 0):
        # Use log-spaced bins for skewed data
        bins = np.logspace(np.log10(max(residuals.min(), 1e-10)), np.log10(residuals.max()), 30)
        n, bins, patches = axes[1, 0].hist(residuals, bins=bins, alpha=0.7, edgecolor='black', density=True)
        axes[1, 0].set_xscale('log')
    else:
        n, bins, patches = axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black', density=True)

    # Add normal distribution curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    axes[1, 0].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Distribution')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].legend()

    # Standardized Residuals vs Leverage (Cook's distance approximation)
    try:
        from statsmodels.api import OLS, add_constant
        from statsmodels.stats.outliers_influence import OLSInfluence

        # Create design matrix with constant
        X_design = add_constant(np.column_stack([np.ones(len(y_pred)), y_pred]))

        # Fit simple OLS for influence analysis
        influence_model = OLS(residuals, X_design).fit()
        influence = OLSInfluence(influence_model)

        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]

        axes[1, 1].scatter(leverage, std_residuals, alpha=0.6, color='purple')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='±2 SD')
        axes[1, 1].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Leverage')
        axes[1, 1].set_ylabel('Standardized Residuals')
        axes[1, 1].set_title('Residuals vs Leverage')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Cook's Distance plot
        axes[1, 2].stem(range(len(cooks_d)), cooks_d, markerfmt=' ', basefmt=' ')
        axes[1, 2].axhline(y=4/len(cooks_d), color='red', linestyle='--', label='Threshold')
        axes[1, 2].set_xlabel('Observation Index')
        axes[1, 2].set_ylabel("Cook's Distance")
        axes[1, 2].set_title("Cook's Distance")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    except ImportError:
        # Fallback plots if statsmodels not available
        axes[1, 1].scatter(range(len(std_residuals)), std_residuals, alpha=0.6, color='purple')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='±2 SD')
        axes[1, 1].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Observation Index')
        axes[1, 1].set_ylabel('Standardized Residuals')
        axes[1, 1].set_title('Standardized Residuals')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Residuals vs Order
        axes[1, 2].scatter(range(len(residuals)), residuals, alpha=0.6, color='orange')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 2].set_xlabel('Observation Order')
        axes[1, 2].set_ylabel('Residuals')
        axes[1, 2].set_title('Residuals vs Order')
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Comprehensive Residual Diagnostics - {model_name}', fontsize=16, fontweight='bold')

    # Add skewness information to title if skewed
    if use_log_scale:
        plt.suptitle(f'Comprehensive Residual Diagnostics - {model_name}\n(Skewed data - log scale used)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        evaluation_logger.info(f"Saved comprehensive residual diagnostics plot to {save_path}")

    return fig

def plot_feature_importance(feature_names: List[str], importance_values: np.ndarray,
                           model_name: str, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot feature importance for tree-based models.

    Args:
        feature_names: List of feature names
        importance_values: Feature importance values
        model_name: Name of the model
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1]
    features_sorted = [feature_names[i] for i in indices]
    importance_sorted = importance_values[indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(features_sorted)), importance_sorted)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance - {model_name}')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(importance_sorted) * 0.01, bar.get_y() + bar.get_height()/2,
                '.3f', ha='left', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        evaluation_logger.info(f"Saved feature importance plot to {save_path}")

    return fig

def plot_bias_variance_tradeoff(bias_scores: List[float], variance_scores: List[float],
                               model_complexities: List[float], model_name: str,
                               save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot bias-variance tradeoff.

    Args:
        bias_scores: Bias scores for different model complexities
        variance_scores: Variance scores for different model complexities
        model_complexities: Model complexity values
        model_name: Name of the model
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(model_complexities, bias_scores, 'b-', label='Bias²', marker='o')
    ax.plot(model_complexities, variance_scores, 'r-', label='Variance', marker='s')
    ax.plot(model_complexities, np.array(bias_scores) + np.array(variance_scores),
            'g-', label='Total Error', marker='^')

    ax.set_xlabel('Model Complexity')
    ax.set_ylabel('Error')
    ax.set_title(f'Bias-Variance Tradeoff - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        evaluation_logger.info(f"Saved bias-variance plot to {save_path}")

    return fig

def create_model_comparison_plot(results_dict: Dict[str, Dict[str, float]],
                                metric: str = 'r2', save_path: Optional[Path] = None) -> go.Figure:
    """
    Create interactive model comparison plot using Plotly.

    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        metric: Metric to compare (default: 'r2')
        save_path: Optional path to save the plot

    Returns:
        Plotly figure object
    """
    models = list(results_dict.keys())
    values = [results_dict[model].get(metric, 0) for model in models]

    fig = go.Figure(data=[
        go.Bar(x=models, y=values, marker_color='lightblue')
    ])

    fig.update_layout(
        title=f'Model Comparison - {metric.upper()}',
        xaxis_title='Models',
        yaxis_title=metric.upper(),
        template='plotly_white'
    )

    if save_path:
        fig.write_html(save_path)
        evaluation_logger.info(f"Saved model comparison plot to {save_path}")

    return fig

def plot_shap_summary(shap_values: np.ndarray, feature_names: List[str],
                     X_sample: pd.DataFrame, model_name: str,
                     save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot SHAP summary plot.

    Args:
        shap_values: SHAP values
        feature_names: Feature names
        X_sample: Sample of input data
        model_name: Name of the model
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    try:
        import shap

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create summary plot
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         show=False, ax=ax)
        ax.set_title(f'SHAP Summary Plot - {model_name}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            evaluation_logger.info(f"Saved SHAP summary plot to {save_path}")

        return fig

    except ImportError:
        evaluation_logger.warning("SHAP not available for plotting")
        return None

def plot_learning_curve(train_scores: np.ndarray, val_scores: np.ndarray,
                        train_sizes: np.ndarray, model_name: str,
                        save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot learning curve with overfitting indicators.

    Args:
        train_scores: Training scores for different training sizes
        val_scores: Validation scores for different training sizes
        train_sizes: Training set sizes
        model_name: Name of the model
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                   alpha=0.1, color='blue')

    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                   alpha=0.1, color='red')

    # Add overfitting indicators
    gap = train_mean - val_mean
    max_gap_idx = np.argmax(gap)
    if gap[max_gap_idx] > 0.1:  # Significant gap indicating overfitting
        ax.axvline(x=train_sizes[max_gap_idx], color='orange', linestyle='--',
                  alpha=0.7, label=f'Potential Overfitting (gap={gap[max_gap_idx]:.3f})')
        ax.scatter(train_sizes[max_gap_idx], val_mean[max_gap_idx],
                  color='orange', s=100, zorder=5)

    # Add convergence indicator
    if len(train_sizes) > 3:
        # Check if validation score is still improving
        recent_improvement = val_mean[-1] - val_mean[-3] if len(val_mean) >= 3 else 0
        if abs(recent_improvement) < 0.01:  # Minimal improvement
            ax.axhline(y=val_mean[-1], color='green', linestyle=':', alpha=0.7,
                      label='Possible Convergence')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(f'Learning Curve - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        evaluation_logger.info(f"Saved learning curve plot to {save_path}")

    return fig

def plot_logistic_residuals(model, X_test: pd.DataFrame, y_test: np.ndarray,
                           model_name: str, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot comprehensive residual diagnostics for logistic regression models.

    Args:
        model: Trained logistic regression model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    try:
        from statsmodels.api import GLM, families
        from statsmodels.genmod.generalized_linear_model import GLMResults
        import statsmodels.api as sm

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calculate deviance residuals
        deviance_residuals = []
        pearson_residuals = []

        for i in range(len(y_test)):
            if y_test[i] == 1:
                deviance_residuals.append(np.sqrt(-2 * np.log(1 - y_pred_proba[i])))
                pearson_residuals.append((1 - y_pred_proba[i]) / np.sqrt(y_pred_proba[i] * (1 - y_pred_proba[i])))
            else:
                deviance_residuals.append(-np.sqrt(-2 * np.log(y_pred_proba[i])))
                pearson_residuals.append(-(y_pred_proba[i]) / np.sqrt(y_pred_proba[i] * (1 - y_pred_proba[i])))

        deviance_residuals = np.array(deviance_residuals)
        pearson_residuals = np.array(pearson_residuals)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Deviance Residuals vs Linear Predictor
        linear_pred = model.decision_function(X_test)
        axes[0, 0].scatter(linear_pred, deviance_residuals, alpha=0.6, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Linear Predictor')
        axes[0, 0].set_ylabel('Deviance Residuals')
        axes[0, 0].set_title('Deviance Residuals vs Linear Predictor')
        axes[0, 0].grid(True, alpha=0.3)

        # Pearson Residuals vs Linear Predictor
        axes[0, 1].scatter(linear_pred, pearson_residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Linear Predictor')
        axes[0, 1].set_ylabel('Pearson Residuals')
        axes[0, 1].set_title('Pearson Residuals vs Linear Predictor')
        axes[0, 1].grid(True, alpha=0.3)

        # Deviance Residuals Q-Q plot
        from scipy import stats
        stats.probplot(deviance_residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Deviance Residuals Q-Q Plot')

        # Residuals histogram
        axes[1, 0].hist(deviance_residuals, bins=30, alpha=0.7, edgecolor='black', density=True, label='Deviance')
        axes[1, 0].hist(pearson_residuals, bins=30, alpha=0.5, edgecolor='black', density=True, label='Pearson')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].legend()

        # Cook's Distance for Logistic Regression
        try:
            # Fit GLM for influence analysis
            X_with_const = sm.add_constant(X_test)
            glm_model = GLM(y_test, X_with_const, family=families.Binomial()).fit()

            # Get influence measures
            influence = glm_model.get_influence()
            cooks_d = influence.cooks_distance[0]
            leverage = influence.hat_matrix_diag

            axes[1, 1].scatter(leverage, deviance_residuals, alpha=0.6, color='purple')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Leverage')
            axes[1, 1].set_ylabel('Deviance Residuals')
            axes[1, 1].set_title('Deviance Residuals vs Leverage')
            axes[1, 1].grid(True, alpha=0.3)

            # Cook's Distance plot
            axes[1, 2].stem(range(len(cooks_d)), cooks_d, markerfmt=' ', basefmt=' ')
            axes[1, 2].axhline(y=4/len(cooks_d), color='red', linestyle='--', label='Threshold')
            axes[1, 2].set_xlabel('Observation Index')
            axes[1, 2].set_ylabel("Cook's Distance")
            axes[1, 2].set_title("Cook's Distance")
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        except:
            # Fallback plots
            axes[1, 1].scatter(range(len(deviance_residuals)), deviance_residuals, alpha=0.6, color='purple')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Observation Index')
            axes[1, 1].set_ylabel('Deviance Residuals')
            axes[1, 1].set_title('Deviance Residuals vs Index')
            axes[1, 1].grid(True, alpha=0.3)

            # Residuals vs Order
            axes[1, 2].scatter(range(len(deviance_residuals)), deviance_residuals, alpha=0.6, color='orange')
            axes[1, 2].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 2].set_xlabel('Observation Order')
            axes[1, 2].set_ylabel('Deviance Residuals')
            axes[1, 2].set_title('Deviance Residuals vs Order')
            axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'Logistic Regression Residual Diagnostics - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            evaluation_logger.info(f"Saved logistic regression residual diagnostics plot to {save_path}")

        return fig

    except ImportError:
        evaluation_logger.warning("statsmodels not available for logistic regression diagnostics")
        # Fallback to basic plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Basic residual plot (difference from 0.5 for binary classification)
        binary_residuals = y_test - y_pred_proba
        axes[0].scatter(y_pred_proba, binary_residuals, alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Residuals (Observed - Predicted)')
        axes[0].set_title('Basic Residuals vs Predicted Probability')

        # Confusion matrix style residual plot
        correct_predictions = (y_pred == y_test)
        axes[1].scatter(range(len(y_test)), binary_residuals, c=correct_predictions,
                       cmap='RdYlGn', alpha=0.6)
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_xlabel('Observation Index')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals by Prediction Accuracy')

        plt.suptitle(f'Basic Logistic Regression Residuals - {model_name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            evaluation_logger.info(f"Saved basic logistic regression residual plot to {save_path}")

        return fig

def save_evaluation_report(results: Dict[str, Any], save_path: Path):
    """
    Save evaluation results as JSON report.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save the report
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    evaluation_logger.info(f"Saved evaluation report to {save_path}")