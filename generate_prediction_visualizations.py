#!/usr/bin/env python3
"""
Comprehensive Prediction and Feature Visualization Generator

This script creates detailed visualizations of model predictions, feature importance,
and prediction analysis for all trained models in the improved pipeline.

Author: Kilo Code
Date: 2025-11-04
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os
import gc
from datetime import datetime

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import (
    PREPROCESSED_DATA_FILE, TARGET_COLUMN, NUMERICAL_FEATURES,
    RANDOM_STATE, TEST_SIZE
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette('husl')

# Memory-safe sample size
SAMPLE_SIZE = 25000  # Balanced for visualization quality and memory

class PredictionVisualizer:
    """Generate comprehensive prediction and feature visualizations."""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.models = {}
        self.predictions = {}
        self.feature_names = None
        self.viz_dir = Path("visualizations/predictions")
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        """Load data with improved preprocessing."""
        print("📊 Loading and preparing data...")

        # Load in chunks for memory efficiency
        chunks = []
        chunk_size = 5000
        total_read = 0

        for chunk in pd.read_csv(PREPROCESSED_DATA_FILE, chunksize=chunk_size):
            if total_read >= SAMPLE_SIZE:
                break
            chunks.append(chunk)
            total_read += len(chunk)

        data = pd.concat(chunks, ignore_index=True)
        if len(data) > SAMPLE_SIZE:
            data = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

        print(f"✅ Loaded {len(data)} samples")

        # Target identification
        target_col = 'Total_Revenue'
        if target_col not in data.columns:
            target_col = TARGET_COLUMN

        # Get available features
        available_features = [f for f in NUMERICAL_FEATURES if f in data.columns]

        # Extract data
        X = data[available_features].copy()
        y = data[target_col].copy()

        # Data cleaning
        mask = ~(X.isna().any(axis=1) | y.isna() | np.isinf(y))
        X = X[mask]
        y = y[mask]

        # Outlier removal
        z_scores = np.abs((y - y.mean()) / y.std())
        mask = z_scores < 3
        X = X[mask]
        y = y[mask]

        print(f"✅ After cleaning: {len(X)} samples, {len(available_features)} features")

        # Feature engineering
        X = self._engineer_features(X)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        self.feature_names = self.X_train.columns.tolist()
        print(f"✅ Final features: {len(self.feature_names)}")

        del data, chunks, X, y
        gc.collect()

    def _engineer_features(self, X):
        """Add engineered features."""
        X_engineered = X.copy()

        # Interaction terms
        if 'Unit_Price' in X.columns and 'Order Quantity' in X.columns:
            X_engineered['Price_Quantity_Interaction'] = X['Unit_Price'] * X['Order Quantity']

        if 'Unit_Cost' in X.columns and 'Unit_Price' in X.columns:
            X_engineered['Profit_Per_Unit'] = X['Unit_Price'] - X['Unit_Cost']

        if 'Discount Applied' in X.columns and 'Unit_Price' in X.columns:
            X_engineered['Discounted_Price'] = X['Unit_Price'] * (1 - X['Discount Applied'])

        # Polynomial features for top features (use target if available, otherwise skip)
        if hasattr(self, 'y_train') and self.y_train is not None:
            selector = SelectKBest(f_regression, k=min(3, len(X.columns)))
            selector.fit(X, self.y_train)
            top_features = X.columns[selector.get_support(indices=True)].tolist()
        else:
            # Fallback: use first few features
            top_features = X.columns[:3].tolist()

        for feature in top_features[:2]:  # Limit to avoid memory issues
            X_engineered[f'{feature}_squared'] = X[feature] ** 2

        return X_engineered

    def train_models(self):
        """Train all available models."""
        print("\n🚀 Training models...")

        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1)': Ridge(alpha=1.0, random_state=RANDOM_STATE),
            'Ridge (α=10)': Ridge(alpha=10.0, random_state=RANDOM_STATE),
            'Lasso (α=1)': Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=2000),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000),
            'Decision Tree': DecisionTreeRegressor(max_depth=20, min_samples_split=10,
                                                   min_samples_leaf=5, random_state=RANDOM_STATE),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20,
                                                  min_samples_split=10, min_samples_leaf=5,
                                                  random_state=RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                          learning_rate=0.1, random_state=RANDOM_STATE)
        }

        for name, model in models_config.items():
            print(f"  ↳ Training {name}...")
            model.fit(self.X_train, self.y_train)

            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            # Calculate metrics
            metrics = {
                'train_r2': r2_score(self.y_train, y_pred_train),
                'test_r2': r2_score(self.y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'train_mae': mean_absolute_error(self.y_train, y_pred_train),
                'test_mae': mean_absolute_error(self.y_test, y_pred_test),
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test
                }
            }

            # Feature importance for tree models
            if hasattr(model, 'feature_importances_'):
                metrics['feature_importance'] = dict(zip(self.feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                metrics['feature_importance'] = dict(zip(self.feature_names, np.abs(coef)))

            self.models[name] = model
            self.predictions[name] = metrics

            print(f"     Test R²: {metrics['test_r2']:.4f}")
            del y_pred_train, y_pred_test
            gc.collect()

    def create_prediction_scatter_plots(self):
        """Create prediction vs actual scatter plots for all models."""
        print("\n📊 Creating prediction scatter plots...")

        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (name, metrics) in enumerate(self.predictions.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            y_pred = metrics['predictions']['test']
            r2 = metrics['test_r2']
            rmse = metrics['test_rmse']

            # Scatter plot
            ax.scatter(self.y_test, y_pred, alpha=0.6, s=10, color='steelblue')

            # Perfect prediction line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{name}\nR²: {r2:.4f}, RMSE: {rmse:.2f}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide empty subplots
        for idx in range(len(self.models), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'prediction_scatter_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved prediction_scatter_plots.png")

    def create_residual_plots(self):
        """Create residual analysis plots."""
        print("📊 Creating residual plots...")

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        model_names = list(self.predictions.keys())[:8]  # Limit to 8 models

        for idx, name in enumerate(model_names):
            row, col = idx // 4, idx % 4
            ax = axes[row, col]

            y_pred = self.predictions[name]['predictions']['test']
            residuals = self.y_test - y_pred

            ax.scatter(y_pred, residuals, alpha=0.6, s=8, color='darkred')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('.4f')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'residual_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved residual_analysis.png")

    def create_feature_importance_plots(self):
        """Create feature importance visualizations."""
        print("📊 Creating feature importance plots...")

        tree_models = [name for name in self.models.keys()
                      if 'Tree' in name or 'Forest' in name or 'Boosting' in name]

        if not tree_models:
            print("⚠️ No tree-based models found for feature importance")
            return

        n_models = len(tree_models)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, name in enumerate(tree_models):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            if 'feature_importance' in self.predictions[name]:
                importance_dict = self.predictions[name]['feature_importance']
                features = list(importance_dict.keys())
                importance = list(importance_dict.values())

                # Sort by importance
                sorted_idx = np.argsort(importance)[::-1]
                features_sorted = [features[i] for i in sorted_idx]
                importance_sorted = [importance[i] for i in sorted_idx]

                bars = ax.barh(range(len(features_sorted)), importance_sorted,
                              color='forestgreen', alpha=0.7)
                ax.set_yticks(range(len(features_sorted)))
                ax.set_yticklabels(features_sorted)
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{name} Feature Importance')

                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + max(importance_sorted) * 0.01, bar.get_y() + bar.get_height()/2,
                           '.3f', ha='left', va='center', fontsize=8)

        # Hide empty subplots
        for idx in range(len(tree_models), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved feature_importance_comparison.png")

    def create_prediction_error_distribution(self):
        """Create prediction error distribution histograms."""
        print("📊 Creating prediction error distributions...")

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        model_names = list(self.predictions.keys())[:8]

        for idx, name in enumerate(model_names):
            row, col = idx // 4, idx % 4
            ax = axes[row, col]

            y_pred = self.predictions[name]['predictions']['test']
            errors = self.y_test - y_pred

            ax.hist(errors, bins=30, alpha=0.7, color='salmon', edgecolor='black')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}\nMean Error: {errors.mean():.2f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'prediction_error_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved prediction_error_distributions.png")

    def create_model_performance_dashboard(self):
        """Create comprehensive model performance dashboard."""
        print("📊 Creating model performance dashboard...")

        # Extract metrics
        model_names = list(self.predictions.keys())
        test_r2 = [self.predictions[name]['test_r2'] for name in model_names]
        test_rmse = [self.predictions[name]['test_rmse'] for name in model_names]
        test_mae = [self.predictions[name]['test_mae'] for name in model_names]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # R² Comparison
        bars1 = axes[0, 0].barh(model_names, test_r2, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Test R² Score')
        axes[0, 0].set_title('Model Performance Comparison (R²)')
        axes[0, 0].axvline(x=0.8, color='red', linestyle='--', label='Good performance')
        axes[0, 0].legend()

        # Add value labels
        for bar, value in zip(bars1, test_r2):
            axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           '.3f', ha='left', va='center')

        # RMSE Comparison
        bars2 = axes[0, 1].barh(model_names, test_rmse, color='darkorange', alpha=0.7)
        axes[0, 1].set_xlabel('Test RMSE')
        axes[0, 1].set_title('Model Performance Comparison (RMSE)')
        axes[0, 1].axvline(x=test_rmse[np.argmin(test_r2)], color='green', linestyle='--',
                          label='Best model RMSE')
        axes[0, 1].legend()

        # MAE Comparison
        bars3 = axes[1, 0].barh(model_names, test_mae, color='darkgreen', alpha=0.7)
        axes[1, 0].set_xlabel('Test MAE')
        axes[1, 0].set_title('Model Performance Comparison (MAE)')

        # R² vs RMSE scatter
        axes[1, 1].scatter(test_rmse, test_r2, s=100, alpha=0.7, c=range(len(model_names)),
                          cmap='viridis')
        axes[1, 1].set_xlabel('Test RMSE')
        axes[1, 1].set_ylabel('Test R²')
        axes[1, 1].set_title('Performance Trade-off (R² vs RMSE)')
        axes[1, 1].grid(True, alpha=0.3)

        # Add model labels
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name.split('(')[0].strip(),
                               (test_rmse[i], test_r2[i]),
                               fontsize=8, alpha=0.8)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'model_performance_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved model_performance_dashboard.png")

    def create_feature_vs_prediction_plots(self):
        """Create scatter plots of features vs predictions for top features."""
        print("📊 Creating feature vs prediction plots...")

        # Get top features from Random Forest if available
        top_features = []
        for name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            if name in self.predictions and 'feature_importance' in self.predictions[name]:
                importance_dict = self.predictions[name]['feature_importance']
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_features = [f[0] for f in sorted_features[:4]]
                break

        if not top_features:
            top_features = self.feature_names[:4]

        n_features = len(top_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

        # Use Random Forest predictions for visualization
        if 'Random Forest' in self.predictions:
            y_pred = self.predictions['Random Forest']['predictions']['test']
        else:
            y_pred = self.predictions[list(self.predictions.keys())[0]]['predictions']['test']

        for idx, feature in enumerate(top_features):
            if feature not in self.X_test.columns:
                continue

            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            feature_values = self.X_test[feature].values

            ax.scatter(feature_values, y_pred, alpha=0.6, s=8, color='purple')
            ax.set_xlabel(f'Feature: {feature}')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{feature} vs Predictions')
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            corr = np.corrcoef(feature_values, y_pred)[0, 1]
            ax.text(0.05, 0.95, '.3f',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Hide empty subplots
        total_plots = n_rows * n_cols
        for idx in range(n_features, total_plots):
            row, col = idx // n_cols, idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_vs_prediction_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved feature_vs_prediction_scatter.png")

    def create_error_analysis_by_ranges(self):
        """Create error analysis by feature value ranges."""
        print("📊 Creating error analysis by feature ranges...")

        # Use Random Forest for error analysis
        if 'Random Forest' not in self.predictions:
            print("⚠️ Random Forest not available for error analysis")
            return

        y_pred = self.predictions['Random Forest']['predictions']['test']
        errors = np.abs(self.y_test - y_pred)

        # Get top 3 features
        importance_dict = self.predictions['Random Forest']['feature_importance']
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        top_feature_names = [f[0] for f in top_features]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, feature in enumerate(top_feature_names):
            ax = axes[idx]

            if feature not in self.X_test.columns:
                continue

            feature_values = self.X_test[feature].values

            # Create bins
            try:
                bins = pd.qcut(feature_values, q=5, duplicates='drop')
                bin_labels = [f'{b.left:.2f}-{b.right:.2f}' for b in bins.cat.categories]

                # Calculate mean error per bin
                bin_errors = []
                for bin_val in bins.cat.categories:
                    mask = bins == bin_val
                    mean_error = errors[mask].mean()
                    bin_errors.append(mean_error)

                ax.bar(range(len(bin_errors)), bin_errors, alpha=0.7, color='coral')
                ax.set_xlabel(f'{feature} Range')
                ax.set_ylabel('Mean Absolute Error')
                ax.set_title(f'Error Analysis by {feature}')
                ax.set_xticks(range(len(bin_labels)))
                ax.set_xticklabels(bin_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error analyzing {feature}\n{str(e)}',
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'Error Analysis by {feature} (Failed)')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'error_analysis_by_ranges.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved error_analysis_by_ranges.png")

    def create_prediction_summary_report(self):
        """Create a summary report of all generated visualizations."""
        print("📝 Creating prediction summary report...")

        report = f"""# Prediction and Feature Visualization Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** {len(self.X_train) + len(self.X_test)} samples ({len(self.X_train)} train, {len(self.X_test)} test)
**Features:** {len(self.feature_names)}
**Models Evaluated:** {len(self.models)}

## Generated Visualizations

### 1. Prediction Scatter Plots (`prediction_scatter_plots.png`)
- Scatter plots of predicted vs actual values for all {len(self.models)} models
- Includes perfect prediction reference line
- Shows model accuracy visually

### 2. Residual Analysis (`residual_analysis.png`)
- Residual plots showing prediction errors vs predicted values
- Helps identify heteroscedasticity and systematic errors
- Reference line at zero error

### 3. Feature Importance Comparison (`feature_importance_comparison.png`)
- Feature importance plots for tree-based models
- Shows which features contribute most to predictions
- Includes importance values

### 4. Prediction Error Distributions (`prediction_error_distributions.png`)
- Histograms of prediction errors for each model
- Shows error distribution and central tendency
- Reference line at zero error

### 5. Model Performance Dashboard (`model_performance_dashboard.png`)
- Comprehensive dashboard with:
  - R² score comparison
  - RMSE comparison
  - MAE comparison
  - Performance trade-off scatter plot

### 6. Feature vs Prediction Scatter (`feature_vs_prediction_scatter.png`)
- Scatter plots of top features vs model predictions
- Shows relationship between features and predictions
- Includes correlation coefficients

### 7. Error Analysis by Ranges (`error_analysis_by_ranges.png`)
- Error analysis across different feature value ranges
- Helps identify where models perform poorly
- Shows mean absolute error by feature bins

## Model Performance Summary

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
"""

        for name, metrics in self.predictions.items():
            report += f"| {name} | {metrics['test_r2']:.4f} | {metrics['test_rmse']:.2f} | {metrics['test_mae']:.2f} |\n"

        # Find best model
        best_model = max(self.predictions.items(), key=lambda x: x[1]['test_r2'])

        report += f"""

## Best Performing Model

**🏆 Winner:** {best_model[0]}
- **Test R²:** {best_model[1]['test_r2']:.4f}
- **Test RMSE:** {best_model[1]['test_rmse']:.2f}
- **Test MAE:** {best_model[1]['test_mae']:.2f}

## Key Insights

1. **Best Model Performance:** {best_model[0]} shows the highest prediction accuracy
2. **Feature Importance:** Top features driving predictions (from tree models)
3. **Error Patterns:** Models tend to perform better/worse in certain feature ranges
4. **Prediction Reliability:** Error distributions show model consistency

## Usage Recommendations

- Use {best_model[0]} for production predictions
- Monitor prediction errors in identified problematic ranges
- Consider feature engineering for underperforming feature ranges
- Regular model retraining with new data

---

*Generated automatically by PredictionVisualizer*
*All visualizations saved in `visualizations/predictions/` directory*
"""

        with open(self.viz_dir / 'prediction_visualization_summary.md', 'w') as f:
            f.write(report)

        print("✅ Saved prediction_visualization_summary.md")

    def run_complete_analysis(self):
        """Run the complete prediction visualization pipeline."""
        print("=" * 70)
        print("🎨 PREDICTION AND FEATURE VISUALIZATION GENERATOR")
        print("=" * 70)

        try:
            # Load and prepare data
            self.load_and_prepare_data()

            # Train models
            self.train_models()

            # Generate all visualizations
            self.create_prediction_scatter_plots()
            self.create_residual_plots()
            self.create_feature_importance_plots()
            self.create_prediction_error_distribution()
            self.create_model_performance_dashboard()
            self.create_feature_vs_prediction_plots()
            self.create_error_analysis_by_ranges()

            # Create summary report
            self.create_prediction_summary_report()

            print("\n" + "=" * 70)
            print("✅ VISUALIZATION GENERATION COMPLETED!")
            print("=" * 70)
            print(f"\n📂 Generated {len(list(self.viz_dir.glob('*.png')))} visualizations in {self.viz_dir}/")
            print("📊 Visualizations created:")
            for viz_file in sorted(self.viz_dir.glob('*.png')):
                print(f"   - {viz_file.name}")
            print("📝 Summary report: prediction_visualization_summary.md")

            # Show best model
            best_model = max(self.predictions.items(), key=lambda x: x[1]['test_r2'])
            print(f"\n🏆 Best Model: {best_model[0]}")
            print(f"   Test R²: {best_model[1]['test_r2']:.4f}")
            return 0

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1

def main():
    """Main execution."""
    visualizer = PredictionVisualizer()
    return visualizer.run_complete_analysis()

if __name__ == "__main__":
    sys.exit(main())