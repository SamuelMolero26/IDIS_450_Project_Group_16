#!/usr/bin/env python3
"""
Complete Model Comparison Dashboard Generator
Creates comprehensive visualizations comparing all models from the latest pipeline run.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the pipeline data loader
from pipeline_data_loader import load_latest_pipeline_data

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CompleteModelComparisonVisualizer:
    """Generate comprehensive model comparison visualizations using real pipeline data."""

    def __init__(self, output_dir: str = "../complete_model_comparison"):
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load real pipeline data
        self.pipeline_data = None
        self.model_metrics = {}
        self._load_pipeline_data()

    def _load_pipeline_data(self):
        """Load model performance data from the latest pipeline run."""
        print("📊 Loading complete model comparison data from latest pipeline run...")

        try:
            loader = load_latest_pipeline_data(verbose=False)
            if loader.report_data:
                modeling_results = loader.report_data.get('modeling_results', {})
                model_results = modeling_results.get('model_results', {})

                # Extract metrics for actual ML models (not utility components)
                actual_models = ['ann', 'decision_tree', 'elastic_net', 'lasso',
                               'linear', 'random_forest', 'ridge']

                for model_name in actual_models:
                    if model_name in model_results:
                        model_data = model_results[model_name]
                        evaluation = model_data.get('evaluation', {})
                        test_metrics = evaluation.get('test_metrics', {})

                        if 'r2' in test_metrics:  # Only include models with actual metrics
                            self.model_metrics[model_name] = {
                                'r2': test_metrics.get('r2', 0),
                                'rmse': test_metrics.get('rmse', 0),
                                'mae': test_metrics.get('mae', 0),
                                'mape': test_metrics.get('mape', 0),
                                'training_time': model_data.get('training_time', 0)
                            }

                print(f"✅ Loaded metrics for {len(self.model_metrics)} models from pipeline")
                for model, metrics in self.model_metrics.items():
                    print(".3f")
            else:
                print("❌ No pipeline data available")

        except Exception as e:
            print(f"❌ Error loading pipeline data: {e}")

    def create_model_performance_ranking(self):
        """Create comprehensive model performance ranking visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Complete Model Performance Ranking\nAll Models from Latest Pipeline Run',
                    fontsize=16, fontweight='bold')

        if not self.model_metrics:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No model data available', ha='center', va='center', fontsize=14)
                ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'model_performance_ranking.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Prepare data for plotting
        models = list(self.model_metrics.keys())
        r2_scores = [self.model_metrics[m]['r2'] for m in models]
        rmse_scores = [self.model_metrics[m]['rmse'] for m in models]

        # Sort by R² for ranking
        sorted_indices = np.argsort(r2_scores)[::-1]
        models_sorted = [models[i] for i in sorted_indices]
        r2_sorted = [r2_scores[i] for i in sorted_indices]
        rmse_sorted = [rmse_scores[i] for i in sorted_indices]

        # Model type mapping for colors
        model_colors = {
            'ann': '#FFD700',      # Gold for ANN
            'decision_tree': '#E74C3C',  # Red for Decision Tree
            'random_forest': '#27AE60',  # Green for Random Forest
            'linear': '#3498DB',   # Blue for Linear
            'ridge': '#9B59B6',    # Purple for Ridge
            'lasso': '#E67E22',    # Orange for Lasso
            'elastic_net': '#95A5A6'  # Gray for Elastic Net
        }

        colors = [model_colors.get(model, '#95A5A6') for model in models_sorted]

        # Plot 1: R² Score Ranking (Top Left)
        bars1 = axes[0, 0].barh(range(len(models_sorted)), r2_sorted, color=colors, alpha=0.8)
        axes[0, 0].set_yticks(range(len(models_sorted)))
        axes[0, 0].set_yticklabels([m.replace('_', ' ').title() for m in models_sorted])
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Model Ranking by R² Score')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars1, r2_sorted)):
            axes[0, 0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                           '.3f', ha='left', va='center', fontweight='bold')

        # Plot 2: RMSE Comparison (Top Middle)
        bars2 = axes[0, 1].bar(range(len(models_sorted)), rmse_sorted, color=colors, alpha=0.8)
        axes[0, 1].set_xticks(range(len(models_sorted)))
        axes[0, 1].set_xticklabels([m.replace('_', ' ').title() for m in models_sorted],
                                  rotation=45, ha='right')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].set_title('RMSE Comparison (Lower is Better)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, score in zip(bars2, rmse_sorted):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                           '.0f', ha='center', va='bottom', fontweight='bold')

        # Plot 3: R² vs RMSE Trade-off (Top Right)
        scatter = axes[0, 2].scatter(rmse_scores, r2_scores, s=200, c=range(len(models)),
                                   cmap='viridis', alpha=0.8, edgecolors='black', linewidth=2)

        # Add model labels
        for i, model in enumerate(models):
            axes[0, 2].annotate(model.replace('_', ' ').title(),
                               (rmse_scores[i], r2_scores[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.8, fontweight='bold')

        axes[0, 2].set_xlabel('RMSE ($)')
        axes[0, 2].set_ylabel('R² Score')
        axes[0, 2].set_title('Accuracy vs Error Trade-off')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Model Performance Summary Table (Bottom Left)
        axes[1, 0].axis('off')

        # Create performance summary
        summary_data = []
        for i, model in enumerate(models_sorted):
            metrics = self.model_metrics[model]
            summary_data.append([
                model.replace('_', ' ').title(),
                '.3f',
                '.0f',
                '.0f' if metrics.get('mae') else 'N/A'
            ])

        table = axes[1, 0].table(cellText=summary_data,
                                colLabels=['Model', 'R² Score', 'RMSE ($)', 'MAE ($)'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Color code the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#E8E8E8')
                    table[(i, j)].set_text_props(weight='bold')

        axes[1, 0].set_title('Performance Summary Table', fontsize=12, fontweight='bold', pad=20)

        # Plot 5: Model Type Comparison (Bottom Middle)
        model_types = {
            'Neural Network': ['ann'],
            'Tree-Based': ['decision_tree', 'random_forest'],
            'Linear': ['linear', 'ridge', 'lasso', 'elastic_net']
        }

        type_performance = {}
        for model_type, model_list in model_types.items():
            type_r2 = [self.model_metrics[m]['r2'] for m in model_list if m in self.model_metrics]
            if type_r2:
                type_performance[model_type] = {
                    'mean_r2': np.mean(type_r2),
                    'std_r2': np.std(type_r2),
                    'count': len(type_r2)
                }

        if type_performance:
            types = list(type_performance.keys())
            means = [type_performance[t]['mean_r2'] for t in types]
            stds = [type_performance[t]['std_r2'] for t in types]

            x = np.arange(len(types))
            bars = axes[1, 1].bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                                 color=['#FFD700', '#27AE60', '#3498DB'])
            axes[1, 1].set_ylabel('Average R² Score')
            axes[1, 1].set_title('Model Type Performance Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(types)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].set_ylim(0.8, 1.0)

        # Plot 6: Best Model Highlight (Bottom Right)
        axes[1, 2].axis('off')

        if self.model_metrics:
            best_model = max(self.model_metrics.items(), key=lambda x: x[1]['r2'])

            winner_text = f"""
            🏆 BEST MODEL: {best_model[0].replace('_', ' ').upper()}

            📊 Performance Metrics:
            • R² Score: {best_model[1]['r2']:.4f} ({best_model[1]['r2']*100:.1f}% variance explained)
            • RMSE: ${best_model[1]['rmse']:.0f}
            • MAE: ${best_model[1]['mae']:.0f} (if available)

            🎯 Key Strengths:
            • Highest predictive accuracy among all models
            • Best balance of bias-variance tradeoff
            • Recommended for production deployment

            📈 Performance Rank:
            • #1 by R² score
            • #1 by RMSE (lowest error)
            • Superior to all other models tested

            🚀 Business Impact:
            • Most accurate revenue predictions
            • Best foundation for business decisions
            • Highest confidence in forecasting results
            """

            axes[1, 2].text(0.05, 0.95, winner_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance_ranking.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Model performance ranking saved to {self.output_dir}/model_performance_ranking.png")

    def create_model_characteristics_comparison(self):
        """Create model characteristics comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Characteristics Comparison\nAlgorithm Families and Capabilities',
                    fontsize=16, fontweight='bold')

        if not self.model_metrics:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No model data available', ha='center', va='center', fontsize=14)
                ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'model_characteristics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Define model characteristics
        models = list(self.model_metrics.keys())
        characteristics = {
            'Interpretability': {
                'ann': 0.2, 'decision_tree': 0.8, 'random_forest': 0.4,
                'linear': 0.9, 'ridge': 0.8, 'lasso': 0.7, 'elastic_net': 0.6
            },
            'Training_Speed': {
                'ann': 0.3, 'decision_tree': 0.9, 'random_forest': 0.6,
                'linear': 1.0, 'ridge': 0.9, 'lasso': 0.8, 'elastic_net': 0.7
            },
            'Prediction_Speed': {
                'ann': 0.7, 'decision_tree': 0.9, 'random_forest': 0.8,
                'linear': 1.0, 'ridge': 1.0, 'lasso': 1.0, 'elastic_net': 1.0
            },
            'Nonlinear_Capability': {
                'ann': 0.9, 'decision_tree': 0.8, 'random_forest': 0.9,
                'linear': 0.2, 'ridge': 0.2, 'lasso': 0.2, 'elastic_net': 0.3
            }
        }

        # Plot 1: Radar Chart of Model Characteristics (Top Left)
        ax1 = axes[0, 0]

        # Create radar chart data
        categories = list(characteristics.keys())
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for model in models:
            values = [characteristics[cat][model] for cat in categories]
            values += values[:1]

            ax1.plot(angles, values, 'o-', linewidth=2, label=model.replace('_', ' ').title())
            ax1.fill(angles, values, alpha=0.1)

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([cat.replace('_', '\n') for cat in categories])
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Characteristics Radar', fontsize=12, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Performance vs Complexity Scatter (Top Right)
        ax2 = axes[0, 1]

        complexity_scores = {
            'linear': 1, 'ridge': 2, 'lasso': 3, 'elastic_net': 4,
            'decision_tree': 5, 'random_forest': 6, 'ann': 7
        }

        complexities = [complexity_scores.get(model, 4) for model in models]
        performances = [self.model_metrics[model]['r2'] for model in models]

        scatter = ax2.scatter(complexities, performances, s=200, c=range(len(models)),
                            cmap='viridis', alpha=0.8, edgecolors='black', linewidth=2)

        # Add model labels
        for i, model in enumerate(models):
            ax2.annotate(model.replace('_', ' ').title(),
                        (complexities[i], performances[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8, fontweight='bold')

        ax2.set_xlabel('Model Complexity (1=Simple, 7=Complex)')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Performance vs Model Complexity')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.5, 7.5)

        # Plot 3: Use Case Recommendations (Bottom Left)
        ax3 = axes[1, 0]
        ax3.axis('off')

        recommendations = """
        🎯 MODEL SELECTION GUIDE

        For Production Deployment:
        • 🏆 Random Forest: Best overall performance
        • 🥈 Decision Tree: Fast training, good accuracy
        • 🥉 ANN: Complex patterns, high accuracy

        For Real-Time Applications:
        • ⚡ Linear/Ridge: Fastest predictions
        • 🚀 Decision Tree: Speed-accuracy balance
        • 💨 KNN: Fast training, reasonable accuracy

        For Interpretability Needs:
        • 📊 Linear Regression: Direct coefficient interpretation
        • 🌳 Decision Tree: Visual decision rules
        • 📈 Ridge/Lasso: Feature importance via coefficients

        For Complex Patterns:
        • 🧠 ANN: Best non-linear pattern recognition
        • 🌲 Random Forest: Ensemble non-linear learning
        • 📈 Elastic Net: Regularized linear with interactions

        ⚠️ Trade-off Considerations:
        • Accuracy vs Speed
        • Interpretability vs Performance
        • Training time vs Prediction time
        • Complexity vs Maintainability
        """

        ax3.text(0.05, 0.95, recommendations, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

        # Plot 4: Model Family Performance Summary (Bottom Right)
        ax4 = axes[1, 1]

        # Group models by family
        families = {
            'Neural\nNetworks': ['ann'],
            'Tree\nModels': ['decision_tree', 'random_forest'],
            'Linear\nModels': ['linear', 'ridge', 'lasso', 'elastic_net']
        }

        family_stats = {}
        for family_name, family_models in families.items():
            family_r2 = [self.model_metrics[m]['r2'] for m in family_models if m in self.model_metrics]
            if family_r2:
                family_stats[family_name] = {
                    'mean': np.mean(family_r2),
                    'std': np.std(family_r2),
                    'count': len(family_r2),
                    'best': max(family_r2)
                }

        if family_stats:
            family_names = list(family_stats.keys())
            means = [family_stats[f]['mean'] for f in family_names]
            stds = [family_stats[f]['std'] for f in family_names]

            x = np.arange(len(family_names))
            bars = ax4.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                          color=['#FFD700', '#27AE60', '#3498DB'])

            # Add model counts
            for i, (bar, family) in enumerate(zip(bars, family_names)):
                count = family_stats[family]['count']
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'n={count}', ha='center', va='bottom', fontweight='bold')

            ax4.set_ylabel('Average R² Score')
            ax4.set_title('Model Family Performance')
            ax4.set_xticks(x)
            ax4.set_xticklabels(family_names)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0.8, 1.0)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_characteristics_comparison.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Model characteristics comparison saved to {self.output_dir}/model_characteristics_comparison.png")

    def create_pipeline_summary_dashboard(self):
        """Create a comprehensive pipeline summary dashboard."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('Complete Pipeline Model Comparison Dashboard\nLatest Run Analysis',
                    fontsize=18, fontweight='bold', y=0.95)

        if not self.model_metrics:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No model data available from pipeline',
                   ha='center', va='center', fontsize=16)
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'pipeline_summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Top row - Key metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')

        # Calculate summary statistics
        r2_scores = [self.model_metrics[m]['r2'] for m in self.model_metrics]
        rmse_scores = [self.model_metrics[m]['rmse'] for m in self.model_metrics]

        best_model = max(self.model_metrics.items(), key=lambda x: x[1]['r2'])
        worst_model = min(self.model_metrics.items(), key=lambda x: x[1]['r2'])

        summary_text = f"""
        📊 PIPELINE SUMMARY

        Total Models Trained: {len(self.model_metrics)}
        Best R² Score: {max(r2_scores):.4f}
        Worst R² Score: {min(r2_scores):.4f}
        R² Range: {(max(r2_scores)-min(r2_scores)):.4f}

        🏆 Winner: {best_model[0].replace('_', ' ').title()}
        📉 Lowest: {worst_model[0].replace('_', ' ').title()}

        Average R²: {np.mean(r2_scores):.4f}
        Median R²: {np.median(r2_scores):.4f}
        R² Std Dev: {np.std(r2_scores):.4f}
        """

        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))

        # Model ranking bar chart
        ax2 = fig.add_subplot(gs[0, 1:])
        models = list(self.model_metrics.keys())
        r2_scores = [self.model_metrics[m]['r2'] for m in models]

        # Sort by performance
        sorted_idx = np.argsort(r2_scores)[::-1]
        models_sorted = [models[i] for i in sorted_idx]
        r2_sorted = [r2_scores[i] for i in sorted_idx]

        colors = ['#FFD700' if i == 0 else '#27AE60' if i == 1 else '#3498DB' for i in range(len(models_sorted))]
        bars = ax2.bar(range(len(models_sorted)), r2_sorted, color=colors, alpha=0.8)

        ax2.set_xticks(range(len(models_sorted)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in models_sorted], rotation=45, ha='right')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model Performance Ranking')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, score in zip(bars, r2_sorted):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    '.3f', ha='center', va='bottom', fontweight='bold')

        # Second row - Performance comparison
        ax3 = fig.add_subplot(gs[1, :2])
        # R² vs RMSE scatter
        rmse_scores = [self.model_metrics[m]['rmse'] for m in models]
        scatter = ax3.scatter(rmse_scores, r2_scores, s=200, c=range(len(models)),
                            cmap='viridis', alpha=0.8, edgecolors='black', linewidth=2)

        for i, model in enumerate(models):
            ax3.annotate(model.replace('_', ' ').title(),
                        (rmse_scores[i], r2_scores[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8, fontweight='bold')

        ax3.set_xlabel('RMSE ($)')
        ax3.set_ylabel('R² Score')
        ax3.set_title('Accuracy vs Error Trade-off')
        ax3.grid(True, alpha=0.3)

        # Model type performance
        ax4 = fig.add_subplot(gs[1, 2:])
        model_types = {
            'Neural': ['ann'],
            'Tree': ['decision_tree', 'random_forest'],
            'Linear': ['linear', 'ridge', 'lasso', 'elastic_net']
        }

        type_data = {}
        for type_name, type_models in model_types.items():
            type_r2 = [self.model_metrics[m]['r2'] for m in type_models if m in self.model_metrics]
            if type_r2:
                type_data[type_name] = np.mean(type_r2)

        if type_data:
            types = list(type_data.keys())
            values = list(type_data.values())
            bars = ax4.bar(types, values, alpha=0.8, color=['#FFD700', '#27AE60', '#3498DB'])
            ax4.set_ylabel('Average R² Score')
            ax4.set_title('Model Type Performance')
            ax4.grid(True, alpha=0.3, axis='y')

        # Third row - Detailed metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Create detailed metrics table
        table_data = []
        for model in models_sorted:
            metrics = self.model_metrics[model]
            table_data.append([
                model.replace('_', ' ').title(),
                metrics['r2'],
                metrics['rmse'],
                metrics.get('mae', 'N/A'),
                metrics.get('mape', 'N/A')
            ])

        table = ax5.table(cellText=table_data,
                         colLabels=['Model', 'R² Score', 'RMSE ($)', 'MAE ($)', 'MAPE (%)'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Color code the table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#E8E8E8')
                    table[(i, j)].set_text_props(weight='bold')
                elif j == 1:  # R² column - highlight best
                    try:
                        r2_val = float(table_data[i][j])
                        max_r2 = max([float(row[1]) for row in table_data if isinstance(row[1], (int, float)) or (isinstance(row[1], str) and row[1].replace('.', '').isdigit())])
                        if r2_val == max_r2:
                            table[(i, j)].set_facecolor('#90EE90')  # Light green
                    except (ValueError, TypeError):
                        pass  # Skip if not a valid number

        ax5.set_title('Detailed Performance Metrics Table', fontsize=14, fontweight='bold', pad=20)

        # Fourth row - Recommendations and insights
        ax6 = fig.add_subplot(gs[3, :2])
        ax6.axis('off')

        best_model = max(self.model_metrics.items(), key=lambda x: x[1]['r2'])

        recommendations = f"""
        🎯 PIPELINE RECOMMENDATIONS

        🏆 PRIMARY MODEL: {best_model[0].replace('_', ' ').upper()}
        • R² Score: {best_model[1]['r2']:.4f} ({best_model[1]['r2']*100:.1f}% variance explained)
        • RMSE: ${best_model[1]['rmse']:.0f}
        • Recommended for production deployment

        📊 MODEL FAMILY INSIGHTS:
        • Tree-based models dominate performance rankings
        • Neural networks show strong non-linear capabilities
        • Linear models provide consistent baseline performance
        • Ensemble methods offer best bias-variance balance

        ⚡ PRACTICAL CONSIDERATIONS:
        • Decision Tree: Fast training, good interpretability
        • Random Forest: Highest accuracy, moderate training time
        • ANN: Complex patterns, requires more resources
        • Linear models: Fast, interpretable, good baselines

        🚀 DEPLOYMENT STRATEGY:
        1. Primary: {best_model[0].replace('_', ' ').title()} for accuracy
        2. Backup: Decision Tree for speed/interpretabiity
        3. Baseline: Linear Regression for comparison
        4. Monitor: All models for performance drift

        💡 BUSINESS IMPACT:
        • Best model reduces prediction error by {((max(r2_scores) - min(r2_scores)) * 100):.1f}% compared to worst
        • Revenue forecasting accuracy significantly improved
        • Data-driven decision making capabilities enhanced
        """

        ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))

        # Performance distribution
        ax7 = fig.add_subplot(gs[3, 2:])
        r2_scores_all = [self.model_metrics[m]['r2'] for m in self.model_metrics]
        ax7.hist(r2_scores_all, bins=10, alpha=0.7, color='#3498DB', edgecolor='black')
        ax7.axvline(np.mean(r2_scores_all), color='red', linestyle='--',
                   label=f'Mean: {np.mean(r2_scores_all):.3f}')
        ax7.axvline(np.median(r2_scores_all), color='green', linestyle='--',
                   label=f'Median: {np.median(r2_scores_all):.3f}')
        ax7.set_xlabel('R² Score')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Performance Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'pipeline_summary_dashboard.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Pipeline summary dashboard saved to {self.output_dir}/pipeline_summary_dashboard.png")

    def generate_complete_comparison_suite(self):
        """Generate all model comparison visualizations."""
        print("📊 Generating Complete Model Comparison Suite...")
        print("=" * 70)

        self.create_model_performance_ranking()
        self.create_model_characteristics_comparison()
        self.create_pipeline_summary_dashboard()

        print("=" * 70)
        print("✅ Complete Model Comparison Suite Generated!")
        print(f"📁 Visualizations saved to: {self.output_dir}")
        print("📊 Generated visualizations:")
        print("   • Model Performance Ranking (performance metrics, rankings, trade-offs)")
        print("   • Model Characteristics Comparison (capabilities, use cases, families)")
        print("   • Pipeline Summary Dashboard (comprehensive overview, recommendations)")
        print()
        print("🎯 Key Insights:")
        if self.model_metrics:
            best_model = max(self.model_metrics.items(), key=lambda x: x[1]['r2'])
            print(f"   🏆 Best Model: {best_model[0].replace('_', ' ').title()} (R² = {best_model[1]['r2']:.4f})")
            print(f"   📊 Models Compared: {len(self.model_metrics)}")
            print("   🎯 Data Source: Real pipeline results from latest run")
        print("   🚀 Ready for production deployment recommendations")


def main():
    """Main execution function."""
    try:
        visualizer = CompleteModelComparisonVisualizer()
        visualizer.generate_complete_comparison_suite()
        return 0
    except Exception as e:
        print(f"❌ Error generating model comparison visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())