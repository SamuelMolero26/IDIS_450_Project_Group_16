#!/usr/bin/env python3
"""
Generate visualizations specifically for KNN and ANN models.

This script creates individual and comparative visualizations for:
- KNN performance and optimal K analysis
- ANN/MLP performance and architecture analysis
- KNN vs ANN comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KNNANNVisualizer:
    """Generate visualizations for KNN and ANN models."""

    def __init__(self):
        self.report_data = None
        self.knn_data = None
        self.ann_data = None
        self.all_models_data = {}
        self.viz_dir = Path(project_root) / "visualizations" / "model_comparison_dashboard"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def load_latest_report(self):
        """Load the most recent pipeline report."""
        reports_dir = Path(project_root) / "reports"
        json_files = list(reports_dir.glob("pipeline_report_*.json"))

        if not json_files:
            print("❌ No pipeline reports found")
            return False

        latest_report = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📊 Loading: {latest_report.name}")

        with open(latest_report, 'r') as f:
            self.report_data = json.load(f)

        return True

    def extract_model_data(self):
        """Extract KNN, ANN, and all model data from report."""
        print("📈 Extracting model data...")

        modeling_results = self.report_data.get('modeling_results', {})
        model_results = modeling_results.get('model_results', {})

        # Get comprehensive comparison
        comp = model_results.get('comprehensive_comparison', {})
        if comp and 'model_rankings' in comp:
            for rank_data in comp['model_rankings']:
                model_name = rank_data['model_name']
                self.all_models_data[model_name] = rank_data

                if model_name.lower() == 'knn':
                    self.knn_data = rank_data
                elif model_name.lower() in ['ann', 'mlp', 'neural_network']:
                    self.ann_data = rank_data

        # Also check for KNN-specific analysis
        knn_analysis = comp.get('knn_specific_analysis', {})
        if knn_analysis and self.knn_data:
            self.knn_data['knn_analysis'] = knn_analysis

        print(f"✅ Found KNN data: {self.knn_data is not None}")
        print(f"✅ Found ANN data: {self.ann_data is not None}")
        print(f"✅ Total models: {len(self.all_models_data)}")

        return (self.knn_data is not None) or (self.ann_data is not None)

    def create_knn_visualization(self):
        """Create comprehensive KNN visualization."""
        if not self.knn_data:
            print("⏭️  Skipping KNN visualization (no data)")
            return

        print("\n🎨 Creating KNN visualization...")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. KNN vs Other Models - R² Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_knn_vs_others_r2(ax1)

        # 2. KNN vs Other Models - RMSE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_knn_vs_others_rmse(ax2)

        # 3. KNN Performance Metrics Table
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_knn_metrics_table(ax3)

        # 4. KNN Ranking Position
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_knn_ranking(ax4)

        # 5. KNN Specific Analysis
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_knn_analysis(ax5)

        plt.suptitle('K-Nearest Neighbors (KNN) - Comprehensive Performance Analysis',
                     fontsize=18, fontweight='bold', y=0.995)

        save_path = self.viz_dir / 'knn_comprehensive_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def create_ann_visualization(self):
        """Create comprehensive ANN visualization."""
        if not self.ann_data:
            print("⏭️  Skipping ANN visualization (no data)")
            return

        print("\n🎨 Creating ANN visualization...")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. ANN vs Other Models - R² Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_ann_vs_others_r2(ax1)

        # 2. ANN vs Other Models - RMSE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_ann_vs_others_rmse(ax2)

        # 3. ANN Performance Metrics Table
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_ann_metrics_table(ax3)

        # 4. ANN Ranking Position
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_ann_ranking(ax4)

        # 5. Training Time Comparison
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_ann_training_time(ax5)

        plt.suptitle('Artificial Neural Network (ANN/MLP) - Comprehensive Performance Analysis',
                     fontsize=18, fontweight='bold', y=0.995)

        save_path = self.viz_dir / 'ann_comprehensive_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def create_knn_ann_comparison(self):
        """Create KNN vs ANN comparison visualization."""
        if not self.knn_data or not self.ann_data:
            print("⏭️  Skipping KNN vs ANN comparison (need both models)")
            return

        print("\n🎨 Creating KNN vs ANN comparison...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('KNN vs ANN (MLP) - Head-to-Head Comparison',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. R² Comparison
        models = ['KNN', 'ANN']
        r2_scores = [self.knn_data['r2_score'], self.ann_data['r2_score']]
        axes[0, 0].bar(models, r2_scores, color=['#3498db', '#e74c3c'], alpha=0.8)
        axes[0, 0].set_ylabel('R² Score', fontweight='bold')
        axes[0, 0].set_title('Accuracy Comparison (R²)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(r2_scores):
            axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

        # 2. RMSE Comparison
        rmse_scores = [self.knn_data['rmse_score'], self.ann_data['rmse_score']]
        axes[0, 1].bar(models, rmse_scores, color=['#3498db', '#e74c3c'], alpha=0.8)
        axes[0, 1].set_ylabel('RMSE ($)', fontweight='bold')
        axes[0, 1].set_title('Error Comparison (RMSE)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rmse_scores):
            axes[0, 1].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')

        # 3. Training Time Comparison
        train_times = [self.knn_data['training_time'], self.ann_data['training_time']]
        axes[0, 2].bar(models, train_times, color=['#3498db', '#e74c3c'], alpha=0.8)
        axes[0, 2].set_ylabel('Training Time (s)', fontweight='bold')
        axes[0, 2].set_title('Training Speed', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(train_times):
            axes[0, 2].text(i, v, f'{v:.3f}s', ha='center', va='bottom', fontweight='bold')

        # 4. Ranking Comparison
        knn_rank = self.knn_data.get('rank_by_r2', 0)
        ann_rank = self.ann_data.get('rank_by_r2', 0)
        ranks = [knn_rank, ann_rank]
        axes[1, 0].barh(models, ranks, color=['#3498db', '#e74c3c'], alpha=0.8)
        axes[1, 0].set_xlabel('Rank (1 = Best)', fontweight='bold')
        axes[1, 0].set_title('Overall Ranking', fontweight='bold')
        axes[1, 0].invert_xaxis()  # Lower rank is better
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(ranks):
            axes[1, 0].text(v, i, f'  Rank #{v}', ha='left', va='center', fontweight='bold')

        # 5. Metrics Radar Chart
        self._plot_knn_ann_radar(axes[1, 1])

        # 6. Summary Recommendations
        self._plot_knn_ann_recommendations(axes[1, 2])

        plt.tight_layout()
        save_path = self.viz_dir / 'knn_vs_ann_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def create_all_models_comparison(self):
        """Create comprehensive all-models comparison."""
        print("\n🎨 Creating all models comparison...")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        # Sort models by R²
        sorted_models = sorted(self.all_models_data.items(),
                              key=lambda x: x[1]['r2_score'], reverse=True)

        # 1. R² Comparison (all models)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_all_models_r2(ax1, sorted_models)

        # 2. RMSE Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_all_models_rmse(ax2, sorted_models)

        # 3. Training Time Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_all_models_training_time(ax3, sorted_models)

        # 4. Model Rankings Table
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_all_models_rankings(ax4, sorted_models)

        # 5. R² vs Training Time Scatter
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_r2_vs_time_scatter(ax5, sorted_models)

        # 6. R² vs RMSE Scatter
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_r2_vs_rmse_scatter(ax6, sorted_models)

        # 7. Model Categories Performance
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_model_categories(ax7, sorted_models)

        # 8. Detailed Metrics Table
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_all_models_table(ax8, sorted_models)

        plt.suptitle('Comprehensive Model Performance Comparison\nAll Models Evaluated',
                     fontsize=20, fontweight='bold', y=0.998)

        save_path = self.viz_dir / 'comprehensive_model_dashboard.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    # Helper plotting methods
    def _plot_knn_vs_others_r2(self, ax):
        """Plot KNN R² vs other models."""
        models = list(self.all_models_data.keys())
        r2_scores = [self.all_models_data[m]['r2_score'] for m in models]

        colors = ['#2ecc71' if m.lower() == 'knn' else '#95a5a6' for m in models]
        bars = ax.bar(models, r2_scores, color=colors, alpha=0.8)

        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('KNN vs Other Models - R² Comparison', fontweight='bold', fontsize=12)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

        # Highlight KNN
        knn_idx = [i for i, m in enumerate(models) if m.lower() == 'knn'][0]
        bars[knn_idx].set_edgecolor('black')
        bars[knn_idx].set_linewidth(2)

    def _plot_knn_vs_others_rmse(self, ax):
        """Plot KNN RMSE vs other models."""
        models = list(self.all_models_data.keys())
        rmse_scores = [self.all_models_data[m]['rmse_score'] for m in models]

        colors = ['#2ecc71' if m.lower() == 'knn' else '#95a5a6' for m in models]
        bars = ax.bar(models, rmse_scores, color=colors, alpha=0.8)

        ax.set_ylabel('RMSE ($)', fontweight='bold')
        ax.set_title('KNN vs Other Models - Error Comparison', fontweight='bold', fontsize=12)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight KNN
        knn_idx = [i for i, m in enumerate(models) if m.lower() == 'knn'][0]
        bars[knn_idx].set_edgecolor('black')
        bars[knn_idx].set_linewidth(2)

    def _plot_knn_metrics_table(self, ax):
        """Create metrics table for KNN."""
        ax.axis('off')

        data = [
            ['Metric', 'Value', 'Rank'],
            ['R² Score', f"{self.knn_data['r2_score']:.4f}", f"#{self.knn_data.get('rank_by_r2', 'N/A')}"],
            ['RMSE', f"${self.knn_data['rmse_score']:,.2f}", f"#{self.knn_data.get('rank_by_rmse', 'N/A')}"],
            ['Training Time', f"{self.knn_data['training_time']:.3f}s", '-'],
        ]

        # Add KNN-specific metrics if available
        if 'knn_analysis' in self.knn_data:
            knn_analysis = self.knn_data['knn_analysis']
            if 'optimal_k' in knn_analysis:
                data.append(['Optimal K', str(knn_analysis.get('optimal_k', 'N/A')), '-'])

        table = ax.table(cellText=data, cellLoc='left', loc='center',
                        colWidths=[0.4, 0.3, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.5)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#2ecc71')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data rows
        for i in range(1, len(data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        ax.set_title('KNN Performance Metrics', fontweight='bold', fontsize=14, pad=10)

    def _plot_knn_ranking(self, ax):
        """Plot KNN ranking among all models."""
        total_models = len(self.all_models_data)
        knn_rank = self.knn_data.get('rank_by_r2', total_models)

        # Create ranking visualization
        models_sorted = sorted(self.all_models_data.items(),
                              key=lambda x: x[1]['r2_score'], reverse=True)

        y_positions = range(len(models_sorted))
        colors = ['#2ecc71' if m[0].lower() == 'knn' else '#95a5a6' for m in models_sorted]

        ax.barh(y_positions, [m[1]['r2_score'] for m in models_sorted], color=colors, alpha=0.8)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([m[0] for m in models_sorted])
        ax.set_xlabel('R² Score', fontweight='bold')
        ax.set_title(f'KNN Overall Ranking: #{knn_rank} of {total_models}',
                     fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_knn_analysis(self, ax):
        """Plot KNN-specific analysis."""
        ax.axis('off')

        if 'knn_analysis' in self.knn_data:
            knn_analysis = self.knn_data['knn_analysis']

            text = "KNN-Specific Analysis\n\n"
            text += f"Optimal K: {knn_analysis.get('optimal_k', 'N/A')}\n\n"
            text += "Key Insights:\n"
            text += "• Distance-based algorithm\n"
            text += "• Non-parametric method\n"
            text += "• Sensitive to feature scaling\n"
            text += "• Good for local patterns\n"
        else:
            text = "KNN Performance Summary\n\n"
            text += f"R² Score: {self.knn_data['r2_score']:.4f}\n"
            text += f"RMSE: ${self.knn_data['rmse_score']:,.2f}\n"
            text += f"Rank: #{self.knn_data.get('rank_by_r2', 'N/A')}\n"

        ax.text(0.1, 0.9, text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title('KNN Insights', fontweight='bold', fontsize=12)

    # Similar methods for ANN visualizations
    def _plot_ann_vs_others_r2(self, ax):
        """Plot ANN R² vs other models."""
        models = list(self.all_models_data.keys())
        r2_scores = [self.all_models_data[m]['r2_score'] for m in models]

        colors = ['#e74c3c' if m.lower() in ['ann', 'mlp'] else '#95a5a6' for m in models]
        bars = ax.bar(models, r2_scores, color=colors, alpha=0.8)

        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('ANN vs Other Models - R² Comparison', fontweight='bold', fontsize=12)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

    def _plot_ann_vs_others_rmse(self, ax):
        """Plot ANN RMSE vs other models."""
        models = list(self.all_models_data.keys())
        rmse_scores = [self.all_models_data[m]['rmse_score'] for m in models]

        colors = ['#e74c3c' if m.lower() in ['ann', 'mlp'] else '#95a5a6' for m in models]
        bars = ax.bar(models, rmse_scores, color=colors, alpha=0.8)

        ax.set_ylabel('RMSE ($)', fontweight='bold')
        ax.set_title('ANN vs Other Models - Error Comparison', fontweight='bold', fontsize=12)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_ann_metrics_table(self, ax):
        """Create metrics table for ANN."""
        ax.axis('off')

        data = [
            ['Metric', 'Value', 'Rank'],
            ['R² Score', f"{self.ann_data['r2_score']:.4f}", f"#{self.ann_data.get('rank_by_r2', 'N/A')}"],
            ['RMSE', f"${self.ann_data['rmse_score']:,.2f}", f"#{self.ann_data.get('rank_by_rmse', 'N/A')}"],
            ['Training Time', f"{self.ann_data['training_time']:.3f}s", '-'],
        ]

        table = ax.table(cellText=data, cellLoc='left', loc='center',
                        colWidths=[0.4, 0.3, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.5)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#e74c3c')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('ANN Performance Metrics', fontweight='bold', fontsize=14, pad=10)

    def _plot_ann_ranking(self, ax):
        """Plot ANN ranking among all models."""
        total_models = len(self.all_models_data)
        ann_rank = self.ann_data.get('rank_by_r2', total_models)

        models_sorted = sorted(self.all_models_data.items(),
                              key=lambda x: x[1]['r2_score'], reverse=True)

        y_positions = range(len(models_sorted))
        colors = ['#e74c3c' if m[0].lower() in ['ann', 'mlp'] else '#95a5a6' for m in models_sorted]

        ax.barh(y_positions, [m[1]['r2_score'] for m in models_sorted], color=colors, alpha=0.8)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([m[0] for m in models_sorted])
        ax.set_xlabel('R² Score', fontweight='bold')
        ax.set_title(f'ANN Overall Ranking: #{ann_rank} of {total_models}',
                     fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_ann_training_time(self, ax):
        """Plot ANN training time vs other models."""
        models = list(self.all_models_data.keys())
        times = [self.all_models_data[m]['training_time'] for m in models]

        colors = ['#e74c3c' if m.lower() in ['ann', 'mlp'] else '#95a5a6' for m in models]
        bars = ax.bar(models, times, color=colors, alpha=0.8)

        ax.set_ylabel('Training Time (s)', fontweight='bold')
        ax.set_title('Training Speed Comparison', fontweight='bold', fontsize=12)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        if max(times) / min(times) > 100:
            ax.set_yscale('log')

    def _plot_knn_ann_radar(self, ax):
        """Create radar chart comparing KNN and ANN."""
        categories = ['Accuracy\n(R²)', 'Speed\n(inverse time)', 'Low Error\n(inverse RMSE)']

        # Normalize metrics to 0-1 scale
        knn_r2 = self.knn_data['r2_score']
        ann_r2 = self.ann_data['r2_score']

        knn_speed = 1 / (self.knn_data['training_time'] + 0.001)
        ann_speed = 1 / (self.ann_data['training_time'] + 0.001)
        max_speed = max(knn_speed, ann_speed)
        knn_speed_norm = knn_speed / max_speed
        ann_speed_norm = ann_speed / max_speed

        max_rmse = max(self.knn_data['rmse_score'], self.ann_data['rmse_score'])
        knn_error_inv = 1 - (self.knn_data['rmse_score'] / max_rmse)
        ann_error_inv = 1 - (self.ann_data['rmse_score'] / max_rmse)

        knn_values = [knn_r2, knn_speed_norm, knn_error_inv]
        ann_values = [ann_r2, ann_speed_norm, ann_error_inv]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        knn_values += knn_values[:1]
        ann_values += ann_values[:1]
        angles += angles[:1]

        ax.plot(angles, knn_values, 'o-', linewidth=2, label='KNN', color='#3498db')
        ax.fill(angles, knn_values, alpha=0.25, color='#3498db')
        ax.plot(angles, ann_values, 'o-', linewidth=2, label='ANN', color='#e74c3c')
        ax.fill(angles, ann_values, alpha=0.25, color='#e74c3c')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Comparison', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    def _plot_knn_ann_recommendations(self, ax):
        """Plot recommendations for KNN vs ANN."""
        ax.axis('off')

        knn_better = self.knn_data['r2_score'] > self.ann_data['r2_score']

        if knn_better:
            winner = "KNN"
            winner_r2 = self.knn_data['r2_score']
            margin = (self.knn_data['r2_score'] - self.ann_data['r2_score']) * 100
        else:
            winner = "ANN"
            winner_r2 = self.ann_data['r2_score']
            margin = (self.ann_data['r2_score'] - self.knn_data['r2_score']) * 100

        text = f"RECOMMENDATION\n\n"
        text += f"🏆 Winner: {winner}\n"
        text += f"R² Score: {winner_r2:.4f}\n"
        text += f"Margin: {margin:.2f}%\n\n"

        text += "Use KNN when:\n"
        text += "• Local patterns matter\n"
        text += "• Interpretability needed\n"
        text += "• Small to medium data\n\n"

        text += "Use ANN when:\n"
        text += "• Complex patterns exist\n"
        text += "• Large datasets available\n"
        text += "• Maximum accuracy needed\n"

        ax.text(0.1, 0.95, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        ax.set_title('Model Selection Guide', fontweight='bold', fontsize=12)

    def _plot_all_models_r2(self, ax, sorted_models):
        """Plot R² for all models."""
        models = [m[0] for m in sorted_models]
        r2_scores = [m[1]['r2_score'] for m in sorted_models]

        bars = ax.bar(models, r2_scores, alpha=0.8)

        # Color code bars
        colors = []
        for m in models:
            if m.lower() == 'knn':
                colors.append('#2ecc71')
            elif m.lower() in ['ann', 'mlp']:
                colors.append('#e74c3c')
            elif 'forest' in m.lower():
                colors.append('#9b59b6')
            elif 'tree' in m.lower():
                colors.append('#f39c12')
            else:
                colors.append('#3498db')

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('Model Accuracy Comparison - R² Scores (All Models)',
                     fontweight='bold', fontsize=14)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

        # Add value labels
        for i, v in enumerate(r2_scores):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    def _plot_all_models_rmse(self, ax, sorted_models):
        """Plot RMSE for all models."""
        models = [m[0] for m in sorted_models]
        rmse_scores = [m[1]['rmse_score'] for m in sorted_models]

        ax.barh(models, rmse_scores, alpha=0.8, color='coral')

        ax.set_xlabel('RMSE ($)', fontweight='bold')
        ax.set_title('Error Metrics (RMSE)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(rmse_scores):
            ax.text(v, i, f' ${v:,.0f}', ha='left', va='center', fontsize=9)

    def _plot_all_models_training_time(self, ax, sorted_models):
        """Plot training time for all models."""
        models = [m[0] for m in sorted_models]
        times = [m[1]['training_time'] for m in sorted_models]

        ax.bar(models, times, alpha=0.8, color='mediumpurple')

        ax.set_ylabel('Training Time (s)', fontweight='bold')
        ax.set_title('Training Efficiency', fontweight='bold', fontsize=12)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        if max(times) / min(times) > 100:
            ax.set_yscale('log')

    def _plot_all_models_rankings(self, ax, sorted_models):
        """Plot model rankings table."""
        ax.axis('off')

        data = [['Rank', 'Model', 'R² Score']]
        for i, (model, metrics) in enumerate(sorted_models[:10], 1):
            data.append([f'#{i}', model, f"{metrics['r2_score']:.4f}"])

        table = ax.table(cellText=data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.5, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Model Rankings (Top 10)', fontweight='bold', fontsize=12, pad=10)

    def _plot_r2_vs_time_scatter(self, ax, sorted_models):
        """Plot R² vs training time scatter."""
        r2_scores = [m[1]['r2_score'] for m in sorted_models]
        times = [m[1]['training_time'] for m in sorted_models]
        models = [m[0] for m in sorted_models]

        for i, (r2, time, model) in enumerate(zip(r2_scores, times, models)):
            ax.scatter(time, r2, s=100, alpha=0.7, label=model if i < 5 else "")
            if i < 5:  # Label top 5
                ax.annotate(model, (time, r2), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Training Time (s)', fontweight='bold')
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('Performance vs Speed', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)

        if max(times) / min(times) > 100:
            ax.set_xscale('log')

    def _plot_r2_vs_rmse_scatter(self, ax, sorted_models):
        """Plot R² vs RMSE scatter."""
        r2_scores = [m[1]['r2_score'] for m in sorted_models]
        rmse_scores = [m[1]['rmse_score'] for m in sorted_models]
        models = [m[0] for m in sorted_models]

        for i, (r2, rmse, model) in enumerate(zip(r2_scores, rmse_scores, models)):
            ax.scatter(rmse, r2, s=100, alpha=0.7)
            if i < 3:  # Label top 3
                ax.annotate(model, (rmse, r2), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('RMSE ($)', fontweight='bold')
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('Accuracy vs Error', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _plot_model_categories(self, ax, sorted_models):
        """Plot performance by model category."""
        categories = {
            'Linear': [],
            'Tree-based': [],
            'Instance-based': [],
            'Neural': []
        }

        for model, metrics in sorted_models:
            if any(x in model.lower() for x in ['linear', 'ridge', 'lasso', 'elastic']):
                categories['Linear'].append(metrics['r2_score'])
            elif any(x in model.lower() for x in ['tree', 'forest']):
                categories['Tree-based'].append(metrics['r2_score'])
            elif 'knn' in model.lower():
                categories['Instance-based'].append(metrics['r2_score'])
            elif any(x in model.lower() for x in ['ann', 'mlp', 'neural']):
                categories['Neural'].append(metrics['r2_score'])

        cat_names = []
        cat_means = []
        for cat, scores in categories.items():
            if scores:
                cat_names.append(cat)
                cat_means.append(np.mean(scores))

        ax.bar(cat_names, cat_means, alpha=0.8, color='teal')

        ax.set_ylabel('Average R² Score', fontweight='bold')
        ax.set_title('Performance by Model Category', fontweight='bold', fontsize=12)
        ax.set_xticklabels(cat_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(cat_means):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    def _plot_all_models_table(self, ax, sorted_models):
        """Create detailed metrics table for all models."""
        ax.axis('off')

        data = [['Model', 'R² Score', 'RMSE', 'Training Time', 'Rank']]
        for i, (model, metrics) in enumerate(sorted_models, 1):
            data.append([
                model,
                f"{metrics['r2_score']:.4f}",
                f"${metrics['rmse_score']:,.0f}",
                f"{metrics['training_time']:.3f}s",
                f"#{i}"
            ])

        table = ax.table(cellText=data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.2, 0.2, 0.1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style rows
        for i in range(1, len(data)):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        ax.set_title('Complete Model Performance Metrics', fontweight='bold', fontsize=14, pad=10)

    def run(self):
        """Run complete visualization generation."""
        print("=" * 80)
        print("KNN & ANN VISUALIZATION GENERATOR")
        print("=" * 80)

        if not self.load_latest_report():
            return False

        if not self.extract_model_data():
            print("❌ Could not extract KNN or ANN data")
            return False

        # Generate individual visualizations
        self.create_knn_visualization()
        self.create_ann_visualization()

        # Generate comparison visualizations
        self.create_knn_ann_comparison()
        self.create_all_models_comparison()

        print("\n" + "=" * 80)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nOutputs saved to: {self.viz_dir}/")
        print("\nGenerated files:")
        print("  1. knn_comprehensive_analysis.png")
        print("  2. ann_comprehensive_analysis.png")
        print("  3. knn_vs_ann_comparison.png")
        print("  4. comprehensive_model_dashboard.png")

        return True


def main():
    """Main execution."""
    visualizer = KNNANNVisualizer()
    success = visualizer.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
