#!/usr/bin/env python3
"""
Common Pipeline Data Loader for Visualization Scripts

This module provides a unified way for all visualization scripts to load
the latest pipeline report data.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class PipelineDataLoader:
    """Load and extract data from pipeline reports."""

    def __init__(self, reports_dir: Optional[Path] = None):
        """
        Initialize the data loader.

        Args:
            reports_dir: Directory containing pipeline reports (default: ../../reports)
        """
        if reports_dir is None:
            # Default to project root / reports
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            reports_dir = project_root / "reports"

        self.reports_dir = Path(reports_dir)
        self.report_data = None
        self.latest_report_path = None

    def load_latest_report(self, verbose: bool = True) -> bool:
        """
        Load the most recent pipeline report.

        Args:
            verbose: Print loading status

        Returns:
            True if report loaded successfully, False otherwise
        """
        json_files = list(self.reports_dir.glob("pipeline_report_*.json"))

        if not json_files:
            if verbose:
                print(f"❌ No pipeline reports found in {self.reports_dir}")
            return False

        self.latest_report_path = max(json_files, key=lambda p: p.stat().st_mtime)

        if verbose:
            print(f"📊 Loading latest report: {self.latest_report_path.name}")

        try:
            with open(self.latest_report_path, 'r') as f:
                self.report_data = json.load(f)

            if verbose:
                exp_id = self.report_data.get('experiment_id', 'Unknown')
                timestamp = self.report_data.get('timestamp', 'Unknown')
                print(f"✅ Loaded experiment: {exp_id} from {timestamp}")

            return True
        except Exception as e:
            if verbose:
                print(f"❌ Error loading report: {e}")
            return False

    def get_experiment_info(self) -> Dict[str, Any]:
        """Get basic experiment information."""
        if not self.report_data:
            return {}

        return {
            'experiment_id': self.report_data.get('experiment_id', 'Unknown'),
            'timestamp': self.report_data.get('timestamp', 'Unknown'),
            'pipeline_version': self.report_data.get('pipeline_version', 'Unknown')
        }

    def get_all_models_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract comprehensive data for all models from comprehensive_comparison.

        Returns:
            Dictionary mapping model names to their complete metrics
        """
        if not self.report_data:
            return {}

        all_models = {}
        modeling_results = self.report_data.get('modeling_results', {})

        # Priority 1: Get from comprehensive_comparison
        comp_comparison = modeling_results.get('comprehensive_comparison', {})
        if comp_comparison and 'model_rankings' in comp_comparison:
            for rank_data in comp_comparison['model_rankings']:
                model_name = rank_data['model_name']
                all_models[model_name] = {
                    'test_r2': rank_data.get('r2_score', 0),
                    'test_rmse': rank_data.get('rmse_score', 0),
                    'rank_by_r2': rank_data.get('rank_by_r2', 0),
                    'rank_by_rmse': rank_data.get('rank_by_rmse', 0),
                    'training_time': rank_data.get('training_time', 0),
                    'is_knn': rank_data.get('is_knn', False)
                }

        # Priority 2: Augment with detailed metrics from model_results
        model_results = modeling_results.get('model_results', {})
        for model_name in all_models.keys():
            if model_name in model_results:
                model_data = model_results[model_name]
                evaluation = model_data.get('evaluation', {})

                train_metrics = evaluation.get('train_metrics', {})
                test_metrics = evaluation.get('test_metrics', {})
                cv_metrics = evaluation.get('cv_metrics', {})

                # Add detailed metrics
                all_models[model_name].update({
                    'train_r2': train_metrics.get('r2', 0),
                    'train_rmse': train_metrics.get('rmse', 0),
                    'train_mae': train_metrics.get('mae', 0),
                    'train_mape': train_metrics.get('mape', 0),
                    'test_mae': test_metrics.get('mae', 0),
                    'test_mape': test_metrics.get('mape', 0),
                    'cv_stability': cv_metrics.get('cv_stability', 'unknown'),
                    'overfitting_gap': train_metrics.get('r2', 0) - test_metrics.get('r2', 0),
                })

                # Get CV summary if available
                comp_cv = evaluation.get('comprehensive_cv', {})
                cv_summary = comp_cv.get('cv_summary', {})
                if cv_summary and 'r2' in cv_summary:
                    all_models[model_name].update({
                        'cv_r2_mean': cv_summary['r2'].get('mean', 0),
                        'cv_r2_std': cv_summary['r2'].get('std', 0),
                    })

        return all_models

    def get_model_data(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific model.

        Args:
            model_name: Name of the model (case-insensitive)

        Returns:
            Model data dictionary or None if not found
        """
        all_models = self.get_all_models_data()

        # Try exact match first
        if model_name in all_models:
            return all_models[model_name]

        # Try case-insensitive match
        for name, data in all_models.items():
            if name.lower() == model_name.lower():
                return data

        return None

    def get_knn_specific_data(self) -> Optional[Dict[str, Any]]:
        """Get KNN-specific analysis data."""
        if not self.report_data:
            return None

        modeling_results = self.report_data.get('modeling_results', {})
        comp_comparison = modeling_results.get('comprehensive_comparison', {})

        return comp_comparison.get('knn_specific_analysis', None)

    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """Get best model information."""
        if not self.report_data:
            return None

        modeling_results = self.report_data.get('modeling_results', {})
        return modeling_results.get('best_model', None)

    def get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Get performance summary from comprehensive comparison."""
        if not self.report_data:
            return None

        modeling_results = self.report_data.get('modeling_results', {})
        comp_comparison = modeling_results.get('comprehensive_comparison', {})

        return comp_comparison.get('performance_summary', None)


def load_latest_pipeline_data(reports_dir: Optional[Path] = None, verbose: bool = True) -> PipelineDataLoader:
    """
    Convenience function to load latest pipeline data.

    Args:
        reports_dir: Directory containing pipeline reports
        verbose: Print loading status

    Returns:
        PipelineDataLoader instance with loaded data
    """
    loader = PipelineDataLoader(reports_dir)
    loader.load_latest_report(verbose=verbose)
    return loader


if __name__ == "__main__":
    # Test the loader
    print("Testing Pipeline Data Loader...")
    loader = load_latest_pipeline_data()

    if loader.report_data:
        print(f"\n📋 Experiment Info:")
        info = loader.get_experiment_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        print(f"\n📊 Models Found:")
        all_models = loader.get_all_models_data()
        for model_name, data in all_models.items():
            print(f"  {model_name}: R² = {data.get('test_r2', 0):.4f}")

        best = loader.get_best_model()
        if best:
            if isinstance(best, dict):
                print(f"\n🏆 Best Model: {best.get('name', 'Unknown')}")
            else:
                print(f"\n🏆 Best Model: {best}")

        knn_data = loader.get_knn_specific_data()
        if knn_data:
            print(f"\n🤖 KNN Optimal K: {knn_data.get('optimal_k', 'N/A')}")
    else:
        print("\n❌ No pipeline data loaded")
