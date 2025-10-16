"""
Main pipeline orchestrator for the advanced modeling system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import uuid

import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import (
        PREPROCESSED_DATA_FILE, TARGET_COLUMN, REPORTS_DIR
    )
    from src.logger import log_experiment_start, log_experiment_end
    from src.logger import pipeline_logger
    from src.data_loader import create_data_loader
    from redis_cache import cache, cache_evaluation_metrics
    from src.model_pipeline import create_model_pipeline
    from src.evaluation_engine import create_evaluation_engine
    from src.qualitative_evaluator import create_qualitative_evaluator
    from src.meta_learner import create_meta_learner
    from src.version_control import create_version_control
    from src.continuous_learning import create_continuous_learning
except ImportError as e:
    print(f"Import error in main_pipeline.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

from tqdm.auto import tqdm

class AdvancedModelingPipeline:
    """
    Main orchestrator for the advanced modeling pipeline.
    """

    def __init__(self, show_progress: bool = True):
        # Toggle terminal progress bars (useful for CI or non-interactive runs)
        self.show_progress = show_progress

        self.data_loader = create_data_loader()
        self.model_pipeline = create_model_pipeline()
        self.evaluation_engine = create_evaluation_engine()
        self.qualitative_evaluator = create_qualitative_evaluator()
        self.meta_learner = create_meta_learner()
        self.version_control = create_version_control()
        self.continuous_learning = create_continuous_learning()

        self.current_experiment_id = None
        self.experiment_results = {}

    def run_complete_pipeline(self, experiment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete advanced modeling pipeline.

        Args:
            experiment_config: Optional configuration for the experiment

        Returns:
            Complete pipeline results
        """
        self.current_experiment_id = str(uuid.uuid4())[:8]

        pipeline_logger.info(f"Starting complete pipeline run: {self.current_experiment_id}")

        try:
            # Top-level pipeline phases (for high-level progress tracking)
            phases = [
                ("Data", self._run_data_pipeline),
                ("Modeling", lambda: self._run_modeling_pipeline(data_results, experiment_config)),
                ("Qualitative", lambda: self._run_qualitative_evaluation(modeling_results)),
                ("Learning", lambda: self._run_learning_pipeline(modeling_results, qualitative_results)),
                ("Finalizing", lambda: self._finalize_pipeline(data_results, modeling_results, qualitative_results, learning_results))
            ]

            data_results = {}
            modeling_results = {}
            qualitative_results = {}
            learning_results = {}

            phase_iterator = tqdm(phases, desc="Pipeline phases", leave=True) if self.show_progress else phases

            for phase_name, phase_fn in phase_iterator:
                pipeline_logger.info(f"Starting phase: {phase_name}")

                if phase_name == "Data":
                    data_results = phase_fn()
                    if 'error' in data_results:
                        return {'error': f"Data pipeline failed: {data_results['error']}"}

                elif phase_name == "Modeling":
                    modeling_results = phase_fn()
                    if 'error' in modeling_results:
                        return {'error': f"Modeling pipeline failed: {modeling_results['error']}"}

                elif phase_name == "Qualitative":
                    qualitative_results = phase_fn()
                    if 'error' in qualitative_results:
                        pipeline_logger.warning(f"Qualitative evaluation failed: {qualitative_results['error']}")

                elif phase_name == "Learning":
                    learning_results = phase_fn()

                elif phase_name == "Finalizing":
                    final_results = phase_fn()
            if 'error' in data_results:
                return {'error': f"Data pipeline failed: {data_results['error']}"}

            pipeline_logger.info(f"Pipeline run {self.current_experiment_id} completed successfully")

            return final_results

        except Exception as e:
            pipeline_logger.error(f"Pipeline run {self.current_experiment_id} failed: {e}")
            return {'error': str(e), 'experiment_id': self.current_experiment_id}

    def _run_data_pipeline(self) -> Dict[str, Any]:
        """
        Run the data loading and preparation pipeline.

        Returns:
            Data pipeline results
        """
        pipeline_logger.info("Running data pipeline")

        try:
            # Load and preprocess data
            df = self.data_loader.load_data()
            X, y = self.data_loader.preprocess_features(df)

            # Split data
            X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y)

            # Validate data quality
            quality_report = self.data_loader.validate_data_quality(df)

            # Create dataset version
            dataset_version = self.version_control.create_dataset_version(
                self.data_loader.data_hash,
                {
                    'total_samples': len(df),
                    'features': list(X.columns),
                    'target': TARGET_COLUMN,
                    'quality_report': quality_report
                }
            )

            results = {
                'dataset_version': dataset_version,
                'data_hash': self.data_loader.data_hash,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_columns': self.data_loader.feature_columns,
                'data_quality': quality_report,
                'data_info': self.data_loader.get_data_info()
            }

            pipeline_logger.info("Data pipeline completed successfully")
            return results

        except Exception as e:
            pipeline_logger.error(f"Data pipeline failed: {e}")
            return {'error': str(e)}

    def _run_modeling_pipeline(self, data_results: Dict[str, Any],
                             experiment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the modeling pipeline with training and evaluation.

        Args:
            data_results: Results from data pipeline
            experiment_config: Optional experiment configuration

        Returns:
            Modeling pipeline results
        """
        pipeline_logger.info("Running modeling pipeline")

        try:
            X_train = data_results['X_train']
            X_test = data_results['X_test']
            y_train = data_results['y_train']
            y_test = data_results['y_test']

            # Default model types to train
            model_types = experiment_config.get('model_types', ['linear', 'decision_tree', 'random_forest']) if experiment_config else ['linear', 'decision_tree', 'random_forest']

            model_results = {}
            evaluation_results = {}

            iterator = tqdm(model_types, desc="Training models", leave=False) if self.show_progress else model_types

            for model_type in iterator:
                try:
                    # Get warm start configuration
                    dataset_features = {
                        'dataset_size': len(X_train),
                        'n_features': X_train.shape[1],
                        'target_mean': np.mean(y_train),
                        'target_std': np.std(y_train)
                    }

                    warm_start_config = self.continuous_learning.warm_start_model(
                        model_type, dataset_features, list(self.experiment_results.values())
                    )

                    # Use warm start parameters if available
                    params = warm_start_config.get('recommended_config', {})

                    # Train model
                    train_results = self.model_pipeline.train_model(model_type, X_train, y_train, params)
                    model_id = train_results['model_id']

                    # Evaluate model
                    pipeline_logger.info(f"Evaluating model {model_id} - checking if exists in trained_models: {model_id in self.model_pipeline.trained_models}")
                    if model_id not in self.model_pipeline.trained_models:
                        pipeline_logger.error(f"Model {model_id} not found in trained_models during evaluation!")
                        raise ValueError(f"Model {model_id} not found in trained_models")

                    eval_results = self.evaluation_engine.evaluate_regression_model(
                        self.model_pipeline.trained_models[model_id],
                        X_train, X_test, y_train, y_test,
                        model_name=f"{model_type}_{model_id}"
                    )

                    # Store evaluation results in cache with experiment ID
                    cache_evaluation_metrics(f"{self.current_experiment_id}_{model_type}", eval_results)

                    # Store evaluation results in cache with experiment ID
                    cache_evaluation_metrics(self.current_experiment_id, eval_results)

                    # Create model version
                    model_version = self.version_control.create_model_version(
                        model_type,
                        {
                            'model_id': model_id,
                            'parameters': params,
                            'warm_start_config': warm_start_config
                        },
                        eval_results
                    )

                    model_results[model_type] = {
                        'model_id': model_id,
                        'version': model_version,
                        'training': train_results,
                        'evaluation': eval_results,
                        'warm_start': warm_start_config
                    }

                    evaluation_results[model_type] = eval_results

                except Exception as e:
                    pipeline_logger.error(f"Failed to train/evaluate {model_type}: {e}")
                    model_results[model_type] = {'error': str(e)}

            # Compare models
            if len([r for r in model_results.values() if 'error' not in r]) > 1:
                valid_model_ids = [r['model_id'] for r in model_results.values() if 'error' not in r]
                comparison_results = self.model_pipeline.compare_models(valid_model_ids, X_test, y_test)
                model_results['comparison'] = comparison_results

            results = {
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'best_model': model_results.get('comparison', {}).get('best_model')
            }

            pipeline_logger.info("Modeling pipeline completed successfully")
            return results

        except Exception as e:
            pipeline_logger.error(f"Modeling pipeline failed: {e}")
            return {'error': str(e)}

    def _run_qualitative_evaluation(self, modeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run qualitative evaluation pipeline.

        Args:
            modeling_results: Results from modeling pipeline

        Returns:
            Qualitative evaluation results
        """
        pipeline_logger.info("Running qualitative evaluation")

        try:
            qualitative_results = {}

            # Get data from modeling results (assuming we can access it)
            # In a full implementation, we'd pass the data through
            data_results = self._get_current_data()

            if 'error' in data_results:
                return {'error': 'Cannot access data for qualitative evaluation'}

            X_test = data_results['X_test']
            y_test = data_results['y_test']

            for model_name, model_data in modeling_results.get('model_results', {}).items():
                if 'error' in model_data:
                    continue

                model_id = model_data['model_id']
                model = self.model_pipeline.trained_models.get(model_id)

                if model is None:
                    pipeline_logger.warning(f"Model {model_id} not found in trained_models, skipping qualitative evaluation")
                    continue

                if model:
                    try:
                        y_pred = model.predict(X_test)

                        # SHAP Analysis
                        shap_results = self.qualitative_evaluator.perform_shap_analysis(
                            model, data_results['X_train'], X_test, model_name
                        )

                        # Error Analysis
                        error_results = self.qualitative_evaluator.perform_error_analysis(
                            model, X_test, y_test, y_pred, model_name
                        )

                        # Business Alignment
                        business_results = self.qualitative_evaluator.check_business_alignment(
                            model, X_test, y_test, y_pred, model_name
                        )

                        qualitative_results[model_name] = {
                            'shap_analysis': shap_results,
                            'error_analysis': error_results,
                            'business_alignment': business_results
                        }

                    except Exception as e:
                        pipeline_logger.error(f"Qualitative evaluation failed for {model_name}: {e}")
                        qualitative_results[model_name] = {'error': str(e)}

            # Generate qualitative report
            if qualitative_results:
                report = self.qualitative_evaluator.generate_qualitative_report(qualitative_results)
                qualitative_results['report'] = report

            pipeline_logger.info("Qualitative evaluation completed")
            return qualitative_results

        except Exception as e:
            pipeline_logger.error(f"Qualitative evaluation failed: {e}")
            return {'error': str(e)}

    def _run_learning_pipeline(self, modeling_results: Dict[str, Any],
                             qualitative_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the meta-learning and continuous learning pipeline.

        Args:
            modeling_results: Results from modeling pipeline
            qualitative_results: Results from qualitative evaluation

        Returns:
            Learning pipeline results
        """
        pipeline_logger.info("Running learning pipeline")

        try:
            # Combine all results for learning
            experiment_data = {
                'experiment_id': self.current_experiment_id,
                'model_results': modeling_results.get('model_results', {}),
                'qualitative_results': qualitative_results,
                'data_info': self._get_current_data().get('data_info', {})
            }

            # Update meta-learner
            meta_update = self.meta_learner.collect_experiment_data(experiment_data)

            # Evaluate learning progress
            progress_eval = self.continuous_learning.evaluate_learning_progress(
                experiment_data, list(self.experiment_results.values())
            )

            # Adapt learning strategy
            strategy_adapt = self.continuous_learning.adapt_strategy(progress_eval, {})

            # Update continuous learning system
            self.continuous_learning.update_learning_system(
                experiment_data, progress_eval, strategy_adapt
            )

            results = {
                'meta_update': meta_update,
                'progress_evaluation': progress_eval,
                'strategy_adaptation': strategy_adapt,
                'learning_report': self.continuous_learning.generate_learning_report()
            }

            pipeline_logger.info("Learning pipeline completed")
            return results

        except Exception as e:
            pipeline_logger.error(f"Learning pipeline failed: {e}")
            return {'error': str(e)}

    def _finalize_pipeline(self, data_results: Dict[str, Any], modeling_results: Dict[str, Any],
                         qualitative_results: Dict[str, Any], learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize the pipeline with reporting and version control.

        Args:
            data_results: Data pipeline results
            modeling_results: Modeling pipeline results
            qualitative_results: Qualitative evaluation results
            learning_results: Learning pipeline results

        Returns:
            Final pipeline results
        """
        pipeline_logger.info("Finalizing pipeline")

        # Compile complete results
        final_results = {
            'experiment_id': self.current_experiment_id,
            'timestamp': datetime.now().isoformat(),
            'data_results': data_results,
            'modeling_results': modeling_results,
            'qualitative_results': qualitative_results,
            'learning_results': learning_results,
            'pipeline_version': '1.0.0'
        }

        # Store experiment results
        self.experiment_results[self.current_experiment_id] = final_results

        # Generate comprehensive report
        report_path = REPORTS_DIR / f"pipeline_report_{self.current_experiment_id}.json"
        with open(report_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        # Log completion
        log_experiment_end(self.current_experiment_id, {
            'status': 'completed',
            'best_model': modeling_results.get('best_model'),
            'total_models': len([r for r in modeling_results.get('model_results', {}).values() if 'error' not in r])
        })

        pipeline_logger.info(f"Pipeline finalized. Report saved to {report_path}")

        return final_results

    def _get_current_data(self) -> Dict[str, Any]:
        """
        Get current data information (helper method).

        Returns:
            Current data information
        """
        try:
            # This is a simplified version - in practice, we'd store the data
            df = self.data_loader.load_data()
            X, y = self.data_loader.preprocess_features(df)
            _, X_test, _, y_test = self.data_loader.split_data(X, y)
            return {
                'X_test': X_test,
                'y_test': y_test,
                'X_train': X,  # Simplified
                'data_info': self.data_loader.get_data_info()
            }
        except Exception as e:
            return {'error': str(e)}

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and statistics.

        Returns:
            Pipeline status information
        """
        return {
            'total_experiments': len(self.experiment_results),
            'current_experiment': self.current_experiment_id,
            'cache_stats': cache.get_stats(),
            'version_stats': self.version_control.get_version_stats(),
            'meta_learner_stats': self.meta_learner.get_meta_learning_stats(),
            'learning_report': self.continuous_learning.generate_learning_report()
        }

    def run_quick_evaluation(self, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Run a quick evaluation with specified model types.

        Args:
            model_types: List of model types to evaluate

        Returns:
            Quick evaluation results
        """
        if model_types is None:
            model_types = ['linear', 'random_forest']

        config = {'model_types': model_types}
        return self.run_complete_pipeline(config)

def create_advanced_pipeline(show_progress: bool = True) -> AdvancedModelingPipeline:
    """
    Factory function to create AdvancedModelingPipeline instance.

    Returns:
        AdvancedModelingPipeline instance
    """
    return AdvancedModelingPipeline(show_progress=show_progress)

# Convenience functions for different use cases
def run_standard_pipeline(show_progress: bool = True) -> Dict[str, Any]:
    """
    Run the standard pipeline with default settings.

    Returns:
        Pipeline results
    """
    pipeline = create_advanced_pipeline(show_progress=show_progress)
    return pipeline.run_complete_pipeline()

def run_quick_comparison(model_types: List[str] = None, show_progress: bool = True) -> Dict[str, Any]:
    """
    Run a quick model comparison.

    Args:
        model_types: List of model types to compare

    Returns:
        Comparison results
    """
    pipeline = create_advanced_pipeline(show_progress=show_progress)
    return pipeline.run_quick_evaluation(model_types)

if __name__ == "__main__":
    # Example usage
    print("Starting Advanced Modeling Pipeline...")

    # Run standard pipeline
    results = run_standard_pipeline()

    print("🚀 Starting Advanced Modeling Pipeline...")
    result = run_standard_pipeline()

    if 'error' in result:
        print(f"❌ Pipeline failed: {result['error']}")
        
    else:
        print("✅ Pipeline completed successfully!")
        print(f"📊 Experiment ID: {result['experiment_id']}")
        print(f"🏆 Best model: {result.get('modeling_results', {}).get('best_model', 'N/A')}")

        # Print summary
        modeling = results.get('modeling_results', {})
        if 'evaluation_results' in modeling:
            print("\nModel Performance Summary:")
            for model_name, eval_result in modeling['evaluation_results'].items():
                if 'test_r2' in eval_result:
                    print(".4f")
                elif 'test_accuracy' in eval_result:
                    print(".4f")