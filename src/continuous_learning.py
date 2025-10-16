"""
Continuous learning module for self-improvement cycle with warm starts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json
import joblib

import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import MODELS_DIR, CACHE_DIR, TARGET_COLUMN
    from src.logger import pipeline_logger
    from redis_cache import cache
    from src.meta_learner import create_meta_learner
    from src.version_control import create_version_control
except ImportError as e:
    print(f"Import error in continuous_learning.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

class ContinuousLearning:
    """
    Continuous learning system with warm starts and self-improvement.
    """

    def __init__(self):
        self.meta_learner = create_meta_learner()
        self.version_control = create_version_control()
        self.learning_history = []
        self.performance_threshold = 0.75  # Minimum acceptable performance
        self.improvement_threshold = 0.02  # Minimum improvement to consider significant

    def initialize_learning(self, initial_data_info: Dict[str, Any]):
        """
        Initialize the continuous learning system.

        Args:
            initial_data_info: Information about the initial dataset
        """
        pipeline_logger.info("Initializing continuous learning system")

        # Load existing meta-learner if available
        self.meta_learner.load_meta_model()

        # Record initialization
        self.learning_history.append({
            'event': 'initialization',
            'timestamp': datetime.now().isoformat(),
            'data_info': initial_data_info
        })

        pipeline_logger.info("Continuous learning system initialized")

    def warm_start_model(self, model_type: str, dataset_features: Dict[str, Any],
                        previous_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide warm start configuration for a new model.

        Args:
            model_type: Type of model to train
            dataset_features: Features describing the current dataset
            previous_experiments: Results from previous experiments

        Returns:
            Warm start configuration
        """
        pipeline_logger.info(f"Generating warm start configuration for {model_type}")

        # Get meta-learning recommendations
        meta_recommendations = self.meta_learner.predict_optimal_config(dataset_features, model_type)

        # Analyze previous experiments for this model type
        relevant_experiments = [exp for exp in previous_experiments
                              if any(model_name.startswith(model_type) for model_name in exp.get('model_results', {}))]

        warm_start_config = {
            'model_type': model_type,
            'recommended_config': meta_recommendations.get('recommended_config', {}),
            'confidence': meta_recommendations.get('confidence', 0.5),
            'inspired_by_experiments': len(relevant_experiments),
            'performance_expectation': meta_recommendations.get('predicted_performance', 0.5)
        }

        # Extract successful patterns from previous experiments
        if relevant_experiments:
            successful_patterns = self._extract_successful_patterns(relevant_experiments, model_type)
            warm_start_config['successful_patterns'] = successful_patterns

        pipeline_logger.info(f"Warm start configuration generated with confidence {warm_start_config['confidence']:.2f}")

        return warm_start_config

    def _extract_successful_patterns(self, experiments: List[Dict[str, Any]], model_type: str) -> Dict[str, Any]:
        """
        Extract successful patterns from previous experiments.

        Args:
            experiments: List of previous experiment results
            model_type: Type of model

        Returns:
            Dictionary with successful patterns
        """
        patterns = {
            'best_performing_configs': [],
            'common_hyperparameters': {},
            'performance_trends': []
        }

        successful_experiments = []

        for exp in experiments:
            model_results = exp.get('model_results', {})
            for model_name, results in model_results.items():
                if model_name.startswith(model_type):
                    evaluation = results.get('evaluation', {})
                    if evaluation.get('evaluation_type') == 'regression':
                        performance = evaluation.get('test_r2', 0)
                    else:
                        performance = evaluation.get('test_accuracy', 0)

                    if performance > self.performance_threshold:
                        successful_experiments.append({
                            'config': results.get('info', {}).get('parameters', {}),
                            'performance': performance,
                            'experiment_id': exp.get('experiment_id')
                        })

        # Sort by performance and get top configurations
        successful_experiments.sort(key=lambda x: x['performance'], reverse=True)
        patterns['best_performing_configs'] = successful_experiments[:3]

        # Extract common hyperparameters
        if successful_experiments:
            all_params = [exp['config'] for exp in successful_experiments]
            common_params = {}

            for param in set().union(*[set(p.keys()) for p in all_params]):
                values = [p.get(param) for p in all_params if param in p]
                if values:
                    # Find most common value
                    most_common = max(set(values), key=values.count)
                    common_params[param] = {
                        'most_common_value': most_common,
                        'frequency': values.count(most_common) / len(values)
                    }

            patterns['common_hyperparameters'] = common_params

        return patterns

    def evaluate_learning_progress(self, current_results: Dict[str, Any],
                                 historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate progress in the learning system.

        Args:
            current_results: Results from current experiment
            historical_results: Results from previous experiments

        Returns:
            Learning progress evaluation
        """
        pipeline_logger.info("Evaluating learning progress")

        progress_evaluation = {
            'current_performance': {},
            'historical_comparison': {},
            'improvement_metrics': {},
            'learning_insights': []
        }

        # Extract current performance
        for model_name, results in current_results.get('model_results', {}).items():
            evaluation = results.get('evaluation', {})
            if evaluation.get('evaluation_type') == 'regression':
                progress_evaluation['current_performance'][model_name] = evaluation.get('test_r2', 0)
            else:
                progress_evaluation['current_performance'][model_name] = evaluation.get('test_accuracy', 0)

        # Compare with historical performance
        if historical_results:
            for model_name in progress_evaluation['current_performance']:
                historical_performances = []

                for hist_result in historical_results[-5:]:  # Last 5 experiments
                    hist_models = hist_result.get('model_results', {})
                    if model_name in hist_models:
                        hist_eval = hist_models[model_name].get('evaluation', {})
                        if hist_eval.get('evaluation_type') == 'regression':
                            perf = hist_eval.get('test_r2', 0)
                        else:
                            perf = hist_eval.get('test_accuracy', 0)
                        historical_performances.append(perf)

                if historical_performances:
                    current_perf = progress_evaluation['current_performance'][model_name]
                    avg_hist_perf = np.mean(historical_performances)
                    improvement = current_perf - avg_hist_perf

                    progress_evaluation['historical_comparison'][model_name] = {
                        'current': current_perf,
                        'historical_avg': avg_hist_perf,
                        'improvement': improvement,
                        'improvement_percentage': (improvement / avg_hist_perf * 100) if avg_hist_perf > 0 else 0
                    }

                    # Check for significant improvement
                    if abs(improvement) > self.improvement_threshold:
                        if improvement > 0:
                            progress_evaluation['learning_insights'].append(
                                f"Significant improvement in {model_name} performance (+{improvement:.3f})"
                            )
                        else:
                            progress_evaluation['learning_insights'].append(
                                f"Performance decline in {model_name} (-{abs(improvement):.3f})"
                            )

        # Overall learning metrics
        all_current_perfs = list(progress_evaluation['current_performance'].values())
        if all_current_perfs:
            progress_evaluation['improvement_metrics']['average_current_performance'] = np.mean(all_current_perfs)
            progress_evaluation['improvement_metrics']['best_current_performance'] = np.max(all_current_perfs)

        # Learning stability (consistency across models)
        if len(all_current_perfs) > 1:
            progress_evaluation['improvement_metrics']['performance_consistency'] = np.std(all_current_perfs)

        pipeline_logger.info("Learning progress evaluation completed")

        return progress_evaluation

    def adapt_strategy(self, progress_evaluation: Dict[str, Any],
                      current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt learning strategy based on progress evaluation.

        Args:
            progress_evaluation: Results from progress evaluation
            current_context: Current learning context

        Returns:
            Adapted strategy recommendations
        """
        pipeline_logger.info("Adapting learning strategy")

        strategy_adaptations = {
            'strategy_changes': [],
            'model_priorities': {},
            'exploration_suggestions': [],
            'risk_assessment': 'low'
        }

        # Analyze performance trends
        historical_comp = progress_evaluation.get('historical_comparison', {})

        improving_models = []
        declining_models = []

        for model_name, comp in historical_comp.items():
            improvement = comp.get('improvement', 0)
            if improvement > self.improvement_threshold:
                improving_models.append((model_name, improvement))
            elif improvement < -self.improvement_threshold:
                declining_models.append((model_name, improvement))

        # Strategy adaptations based on trends
        if improving_models:
            strategy_adaptations['strategy_changes'].append(
                f"Continue focusing on improving models: {', '.join([m[0] for m in improving_models])}"
            )
            # Prioritize improving models
            for model_name, improvement in improving_models:
                strategy_adaptations['model_priorities'][model_name] = 'high'

        if declining_models:
            strategy_adaptations['strategy_changes'].append(
                f"Investigate performance decline in: {', '.join([m[0] for m in declining_models])}"
            )
            strategy_adaptations['risk_assessment'] = 'medium'

            # Suggest exploration for declining models
            for model_name, _ in declining_models:
                strategy_adaptations['exploration_suggestions'].append(
                    f"Try different hyperparameters or feature engineering for {model_name}"
                )

        # Check for overfitting/underfitting patterns
        current_perf = progress_evaluation.get('current_performance', {})
        historical_comp = progress_evaluation.get('historical_comparison', {})

        for model_name in current_perf:
            if model_name in historical_comp:
                comp = historical_comp[model_name]
                current = comp['current']
                hist_avg = comp['historical_avg']

                # Potential overfitting (much better than historical average)
                if current > hist_avg + 0.1:
                    strategy_adaptations['exploration_suggestions'].append(
                        f"Validate {model_name} performance on additional test sets to check for overfitting"
                    )

                # Potential underfitting (consistently poor performance)
                elif current < 0.6 and hist_avg < 0.6:
                    strategy_adaptations['exploration_suggestions'].append(
                        f"Consider more complex models or additional features for {model_name}"
                    )

        # Meta-learning insights
        meta_insights = self.meta_learner.generate_meta_insights()
        strategy_adaptations['meta_insights'] = meta_insights.get('insights', [])
        strategy_adaptations['meta_recommendations'] = meta_insights.get('recommendations', [])

        pipeline_logger.info("Strategy adaptation completed")

        return strategy_adaptations

    def update_learning_system(self, experiment_results: Dict[str, Any],
                             progress_evaluation: Dict[str, Any],
                             strategy_adaptations: Dict[str, Any]):
        """
        Update the learning system with new insights.

        Args:
            experiment_results: Results from the latest experiment
            progress_evaluation: Progress evaluation results
            strategy_adaptations: Strategy adaptation recommendations
        """
        pipeline_logger.info("Updating learning system")

        # Update meta-learner with new experiment data
        update_result = self.meta_learner.update_meta_model(experiment_results)

        # Record learning event (convert any non-serializable objects)
        learning_event = {
            'event': 'system_update',
            'timestamp': datetime.now().isoformat(),
            'experiment_results': self._make_serializable(experiment_results),
            'progress_evaluation': self._make_serializable(progress_evaluation),
            'strategy_adaptations': self._make_serializable(strategy_adaptations),
            'meta_update_result': self._make_serializable(update_result)
        }

        self.learning_history.append(learning_event)

        # Save learning history
        self._save_learning_history()

        pipeline_logger.info("Learning system updated successfully")

    def _save_learning_history(self):
        """Save learning history to disk."""
        history_file = CACHE_DIR / "learning_history.json"

        try:
            # Keep only recent history to avoid file bloat
            recent_history = self.learning_history[-50:]  # Last 50 events

            # Convert any non-serializable objects (like pandas Intervals) to strings
            serializable_history = []
            for event in recent_history:
                serializable_event = {}
                for key, value in event.items():
                    if hasattr(value, 'dtype') and 'interval' in str(value.dtype).lower():
                        serializable_event[key] = str(value)
                    else:
                        serializable_event[key] = value
                serializable_history.append(serializable_event)

            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2, default=str)

        except Exception as e:
            pipeline_logger.error(f"Error saving learning history: {e}")

    def load_learning_history(self):
        """Load learning history from disk."""
        history_file = CACHE_DIR / "learning_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.learning_history = json.load(f)
                pipeline_logger.info("Learning history loaded")
            except Exception as e:
                pipeline_logger.error(f"Error loading learning history: {e}")

    def generate_learning_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive learning report.

        Returns:
            Dictionary with learning system status and insights
        """
        report = {
            'learning_system_status': {
                'total_learning_events': len(self.learning_history),
                'meta_learner_status': 'trained' if self.meta_learner.meta_model else 'not_trained',
                'experiment_history_size': len(self.meta_learner.experiment_history)
            },
            'current_insights': self.meta_learner.generate_meta_insights(),
            'learning_trends': self._analyze_learning_trends(),
            'recommendations': []
        }

        # Generate recommendations based on current state
        if not self.meta_learner.meta_model:
            report['recommendations'].append("Train meta-learner with more experiment data")
        else:
            meta_stats = self.meta_learner.get_meta_learning_stats()
            if meta_stats.get('total_experiments', 0) < 20:
                report['recommendations'].append("Continue collecting experiment data for better meta-learning")

        # Performance trends
        trends = report['learning_trends']
        if trends.get('overall_trend') == 'improving':
            report['recommendations'].append("Current learning trajectory is positive - maintain current approach")
        elif trends.get('overall_trend') == 'declining':
            report['recommendations'].append("Performance is declining - review recent changes and experiment configurations")

        return report

    def _analyze_learning_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in the learning system.

        Returns:
            Dictionary with trend analysis
        """
        trends = {
            'overall_trend': 'stable',
            'performance_trend': [],
            'consistency_trend': []
        }

        if len(self.learning_history) < 3:
            return trends

        # Analyze recent learning events
        recent_events = [event for event in self.learning_history[-10:]
                        if event.get('event') == 'system_update']

        if recent_events:
            performances = []
            consistencies = []

            for event in recent_events:
                progress = event.get('progress_evaluation', {})
                avg_perf = progress.get('improvement_metrics', {}).get('average_current_performance')
                consistency = progress.get('improvement_metrics', {}).get('performance_consistency')

                if avg_perf is not None:
                    performances.append(avg_perf)
                if consistency is not None:
                    consistencies.append(consistency)

            # Calculate trends
            if len(performances) > 1:
                perf_trend = np.polyfit(range(len(performances)), performances, 1)[0]
                trends['performance_trend'] = performances
                trends['performance_slope'] = float(perf_trend)

                if perf_trend > 0.01:
                    trends['overall_trend'] = 'improving'
                elif perf_trend < -0.01:
                    trends['overall_trend'] = 'declining'

            if consistencies:
                trends['consistency_trend'] = consistencies

        return trends

    def get_warm_start_recommendations(self, model_type: str,
                                     dataset_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive warm start recommendations.

        Args:
            model_type: Type of model
            dataset_features: Dataset characteristics

        Returns:
            Warm start recommendations
        """
        # Get meta-learning recommendations
        meta_rec = self.meta_learner.predict_optimal_config(dataset_features, model_type)

        # Get learning system insights
        learning_report = self.generate_learning_report()

        recommendations = {
            'model_type': model_type,
            'meta_recommendations': meta_rec,
            'learning_insights': learning_report.get('current_insights', {}),
            'system_recommendations': learning_report.get('recommendations', []),
            'confidence_score': self._calculate_overall_confidence(meta_rec, learning_report)
        }

        return recommendations

    def _calculate_overall_confidence(self, meta_rec: Dict[str, Any],
                                    learning_report: Dict[str, Any]) -> float:
        """
        Calculate overall confidence in recommendations.

        Args:
            meta_rec: Meta-learning recommendations
            learning_report: Learning system report

        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []

        # Meta-learning confidence
        meta_conf = meta_rec.get('confidence', 0.5)
        confidence_factors.append(meta_conf)

        # Learning system maturity
        meta_stats = self.meta_learner.get_meta_learning_stats()
        experiment_count = meta_stats.get('total_experiments', 0)

        # Confidence increases with more experiments
        maturity_conf = min(1.0, experiment_count / 50)  # Full confidence at 50 experiments
        confidence_factors.append(maturity_conf)

        # Recent performance consistency
        trends = learning_report.get('learning_trends', {})
        consistency_trend = trends.get('consistency_trend', [])

        if consistency_trend:
            # Lower consistency (more variation) reduces confidence
            avg_consistency = np.mean(consistency_trend)
            consistency_conf = max(0.3, 1.0 - avg_consistency)  # Higher consistency = higher confidence
            confidence_factors.append(consistency_conf)

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _make_serializable(self, obj):
        """
        Convert object to JSON-serializable format.

        Args:
            obj: Object to make serializable

        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            # For custom objects, try to serialize their __dict__
            try:
                return self._make_serializable(obj.__dict__)
            except:
                return str(obj)
        else:
            # For other types, try to convert or return string representation
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

def create_continuous_learning() -> ContinuousLearning:
    """
    Factory function to create ContinuousLearning instance.

    Returns:
        ContinuousLearning instance
    """
    return ContinuousLearning()