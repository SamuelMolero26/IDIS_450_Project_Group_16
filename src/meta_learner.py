"""
Meta-learning module for configuration optimization and self-improvement.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import META_LEARNER_FEATURES, RANDOM_STATE, MODELS_DIR
    from src.logger import meta_logger
    from redis_cache import cache
except ImportError as e:
    print(f"Import error in meta_learner.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

class MetaLearner:
    """
    Meta-learning system for optimizing model configurations.
    """

    def __init__(self):
        self.meta_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.experiment_history = []
        self.performance_threshold = 0.7  # Minimum acceptable performance

    def collect_experiment_data(self, experiment_results: Dict[str, Any]) -> None:
        """
        Collect data from completed experiments for meta-learning.

        Args:
            experiment_results: Results from a modeling experiment
        """
        meta_logger.info("Collecting experiment data for meta-learning")

        # Extract relevant features for meta-learning
        meta_features = {}

        # Dataset characteristics
        if 'data_info' in experiment_results:
            data_info = experiment_results['data_info']
            meta_features['dataset_size'] = data_info.get('total_rows', 0)
            meta_features['n_features'] = len(data_info.get('feature_columns', []))

        # Target statistics
        if 'target_stats' in experiment_results:
            target_stats = experiment_results['target_stats']
            meta_features['target_mean'] = target_stats.get('mean', 0)
            meta_features['target_std'] = target_stats.get('std', 1)

        # Model information
        if 'model_results' in experiment_results:
            for model_name, model_data in experiment_results['model_results'].items():
                model_features = meta_features.copy()
                model_features['model_type'] = model_name

                # Performance metrics including CV stability
                if 'evaluation' in model_data:
                    eval_data = model_data['evaluation']
                    if eval_data.get('evaluation_type') == 'regression':
                        model_features['cv_mean_score'] = eval_data.get('cv_metrics', {}).get('rmse_mean', 0)
                        model_features['cv_std_score'] = eval_data.get('cv_metrics', {}).get('rmse_std', 0)
                        model_features['cv_stability_score'] = eval_data.get('cv_metrics', {}).get('cv_stability_score', 0)
                        model_features['test_score'] = eval_data.get('test_metrics', {}).get('r2', 0)
                    else:
                        model_features['cv_mean_score'] = eval_data.get('cv_metrics', {}).get('accuracy_mean', 0)
                        model_features['cv_std_score'] = eval_data.get('cv_metrics', {}).get('accuracy_std', 0)
                        model_features['cv_stability_score'] = eval_data.get('cv_metrics', {}).get('cv_stability_score', 0)
                        model_features['test_score'] = eval_data.get('test_metrics', {}).get('accuracy', 0)

                # Model parameters
                if 'info' in model_data and 'parameters' in model_data['info']:
                    params = model_data['info']['parameters']
                    model_features.update({f'param_{k}': v for k, v in params.items() if isinstance(v, (int, float))})

                # Qualitative insights
                if 'qualitative' in model_data:
                    qual_data = model_data['qualitative']
                    model_features['business_alignment_score'] = qual_data.get('business_alignment_score', 50)

                    # SHAP feature importance (top feature)
                    if 'shap_statistics' in qual_data:
                        shap_stats = qual_data['shap_statistics']
                        if 'mean_abs_shap' in shap_stats and shap_stats['mean_abs_shap']:
                            model_features['top_shap_importance'] = max(shap_stats['mean_abs_shap'])

                # Add to experiment history
                experiment_record = {
                    'features': model_features,
                    'performance': model_features.get('test_score', 0),
                    'timestamp': datetime.now().isoformat(),
                    'experiment_id': experiment_results.get('experiment_id', 'unknown')
                }

                self.experiment_history.append(experiment_record)

        meta_logger.info(f"Collected data from {len(self.experiment_history)} experiments")

    def train_meta_model(self) -> Dict[str, Any]:
        """
        Train the meta-learning model on collected experiment data.

        Returns:
            Training results and performance metrics
        """
        if len(self.experiment_history) < 10:
            meta_logger.warning("Insufficient experiment data for meta-learning training")
            return {'error': 'Insufficient data'}

        meta_logger.info("Training meta-learning model")

        # Prepare training data
        X_data = []
        y_data = []

        for record in self.experiment_history:
            features = record['features']
            performance = record['performance']

            # Filter to known meta-learning features including CV stability
            filtered_features = {k: v for k, v in features.items() if k in META_LEARNER_FEATURES or k.startswith('param_') or k == 'cv_stability_score'}

            X_data.append(filtered_features)
            y_data.append(performance)

        # Convert to DataFrame
        X_df = pd.DataFrame(X_data)
        y = np.array(y_data)

        # Handle missing values
        X_df = X_df.fillna(X_df.mean())

        # Encode categorical features
        for col in X_df.select_dtypes(include=['object', 'category']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_df[col] = self.label_encoders[col].fit_transform(X_df[col])
            else:
                X_df[col] = self.label_encoders[col].transform(X_df[col])

        # Scale features
        X_scaled = self.scaler.fit_transform(X_df)
        X_scaled = pd.DataFrame(X_scaled, columns=X_df.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
        )

        # Train meta-model
        self.meta_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        )

        self.meta_model.fit(X_train, y_train)

        # Evaluate meta-model
        y_pred = self.meta_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        feature_importance = dict(zip(X_df.columns, self.meta_model.feature_importances_))

        results = {
            'meta_model_trained': True,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'test_mse': mse,
            'test_r2': r2,
            'feature_importance': feature_importance,
            'training_timestamp': datetime.now().isoformat()
        }

        meta_logger.info(f"Meta-learning model trained. Test R²: {r2:.4f}")

        return results

    def predict_optimal_config(self, dataset_features: Dict[str, Any],
                             model_type: str) -> Dict[str, Any]:
        """
        Predict optimal configuration for a new dataset and model type.

        Args:
            dataset_features: Features describing the dataset
            model_type: Type of model to optimize

        Returns:
            Predicted optimal configuration
        """
        if self.meta_model is None:
            meta_logger.warning("Meta-model not trained, returning default configuration")
            return self._get_default_config(model_type)

        meta_logger.info(f"Predicting optimal config for {model_type}")

        # Prepare input features
        input_features = {}

        # Dataset features
        for feature in META_LEARNER_FEATURES:
            if feature in dataset_features:
                input_features[feature] = dataset_features[feature]
            else:
                input_features[feature] = 0  # Default value

        # Model type encoding
        input_features['model_type'] = model_type

        # Convert to DataFrame
        X_input = pd.DataFrame([input_features])

        # Handle categorical encoding
        for col in X_input.select_dtypes(include=['object', 'category']).columns:
            if col in self.label_encoders:
                try:
                    X_input[col] = self.label_encoders[col].transform(X_input[col])
                except:
                    X_input[col] = 0  # Unknown category
            else:
                X_input[col] = 0  # No encoder available

        # Handle missing columns
        trained_features = self.scaler.feature_names_in_
        for feature in trained_features:
            if feature not in X_input.columns:
                X_input[feature] = 0

        # Reorder columns to match training
        X_input = X_input[trained_features]

        # Scale features
        X_scaled = self.scaler.transform(X_input)

        # Predict performance for different configurations
        config_predictions = self._generate_config_predictions(X_scaled, model_type)

        # Select best configuration
        best_config = max(config_predictions.items(), key=lambda x: x[1]['predicted_performance'])

        meta_logger.info(f"Optimal config predicted for {model_type}: {best_config[0]}")

        return {
            'recommended_config': best_config[0],
            'predicted_performance': best_config[1]['predicted_performance'],
            'confidence': best_config[1]['confidence'],
            'alternative_configs': list(config_predictions.keys())[:3]
        }

    def _generate_config_predictions(self, X_scaled: np.ndarray, model_type: str) -> Dict[str, Dict]:
        """
        Generate predictions for different configurations.

        Args:
            X_scaled: Scaled input features
            model_type: Type of model

        Returns:
            Dictionary of configuration predictions
        """
        from .config import MODEL_CONFIGS

        config_predictions = {}

        if model_type in MODEL_CONFIGS:
            param_grid = MODEL_CONFIGS[model_type]

            # Generate all combinations (simplified - just use first value for each param)
            base_config = {param: values[0] for param, values in param_grid.items()}

            # Predict performance for base config
            predicted_perf = self.meta_model.predict(X_scaled)[0]

            config_predictions[str(base_config)] = {
                'predicted_performance': predicted_perf,
                'confidence': 0.8  # Placeholder confidence score
            }

            # Generate variations
            for param, values in param_grid.items():
                if len(values) > 1:
                    for value in values[1:3]:  # Test up to 3 values
                        config = base_config.copy()
                        config[param] = value
                        predicted_perf = self.meta_model.predict(X_scaled)[0]

                        config_predictions[str(config)] = {
                            'predicted_performance': predicted_perf,
                            'confidence': 0.7
                        }

        return config_predictions

    def _get_default_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get default configuration when meta-model is not available.

        Args:
            model_type: Type of model

        Returns:
            Default configuration
        """
        from .config import MODEL_CONFIGS

        default_config = {}
        if model_type in MODEL_CONFIGS:
            default_config = {param: values[0] for param, values in MODEL_CONFIGS[model_type].items()}

        return {
            'recommended_config': default_config,
            'predicted_performance': 0.5,
            'confidence': 0.5,
            'note': 'Using default configuration - meta-model not trained'
        }

    def update_meta_model(self, new_experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update meta-model with new experiment results.

        Args:
            new_experiment_results: Results from new experiments

        Returns:
            Update results
        """
        meta_logger.info("Updating meta-learning model")

        # Collect new data
        self.collect_experiment_data(new_experiment_results)

        # Retrain if we have enough data
        if len(self.experiment_history) >= 10:
            training_results = self.train_meta_model()

            # Save updated model
            self.save_meta_model()

            return {
                'updated': True,
                'training_results': training_results,
                'total_experiments': len(self.experiment_history)
            }
        else:
            return {
                'updated': False,
                'reason': 'Insufficient experiment data',
                'total_experiments': len(self.experiment_history)
            }

    def save_meta_model(self, save_path: Optional[Path] = None):
        """
        Save the meta-learning model and related data.

        Args:
            save_path: Optional path to save the model
        """
        if save_path is None:
            save_path = MODELS_DIR / "meta_learner.pkl"

        MODELS_DIR.mkdir(exist_ok=True)

        model_data = {
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'experiment_history': self.experiment_history,
            'saved_timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, save_path)
        meta_logger.info(f"Meta-learning model saved to {save_path}")

    def load_meta_model(self, load_path: Optional[Path] = None):
        """
        Load the meta-learning model and related data.

        Args:
            load_path: Optional path to load the model from
        """
        if load_path is None:
            load_path = MODELS_DIR / "meta_learner.pkl"

        if not load_path.exists():
            meta_logger.warning(f"Meta-model file not found: {load_path}")
            return

        model_data = joblib.load(load_path)

        self.meta_model = model_data['meta_model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.experiment_history = model_data['experiment_history']

        meta_logger.info(f"Meta-learning model loaded from {load_path}")

    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the meta-learning system.

        Returns:
            Dictionary with meta-learning statistics
        """
        stats = {
            'total_experiments': len(self.experiment_history),
            'meta_model_trained': self.meta_model is not None,
            'experiment_history_length': len(self.experiment_history)
        }

        if self.experiment_history:
            performances = [exp['performance'] for exp in self.experiment_history]
            stats.update({
                'avg_performance': np.mean(performances),
                'best_performance': np.max(performances),
                'performance_std': np.std(performances)
            })

        if self.meta_model:
            stats['feature_importance'] = dict(zip(
                self.scaler.feature_names_in_,
                self.meta_model.feature_importances_
            ))

        return stats

    def generate_meta_insights(self) -> Dict[str, Any]:
        """
        Generate insights from meta-learning analysis.

        Returns:
            Dictionary with meta-learning insights
        """
        insights = {
            'insights': [],
            'recommendations': []
        }

        if not self.experiment_history:
            insights['insights'].append("No experiment data available for meta-learning")
            return insights

        # Analyze performance patterns including CV stability
        performances = [exp['performance'] for exp in self.experiment_history]
        cv_stabilities = [exp['features'].get('cv_stability_score', 0) for exp in self.experiment_history]

        if len(performances) > 5:
            # Performance trends
            recent_performances = performances[-10:]
            trend = np.polyfit(range(len(recent_performances)), recent_performances, 1)[0]

            if trend > 0.01:
                insights['insights'].append("Model performance is improving over time")
                insights['recommendations'].append("Continue current meta-learning approach")
            elif trend < -0.01:
                insights['insights'].append("Model performance is declining")
                insights['recommendations'].append("Review recent experiments and adjust strategy")

            # CV stability trends
            if cv_stabilities:
                recent_stabilities = cv_stabilities[-10:]
                stability_trend = np.polyfit(range(len(recent_stabilities)), recent_stabilities, 1)[0]

                if stability_trend > 0.01:
                    insights['insights'].append("CV stability is improving over time")
                    insights['recommendations'].append("Model configurations are becoming more robust")
                elif stability_trend < -0.01:
                    insights['insights'].append("CV stability is declining")
                    insights['recommendations'].append("Review model configurations for consistency issues")

            # Best performing model types with stability consideration
            model_performances = {}
            model_stabilities = {}
            for exp in self.experiment_history:
                model_type = exp['features'].get('model_type', 'unknown')
                performance = exp['performance']
                stability = exp['features'].get('cv_stability_score', 0)

                if model_type not in model_performances:
                    model_performances[model_type] = []
                    model_stabilities[model_type] = []
                model_performances[model_type].append(performance)
                model_stabilities[model_type].append(stability)

            if model_performances:
                # Calculate combined score (performance + stability)
                combined_scores = {}
                for model_type in model_performances:
                    avg_perf = np.mean(model_performances[model_type])
                    avg_stability = np.mean(model_stabilities[model_type])
                    combined_scores[model_type] = avg_perf * 0.7 + avg_stability * 0.3  # Weighted combination

                best_model = max(combined_scores.items(), key=lambda x: x[1])
                best_avg_perf = np.mean(model_performances[best_model[0]])
                best_avg_stability = np.mean(model_stabilities[best_model[0]])

                insights['insights'].append(
                    f"Best performing model type: {best_model[0]} "
                    f"(avg performance: {best_avg_perf:.4f}, avg CV stability: {best_avg_stability:.3f})"
                )
                insights['recommendations'].append(f"Prioritize {best_model[0]} models for future experiments")

        # Meta-model effectiveness
        if self.meta_model:
            insights['insights'].append("Meta-learning model is active and providing configuration recommendations")
        else:
            insights['recommendations'].append("Train meta-learning model with more experiment data")

        return insights

def create_meta_learner() -> MetaLearner:
    """
    Factory function to create MetaLearner instance.

    Returns:
        MetaLearner instance
    """
    return MetaLearner()