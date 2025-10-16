"""
Core modeling pipeline for Linear/Logistic Regression and Decision Tree/Random Forest.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
from pathlib import Path
import json
from datetime import datetime

import sys
import os
# Ensure project root is in path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import (
        RANDOM_STATE, CV_FOLDS, MODEL_CONFIGS, MODELS_DIR,
        TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    )
    from src.logger import model_logger
    from redis_cache import cache_model_results, get_cached_model_results
except ImportError as e:
    print(f"Import error in model_pipeline.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise

class ModelPipeline:
    """
    Core modeling pipeline supporting multiple algorithms.
    """

    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_configs = MODEL_CONFIGS
        self.cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def _get_model_class(self, model_type: str, task: str = 'regression'):
        """
        Get the appropriate model class based on type and task.

        Args:
            model_type: Type of model ('linear', 'decision_tree', 'random_forest')
            task: Task type ('regression' or 'classification')

        Returns:
            Model class
        """
        model_map = {
            'linear': {
                'regression': LinearRegression,
                'classification': LogisticRegression
            },
            'decision_tree': {
                'regression': DecisionTreeRegressor,
                'classification': DecisionTreeClassifier
            },
            'random_forest': {
                'regression': RandomForestRegressor,
                'classification': RandomForestClassifier
            }
        }

        return model_map.get(model_type, {}).get(task)

    def _determine_task_type(self, y: np.ndarray) -> str:
        """
        Determine if the task is regression or classification.

        Args:
            y: Target values

        Returns:
            Task type string
        """
        unique_values = len(np.unique(y))

        # If target has few unique values or is boolean, treat as classification
        if unique_values <= 10 or y.dtype == bool:
            return 'classification'
        else:
            return 'regression'

    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: np.ndarray,
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a single model.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            params: Model parameters

        Returns:
            Dictionary with model info and training results
        """
        task_type = self._determine_task_type(y_train)
        model_class = self._get_model_class(model_type, task_type)

        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type} for task: {task_type}")

        # Use default params if none provided
        if params is None:
            params = {}

        # Check cache first
        cache_key = f"{model_type}_{task_type}_{str(sorted(params.items()))}"
        model_logger.info(f"Checking cache for {model_type} with key: {cache_key}")
        cached_results = get_cached_model_results(model_type, params)

        if cached_results:
            model_logger.info(f"Using cached results for {model_type}")
            # Use cached results and create a dummy model for evaluation
            # We'll create a minimal model object that can be used for prediction
            model_id = cached_results.get('model_id', f"{model_type}_{len(self.trained_models)}")

            # For cached results, we need to create a model instance for evaluation
            # We'll retrain with the same parameters to get the model object
            model_logger.info(f"Retraining model with cached parameters for evaluation")
        else:
            model_logger.info(f"Cache miss for {model_type} - training new model")

        # Create and train model - handle random_state appropriately
        if model_type == 'linear':
            # LinearRegression doesn't accept random_state
            model = model_class(**params)
        else:
            # Tree-based and ensemble models do accept random_state
            model = model_class(random_state=RANDOM_STATE, **params)

        model_logger.info(f"Training {model_type} model with params: {params}")

        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Generate consistent model ID
        if cached_results:
            # Use the cached model ID for consistency
            model_id = cached_results.get('model_id', f"{model_type}_{len(self.trained_models)}")
        else:
            # Generate new model ID for fresh training
            model_id = f"{model_type}_{len(self.trained_models)}"

        # Store trained model
        self.trained_models[model_id] = model
        model_logger.info(f"Stored trained model with ID: {model_id}")

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv,
                                  scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy')

        # Calculate training metrics
        y_pred_train = model.predict(X_train)

        if task_type == 'regression':
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            cv_mean = -cv_scores.mean()  # Convert back from negative MSE
            cv_std = cv_scores.std()

            metrics = {
                'train_mse': train_mse,
                'train_rmse': np.sqrt(train_mse),
                'train_r2': train_r2,
                'cv_mse_mean': cv_mean,
                'cv_mse_std': cv_std,
                'cv_rmse_mean': np.sqrt(cv_mean)
            }
        else:
            train_accuracy = accuracy_score(y_train, y_pred_train)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            metrics = {
                'train_accuracy': train_accuracy,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std
            }

        results = {
            'model_id': model_id,
            'model_type': model_type,
            'task_type': task_type,
            'parameters': params,
            'training_time': training_time,
            'metrics': metrics,
            'feature_importance': self._get_feature_importance(model, X_train.columns.tolist()) if hasattr(model, 'feature_importances_') else None,
            'coefficients': self._get_coefficients(model) if hasattr(model, 'coef_') else None,
            'training_timestamp': datetime.now().isoformat()
        }

        # Cache results
        cache_model_results(model_type, params, results)

        # Also cache the model object separately for potential future retrieval
        model_cache_key = f"model_object_{model_id}"
        try:
            import joblib
            model_bytes = joblib.dumps(model)
            from src.config import CACHE_TTL
            cache.set(model_cache_key, model_bytes, ttl=CACHE_TTL)
            model_logger.info(f"Cached model object for {model_id}")
        except Exception as e:
            model_logger.warning(f"Failed to cache model object for {model_id}: {e}")

        model_logger.info(f"Model {model_type} trained successfully. CV Score: {cv_mean:.4f}")

        return results

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from tree-based models.

        Args:
            model: Trained model
            feature_names: Feature names

        Returns:
            Dictionary of feature importance
        """
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        return {}

    def _get_coefficients(self, model) -> Dict[str, float]:
        """
        Extract coefficients from linear models.

        Args:
            model: Trained model

        Returns:
            Dictionary of coefficients
        """
        if hasattr(model, 'coef_'):
            # Handle multi-class case
            coef = model.coef_
            if coef.ndim > 1:
                # For multi-class, return the mean absolute coefficient across classes
                coef = np.mean(np.abs(coef), axis=0)
            return dict(zip(model.feature_names_in_, coef))
        return {}

    def predict(self, model_id: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_id: ID of the trained model
            X: Input features

        Returns:
            Predictions array
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        model = self.trained_models[model_id]
        return model.predict(X)

    def evaluate_model(self, model_id: str, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            model_id: ID of the trained model
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        model = self.trained_models[model_id]
        y_pred = model.predict(X_test)

        task_type = self._determine_task_type(y_test)

        if task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                'test_mse': mse,
                'test_rmse': np.sqrt(mse),
                'test_r2': r2,
                'test_mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            metrics = {
                'test_accuracy': accuracy,
                'classification_report': report
            }

        model_logger.info(f"Model {model_id} evaluation completed. Test Score: {metrics.get('test_r2', metrics.get('test_accuracy', 'N/A')):.4f}")

        return metrics

    def hyperparameter_tuning(self, model_type: str, X_train: pd.DataFrame, y_train: np.ndarray,
                            param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search with cross-validation.

        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter grid to search

        Returns:
            Best parameters and scores
        """
        from sklearn.model_selection import GridSearchCV

        task_type = self._determine_task_type(y_train)
        model_class = self._get_model_class(model_type, task_type)

        if param_grid is None:
            param_grid = self.model_configs.get(model_type, {})

        model_logger.info(f"Starting hyperparameter tuning for {model_type}")

        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'

        grid_search = GridSearchCV(
            model_class(random_state=RANDOM_STATE),
            param_grid,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_score = -grid_search.best_score_ if task_type == 'regression' else grid_search.best_score_

        results = {
            'best_params': grid_search.best_params_,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_,
            'all_scores': grid_search.cv_results_['mean_test_score']
        }

        model_logger.info(f"Hyperparameter tuning completed. Best score: {best_score:.4f}")

        return results

    def save_model(self, model_id: str, save_path: Path):
        """
        Save a trained model to disk.

        Args:
            model_id: ID of the trained model
            save_path: Path to save the model
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        model = self.trained_models[model_id]
        joblib.dump(model, save_path)
        model_logger.info(f"Model {model_id} saved to {save_path}")

    def load_model(self, model_id: str, load_path: Path):
        """
        Load a model from disk.

        Args:
            model_id: ID to assign to the loaded model
            load_path: Path to load the model from
        """
        model = joblib.load(load_path)
        self.trained_models[model_id] = model
        model_logger.info(f"Model loaded from {load_path} as {model_id}")

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a trained model.

        Args:
            model_id: ID of the trained model

        Returns:
            Model information dictionary
        """
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        model = self.trained_models[model_id]

        info = {
            'model_id': model_id,
            'model_type': type(model).__name__,
            'parameters': model.get_params() if hasattr(model, 'get_params') else {},
            'n_features': getattr(model, 'n_features_in_', None),
            'feature_names': getattr(model, 'feature_names_in_', None)
        }

        return info

    def compare_models(self, model_ids: List[str], X_test: pd.DataFrame,
                      y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compare multiple trained models.

        Args:
            model_ids: List of model IDs to compare
            X_test: Test features
            y_test: Test targets

        Returns:
            Comparison results
        """
        results = {}

        for model_id in model_ids:
            try:
                evaluation = self.evaluate_model(model_id, X_test, y_test)
                info = self.get_model_info(model_id)
                results[model_id] = {
                    'info': info,
                    'evaluation': evaluation
                }
            except Exception as e:
                model_logger.error(f"Error evaluating model {model_id}: {e}")
                results[model_id] = {'error': str(e)}

        # Determine best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if valid_results:
            task_type = self._determine_task_type(y_test)
            metric = 'test_r2' if task_type == 'regression' else 'test_accuracy'

            best_model = max(valid_results.items(),
                           key=lambda x: x[1]['evaluation'].get(metric, -np.inf))
            results['best_model'] = best_model[0]

        return results

def create_model_pipeline() -> ModelPipeline:
    """
    Factory function to create ModelPipeline instance.

    Returns:
        ModelPipeline instance
    """
    return ModelPipeline()