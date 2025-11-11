"""
Core modeling pipeline for Linear/Logistic Regression and Decision Tree/Random Forest.

Supports four model types:
- Linear Regression (for regression tasks)
- Logistic Regression (for classification tasks)
- Decision Tree (regression and classification)
- Random Forest (regression and classification)

Refactored for improved maintainability and reduced complexity.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
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
        TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CACHE_TTL
    )
    from src.logger import model_logger
    from redis_cache import cache, cache_model_results, get_cached_model_results
except ImportError as e:
    print(f"Import error in model_pipeline.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root: {project_root}")
    raise


class ModelPipeline:
    """
    Core modeling pipeline supporting multiple algorithms.

    Responsibilities:
    - Model creation and training
    - Hyperparameter tuning
    - Model evaluation and comparison
    - Caching and persistence
    """

    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_configs = MODEL_CONFIGS
        self.cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        self.poly_features = {}  # Store polynomial feature transformers
        self.scaler = None  # Store scaler for linear models

    # ============================================================================
    # MODEL SELECTION AND CONFIGURATION
    # ============================================================================

    def _get_model_class(self, model_type: str, task: str = 'regression'):
        """
        Get the appropriate model class based on type and task.

        Supported models:
        - 'linear': Linear Regression (regression) / Logistic Regression (classification)
        - 'decision_tree': Decision Tree Regressor/Classifier
        - 'random_forest': Random Forest Regressor/Classifier

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

    def _is_linear_model(self, model_type: str) -> bool:
        """Check if model type is linear-based."""
        return model_type == 'linear'

    def _is_tree_model(self, model_type: str) -> bool:
        """Check if model type is tree-based."""
        return model_type in ['decision_tree', 'random_forest']

    # ============================================================================
    # ADAPTIVE PARAMETERS FOR TREE MODELS
    # ============================================================================

    def _get_adaptive_tree_params(self, model_type: str, X_train: pd.DataFrame,
                                y_train: np.ndarray) -> Dict[str, Any]:
        """
        Get adaptive parameters based on dataset characteristics.

        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets

        Returns:
            Adaptive parameters dictionary
        """
        n_samples, n_features = X_train.shape

        # Adaptive depth based on dataset size
        if n_samples < 1000:
            max_depth_range = [5, 10, 15, 20]
        elif n_samples < 10000:
            max_depth_range = [10, 15, 20, 30]
        else:
            max_depth_range = [15, 20, 30, None]

        # Adaptive min_samples based on dataset size
        if n_samples < 1000:
            min_samples_split_range = [2, 5, 10]
            min_samples_leaf_range = [1, 2, 4]
        else:
            min_samples_split_range = [5, 10, 20, 50]
            min_samples_leaf_range = [2, 4, 8, 16]

        # Adaptive max_features
        if n_features < 10:
            max_features_options = [None, 'sqrt']
        elif n_features < 50:
            max_features_options = ['sqrt', 'log2', None]
        else:
            max_features_options = ['sqrt', 'log2', 0.5, 0.75]

        adaptive_params = {
            'max_depth': max_depth_range,
            'min_samples_split': min_samples_split_range,
            'min_samples_leaf': min_samples_leaf_range,
            'max_features': max_features_options
        }

        if model_type == 'random_forest':
            # Adaptive n_estimators based on dataset size
            if n_samples < 1000:
                n_estimators_range = [50, 100, 150]
            elif n_samples < 10000:
                n_estimators_range = [100, 200, 300]
            else:
                n_estimators_range = [200, 300, 500]

            adaptive_params['n_estimators'] = n_estimators_range
            adaptive_params['bootstrap'] = [True, False]
            adaptive_params['max_samples'] = [0.5, 0.75, 1.0]

        return adaptive_params

    # ============================================================================
    # TREE MODEL REGULARIZATION
    # ============================================================================

    def _apply_tree_regularization(self, model, model_type: str, params: Dict[str, Any],
                                  X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Apply tree-specific regularization techniques.

        Args:
            model: Trained model
            model_type: Type of model
            params: Model parameters
            X_train: Training features
            y_train: Training targets
        """
        if model_type == 'decision_tree':
            # Apply cost complexity pruning if not already done
            if hasattr(model, 'ccp_alpha') and params.get('ccp_alpha', 0.0) == 0.0:
                path = model.cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas = path.ccp_alphas

                # Evaluate different alpha values
                alpha_scores = []
                for alpha in ccp_alphas[::10]:  # Sample every 10th alpha
                    temp_model = type(model)(random_state=RANDOM_STATE, ccp_alpha=alpha)
                    scores = cross_val_score(temp_model, X_train, y_train, cv=5)
                    alpha_scores.append((alpha, scores.mean()))

                # Select best alpha
                best_alpha = max(alpha_scores, key=lambda x: x[1])[0]
                if best_alpha > 0:
                    model.ccp_alpha = best_alpha
                    model_logger.info(f"Applied cost complexity pruning with alpha={best_alpha}")

        elif model_type == 'random_forest':
            # Additional regularization for random forest
            if params.get('bootstrap', True) and 'max_samples' in params:
                n_samples = len(X_train)
                max_samples = params.get('max_samples', 1.0)
                if isinstance(max_samples, float):
                    max_samples = int(max_samples * n_samples)
                model.max_samples = max_samples
                model_logger.info(f"Applied bootstrap sampling with max_samples={max_samples}")

    # ============================================================================
    # FEATURE ENGINEERING AND PREPROCESSING
    # ============================================================================

    def _apply_feature_engineering(self, X_train: pd.DataFrame, model_type: str,
                                  model_id: str, params: Dict[str, Any],
                                  fit: bool = True) -> pd.DataFrame:
        """
        Apply feature engineering (polynomial features and scaling).

        Args:
            X_train: Training features
            model_type: Type of model
            model_id: Model identifier
            params: Model parameters
            fit: Whether to fit transformers (True) or just transform (False)

        Returns:
            Transformed features
        """
        X_transformed = X_train.copy()

        if self._is_linear_model(model_type):
            # Apply polynomial features if specified
            poly_degree = params.get('polynomial_degree', 1)
            if poly_degree > 1:
                if fit:
                    self.poly_features[model_id] = PolynomialFeatures(
                        degree=poly_degree, include_bias=False
                    )
                    X_poly = self.poly_features[model_id].fit_transform(X_transformed)
                else:
                    if model_id in self.poly_features:
                        X_poly = self.poly_features[model_id].transform(X_transformed)
                    else:
                        X_poly = X_transformed

                poly_feature_names = self.poly_features[model_id].get_feature_names_out(
                    X_transformed.columns
                )
                X_transformed = pd.DataFrame(
                    X_poly, columns=poly_feature_names, index=X_transformed.index
                )
                if fit:
                    model_logger.info(
                        f"Applied polynomial features (degree {poly_degree}) to {model_type}"
                    )

            # Apply RobustScaler
            if fit:
                self.scaler = RobustScaler()
                X_scaled = self.scaler.fit_transform(X_transformed)
                model_logger.info(f"Applied RobustScaler to features for {model_type}")
            else:
                if self.scaler is not None:
                    X_scaled = self.scaler.transform(X_transformed)
                else:
                    X_scaled = X_transformed

            X_transformed = pd.DataFrame(
                X_scaled, columns=X_transformed.columns, index=X_transformed.index
            )

        return X_transformed

    # ============================================================================
    # CACHING
    # ============================================================================

    def _check_cache_for_model(self, model_type: str, params: Dict[str, Any],
                              feature_names: List[str]) -> Tuple[bool, Optional[Dict], Optional[Any]]:
        """
        Check if a cached model exists and is valid.

        Args:
            model_type: Type of model
            params: Model parameters
            feature_names: List of feature names

        Returns:
            Tuple of (use_cache, cached_results, cached_model)
        """
        cache_key = f"{model_type}_{str(sorted(params.items()))}_{hash(str(sorted(feature_names)))}"
        model_logger.info(f"Checking cache for {model_type} with key: {cache_key}")

        cached_results = get_cached_model_results(model_type, params)

        if not cached_results:
            model_logger.info(f"Cache miss for {model_type} - training new model")
            return False, None, None

        # Validate feature set matches
        cached_features = cached_results.get('feature_names', [])
        if sorted(cached_features) != sorted(feature_names):
            model_logger.info(f"Cache feature mismatch, retraining {model_type}")
            return False, None, None

        model_logger.info(f"Using cached results for {model_type}")
        model_id = cached_results.get('model_id', f"{model_type}_cached")

        # Try to load cached model object
        model_cache_key = f"model_object_{model_id}"
        try:
            cached_model_bytes = cache.get(model_cache_key)
            if cached_model_bytes:
                cached_model = joblib.loads(cached_model_bytes)
                model_logger.info(f"Loaded cached model object for {model_id}")
                return True, cached_results, cached_model
            else:
                model_logger.info(f"Model object not cached, retraining {model_type}")
                return False, None, None
        except Exception as e:
            model_logger.warning(f"Failed to load cached model: {e}, retraining")
            return False, None, None

    def _cache_model_and_results(self, model, model_id: str, model_type: str,
                                params: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Cache model object and results.

        Args:
            model: Trained model
            model_id: Model identifier
            model_type: Type of model
            params: Model parameters
            results: Training results
        """
        # Cache results
        cache_model_results(model_type, params, results)

        # Cache model object
        model_cache_key = f"model_object_{model_id}"
        try:
            model_bytes = joblib.dumps(model)
            cache.set(model_cache_key, model_bytes, ttl=CACHE_TTL)
            model_logger.info(f"Cached model object for {model_id}")
        except Exception as e:
            model_logger.warning(f"Failed to cache model object for {model_id}: {e}")

    # ============================================================================
    # MODEL CREATION AND TRAINING
    # ============================================================================

    def _create_model_instance(self, model_class, model_type: str,
                              params: Dict[str, Any]) -> Any:
        """
        Create a model instance with appropriate parameters.

        Args:
            model_class: Model class
            model_type: Type of model ('linear', 'decision_tree', 'random_forest')
            params: Model parameters

        Returns:
            Model instance
        """
        # Filter out polynomial_degree from params (handled separately for linear models)
        filtered_params = {k: v for k, v in params.items() if k != 'polynomial_degree'}

        if self._is_linear_model(model_type):
            # LinearRegression doesn't accept random_state
            # LogisticRegression does, but we handle it automatically
            return model_class(**filtered_params)
        else:
            # Tree-based models (decision_tree, random_forest) accept random_state
            return model_class(random_state=RANDOM_STATE, **params)

    def _perform_cv_with_scaling(self, model, X_train: pd.DataFrame, y_train: np.ndarray,
                                model_type: str, task_type: str,
                                params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Perform cross-validation with proper scaling for linear models.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            model_type: Type of model
            task_type: Task type
            params: Model parameters

        Returns:
            Cross-validation scores
        """
        from sklearn.pipeline import Pipeline

        if self._is_linear_model(model_type):
            # Get polynomial degree from params
            poly_degree = params.get('polynomial_degree', 1) if params else 1

            if poly_degree > 1:
                # Pipeline with polynomial features, scaling, and model
                pipeline = Pipeline([
                    ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                    ('scaler', RobustScaler()),
                    ('model', model.__class__(**model.get_params()))
                ])
            else:
                # Pipeline with just scaling and model
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    ('model', model.__class__(**model.get_params()))
                ])

            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=self.cv,
                scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
            )
        else:
            # For tree-based models, no scaling needed
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=self.cv,
                scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
            )

        return cv_scores

    def _compute_training_metrics(self, model, X_train_scaled: pd.DataFrame,
                                 y_train: np.ndarray, cv_scores: np.ndarray,
                                 task_type: str) -> Dict[str, Any]:
        """
        Compute training and CV metrics.

        Args:
            model: Trained model
            X_train_scaled: Scaled training features
            y_train: Training targets
            cv_scores: Cross-validation scores
            task_type: Task type

        Returns:
            Dictionary of metrics
        """
        y_pred_train = model.predict(X_train_scaled)

        if task_type == 'regression':
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            cv_mean = -cv_scores.mean()
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

        return metrics

    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: np.ndarray,
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a single model with comprehensive tracking.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            params: Model parameters

        Returns:
            Dictionary with model info and training results
        """
        # Initialize
        task_type = self._determine_task_type(y_train)
        model_class = self._get_model_class(model_type, task_type)

        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type} for task: {task_type}")

        params = params or {}
        feature_names = sorted(X_train.columns.tolist())
        model_id = f"{model_type}_{len(self.trained_models)}"

        # Apply adaptive parameters for tree models
        if self._is_tree_model(model_type):
            adaptive_params = self._get_adaptive_tree_params(model_type, X_train, y_train)
            for key, values in adaptive_params.items():
                if key not in params:
                    params[key] = values[0] if isinstance(values, list) else values

        # Check cache
        use_cache, cached_results, cached_model = self._check_cache_for_model(
            model_type, params, feature_names
        )

        if use_cache and cached_model is not None:
            # Use cached model
            model = cached_model
            self.trained_models[model_id] = model

            # Apply feature engineering for consistency
            X_train_scaled = self._apply_feature_engineering(
                X_train, model_type, model_id, params, fit=True
            )

            training_time = cached_results.get('training_time', 0)
            model_logger.info(f"Using cached model for {model_type}")
        else:
            # Train new model
            X_train_scaled = self._apply_feature_engineering(
                X_train, model_type, model_id, params, fit=True
            )

            model = self._create_model_instance(model_class, model_type, params)
            model_logger.info(f"Training {model_type} model with params: {params}")

            start_time = datetime.now()
            model.fit(X_train_scaled, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Apply tree-specific regularization
            if self._is_tree_model(model_type):
                self._apply_tree_regularization(
                    model, model_type, params, X_train_scaled, y_train
                )

            # Store trained model
            self.trained_models[model_id] = model
            model_logger.info(f"Stored trained model with ID: {model_id}")

        # Perform cross-validation
        cv_scores = self._perform_cv_with_scaling(
            model, X_train, y_train, model_type, task_type, params
        )

        # Calculate metrics
        metrics = self._compute_training_metrics(
            model, X_train_scaled, y_train, cv_scores, task_type
        )

        # Calculate overfitting indicators
        overfitting_indicators = self._calculate_overfitting_indicators(
            metrics, task_type, model_type
        )

        # Compile results
        results = {
            'model_id': model_id,
            'model_type': model_type,
            'task_type': task_type,
            'parameters': params,
            'training_time': training_time,
            'metrics': metrics,
            'overfitting_indicators': overfitting_indicators,
            'feature_names': feature_names,
            'feature_importance': self._get_feature_importance(
                model, X_train_scaled.columns.tolist()
            ) if hasattr(model, 'feature_importances_') else None,
            'coefficients': self._get_coefficients(model) if hasattr(model, 'coef_') else None,
            'training_timestamp': datetime.now().isoformat()
        }

        # Cache if new model
        if not use_cache:
            self._cache_model_and_results(model, model_id, model_type, params, results)

        cv_mean = metrics.get('cv_mse_mean', metrics.get('cv_accuracy_mean', 0))
        model_logger.info(f"Model {model_type} trained successfully. CV Score: {cv_mean:.4f}")

        return results

    # ============================================================================
    # OVERFITTING ANALYSIS
    # ============================================================================

    def _calculate_overfitting_indicators(self, metrics: Dict[str, Any], task_type: str,
                                        model_type: str) -> Dict[str, Any]:
        """
        Calculate indicators of overfitting.

        Args:
            metrics: Training and CV metrics
            task_type: Type of task
            model_type: Type of model

        Returns:
            Dictionary with overfitting indicators
        """
        indicators = {}

        if task_type == 'regression':
            train_score = metrics.get('train_r2', 0)
            cv_mse = metrics.get('cv_mse_mean', 0)
            train_mse = metrics.get('train_mse', 0)

            # MSE ratio
            mse_ratio = cv_mse / train_mse if train_mse > 0 else 1.0

            indicators = {
                'mse_train_cv_ratio': mse_ratio,
                'overfitting_risk': (
                    'high' if mse_ratio > 2.0 else
                    'medium' if mse_ratio > 1.5 else
                    'low'
                ),
                'cv_stability_score': 1.0 / (1.0 + metrics.get('cv_mse_std', 0))
            }
        else:
            train_score = metrics.get('train_accuracy', 0)
            cv_score = metrics.get('cv_accuracy_mean', 0)
            acc_diff = train_score - cv_score

            indicators = {
                'accuracy_train_cv_diff': acc_diff,
                'overfitting_risk': (
                    'high' if acc_diff > 0.15 else
                    'medium' if acc_diff > 0.1 else
                    'low'
                ),
                'cv_stability_score': 1.0 / (1.0 + metrics.get('cv_accuracy_std', 0))
            }

        return indicators

    # ============================================================================
    # FEATURE EXTRACTION
    # ============================================================================

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from tree-based models."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        return {}

    def _get_coefficients(self, model) -> Dict[str, float]:
        """Extract coefficients from linear models."""
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            if hasattr(model, 'feature_names_in_'):
                return dict(zip(model.feature_names_in_, coef))
        return {}

    # ============================================================================
    # HYPERPARAMETER TUNING
    # ============================================================================

    def hyperparameter_tuning(self, model_type: str, X_train: pd.DataFrame,
                             y_train: np.ndarray,
                             param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive hyperparameter tuning.

        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter grid to search

        Returns:
            Best parameters and scores
        """
        task_type = self._determine_task_type(y_train)
        model_class = self._get_model_class(model_type, task_type)

        if param_grid is None:
            param_grid = self.model_configs.get(model_type, {})

        # Apply adaptive parameters for tree models
        if self._is_tree_model(model_type):
            adaptive_params = self._get_adaptive_tree_params(model_type, X_train, y_train)
            for key, values in adaptive_params.items():
                if key not in param_grid:
                    param_grid[key] = values

        model_logger.info(f"Starting hyperparameter tuning for {model_type}")

        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'

        # Special handling for linear models with polynomial features
        if self._is_linear_model(model_type) and 'polynomial_degree' in param_grid:
            return self._tune_linear_model_with_polynomials(
                model_class, model_type, X_train, y_train, param_grid, task_type, scoring
            )

        # Standard tuning for other models
        if self._is_tree_model(model_type):
            # Use RandomizedSearchCV for efficiency with tree models
            n_iter = min(50, np.prod([len(v) for v in param_grid.values() if isinstance(v, list)]))
            search_cv = RandomizedSearchCV(
                model_class(random_state=RANDOM_STATE),
                param_grid, n_iter=n_iter, cv=self.cv, scoring=scoring,
                n_jobs=-1, verbose=1, random_state=RANDOM_STATE
            )
            tuning_method = 'randomized_search'
        else:
            # GridSearchCV for linear models
            search_cv = GridSearchCV(
                model_class(), param_grid, cv=self.cv, scoring=scoring, n_jobs=-1, verbose=1
            )
            tuning_method = 'grid_search'

        search_cv.fit(X_train, y_train)

        # Evaluate best model
        best_model = self._create_model_instance(model_class, model_type, search_cv.best_params_)
        best_model.fit(X_train, y_train)

        # Overfitting assessment
        train_pred = best_model.predict(X_train)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=self.cv, scoring=scoring)

        if task_type == 'regression':
            train_score = r2_score(y_train, train_pred)
            cv_score = -cv_scores.mean()
            score_diff = train_score - cv_score
            overfitting_risk = 'high' if score_diff > 0.2 else 'medium' if score_diff > 0.1 else 'low'
        else:
            train_score = accuracy_score(y_train, train_pred)
            cv_score = cv_scores.mean()
            score_diff = train_score - cv_score
            overfitting_risk = 'high' if score_diff > 0.15 else 'medium' if score_diff > 0.1 else 'low'

        best_score = -search_cv.best_score_ if task_type == 'regression' else search_cv.best_score_

        results = {
            'best_params': search_cv.best_params_,
            'best_score': best_score,
            'cv_results': search_cv.cv_results_,
            'all_scores': search_cv.cv_results_['mean_test_score'],
            'tuning_method': tuning_method,
            'overfitting_risk_assessment': {
                'train_score': train_score,
                'cv_score': cv_score,
                'score_difference': score_diff,
                'overfitting_risk': overfitting_risk
            }
        }

        model_logger.info(
            f"Tuning completed. Best score: {best_score:.4f}, "
            f"Overfitting risk: {overfitting_risk}"
        )

        return results

    def _tune_linear_model_with_polynomials(self, model_class, model_type: str,
                                           X_train: pd.DataFrame, y_train: np.ndarray,
                                           param_grid: Dict, task_type: str,
                                           scoring: str) -> Dict[str, Any]:
        """
        Tune linear models with polynomial feature search.

        Args:
            model_class: Model class
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter grid
            task_type: Task type
            scoring: Scoring metric

        Returns:
            Tuning results
        """
        from sklearn.pipeline import Pipeline

        poly_degrees = param_grid.pop('polynomial_degree')
        other_params = param_grid

        best_score = float('-inf')
        best_params = {}
        all_results = []

        for degree in poly_degrees:
            # Create pipeline with polynomial features and scaling
            if degree > 1:
                pipeline = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                    ('scaler', RobustScaler()),
                    ('model', model_class())
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    ('model', model_class())
                ])

            # Grid search for other parameters
            if other_params:
                # Prefix parameter names for pipeline
                pipeline_params = {f'model__{k}': v for k, v in other_params.items()}
                grid_search = GridSearchCV(
                    pipeline, pipeline_params, cv=self.cv, scoring=scoring, n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                current_score = grid_search.best_score_
                current_params = {'polynomial_degree': degree}
                # Remove 'model__' prefix from best params
                for k, v in grid_search.best_params_.items():
                    current_params[k.replace('model__', '')] = v
                cv_results = grid_search.cv_results_
            else:
                # No other parameters, just evaluate pipeline
                scores = cross_val_score(pipeline, X_train, y_train, cv=self.cv, scoring=scoring)
                current_score = scores.mean()
                current_params = {'polynomial_degree': degree}
                cv_results = {'mean_test_score': scores}

            all_results.append({
                'params': current_params,
                'score': current_score,
                'cv_results': cv_results
            })

            if current_score > best_score:
                best_score = current_score
                best_params = current_params

        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': all_results,
            'all_scores': [r['score'] for r in all_results],
            'tuning_method': 'grid_search_with_polynomial'
        }

    # ============================================================================
    # MODEL EVALUATION AND COMPARISON
    # ============================================================================

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

        # Extract model type from model_id
        model_type = model_id.split('_')[0]

        # Apply transformations
        X_transformed = X.copy()

        # Apply polynomial features if exists
        if model_id in self.poly_features:
            X_poly = self.poly_features[model_id].transform(X_transformed)
            poly_feature_names = self.poly_features[model_id].get_feature_names_out(X.columns)
            X_transformed = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)

        # Apply scaling if needed
        if self._is_linear_model(model_type) and self.scaler is not None:
            X_transformed = pd.DataFrame(
                self.scaler.transform(X_transformed),
                columns=X_transformed.columns,
                index=X_transformed.index
            )

        return model.predict(X_transformed)

    def evaluate_model(self, model_id: str, X_test: pd.DataFrame,
                      y_test: np.ndarray) -> Dict[str, Any]:
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

        y_pred = self.predict(model_id, X_test)
        task_type = self._determine_task_type(y_test)

        if task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Calculate MAPE safely
            epsilon = 1e-10
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

            metrics = {
                'test_mse': mse,
                'test_rmse': np.sqrt(mse),
                'test_r2': r2,
                'test_mape': mape
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            metrics = {
                'test_accuracy': accuracy,
                'classification_report': report
            }

        model_logger.info(
            f"Model {model_id} evaluation completed. "
            f"Score: {metrics.get('test_r2', metrics.get('test_accuracy', 'N/A')):.4f}"
        )

        return metrics

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

            # Rank models
            model_rankings = []
            for model_id, model_data in valid_results.items():
                base_score = model_data['evaluation'].get(metric, -np.inf)

                model_rankings.append({
                    'model_id': model_id,
                    'score': base_score
                })

            model_rankings.sort(key=lambda x: x['score'], reverse=True)
            best_model = model_rankings[0]

            results['best_model'] = best_model['model_id']
            results['model_rankings'] = model_rankings
            results['selection_criteria'] = {
                'primary_metric': metric,
                'multi_criteria_scoring': False
            }

        return results

    # ============================================================================
    # ENSEMBLE METHODS
    # ============================================================================

    def create_tree_linear_ensemble(self, tree_model_id: str, linear_model_id: str,
                                   X_train: pd.DataFrame, y_train: np.ndarray,
                                   ensemble_type: str = 'stacking') -> Dict[str, Any]:
        """
        Create an ensemble combining tree-based and linear models.

        Args:
            tree_model_id: ID of the tree model
            linear_model_id: ID of the linear model
            X_train: Training features
            y_train: Training targets
            ensemble_type: Type of ensemble ('stacking' or 'voting')

        Returns:
            Ensemble model results
        """
        from sklearn.ensemble import StackingRegressor, StackingClassifier
        from sklearn.ensemble import VotingRegressor, VotingClassifier

        if tree_model_id not in self.trained_models or linear_model_id not in self.trained_models:
            raise ValueError("Both tree and linear models must be trained first")

        tree_model = self.trained_models[tree_model_id]
        linear_model = self.trained_models[linear_model_id]
        task_type = self._determine_task_type(y_train)
        ensemble_id = f"ensemble_{tree_model_id}_{linear_model_id}_{ensemble_type}"

        if ensemble_type == 'stacking':
            if task_type == 'regression':
                ensemble = StackingRegressor(
                    estimators=[('tree', tree_model), ('linear', linear_model)],
                    final_estimator=LinearRegression(),
                    cv=5
                )
            else:
                ensemble = StackingClassifier(
                    estimators=[('tree', tree_model), ('linear', linear_model)],
                    final_estimator=LogisticRegression(random_state=RANDOM_STATE),
                    cv=5
                )
        elif ensemble_type == 'voting':
            if task_type == 'regression':
                ensemble = VotingRegressor(
                    estimators=[('tree', tree_model), ('linear', linear_model)]
                )
            else:
                ensemble = VotingClassifier(
                    estimators=[('tree', tree_model), ('linear', linear_model)],
                    voting='soft'
                )
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")

        # Train ensemble
        start_time = datetime.now()
        ensemble.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        self.trained_models[ensemble_id] = ensemble

        # Evaluate
        cv_scores = cross_val_score(
            ensemble, X_train, y_train, cv=self.cv,
            scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
        )

        if task_type == 'regression':
            cv_mean = -cv_scores.mean()
            metrics = {
                'cv_mse_mean': cv_mean,
                'cv_mse_std': cv_scores.std(),
                'cv_rmse_mean': np.sqrt(cv_mean)
            }
        else:
            metrics = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std()
            }

        results = {
            'ensemble_id': ensemble_id,
            'ensemble_type': ensemble_type,
            'base_models': [tree_model_id, linear_model_id],
            'training_time': training_time,
            'metrics': metrics,
            'training_timestamp': datetime.now().isoformat()
        }

        model_logger.info(f"Created {ensemble_type} ensemble: {ensemble_id}")
        return results

    # ============================================================================
    # MODEL PERSISTENCE
    # ============================================================================

    def save_model(self, model_id: str, save_path: Path):
        """Save a trained model to disk."""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")

        model = self.trained_models[model_id]
        joblib.dump(model, save_path)
        model_logger.info(f"Model {model_id} saved to {save_path}")

    def load_model(self, model_id: str, load_path: Path):
        """Load a model from disk."""
        model = joblib.load(load_path)
        self.trained_models[model_id] = model
        model_logger.info(f"Model loaded from {load_path} as {model_id}")

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a trained model."""
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


def create_model_pipeline() -> ModelPipeline:
    """
    Factory function to create ModelPipeline instance.

    Returns:
        ModelPipeline instance
    """
    return ModelPipeline()
