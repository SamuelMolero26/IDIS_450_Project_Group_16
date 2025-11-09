"""
Core modeling pipeline for Linear/Logistic Regression and Decision Tree/Random Forest.
Enhanced with Ridge, Lasso, ElasticNet, and GradientBoosting models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
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
        self.poly_features = {}  # Store polynomial feature transformers

    def _get_model_class(self, model_type: str, task: str = 'regression'):
        """
        Get the appropriate model class based on type and task with enhanced configuration.

        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'elasticnet', 'decision_tree', 'random_forest', 'gradient_boosting')
            task: Task type ('regression' or 'classification')

        Returns:
            Model class
        """
        model_map = {
            'linear': {
                'regression': LinearRegression,
                'classification': LogisticRegression
            },
            'ridge': {
                'regression': Ridge,
                'classification': LogisticRegression
            },
            'lasso': {
                'regression': Lasso,
                'classification': LogisticRegression
            },
            'elasticnet': {
                'regression': ElasticNet,
                'classification': LogisticRegression
            },
            'decision_tree': {
                'regression': DecisionTreeRegressor,
                'classification': DecisionTreeClassifier
            },
            'random_forest': {
                'regression': RandomForestRegressor,
                'classification': RandomForestClassifier
            },
            'gradient_boosting': {
                'regression': GradientBoostingRegressor,
                'classification': GradientBoostingClassifier
            }
        }

        return model_map.get(model_type, {}).get(task)

    def _apply_tree_regularization(self, model, model_type: str, params: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Apply tree-specific regularization techniques to prevent overfitting.

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
                # Calculate optimal ccp_alpha using cross-validation
                from sklearn.model_selection import cross_val_score
                path = model.cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas = path.ccp_alphas

                # Evaluate different alpha values
                alpha_scores = []
                for alpha in ccp_alphas[::10]:  # Sample every 10th alpha for efficiency
                    temp_model = type(model)(random_state=self.random_state, ccp_alpha=alpha)
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
                # Ensure max_samples is properly set for bootstrap sampling
                n_samples = len(X_train)
                max_samples = params.get('max_samples', 1.0)
                if isinstance(max_samples, float):
                    max_samples = int(max_samples * n_samples)
                model.max_samples = max_samples
                model_logger.info(f"Applied bootstrap sampling with max_samples={max_samples}")

    def _get_adaptive_tree_params(self, model_type: str, X_train: pd.DataFrame,
                                y_train: np.ndarray) -> Dict[str, Any]:
        """
        Get adaptive parameters based on dataset characteristics to prevent overfitting.

        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets

        Returns:
            Adaptive parameters dictionary
        """
        n_samples, n_features = X_train.shape

        # Adaptive depth based on dataset size - increased for better fit
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
        Train a single model with enhanced regularization and adaptive parameters.

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

        # Get adaptive parameters for tree-based models
        if model_type in ['decision_tree', 'random_forest']:
            adaptive_params = self._get_adaptive_tree_params(model_type, X_train, y_train)
            # Merge adaptive params with provided params
            for key, values in adaptive_params.items():
                if key not in params:
                    params[key] = values[0] if isinstance(values, list) else values

        # Create consistent cache key that includes feature names for proper cache invalidation
        feature_names = sorted(X_train.columns.tolist())
        cache_key = f"{model_type}_{task_type}_{str(sorted(params.items()))}_{hash(str(feature_names))}"
        model_logger.info(f"Checking cache for {model_type} with key: {cache_key}")
        cached_results = get_cached_model_results(model_type, params)

        # Only use cache if feature set matches (prevent stale cache issues)
        use_cache = False
        if cached_results:
            cached_features = cached_results.get('feature_names', [])
            if sorted(cached_features) == feature_names:
                use_cache = True
                model_logger.info(f"Using cached results for {model_type}")
                model_id = cached_results.get('model_id', f"{model_type}_{len(self.trained_models)}")
                # Load cached model object if available
                model_cache_key = f"model_object_{model_id}"
                try:
                    cached_model_bytes = cache.get(model_cache_key)
                    if cached_model_bytes:
                        model = joblib.loads(cached_model_bytes)
                        model_logger.info(f"Loaded cached model object for {model_id}")
                    else:
                        # Retrain if model object not cached
                        use_cache = False
                        model_logger.info(f"Model object not cached, retraining {model_type}")
                except Exception as e:
                    model_logger.warning(f"Failed to load cached model: {e}, retraining")
                    use_cache = False
            else:
                model_logger.info(f"Cache feature mismatch, retraining {model_type}")
                use_cache = False
        else:
            model_logger.info(f"Cache miss for {model_type} - training new model")

        if not use_cache:
            # Generate consistent model ID first
            model_id = f"{model_type}_{len(self.trained_models)}"

            # Apply feature engineering and scaling for all models
            if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
                # Get polynomial degree from params, default to 1 (no polynomial features)
                poly_degree = params.get('polynomial_degree', 1)

                if poly_degree > 1:
                    # Create polynomial features
                    self.poly_features[model_id] = PolynomialFeatures(degree=poly_degree, include_bias=False)
                    X_train_poly = self.poly_features[model_id].fit_transform(X_train)
                    poly_feature_names = self.poly_features[model_id].get_feature_names_out(X_train.columns)
                    X_train = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)
                    model_logger.info(f"Applied polynomial features (degree {poly_degree}) to {model_type} model")

                # Apply RobustScaler instead of StandardScaler for better outlier handling
                self.scaler = RobustScaler()
                X_train_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                model_logger.info(f"Applied RobustScaler to features for {model_type} model")
            else:
                X_train_scaled = X_train

            # Create and train model - handle random_state and parameters appropriately
            if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
                # Filter out polynomial_degree from params as it's handled separately
                filtered_params = {k: v for k, v in params.items() if k != 'polynomial_degree'}
                
                # Ridge, Lasso, ElasticNet accept random_state, LinearRegression doesn't
                if model_type in ['ridge', 'lasso', 'elasticnet']:
                    model = model_class(random_state=RANDOM_STATE, **filtered_params)
                else:
                    model = model_class(**filtered_params)
            else:
                # Tree-based and ensemble models do accept random_state
                model = model_class(random_state=RANDOM_STATE, **params)

            model_logger.info(f"Training {model_type} model with params: {params}")

            start_time = datetime.now()
            model.fit(X_train_scaled, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Apply tree-specific regularization after training
            if model_type in ['decision_tree', 'random_forest']:
                self._apply_tree_regularization(model, model_type, params, X_train_scaled, y_train)
        else:
            # For cached models, we still need to apply polynomial features and scaling
            if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
                # Get polynomial degree from params, default to 1
                poly_degree = params.get('polynomial_degree', 1)

                if poly_degree > 1 and model_id in self.poly_features:
                    # Apply polynomial features
                    X_train_poly = self.poly_features[model_id].transform(X_train)
                    poly_feature_names = self.poly_features[model_id].get_feature_names_out(X_train.columns)
                    X_train = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)

                # Apply scaling if scaler exists
                if hasattr(self, 'scaler'):
                    X_train_scaled = pd.DataFrame(
                        self.scaler.transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index
                    )
                else:
                    X_train_scaled = X_train
            else:
                X_train_scaled = X_train
            training_time = cached_results.get('training_time', 0)

        # Store trained model
        self.trained_models[model_id] = model
        model_logger.info(f"Stored trained model with ID: {model_id}")

        # Perform cross-validation with proper scaling and polynomial features
        cv_scores = self._perform_cv_with_scaling(model, X_train, y_train, model_type, task_type, params)

        # Calculate training metrics
        y_pred_train = model.predict(X_train_scaled)

        if task_type == 'regression':
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            cv_mean = -cv_scores.mean() if 'neg_mean_squared_error' in str(cv_scores) else cv_scores.mean()
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

        # Calculate overfitting indicators
        overfitting_indicators = self._calculate_overfitting_indicators(
            metrics, task_type, model_type
        )

        results = {
            'model_id': model_id,
            'model_type': model_type,
            'task_type': task_type,
            'parameters': params,
            'training_time': training_time,
            'metrics': metrics,
            'overfitting_indicators': overfitting_indicators,
            'feature_names': feature_names,  # Include for cache validation
            'feature_importance': self._get_feature_importance(model, X_train.columns.tolist()) if hasattr(model, 'feature_importances_') else None,
            'coefficients': self._get_coefficients(model) if hasattr(model, 'coef_') else None,
            'training_timestamp': datetime.now().isoformat()
        }

        # Cache results only if not using cache
        if not use_cache:
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

    def _perform_cv_with_scaling(self, model, X_train: pd.DataFrame, y_train: np.ndarray,
                                model_type: str, task_type: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Perform cross-validation with proper scaling for linear models.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            model_type: Type of model
            task_type: Task type ('regression' or 'classification')

        Returns:
            Cross-validation scores
        """
        from sklearn.model_selection import cross_val_score

        # For linear models, we need to apply polynomial features and scaling within each CV fold
        if model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            from sklearn.pipeline import Pipeline

            # Get polynomial degree from params
            poly_degree = params.get('polynomial_degree', 1) if params else 1

            if poly_degree > 1:
                # Create pipeline with polynomial features, scaling, and model
                pipeline = Pipeline([
                    ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                    ('scaler', RobustScaler()),
                    ('model', model.__class__(**model.get_params()))
                ])
            else:
                # Create pipeline with just scaling and model
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

        # Calculate training metrics
        y_pred_train = model.predict(X_train_scaled)

        if task_type == 'regression':
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            cv_mean = -cv_scores.mean() if 'neg_mean_squared_error' in str(cv_scores) else cv_scores.mean()
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
            'feature_names': feature_names,  # Include for cache validation
            'feature_importance': self._get_feature_importance(model, X_train.columns.tolist()) if hasattr(model, 'feature_importances_') else None,
            'coefficients': self._get_coefficients(model) if hasattr(model, 'coef_') else None,
            'training_timestamp': datetime.now().isoformat()
        }

        # Cache results only if not using cache
        if not use_cache:
            cache_model_results(model_type, params, results)

            # Also cache the model object separately for potential future retrieval
            model_cache_key = f"model_object_{model_id}"
            try:
                import joblib
                model_bytes = joblib.dump(model, model_cache_key + '.pkl')
                # Read the file and store in cache
                with open(model_cache_key + '.pkl', 'rb') as f:
                    model_bytes = f.read()
                from src.config import CACHE_TTL
                cache.set(model_cache_key, model_bytes, ttl=CACHE_TTL)
                # Clean up temp file
                import os
                os.remove(model_cache_key + '.pkl')
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

    def _calculate_overfitting_indicators(self, metrics: Dict[str, Any], task_type: str,
                                        model_type: str) -> Dict[str, Any]:
        """
        Calculate indicators of overfitting for the trained model.

        Args:
            metrics: Training and CV metrics
            task_type: Type of task ('regression' or 'classification')
            model_type: Type of model

        Returns:
            Dictionary with overfitting indicators
        """
        indicators = {}

        if task_type == 'regression':
            train_score = metrics.get('train_r2', 0)
            cv_score = metrics.get('cv_mse_mean', 0)
            train_mse = metrics.get('train_mse', 0)
            cv_mse = metrics.get('cv_mse_mean', 0)

            # R² difference (higher difference indicates overfitting)
            r2_diff = train_score - cv_score if cv_score > 0 else 0

            # MSE ratio (higher ratio indicates overfitting)
            mse_ratio = train_mse / cv_mse if cv_mse > 0 else 1.0

            indicators = {
                'r2_train_cv_diff': r2_diff,
                'mse_train_cv_ratio': mse_ratio,
                'overfitting_risk': 'high' if r2_diff > 0.2 or mse_ratio > 2.0 else 'medium' if r2_diff > 0.1 or mse_ratio > 1.5 else 'low',
                'cv_stability_score': 1.0 / (1.0 + metrics.get('cv_mse_std', 0))
            }

        else:  # classification
            train_score = metrics.get('train_accuracy', 0)
            cv_score = metrics.get('cv_accuracy_mean', 0)

            # Accuracy difference (higher difference indicates overfitting)
            acc_diff = train_score - cv_score

            indicators = {
                'accuracy_train_cv_diff': acc_diff,
                'overfitting_risk': 'high' if acc_diff > 0.15 else 'medium' if acc_diff > 0.1 else 'low',
                'cv_stability_score': 1.0 / (1.0 + metrics.get('cv_accuracy_std', 0))
            }

        # Model-specific indicators
        if model_type in ['decision_tree', 'random_forest']:
            # For tree models, check complexity indicators
            indicators.update({
                'complexity_indicators': {
                    'max_depth': getattr(self.trained_models[list(self.trained_models.keys())[-1]], 'max_depth', None),
                    'n_leaves': getattr(self.trained_models[list(self.trained_models.keys())[-1]], 'n_leaves_', None),
                    'complexity_risk': 'high' if indicators.get('overfitting_risk') == 'high' else 'medium'
                }
            })

        return indicators

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
            ensemble_type: Type of ensemble ('stacking', 'blending', 'voting')

        Returns:
            Ensemble model results
        """
        from sklearn.ensemble import StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier

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
                    cv=5,
                    random_state=RANDOM_STATE
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

        # Store ensemble
        self.trained_models[ensemble_id] = ensemble

        # Evaluate ensemble
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=self.cv,
                                  scoring='neg_mean_squared_error' if task_type == 'regression' else 'accuracy')

        if task_type == 'regression':
            cv_mean = -cv_scores.mean()
            cv_std = cv_scores.std()
            metrics = {
                'cv_mse_mean': cv_mean,
                'cv_mse_std': cv_std,
                'cv_rmse_mean': np.sqrt(cv_mean)
            }
        else:
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            metrics = {
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std
            }

        results = {
            'ensemble_id': ensemble_id,
            'ensemble_type': ensemble_type,
            'base_models': [tree_model_id, linear_model_id],
            'training_time': training_time,
            'metrics': metrics,
            'feature_importance': self._get_ensemble_feature_importance(ensemble, X_train.columns.tolist()),
            'training_timestamp': datetime.now().isoformat()
        }

        model_logger.info(f"Created {ensemble_type} ensemble: {ensemble_id}")
        return results

    def _get_ensemble_feature_importance(self, ensemble, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance from ensemble models.

        Args:
            ensemble: Trained ensemble model
            feature_names: Feature names

        Returns:
            Dictionary of feature importance
        """
        # For stacking ensembles, use the final estimator's feature importance if available
        if hasattr(ensemble, 'final_estimator_') and hasattr(ensemble.final_estimator_, 'feature_importances_'):
            return dict(zip(feature_names, ensemble.final_estimator_.feature_importances_))

        # For voting ensembles, average feature importance from base models
        if hasattr(ensemble, 'estimators_'):
            importance_sum = None
            count = 0
            for name, estimator in ensemble.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    if importance_sum is None:
                        importance_sum = np.array(estimator.feature_importances_)
                    else:
                        importance_sum += estimator.feature_importances_
                    count += 1

            if importance_sum is not None and count > 0:
                avg_importance = importance_sum / count
                return dict(zip(feature_names, avg_importance))

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

        # Apply polynomial features and scaling for linear models
        if model_id in self.poly_features:
            # Apply polynomial transformation
            X_poly = self.poly_features[model_id].transform(X)
            poly_feature_names = self.poly_features[model_id].get_feature_names_out(X.columns)
            X_transformed = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
        else:
            X_transformed = X

        # Apply scaling for linear-based models
        if hasattr(self, 'scaler') and any(model_id.startswith(prefix) for prefix in ['linear', 'ridge', 'lasso', 'elasticnet']):
            X_transformed = pd.DataFrame(
                self.scaler.transform(X_transformed),
                columns=X_transformed.columns,
                index=X_transformed.index
            )

        return model.predict(X_transformed)

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

            # Calculate MAPE safely, avoiding division by zero
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

        model_logger.info(f"Model {model_id} evaluation completed. Test Score: {metrics.get('test_r2', metrics.get('test_accuracy', 'N/A')):.4f}")

        return metrics

    def hyperparameter_tuning(self, model_type: str, X_train: pd.DataFrame, y_train: np.ndarray,
                              param_grid: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Perform enhanced hyperparameter tuning with adaptive parameters and regularization.

        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter grid to search

        Returns:
            Best parameters and scores
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        task_type = self._determine_task_type(y_train)
        model_class = self._get_model_class(model_type, task_type)

        if param_grid is None:
            param_grid = self.model_configs.get(model_type, {})

        # For tree-based models, use adaptive parameter grids
        if model_type in ['decision_tree', 'random_forest']:
            adaptive_params = self._get_adaptive_tree_params(model_type, X_train, y_train)
            # Merge adaptive params with provided param_grid
            for key, values in adaptive_params.items():
                if key not in param_grid:
                    param_grid[key] = values

        model_logger.info(f"Starting enhanced hyperparameter tuning for {model_type}")

        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'

        # For linear-based models, we need to handle polynomial_degree separately
        if model_type in ['linear', 'ridge', 'lasso', 'elasticnet'] and 'polynomial_degree' in param_grid:
            # Create custom parameter grid combinations
            poly_degrees = param_grid.pop('polynomial_degree')
            other_params = param_grid

            best_score = float('-inf') if task_type == 'regression' else float('inf')
            best_params = {}
            all_results = []

            for degree in poly_degrees:
                # Create pipeline with polynomial features
                from sklearn.pipeline import Pipeline

                if degree > 1:
                    pipeline = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                        ('scaler', RobustScaler()),
                        ('model', model_class() if model_type == 'linear' else model_class(random_state=RANDOM_STATE))
                    ])
                else:
                    pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        ('model', model_class() if model_type == 'linear' else model_class(random_state=RANDOM_STATE))
                    ])

                # Create parameter grid for other parameters
                if other_params:
                    grid_search = GridSearchCV(
                        pipeline,
                        other_params,
                        cv=self.cv,
                        scoring=scoring,
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    current_score = -grid_search.best_score_ if task_type == 'regression' else grid_search.best_score_
                    current_params = {'polynomial_degree': degree, **grid_search.best_params_}
                else:
                    # No other parameters, just evaluate the pipeline directly
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(pipeline, X_train, y_train, cv=self.cv, scoring=scoring)
                    current_score = -scores.mean() if task_type == 'regression' else scores.mean()
                    current_params = {'polynomial_degree': degree}
                    # Create a mock cv_results for consistency
                    cv_results = {'mean_test_score': scores}

                all_results.append({
                    'params': current_params,
                    'score': current_score,
                    'cv_results': cv_results if 'cv_results' in locals() else grid_search.cv_results_
                })

                if task_type == 'regression':
                    if current_score > best_score:
                        best_score = current_score
                        best_params = current_params
                else:
                    if current_score < best_score:
                        best_score = current_score
                        best_params = current_params

            results = {
                'best_params': best_params,
                'best_score': best_score,
                'cv_results': all_results,
                'all_scores': [r['score'] for r in all_results],
                'tuning_method': 'grid_search_with_polynomial'
            }
        else:
            # Enhanced hyperparameter tuning for tree-based and ensemble models
            if model_type in ['decision_tree', 'random_forest', 'gradient_boosting']:
                # Use RandomizedSearchCV for larger parameter spaces to be more efficient
                n_iter = min(50, len([1 for params in param_grid.values() for _ in params]))

                search_cv = RandomizedSearchCV(
                    model_class(random_state=RANDOM_STATE),
                    param_grid,
                    n_iter=n_iter,
                    cv=self.cv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1,
                    random_state=RANDOM_STATE
                )
                tuning_method = 'randomized_search'
            else:
                # Standard GridSearchCV for linear-based models
                if model_type in ['ridge', 'lasso', 'elasticnet']:
                    search_cv = GridSearchCV(
                        model_class(random_state=RANDOM_STATE),
                        param_grid,
                        cv=self.cv,
                        scoring=scoring,
                        n_jobs=-1,
                        verbose=1
                    )
                else:
                    search_cv = GridSearchCV(
                        model_class(),
                        param_grid,
                        cv=self.cv,
                        scoring=scoring,
                        n_jobs=-1,
                        verbose=1
                    )
                tuning_method = 'grid_search'

            search_cv.fit(X_train, y_train)

            best_score = -search_cv.best_score_ if task_type == 'regression' else search_cv.best_score_

            # Calculate overfitting risk for best parameters
            best_model = model_class(random_state=RANDOM_STATE, **search_cv.best_params_)
            best_model.fit(X_train, y_train)

            # Quick overfitting check
            train_pred = best_model.predict(X_train)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=self.cv, scoring=scoring)

            if task_type == 'regression':
                train_score = r2_score(y_train, train_pred)
                cv_score = -cv_scores.mean()
                overfitting_risk = 'high' if (train_score - cv_score) > 0.2 else 'medium' if (train_score - cv_score) > 0.1 else 'low'
            else:
                train_score = accuracy_score(y_train, train_pred)
                cv_score = cv_scores.mean()
                overfitting_risk = 'high' if (train_score - cv_score) > 0.15 else 'medium' if (train_score - cv_score) > 0.1 else 'low'

            results = {
                'best_params': search_cv.best_params_,
                'best_score': best_score,
                'cv_results': search_cv.cv_results_,
                'all_scores': search_cv.cv_results_['mean_test_score'],
                'tuning_method': tuning_method,
                'overfitting_risk_assessment': {
                    'train_score': train_score,
                    'cv_score': cv_score,
                    'score_difference': train_score - cv_score,
                    'overfitting_risk': overfitting_risk
                }
            }

        model_logger.info(f"Enhanced hyperparameter tuning completed. Best score: {best_score:.4f}, Overfitting risk: {results.get('overfitting_risk_assessment', {}).get('overfitting_risk', 'unknown')}")

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
        Compare multiple trained models with enhanced analysis including overfitting indicators.

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

                # Get overfitting indicators if available
                overfitting_indicators = {}
                if hasattr(self, 'trained_models') and model_id in self.trained_models:
                    # Try to get overfitting indicators from training results
                    # This would need to be stored during training
                    pass

                results[model_id] = {
                    'info': info,
                    'evaluation': evaluation,
                    'overfitting_indicators': overfitting_indicators
                }
            except Exception as e:
                model_logger.error(f"Error evaluating model {model_id}: {e}")
                results[model_id] = {'error': str(e)}

        # Determine best model with enhanced criteria
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if valid_results:
            task_type = self._determine_task_type(y_test)
            metric = 'test_r2' if task_type == 'regression' else 'test_accuracy'

            # Create comprehensive ranking with multiple criteria
            model_rankings = []
            for model_id, model_data in valid_results.items():
                evaluation = model_data['evaluation']
                base_score = evaluation.get(metric, -np.inf)

                # Get overfitting risk assessment
                overfitting_risk = model_data.get('overfitting_indicators', {}).get('overfitting_risk', 'medium')
                risk_penalty = {'low': 0, 'medium': 0.05, 'high': 0.15}.get(overfitting_risk, 0.1)

                # Get CV stability if available
                cv_stability = 0.5  # Default neutral stability
                if 'cv_metrics' in evaluation and 'cv_stability_score' in evaluation['cv_metrics']:
                    cv_stability = evaluation['cv_metrics']['cv_stability_score']

                # Enhanced weighted score: 60% performance + 30% stability + 10% overfitting penalty
                weighted_score = base_score * 0.6 + cv_stability * 0.3 - risk_penalty * 0.1

                # Model type bonus for ensemble compatibility
                model_type_bonus = 0
                model_info = model_data.get('info', {})
                if 'linear' in model_id.lower():
                    model_type_bonus = 0.05  # Bonus for linear models in ensembles
                elif 'random_forest' in model_id.lower():
                    model_type_bonus = 0.03  # Bonus for RF in ensembles

                final_score = weighted_score + model_type_bonus

                model_rankings.append({
                    'model_id': model_id,
                    'final_score': final_score,
                    'base_score': base_score,
                    'cv_stability': cv_stability,
                    'overfitting_risk': overfitting_risk,
                    'model_type_bonus': model_type_bonus,
                    'weighted_components': {
                        'performance': base_score * 0.6,
                        'stability': cv_stability * 0.3,
                        'overfitting_penalty': -risk_penalty * 0.1,
                        'model_bonus': model_type_bonus
                    }
                })

            # Sort by final score and select best
            model_rankings.sort(key=lambda x: x['final_score'], reverse=True)
            best_model = model_rankings[0]

            results['best_model'] = best_model['model_id']
            results['model_rankings'] = model_rankings
            results['selection_criteria'] = {
                'primary_metric': metric,
                'multi_criteria_scoring': True,
                'performance_weight': 0.6,
                'stability_weight': 0.3,
                'overfitting_penalty_weight': 0.1,
                'model_type_bonus': True
            }

            # Generate ensemble recommendations
            results['ensemble_recommendations'] = self._generate_ensemble_recommendations(model_rankings)

        return results

    def _generate_ensemble_recommendations(self, model_rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate recommendations for creating ensembles from top models.

        Args:
            model_rankings: Ranked list of models

        Returns:
            Ensemble recommendations
        """
        recommendations = {
            'suggested_ensembles': [],
            'rationale': []
        }

        if len(model_rankings) >= 2:
            top_models = model_rankings[:3]  # Consider top 3 models

            # Look for complementary model types
            has_linear = any('linear' in m['model_id'].lower() for m in top_models)
            has_tree = any('decision_tree' in m['model_id'].lower() or 'random_forest' in m['model_id'].lower() for m in top_models)

            if has_linear and has_tree:
                recommendations['suggested_ensembles'].append({
                    'type': 'stacking',
                    'models': [m['model_id'] for m in top_models[:2]],
                    'expected_benefit': 'high',
                    'rationale': 'Combining linear and tree-based models for complementary strengths'
                })

                recommendations['suggested_ensembles'].append({
                    'type': 'voting',
                    'models': [m['model_id'] for m in top_models],
                    'expected_benefit': 'medium',
                    'rationale': 'Simple ensemble averaging for robustness'
                })

            recommendations['rationale'].append("Multiple model types available - ensemble methods recommended")
        else:
            recommendations['rationale'].append("Limited model diversity - focus on single best model")

        return recommendations

def create_model_pipeline() -> ModelPipeline:
    """
    Factory function to create ModelPipeline instance.

    Returns:
        ModelPipeline instance
    """
    return ModelPipeline()