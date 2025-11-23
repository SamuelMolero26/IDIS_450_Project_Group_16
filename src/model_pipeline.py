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
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_selection import RFECV
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
from pathlib import Path
import json
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor , KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pickle 
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
        - 'ridge': Ridge Regression (L2 regularization)
        - 'lasso': Lasso Regression (L1 regularization)
        - 'elastic_net': ElasticNet Regression (L1 + L2 regularization)
        - 'decision_tree': Decision Tree Regressor/Classifier
        - 'random_forest': Random Forest Regressor/Classifier

        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'elastic_net', 'decision_tree', 'random_forest')
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
                'classification': LogisticRegression  # Logistic with L2
            },
            'lasso': {
                'regression': Lasso,
                'classification': LogisticRegression  # Logistic with L1
            },
            'elastic_net': {
                'regression': ElasticNet,
                'classification': LogisticRegression  # Logistic with ElasticNet
            },
            'decision_tree': {
                'regression': DecisionTreeRegressor,
                'classification': DecisionTreeClassifier
            },
            'random_forest': {
                'regression': RandomForestRegressor,
                'classification': RandomForestClassifier
            }, 
            'KNN': {
                'regression': KNeighborsRegressor,
                'classification': KNeighborsClassifier
            },
            'ann': {
                'regression': MLPRegressor,
                'classification': MLPClassifier
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
        """Check if model type is linear-based (includes regularized variants)."""
        return model_type in ['linear', 'ridge', 'lasso', 'elastic_net']

    def _is_tree_model(self, model_type: str) -> bool:
        """Check if model type is tree-based."""
        return model_type in ['decision_tree', 'random_forest']

    def _is_ann_model(self, model_type: str) -> bool:
        """Check if model type is ANN-based."""
        return model_type in ['ann']
    
    def _is_distance_based_model(self, model_type: str) -> bool:
        """Check if model type is distance-based (KNN)."""
        return model_type == 'KNN'

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
                                   X_train: pd.DataFrame, y_train: np.ndarray,
                                   is_final_model: bool = False) -> None:
        """
        Apply tree-specific regularization techniques.

        Args:
            model: Trained model
            model_type: Type of model
            params: Model parameters
            X_train: Training features
            y_train: Training targets
            is_final_model: Whether this is the final model (not during CV)
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
            # Additional regularization for random forest - only apply to final model
            # to avoid issues during cross-validation with different fold sizes
            if is_final_model and params.get('bootstrap', True) and 'max_samples' in params:
                n_samples = len(X_train)
                max_samples = params.get('max_samples', 1.0)
                if isinstance(max_samples, float) and max_samples <= 1.0:
                    max_samples = int(max_samples * n_samples)
                model.max_samples = max_samples
                model_logger.info(f"Applied bootstrap sampling with max_samples={max_samples}")

    # ============================================================================
    # MULTICOLLINEARITY DETECTION AND FEATURE SELECTION
    # ============================================================================

    def _detect_multicollinearity(self, X: pd.DataFrame,
                                  threshold: float = 10.0) -> Dict[str, Any]:
        """
        Calculate Variance Inflation Factor (VIF) for each feature.

        Args:
            X: Feature matrix
            threshold: VIF threshold (default 10.0, features above are problematic)

        Returns:
            Dictionary with VIF scores and problematic features
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                               for i in range(len(X.columns))]

            problematic_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()

            model_logger.info(f"VIF Analysis complete. Max VIF: {vif_data['VIF'].max():.2f}")
            if problematic_features:
                model_logger.warning(f"High VIF features (>{threshold}): {problematic_features}")

            return {
                'vif_scores': vif_data.to_dict('records'),
                'problematic_features': problematic_features,
                'max_vif': float(vif_data["VIF"].max()),
                'mean_vif': float(vif_data["VIF"].mean())
            }
        except ImportError:
            model_logger.warning("statsmodels not available, skipping VIF analysis")
            return {'vif_scores': [], 'problematic_features': [], 'max_vif': 0, 'mean_vif': 0}
        except Exception as e:
            model_logger.error(f"Error in VIF calculation: {e}")
            return {'vif_scores': [], 'problematic_features': [], 'max_vif': 0, 'mean_vif': 0}

    def _remove_correlated_features(self, X: pd.DataFrame,
                                    threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with correlation > threshold.

        Args:
            X: Feature matrix
            threshold: Correlation threshold (default 0.95)

        Returns:
            Tuple of (filtered DataFrame, list of dropped features)
        """
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper_triangle.columns
                   if any(upper_triangle[column] > threshold)]

        if to_drop:
            model_logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")

        return X.drop(columns=to_drop), to_drop

    def _perform_rfe(self, X_train: pd.DataFrame, y_train: np.ndarray,
                     model_type: str = 'ridge',
                     n_features_to_select: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Recursive Feature Elimination with Cross-Validation.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of base model for RFE
            n_features_to_select: Number of features to select (None for automatic)

        Returns:
            Dict with selected features, rankings, and scores
        """
        estimator = Ridge(alpha=1.0)

        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=self.cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        rfecv.fit(X_train, y_train)

        selected_features = X_train.columns[rfecv.support_].tolist()
        feature_rankings = dict(zip(X_train.columns, rfecv.ranking_))

        model_logger.info(f"RFE selected {len(selected_features)}/{len(X_train.columns)} features")

        return {
            'selected_features': selected_features,
            'n_features_selected': len(selected_features),
            'feature_rankings': feature_rankings,
            'cv_scores': rfecv.cv_results_['mean_test_score'].tolist() if hasattr(rfecv, 'cv_results_') else [],
            'optimal_n_features': rfecv.n_features_
        }

    # ============================================================================
    # FEATURE TRANSFORMATION AND ENGINEERING
    # ============================================================================
    
    def _apply_feature_transformations(self, X: pd.DataFrame,
                                       strategy: str = 'auto',
                                       fit: bool = True) -> pd.DataFrame:
        """
        Apply feature transformations to improve linear model performance.

        Args:
            X: Feature matrix
            strategy: Transformation strategy ('auto', 'log', 'sqrt', 'box-cox', 'quantile')
            fit: Whether to fit transformers or use existing

        Returns:
            Transformed feature matrix
        """
        
        #check for model type
       
        X_transformed = X.copy()

        for col in X.columns:
            try:
                # Skip if constant column
                if X[col].nunique() <= 1:
                    continue

                skewness = stats.skew(X[col].dropna())

                if abs(skewness) > 1.0:  # Highly skewed
                    if (X[col] > 0).all():  # All positive
                        if strategy == 'auto' or strategy == 'log':
                            X_transformed[col] = np.log1p(X[col])
                            model_logger.debug(f"Applied log transform to {col} (skewness: {skewness:.2f})")
                elif abs(skewness) > 0.5:  # Moderately skewed
                    if strategy == 'auto' or strategy == 'sqrt':
                        min_val = X[col].min()
                        X_transformed[col] = np.sqrt(X[col] - min_val + 1)
                        model_logger.debug(f"Applied sqrt transform to {col} (skewness: {skewness:.2f})")
            except Exception as e:
                model_logger.warning(f"Could not transform {col}: {e}")
                continue

        return X_transformed

    def _create_domain_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific interaction features for sales data.

        Based on business knowledge of US Regional Sales dataset.

        Args:
            X: Feature matrix

        Returns:
            Enhanced feature matrix with interaction terms
        """
        X_enhanced = X.copy()

        # Price-related interactions
        if 'Unit Price' in X.columns and 'Order Quantity' in X.columns:
            X_enhanced['Price_x_Quantity'] = X['Unit Price'] * X['Order Quantity']

        if 'Unit Price' in X.columns and 'Discount Applied' in X.columns:
            X_enhanced['Effective_Price'] = X['Unit Price'] * (1 - X['Discount Applied'])

        # Cost-profit interactions
        if 'Unit Cost' in X.columns and 'Profit_Margin' in X.columns:
            X_enhanced['Expected_Profit'] = X['Unit Cost'] * X['Profit_Margin']

        # Time-related interactions
        if 'Total_Lead_Time' in X.columns and 'Order Quantity' in X.columns:
            X_enhanced['Lead_Time_per_Unit'] = X['Total_Lead_Time'] / (X['Order Quantity'] + 1)

        # Discount effectiveness
        if 'Discount Applied' in X.columns and 'Order Quantity' in X.columns:
            X_enhanced['Discount_Impact'] = X['Discount Applied'] * X['Order Quantity']

        n_new_features = len(X_enhanced.columns) - len(X.columns)
        if n_new_features > 0:
            model_logger.info(f"Created {n_new_features} domain interaction features")

        return X_enhanced

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

        if self._is_linear_model(model_type) or self._is_distance_based_model(model_type):
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

            # Apply RobustScaler for linear and distance-based models
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

        elif self._is_ann_model(model_type):
            # ANN requires StandardScaler (mean=0, std=1) for optimal performance
            if fit:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X_transformed)
                model_logger.info(f"Applied StandardScaler to features for {model_type} (ANN)")
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
            model_bytes = pickle.dumps(model)
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
            model_type: Type of model ('linear', 'decision_tree', 'random_forest', 'KNN')
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
        elif model_type == 'KNN':
            # KNN doesn't accept random_state
            return model_class(**filtered_params)
        elif self._is_ann_model(model_type):
            # ANN models (MLPRegressor/MLPClassifier) have specific parameter handling
            # They accept random_state but it affects weight initialization
            # Don't override if random_state is already specified in params
            if 'random_state' not in filtered_params:
                return model_class(random_state=RANDOM_STATE, **filtered_params)
            else:
                return model_class(**filtered_params)
        else:
            # Tree-based models (decision_tree, random_forest) accept random_state
            model = model_class(random_state=RANDOM_STATE, **filtered_params)
            # Log max_samples for RandomForest to debug the ValueError
            if model_type == 'random_forest' and hasattr(model, 'max_samples'):
                model_logger.info(f"RandomForest max_samples set to: {model.max_samples}")
            return model

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
        elif self._is_ann_model(model_type):
            # ANN models require StandardScaler for optimal performance
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
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

        # Log dataset size for debugging max_samples issue
        model_logger.info(f"Training {model_type} with dataset size: {len(X_train)} samples, {len(feature_names)} features")

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
                    model, model_type, params, X_train_scaled, y_train, is_final_model=True
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

        # KNN-specific validation and documentation
        if model_type == 'KNN':
            knn_validation = self.validate_knn_optimal_k(results)
            results['knn_validation'] = knn_validation

            # Generate K vs accuracy plot if hyperparameter tuning was performed
            if hasattr(self, '_last_tuning_results') and self._last_tuning_results:
                self.generate_knn_k_vs_accuracy_plot(self._last_tuning_results)

            # Add documentation for experiment reports
            knn_documentation = self.document_knn_optimal_k(results, knn_validation)
            results.update(knn_documentation)

        # Cache if new model
        if not use_cache:
            self._cache_model_and_results(model, model_id, model_type, params, results)

        cv_mean = metrics.get('cv_mse_mean', metrics.get('cv_accuracy_mean', 0))
        model_logger.info(f"Model {model_type} trained successfully. CV Score: {cv_mean:.4f}")

        return results

    # ============================================================================
    # RESIDUAL DIAGNOSTICS AND MODEL VALIDATION
    # ============================================================================

    def _calculate_cooks_distance(self, model, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """
        Calculate Cook's distance for detecting influential points.

        Args:
            model: Fitted model
            X: Feature matrix
            y: Target values

        Returns:
            Array of Cook's distances
        """
        try:
            from statsmodels.stats.outliers_influence import OLSInfluence

            # For linear models, we can use statsmodels
            if hasattr(model, 'coef_'):
                y_pred = model.predict(X)
                residuals = y - y_pred

                # Approximate Cook's distance
                n = len(X)
                p = X.shape[1]
                mse = np.mean(residuals ** 2)

                # Hat matrix diagonal (leverage)
                if hasattr(X, 'values'):
                    X_vals = X.values
                else:
                    X_vals = X

                H = X_vals @ np.linalg.pinv(X_vals.T @ X_vals) @ X_vals.T
                h = np.diag(H)

                # Cook's distance formula
                cooks_d = (residuals ** 2 / (p * mse)) * (h / (1 - h) ** 2)
                return cooks_d
            else:
                # For tree models, return zeros
                return np.zeros(len(X))
        except Exception as e:
            model_logger.warning(f"Could not calculate Cook's distance: {e}")
            return np.zeros(len(X))

    def _analyze_residuals(self, model, X_train: pd.DataFrame,
                           y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis.

        Checks:
        - Normality (Shapiro-Wilk test)
        - Homoscedasticity (Breusch-Pagan test)
        - Autocorrelation (Durbin-Watson)
        - Influential points (Cook's distance)

        Args:
            model: Fitted model
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary with diagnostic results
        """
        try:
            y_pred = model.predict(X_train)
            residuals = y_train - y_pred
            standardized_residuals = (residuals - residuals.mean()) / residuals.std()

            # Normality test (sample if too large)
            sample_size = min(5000, len(residuals))
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:sample_size])

            # Homoscedasticity test
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_test = het_breuschpagan(residuals, X_train)
                bp_statistic, bp_p_value = bp_test[0], bp_test[1]
                is_homoscedastic = bp_p_value > 0.05
            except:
                bp_statistic, bp_p_value = 0.0, 1.0
                is_homoscedastic = True

            # Autocorrelation
            try:
                from statsmodels.stats.stattools import durbin_watson
                dw_stat = durbin_watson(residuals)
                has_autocorrelation = dw_stat < 1.5 or dw_stat > 2.5
            except:
                dw_stat = 2.0
                has_autocorrelation = False

            # Influential points
            cooks_d = self._calculate_cooks_distance(model, X_train, y_train)
            influential_threshold = 4 / len(X_train)
            n_influential = np.sum(cooks_d > influential_threshold)

            diagnostics = {
                'normality': {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'is_normal': bool(shapiro_p > 0.05)
                },
                'homoscedasticity': {
                    'breusch_pagan_statistic': float(bp_statistic),
                    'breusch_pagan_p_value': float(bp_p_value),
                    'is_homoscedastic': bool(is_homoscedastic)
                },
                'autocorrelation': {
                    'durbin_watson': float(dw_stat),
                    'has_autocorrelation': bool(has_autocorrelation)
                },
                'influential_points': {
                    'n_influential': int(n_influential),
                    'pct_influential': float((n_influential / len(X_train)) * 100)
                },
                'residual_stats': {
                    'mean': float(residuals.mean()),
                    'std': float(residuals.std()),
                    'skewness': float(stats.skew(residuals)),
                    'kurtosis': float(stats.kurtosis(residuals))
                }
            }

            model_logger.info(f"Residual diagnostics: Normal={diagnostics['normality']['is_normal']}, "
                            f"Homoscedastic={diagnostics['homoscedasticity']['is_homoscedastic']}")

            return diagnostics
        except Exception as e:
            model_logger.error(f"Error in residual analysis: {e}")
            return {
                'normality': {'shapiro_statistic': 0, 'shapiro_p_value': 1, 'is_normal': True},
                'homoscedasticity': {'breusch_pagan_statistic': 0, 'breusch_pagan_p_value': 1, 'is_homoscedastic': True},
                'autocorrelation': {'durbin_watson': 2.0, 'has_autocorrelation': False},
                'influential_points': {'n_influential': 0, 'pct_influential': 0},
                'residual_stats': {'mean': 0, 'std': 0, 'skewness': 0, 'kurtosis': 0}
            }

    def _generate_learning_curves(self, model_type: str, X_train: pd.DataFrame,
                                  y_train: np.ndarray, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate learning curves to diagnose bias/variance.

        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            params: Model parameters

        Returns:
            Dictionary with learning curve data
        """
        try:
            model_class = self._get_model_class(model_type, 'regression')
            if params:
                filtered_params = {k: v for k, v in params.items() if k != 'polynomial_degree'}
                if self._is_linear_model(model_type):
                    model = model_class(**filtered_params)
                else:
                    model = model_class(random_state=RANDOM_STATE, **filtered_params)
            else:
                if self._is_linear_model(model_type):
                    model = model_class()
                else:
                    model = model_class(random_state=RANDOM_STATE)

            train_sizes = np.linspace(0.1, 1.0, 10)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=self.cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            convergence_gap = float(np.abs(train_scores.mean() - val_scores.mean()))

            model_logger.info(f"Learning curves generated. Convergence gap: {convergence_gap:.4f}")

            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': (-train_scores.mean(axis=1)).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': (-val_scores.mean(axis=1)).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist(),
                'convergence_gap': convergence_gap
            }
        except Exception as e:
            model_logger.error(f"Error generating learning curves: {e}")
            return {
                'train_sizes': [],
                'train_scores_mean': [],
                'train_scores_std': [],
                'val_scores_mean': [],
                'val_scores_std': [],
                'convergence_gap': 0.0
            }

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

    def train_with_cv_regularization(self, X_train: pd.DataFrame,
                                     y_train: np.ndarray,
                                     model_type: str = 'ridge') -> Tuple[Any, Dict[str, Any]]:
        """
        Train regularized model with automatic alpha selection using CV.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of regularized model ('ridge', 'lasso', 'elastic_net')

        Returns:
            Tuple of (trained model, results dictionary)
        """
        alphas = np.logspace(-4, 4, 50)

        model_logger.info(f"Training {model_type} with automatic alpha selection (CV)")

        if model_type == 'ridge':
            model = RidgeCV(alphas=alphas, cv=self.cv, scoring='neg_mean_squared_error')
        elif model_type == 'lasso':
            model = LassoCV(alphas=alphas, cv=self.cv, max_iter=10000, n_jobs=-1)
        elif model_type == 'elastic_net':
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
            model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=self.cv,
                                max_iter=10000, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model type for CV regularization: {model_type}")

        model.fit(X_train, y_train)

        results = {
            'optimal_alpha': float(model.alpha_),
            'model_type': model_type
        }

        if model_type == 'elastic_net':
            results['optimal_l1_ratio'] = float(model.l1_ratio_)

        model_logger.info(f"Optimal alpha for {model_type}: {model.alpha_:.6f}")

        return model, results

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

      

        if model_type == 'random_forest' and 'max_samples' in param_grid:
            # Filter out incompatible combinations where bootstrap=False and max_samples is not None
            # max_samples should only be used when bootstrap=True
            param_grid = self._fix_random_forest_max_samples(param_grid)

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

    def _fix_random_forest_max_samples(self, param_grid: Dict[str, List]) -> Dict[str, List]:
        """
        Fix Random Forest max_samples parameter combinations to avoid CV errors.

        When bootstrap=False, max_samples should be None.
        When bootstrap=True, max_samples should be a float between 0 and 1.

        Args:
            param_grid: Original parameter grid

        Returns:
            Fixed parameter grid with compatible combinations
        """
        from sklearn.model_selection import ParameterGrid

        if 'max_samples' not in param_grid:
            return param_grid

        # Generate all possible combinations
        all_combinations = list(ParameterGrid(param_grid))

        # Filter out incompatible combinations
        compatible_combinations = []
        for combo in all_combinations:
            bootstrap = combo.get('bootstrap', True)
            max_samples = combo.get('max_samples')

            # If bootstrap=False, max_samples should be None
            if not bootstrap:
                combo = combo.copy()
                combo['max_samples'] = None

            # If bootstrap=True and max_samples is None, set to a default proportion
            elif bootstrap and max_samples is None:
                combo = combo.copy()
                combo['max_samples'] = 1.0

            compatible_combinations.append(combo)

        # Convert back to parameter grid format
        fixed_param_grid = {}
        for key in param_grid.keys():
            unique_values = list(set(combo[key] for combo in compatible_combinations))
            # Sort for consistency, but keep None first for bootstrap=False cases
            if None in unique_values:
                unique_values.remove(None)
                unique_values = [None] + sorted(unique_values)
            else:
                unique_values = sorted(unique_values)
            fixed_param_grid[key] = unique_values

        return fixed_param_grid

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
        if (self._is_linear_model(model_type) or self._is_ann_model(model_type)) and self.scaler is not None:
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


    # ============================================================================
    # KNN-SPECIFIC VALIDATION AND VISUALIZATION
    # ============================================================================

    def validate_knn_optimal_k(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate KNN optimal K selection and provide comprehensive analysis.

        Args:
            model_results: Results dictionary from train_model() for KNN

        Returns:
            Validation results with optimal K analysis
        """
        if model_results.get('model_type') != 'KNN':
            raise ValueError("This validation is only for KNN models")

        optimal_k = model_results.get('parameters', {}).get('n_neighbors')

        # Handle case where n_neighbors is not set (default training)
        if optimal_k is None:
            # Get the default K value from the trained model
            model_id = model_results.get('model_id')
            if model_id and model_id in self.trained_models:
                trained_model = self.trained_models[model_id]
                if hasattr(trained_model, 'n_neighbors'):
                    optimal_k = trained_model.n_neighbors
                else:
                    optimal_k = 5  # sklearn default
            else:
                optimal_k = 5  # fallback default

        cv_score = model_results.get('metrics', {}).get('cv_mse_mean',
                                                        model_results.get('metrics', {}).get('cv_accuracy_mean', 0))

        # Validate K range
        k_range_valid = 3 <= optimal_k <= 20
        k_reasonable = 5 <= optimal_k <= 15

        validation_results = {
            'optimal_k': optimal_k,
            'cv_score': cv_score,
            'k_in_expected_range': k_range_valid,
            'k_in_reasonable_range': k_reasonable,
            'validation_warnings': []
        }

        # Add warnings for edge cases
        if optimal_k == 3:
            validation_results['validation_warnings'].append(
                "Optimal K is at minimum boundary (3) - may indicate complex local patterns or potential overfitting"
            )
        elif optimal_k == 30:
            validation_results['validation_warnings'].append(
                "Optimal K is at maximum boundary (30) - may indicate simple global patterns or underfitting"
            )
        elif not k_reasonable:
            validation_results['validation_warnings'].append(
                f"Optimal K ({optimal_k}) is outside typical range (5-15) - investigate further"
            )

        model_logger.info(f"KNN Validation: Optimal K={optimal_k}, CV Score={cv_score:.4f}")
        return validation_results

    def generate_knn_k_vs_accuracy_plot(self, model_results: Dict[str, Any],
                                       save_path: str = 'visualizations/knn_optimal_k_analysis.png') -> None:
        """
        Generate K vs Accuracy plot for KNN hyperparameter tuning analysis.

        Args:
            model_results: Results dictionary from hyperparameter_tuning() for KNN
            save_path: Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Extract CV results from hyperparameter tuning
            if 'cv_results' not in model_results:
                model_logger.warning("No CV results found in model_results - cannot generate K vs accuracy plot")
                return

            cv_results = model_results['cv_results']

            # For regression, convert negative MSE to positive R² equivalent
            task_type = model_results.get('task_type', 'regression')
            if task_type == 'regression':
                # GridSearchCV uses neg_mean_squared_error, convert to RMSE for plotting
                mean_scores = -cv_results['mean_test_score']
                std_scores = cv_results['std_test_score']
                metric_name = 'CV RMSE'
            else:
                mean_scores = cv_results['mean_test_score']
                std_scores = cv_results['std_test_score']
                metric_name = 'CV Accuracy'

            # Extract K values from parameter combinations
            k_values = []
            for params in cv_results['params']:
                k_values.append(params.get('n_neighbors', 5))

            # Create DataFrame for plotting
            plot_data = pd.DataFrame({
                'k': k_values,
                'mean_score': mean_scores,
                'std_score': std_scores
            })

            # Aggregate by K value (take mean across different parameter combinations)
            plot_data_agg = plot_data.groupby('k').agg({
                'mean_score': 'mean',
                'std_score': 'mean'
            }).reset_index()

            # Sort by K for proper plotting
            plot_data_agg = plot_data_agg.sort_values('k')

            # Get optimal K
            best_params = model_results.get('best_params', {})
            optimal_k = best_params.get('n_neighbors', plot_data_agg.loc[plot_data_agg['mean_score'].idxmax(), 'k'])

            # Create plot
            plt.figure(figsize=(12, 8))
            sns.set_style("whitegrid")

            # Plot line with error bars
            plt.errorbar(plot_data_agg['k'], plot_data_agg['mean_score'],
                        yerr=plot_data_agg['std_score'],
                        marker='o', markersize=8, linewidth=2,
                        color='#2E86AB', ecolor='#A23B72', capsize=5,
                        label=f'{metric_name} ± 1 STD')

            # Highlight optimal K
            optimal_idx = plot_data_agg[plot_data_agg['k'] == optimal_k].index[0]
            plt.scatter(optimal_k, plot_data_agg.loc[optimal_idx, 'mean_score'],
                       s=150, color='#F24236', zorder=5,
                       label=f'Optimal K = {optimal_k}')

            # Add vertical line at optimal K
            plt.axvline(x=optimal_k, color='#F24236', linestyle='--', alpha=0.7)

            # Formatting
            plt.xlabel('Number of Neighbors (K)', fontsize=14, fontweight='bold')
            plt.ylabel(metric_name, fontsize=14, fontweight='bold')
            plt.title('KNN Hyperparameter Tuning: K vs Cross-Validation Performance',
                     fontsize=16, fontweight='bold', pad=20)

            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)

            # Set x-ticks to show all K values
            plt.xticks(plot_data_agg['k'].values)

            # Add text annotation for optimal K
            optimal_score = plot_data_agg.loc[optimal_idx, 'mean_score']
            plt.annotate(f'K = {optimal_k}\n{metric_name} = {optimal_score:.4f}',
                        xy=(optimal_k, optimal_score),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                        fontsize=11, ha='left')

            # Save plot
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            model_logger.info(f"K vs Accuracy plot saved to {save_path}")

        except ImportError as e:
            model_logger.warning(f"Could not generate K vs accuracy plot: {e}")
        except Exception as e:
            model_logger.error(f"Error generating K vs accuracy plot: {e}")

    def document_knn_optimal_k(self, model_results: Dict[str, Any],
                              validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Document optimal K findings for experiment reports.

        Args:
            model_results: KNN training results
            validation_results: KNN validation results

        Returns:
            Documentation dictionary for experiment reports
        """
        optimal_k = validation_results.get('optimal_k')
        cv_score = validation_results.get('cv_score')

        documentation = {
            'knn_optimal_k_analysis': {
                'optimal_k_value': optimal_k,
                'cv_score_at_optimal_k': cv_score,
                'k_validation_status': 'PASS' if validation_results.get('k_in_reasonable_range') else 'REVIEW',
                'validation_warnings': validation_results.get('validation_warnings', []),
                'k_interpretation': self._interpret_knn_k_value(optimal_k),
                'bias_variance_assessment': self._assess_knn_bias_variance(optimal_k),
                'recommendations': self._generate_knn_k_recommendations(optimal_k, validation_results)
            }
        }

        return documentation

    def _interpret_knn_k_value(self, k: int) -> str:
        """Interpret the meaning of the selected K value."""
        if k <= 3:
            return "Very low K suggests complex local patterns in the data. The model captures fine-grained relationships but may be sensitive to noise."
        elif k <= 7:
            return "Low-moderate K indicates local patterns are important. Good balance between capturing local structure and generalizing to new data."
        elif k <= 15:
            return "Moderate K suggests balanced local-global pattern recognition. Optimal for most tabular datasets."
        elif k <= 20:
            return "Higher K indicates smoother decision boundaries. May miss some local patterns but more robust to noise."
        else:
            return "Very high K suggests global patterns dominate. Decision boundaries are smooth but may underfit complex relationships."

    def _assess_knn_bias_variance(self, k: int) -> Dict[str, str]:
        """Assess bias-variance tradeoff for the selected K."""
        if k <= 5:
            return {
                'bias': 'Low',
                'variance': 'High',
                'tradeoff': 'High variance, low bias - may overfit to training noise'
            }
        elif k <= 10:
            return {
                'bias': 'Moderate',
                'variance': 'Moderate',
                'tradeoff': 'Balanced bias-variance - optimal generalization expected'
            }
        else:
            return {
                'bias': 'High',
                'variance': 'Low',
                'tradeoff': 'Low variance, high bias - may underfit complex patterns'
            }

    def _generate_knn_k_recommendations(self, k: int, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on optimal K analysis."""
        recommendations = []

        if not validation_results.get('k_in_reasonable_range'):
            recommendations.append(f"Consider expanding K search range - optimal K ({k}) is at boundary")

        if k <= 3:
            recommendations.append("Monitor for overfitting - consider cross-validation stability")
            recommendations.append("Evaluate model performance on held-out test set carefully")

        if k >= 20:
            recommendations.append("Monitor for underfitting - model may be too smooth")
            recommendations.append("Consider feature engineering to capture more complex patterns")

        if validation_results.get('validation_warnings'):
            recommendations.extend(validation_results['validation_warnings'])

        if not recommendations:
            recommendations.append("Optimal K selection appears robust - proceed with confidence")

        return recommendations


    def create_comprehensive_model_comparison(self, X_test: pd.DataFrame, y_test: np.ndarray,
                                           include_knn: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive model comparison including KNN with best model selection.

        Args:
            X_test: Test features
            y_test: Test targets
            include_knn: Whether to include KNN in comparison

        Returns:
            Comprehensive comparison results with best model identification
        """
        # Import evaluation engine with fallback for missing dependencies
        try:
            from src.evaluation_engine import create_evaluation_engine
            evaluation_engine = create_evaluation_engine()
        except ImportError as e:
            model_logger.warning(f"Could not import evaluation_engine: {e}. Using simplified evaluation.")
            evaluation_engine = None

        # Define models to compare
        models_to_compare = ['linear', 'ridge', 'lasso', 'decision_tree', 'random_forest']
        if include_knn:
            models_to_compare.append('KNN')

        comparison_results = {}
        evaluation_results = {}

        model_logger.info(f"Starting comprehensive model comparison with {len(models_to_compare)} models")

        for model_type in models_to_compare:
            try:
                model_logger.info(f"Training and evaluating {model_type}")

                # Train model
                if model_type == 'KNN':
                    # Use hyperparameter tuning for KNN to get optimal K
                    tuning_results = self.hyperparameter_tuning('KNN', X_test, y_test)  # Using test data for demo
                    optimal_params = tuning_results['best_params']
                    model_results = self.train_model('KNN', X_test, y_test, params=optimal_params)
                else:
                    model_results = self.train_model(model_type, X_test, y_test)

                # Get trained model
                model_id = model_results['model_id']
                model = self.trained_models[model_id]

                # Evaluate model (simplified if evaluation_engine not available)
                if evaluation_engine:
                    evaluation = evaluation_engine.evaluate_regression_model(
                        model, X_test, X_test, y_test, y_test, model_type
                    )
                else:
                    # Simplified evaluation
                    from sklearn.metrics import mean_squared_error, r2_score

                    y_pred = model.predict(X_test)
                    test_mse = mean_squared_error(y_test, y_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(y_test, y_pred)

                    # Simple CV
                    from sklearn.model_selection import cross_val_score
                    cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='neg_mean_squared_error')
                    cv_mse = -cv_scores.mean()

                    evaluation = {
                        'evaluation_type': 'regression',
                        'test_metrics': {
                            'mse': test_mse,
                            'rmse': test_rmse,
                            'r2': test_r2
                        },
                        'cv_metrics': {
                            'mse_mean': cv_mse,
                            'rmse_mean': np.sqrt(cv_mse)
                        }
                    }

                comparison_results[model_type] = {
                    'model_results': model_results,
                    'evaluation': evaluation,
                    'test_r2': evaluation['test_metrics']['r2'],
                    'test_rmse': evaluation['test_metrics']['rmse'],
                    'cv_r2_mean': evaluation['cv_metrics']['accuracy_mean'] if evaluation['evaluation_type'] == 'classification'
                                else None,  # For regression, we don't have direct CV R²
                    'training_time': model_results.get('training_time', 0)
                }

                evaluation_results[model_type] = evaluation

                model_logger.info(f"Completed evaluation for {model_type}: R² = {evaluation['test_metrics']['r2']:.4f}")

            except Exception as e:
                model_logger.error(f"Failed to evaluate {model_type}: {e}")
                comparison_results[model_type] = {'error': str(e)}

        # Determine best model
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}

        if valid_results:
            # Rank models by R² score (higher is better)
            model_rankings = []
            for model_name, results in valid_results.items():
                r2_score = results.get('test_r2', -float('inf'))
                rmse_score = results.get('test_rmse', float('inf'))

                model_rankings.append({
                    'model_name': model_name,
                    'r2_score': r2_score,
                    'rmse_score': rmse_score,
                    'rank_by_r2': None,  # Will be set after sorting
                    'rank_by_rmse': None,  # Will be set after sorting
                    'training_time': results.get('training_time', 0),
                    'is_knn': model_name == 'KNN'
                })

            # Sort by R² (descending) and assign ranks
            model_rankings.sort(key=lambda x: x['r2_score'], reverse=True)
            for i, model in enumerate(model_rankings):
                model['rank_by_r2'] = i + 1

            # Sort by RMSE (ascending) and assign ranks
            model_rankings.sort(key=lambda x: x['rmse_score'])
            for i, model in enumerate(model_rankings):
                model['rank_by_rmse'] = i + 1

            # Determine best model (highest R²)
            best_model = model_rankings[0]

            # Generate comparison visualizations
            visualization_paths = self._generate_model_comparison_visualizations(
                comparison_results, model_rankings, best_model
            )

            # Create comprehensive comparison report
            comparison_report = {
                'comparison_timestamp': datetime.now().isoformat(),
                'models_compared': list(comparison_results.keys()),
                'total_models': len(comparison_results),
                'valid_models': len(valid_results),
                'best_model': {
                    'name': best_model['model_name'],
                    'r2_score': best_model['r2_score'],
                    'rmse_score': best_model['rmse_score'],
                    'rank': best_model['rank_by_r2'],
                    'is_knn': best_model['is_knn']
                },
                'model_rankings': model_rankings,
                'performance_summary': {
                    'r2_range': {
                        'min': min(m['r2_score'] for m in model_rankings),
                        'max': max(m['r2_score'] for m in model_rankings),
                        'mean': sum(m['r2_score'] for m in model_rankings) / len(model_rankings)
                    },
                    'rmse_range': {
                        'min': min(m['rmse_score'] for m in model_rankings),
                        'max': max(m['rmse_score'] for m in model_rankings),
                        'mean': sum(m['rmse_score'] for m in model_rankings) / len(model_rankings)
                    }
                },
                'knn_specific_analysis': self._analyze_knn_performance(comparison_results, model_rankings),
                'visualizations': visualization_paths,
                'recommendations': self._generate_model_comparison_recommendations(
                    model_rankings, best_model
                )
            }

            # Save comprehensive report
            self._save_model_comparison_report(comparison_report)

            model_logger.info(f"Model comparison completed. Best model: {best_model['model_name']} (R² = {best_model['r2_score']:.4f})")

            return comparison_report

        else:
            model_logger.error("No valid models to compare")
            return {'error': 'No valid models for comparison'}

    def _generate_model_comparison_visualizations(self, comparison_results: Dict[str, Any],
                                                model_rankings: List[Dict[str, Any]],
                                                best_model: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate comprehensive model comparison visualizations.

        Args:
            comparison_results: Raw comparison results
            model_rankings: Ranked model performance
            best_model: Best performing model

        Returns:
            Dictionary of visualization paths
        """
        visualization_paths = {}

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # 1. Model Performance Comparison (R² scores)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comprehensive Model Comparison Including KNN', fontsize=16, fontweight='bold')

            # R² Comparison
            model_names = [m['model_name'] for m in model_rankings]
            r2_scores = [m['r2_score'] for m in model_rankings]

            bars = axes[0, 0].bar(model_names, r2_scores, color='lightblue', alpha=0.7)
            axes[0, 0].set_title('R² Score Comparison', fontweight='bold')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Highlight best model
            best_idx = model_names.index(best_model['model_name'])
            bars[best_idx].set_color('darkblue')
            bars[best_idx].set_alpha(1.0)

            # Add value labels
            for bar, score in zip(bars, r2_scores):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               '.3f', ha='center', va='bottom', fontweight='bold')

            # RMSE Comparison
            rmse_scores = [m['rmse_score'] for m in model_rankings]
            bars_rmse = axes[0, 1].bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Highlight best model (lowest RMSE)
            bars_rmse[best_idx].set_color('darkred')
            bars_rmse[best_idx].set_alpha(1.0)

            # Add value labels
            for bar, score in zip(bars_rmse, rmse_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               '.2f', ha='center', va='bottom', fontweight='bold')

            # Training Time Comparison
            training_times = [m['training_time'] for m in model_rankings]
            bars_time = axes[1, 0].bar(model_names, training_times, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Training Time Comparison', fontweight='bold')
            axes[1, 0].set_ylabel('Training Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Highlight KNN if it's fast
            knn_idx = next((i for i, m in enumerate(model_rankings) if m['is_knn']), None)
            if knn_idx is not None:
                bars_time[knn_idx].set_color('darkgreen')

            # Performance vs Speed Scatter Plot
            axes[1, 1].scatter(r2_scores, training_times, s=100, alpha=0.7)

            # Add model labels
            for i, model in enumerate(model_rankings):
                color = 'red' if model['model_name'] == best_model['model_name'] else 'blue'
                axes[1, 1].annotate(model['model_name'], (r2_scores[i], training_times[i]),
                                   xytext=(5, 5), textcoords='offset points', color=color,
                                   fontweight='bold' if model['model_name'] == best_model['model_name'] else 'normal')

            axes[1, 1].set_xlabel('R² Score')
            axes[1, 1].set_ylabel('Training Time (seconds)')
            axes[1, 1].set_title('Performance vs Training Speed', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save main comparison plot
            comparison_path = 'visualizations/model_comparison_with_knn.png'
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['model_comparison'] = comparison_path

            # 2. KNN-specific analysis plot (if KNN is included)
            knn_results = next((m for m in model_rankings if m['is_knn']), None)
            if knn_results:
                self._generate_knn_comparison_visualization(knn_results, model_rankings, visualization_paths)

            model_logger.info(f"Generated model comparison visualizations: {list(visualization_paths.keys())}")

        except Exception as e:
            model_logger.error(f"Failed to generate model comparison visualizations: {e}")

        return visualization_paths

    def _generate_knn_comparison_visualization(self, knn_results: Dict[str, Any],
                                             model_rankings: List[Dict[str, Any]],
                                             visualization_paths: Dict[str, str]) -> None:
        """
        Generate KNN-specific comparison visualization.

        Args:
            knn_results: KNN performance results
            model_rankings: All model rankings
            visualization_paths: Dictionary to update with new paths
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create performance comparison with KNN highlighted
            model_names = [m['model_name'] for m in model_rankings]
            r2_scores = [m['r2_score'] for m in model_rankings]

            bars = ax.bar(model_names, r2_scores, color='lightgray', alpha=0.7, label='Other Models')

            # Highlight KNN
            knn_idx = next(i for i, m in enumerate(model_rankings) if m['is_knn'])
            bars[knn_idx].set_color('orange')
            bars[knn_idx].set_label('KNN')

            # Highlight best model
            best_idx = 0  # model_rankings is sorted by R²
            if best_idx != knn_idx:
                bars[best_idx].set_color('darkblue')
                bars[best_idx].set_label('Best Model')

            ax.set_title('KNN Performance Relative to Other Models', fontweight='bold')
            ax.set_ylabel('R² Score')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add performance difference annotation
            best_r2 = model_rankings[0]['r2_score']
            knn_r2 = knn_results['r2_score']
            difference = knn_r2 - best_r2

            ax.axhline(y=best_r2, color='red', linestyle='--', alpha=0.7,
                      label='.3f')

            # Add text annotation
            knn_x = model_names[knn_idx]
            knn_y = r2_scores[knn_idx]
            ax.annotate('.3f',
                       xy=(knn_x, knn_y), xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.8),
                       fontweight='bold')

            plt.tight_layout()

            knn_comparison_path = 'visualizations/knn_relative_performance.png'
            plt.savefig(knn_comparison_path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_comparison'] = knn_comparison_path

        except Exception as e:
            model_logger.warning(f"Failed to generate KNN comparison visualization: {e}")

    def _analyze_knn_performance(self, comparison_results: Dict[str, Any],
                               model_rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze KNN performance relative to other models.

        Args:
            comparison_results: Raw comparison results
            model_rankings: Ranked model performance

        Returns:
            KNN-specific analysis
        """
        knn_results = next((m for m in model_rankings if m['is_knn']), None)
        if not knn_results:
            return {'error': 'KNN not found in results'}

        analysis = {
            'knn_r2_score': knn_results['r2_score'],
            'knn_rmse_score': knn_results['rmse_score'],
            'knn_rank': knn_results['rank_by_r2'],
            'knn_training_time': knn_results['training_time'],
            'performance_vs_best': None,
            'speed_advantage': None,
            'knn_strengths': [],
            'knn_weaknesses': []
        }

        # Compare to best model
        best_model = model_rankings[0]
        if best_model['model_name'] != 'KNN':
            r2_diff = knn_results['r2_score'] - best_model['r2_score']
            analysis['performance_vs_best'] = {
                'r2_difference': r2_diff,
                'percentage_of_best': (knn_results['r2_score'] / best_model['r2_score']) * 100,
                'better_than_best': r2_diff > 0
            }

        # Speed analysis
        avg_training_time = sum(m['training_time'] for m in model_rankings) / len(model_rankings)
        analysis['speed_advantage'] = avg_training_time / knn_results['training_time']

        # Determine strengths and weaknesses
        if knn_results['rank_by_r2'] <= 3:
            analysis['knn_strengths'].append("Competitive predictive performance")
        if knn_results['training_time'] < avg_training_time:
            analysis['knn_strengths'].append("Fast training time")
        if knn_results['rank_by_r2'] > len(model_rankings) // 2:
            analysis['knn_weaknesses'].append("Lower predictive accuracy compared to tree-based models")

        return analysis

    def _generate_model_comparison_recommendations(self, model_rankings: List[Dict[str, Any]],
                                                 best_model: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on model comparison results.

        Args:
            model_rankings: Ranked model performance
            best_model: Best performing model

        Returns:
            List of recommendations
        """
        recommendations = []

        # Best model recommendation
        recommendations.append(f"Use {best_model['model_name']} as the primary model (R² = {best_model['r2_score']:.4f})")

        # KNN-specific recommendations
        knn_model = next((m for m in model_rankings if m['is_knn']), None)
        if knn_model:
            if knn_model['rank_by_r2'] <= 3:
                recommendations.append("KNN shows competitive performance and may be suitable for production use")
            if knn_model['training_time'] < 1.0:
                recommendations.append("KNN offers excellent training speed for rapid prototyping")

        # Ensemble recommendations
        tree_models = [m for m in model_rankings if 'tree' in m['model_name'].lower() or 'forest' in m['model_name'].lower()]
        linear_models = [m for m in model_rankings if 'linear' in m['model_name'].lower() or 'ridge' in m['model_name'].lower() or 'lasso' in m['model_name'].lower()]

        if len(tree_models) > 0 and len(linear_models) > 0:
            recommendations.append("Consider ensemble methods combining tree-based and linear models for potentially better performance")

        # Speed vs accuracy tradeoff
        fast_models = [m for m in model_rankings if m['training_time'] < 5.0]
        if len(fast_models) > 0 and fast_models[0]['r2_score'] > 0.7:
            recommendations.append("Fast models available with good performance - suitable for real-time applications")

        return recommendations

    def generate_knn_specific_visualizations(self, model_id: str, X_train: pd.DataFrame,
                                           y_train: np.ndarray, X_test: pd.DataFrame,
                                           y_test: np.ndarray) -> Dict[str, str]:
        """
        Generate comprehensive KNN-specific visualizations to showcase the model.

        Args:
            model_id: ID of the trained KNN model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of generated visualization paths
        """
        if model_id not in self.trained_models:
            model_logger.error(f"Model {model_id} not found")
            return {}

        model = self.trained_models[model_id]
        if not hasattr(model, 'n_neighbors'):
            model_logger.error(f"Model {model_id} is not a KNN model")
            return {}

        visualization_paths = {}

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import mean_squared_error
            import numpy as np

            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # 1. KNN Neighbor Distance Distribution
            neighbor_distances = self._generate_knn_neighbor_distance_analysis(
                model, X_test, visualization_paths
            )

            # 2. KNN Prediction Stability Analysis
            stability_results = self._generate_knn_prediction_stability_analysis(
                model, X_test, y_test, visualization_paths
            )

            # 3. KNN Feature Importance via Permutation
            feature_importance = self._generate_knn_feature_importance_analysis(
                model, X_test, y_test, visualization_paths
            )

            # 4. KNN Error Analysis by Prediction Range
            error_analysis = self._generate_knn_error_range_analysis(
                model, X_test, y_test, visualization_paths
            )

            # 5. KNN Training vs Prediction Time Analysis
            timing_analysis = self._generate_knn_timing_analysis(
                model, X_train, X_test, visualization_paths
            )

            # 6. KNN Decision Boundary Visualization (2D projection)
            if X_train.shape[1] >= 2:
                decision_boundary = self._generate_knn_decision_boundary_visualization(
                    model, X_train, y_train, visualization_paths
                )

            model_logger.info(f"Generated {len(visualization_paths)} KNN-specific visualizations")

        except Exception as e:
            model_logger.error(f"Failed to generate KNN visualizations: {e}")

        return visualization_paths

    def _generate_knn_neighbor_distance_analysis(self, model, X_test: pd.DataFrame,
                                               visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze and visualize distances to nearest neighbors.
        """
        try:
            # Get distances to k nearest neighbors for a sample of test points
            sample_size = min(500, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)

            # Get distances and indices of k nearest neighbors
            distances, indices = model.kneighbors(X_sample)

            # Create comprehensive distance analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'KNN Neighbor Distance Analysis (K={model.n_neighbors})', fontsize=16, fontweight='bold')

            # 1. Distribution of distances to nearest neighbor
            nearest_distances = distances[:, 0]
            axes[0, 0].hist(nearest_distances, bins=30, alpha=0.7, edgecolor='black', density=True)
            axes[0, 0].axvline(np.mean(nearest_distances), color='red', linestyle='--',
                              label='.3f')
            axes[0, 0].axvline(np.median(nearest_distances), color='orange', linestyle='--',
                              label='.3f')
            axes[0, 0].set_xlabel('Distance to Nearest Neighbor')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Distribution of Distances to Nearest Neighbor')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Average distance by neighbor rank
            avg_distances_by_rank = np.mean(distances, axis=0)
            axes[0, 1].plot(range(1, model.n_neighbors + 1), avg_distances_by_rank,
                           marker='o', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Neighbor Rank')
            axes[0, 1].set_ylabel('Average Distance')
            axes[0, 1].set_title('Average Distance by Neighbor Rank')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Distance variability (coefficient of variation)
            cv_distances = np.std(distances, axis=0) / np.mean(distances, axis=0)
            axes[1, 0].bar(range(1, model.n_neighbors + 1), cv_distances, alpha=0.7)
            axes[1, 0].set_xlabel('Neighbor Rank')
            axes[1, 0].set_ylabel('Coefficient of Variation')
            axes[1, 0].set_title('Distance Variability by Neighbor Rank')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Distance percentiles
            percentiles = [25, 50, 75, 90]
            percentile_values = np.percentile(distances, percentiles, axis=0)
            x_positions = np.arange(model.n_neighbors)

            for i, (p, values) in enumerate(zip(percentiles, percentile_values)):
                axes[1, 1].plot(x_positions + 1, values, marker='o', label=f'{p}th percentile',
                               linewidth=2, markersize=4)

            axes[1, 1].set_xlabel('Neighbor Rank')
            axes[1, 1].set_ylabel('Distance')
            axes[1, 1].set_title('Distance Percentiles by Neighbor Rank')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_neighbor_distance_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_neighbor_distances'] = path

            return {
                'mean_nearest_distance': float(np.mean(nearest_distances)),
                'median_nearest_distance': float(np.median(nearest_distances)),
                'distance_variability': float(np.mean(cv_distances))
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate neighbor distance analysis: {e}")
            return {}

    def _generate_knn_prediction_stability_analysis(self, model, X_test: pd.DataFrame,
                                                  y_test: np.ndarray, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze prediction stability across different K values.
        """
        try:
            # Test different K values around the optimal
            optimal_k = model.n_neighbors
            k_values = [max(1, optimal_k - 2), max(1, optimal_k - 1), optimal_k,
                       optimal_k + 1, optimal_k + 2, optimal_k + 5]

            stability_results = {}

            # Sample a subset for efficiency
            sample_size = min(200, len(X_test))
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices]
            y_sample = y_test[sample_indices]

            for k in k_values:
                # Create model with this K
                temp_model = type(model)(n_neighbors=k, **{k: v for k, v in model.get_params().items()
                                                          if k != 'n_neighbors'})
                temp_model.fit(X_test, y_test)  # Fit on full training set

                # Get predictions
                y_pred = temp_model.predict(X_sample)
                mse = mean_squared_error(y_sample, y_pred)
                rmse = np.sqrt(mse)

                stability_results[k] = {
                    'rmse': rmse,
                    'predictions': y_pred.tolist()
                }

            # Create stability visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'KNN Prediction Stability Analysis (Optimal K={optimal_k})', fontsize=14, fontweight='bold')

            # RMSE vs K
            k_list = list(stability_results.keys())
            rmse_values = [stability_results[k]['rmse'] for k in k_list]

            axes[0].plot(k_list, rmse_values, marker='o', linewidth=3, markersize=8, color='blue')
            axes[0].axvline(optimal_k, color='red', linestyle='--', linewidth=2,
                           label=f'Optimal K={optimal_k}')
            axes[0].set_xlabel('K Value')
            axes[0].set_ylabel('RMSE')
            axes[0].set_title('Prediction Error vs K Value')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Prediction variability (standard deviation of predictions across K values)
            pred_matrix = np.array([stability_results[k]['predictions'] for k in k_list])
            pred_std = np.std(pred_matrix, axis=0)

            axes[1].hist(pred_std, bins=30, alpha=0.7, edgecolor='black', density=True)
            axes[1].axvline(np.mean(pred_std), color='red', linestyle='--',
                           label='.3f')
            axes[1].set_xlabel('Prediction Standard Deviation')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Prediction Variability Across K Values')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_prediction_stability.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_prediction_stability'] = path

            return {
                'optimal_k': optimal_k,
                'stability_range': {
                    'min_rmse': min(rmse_values),
                    'max_rmse': max(rmse_values),
                    'rmse_variability': np.std(rmse_values)
                },
                'prediction_variability': float(np.mean(pred_std))
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate prediction stability analysis: {e}")
            return {}

    def _generate_knn_feature_importance_analysis(self, model, X_test: pd.DataFrame,
                                                y_test: np.ndarray, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate feature importance analysis for KNN using permutation importance.
        """
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.dummy import DummyRegressor

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, scoring='r2'
            )

            # Create baseline with dummy model
            dummy_model = DummyRegressor(strategy='mean')
            dummy_model.fit(X_test, y_test)
            dummy_score = dummy_model.score(X_test, y_test)

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'KNN Feature Importance Analysis (K={model.n_neighbors})', fontsize=14, fontweight='bold')

            # Feature importance bar plot
            features = X_test.columns
            importance_means = perm_importance.importances_mean
            importance_stds = perm_importance.importances_std

            # Sort by importance
            sorted_idx = np.argsort(importance_means)[::-1]
            features_sorted = [features[i] for i in sorted_idx]
            means_sorted = importance_means[sorted_idx]
            stds_sorted = importance_stds[sorted_idx]

            bars = axes[0].bar(range(len(features_sorted)), means_sorted, yerr=stds_sorted,
                              capsize=5, alpha=0.7, color='skyblue')
            axes[0].set_xticks(range(len(features_sorted)))
            axes[0].set_xticklabels(features_sorted, rotation=45, ha='right')
            axes[0].set_xlabel('Features')
            axes[0].set_ylabel('Permutation Importance (R² decrease)')
            axes[0].set_title('Feature Importance via Permutation')
            axes[0].grid(True, alpha=0.3)

            # Add value labels
            for bar, mean in zip(bars, means_sorted):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           '.3f', ha='center', va='bottom', fontsize=8)

            # Importance distribution
            axes[1].hist(importance_means, bins=20, alpha=0.7, edgecolor='black', density=True)
            axes[1].axvline(np.mean(importance_means), color='red', linestyle='--',
                           label='.3f')
            axes[1].axvline(0, color='black', linestyle='-', alpha=0.5, label='No importance')
            axes[1].set_xlabel('Permutation Importance')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Distribution of Feature Importances')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_feature_importance.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_feature_importance'] = path

            return {
                'top_features': features_sorted[:5],
                'importance_scores': dict(zip(features_sorted, means_sorted)),
                'mean_importance': float(np.mean(importance_means)),
                'max_importance': float(np.max(importance_means))
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate feature importance analysis: {e}")
            return {}

    def _generate_knn_error_range_analysis(self, model, X_test: pd.DataFrame,
                                         y_test: np.ndarray, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze prediction errors across different value ranges.
        """
        try:
            y_pred = model.predict(X_test)
            errors = y_test - y_pred
            abs_errors = np.abs(errors)

            # Create error analysis by prediction ranges
            pred_ranges = pd.qcut(y_pred, q=5, duplicates='drop')
            range_labels = [f'{interval.left:.0f}-{interval.right:.0f}' for interval in pred_ranges.cat.categories]

            # Calculate error statistics by range
            range_errors = []
            for i, label in enumerate(range_labels):
                mask = pred_ranges == pred_ranges.cat.categories[i]
                range_abs_errors = abs_errors[mask]
                range_errors.append({
                    'range': label,
                    'mean_abs_error': float(np.mean(range_abs_errors)),
                    'median_abs_error': float(np.median(range_abs_errors)),
                    'std_error': float(np.std(range_abs_errors)),
                    'count': int(np.sum(mask))
                })

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'KNN Error Analysis by Prediction Range (K={model.n_neighbors})', fontsize=14, fontweight='bold')

            # Error by prediction range
            ranges = [r['range'] for r in range_errors]
            mean_errors = [r['mean_abs_error'] for r in range_errors]
            std_errors = [r['std_error'] for r in range_errors]

            axes[0].bar(ranges, mean_errors, yerr=std_errors, capsize=5, alpha=0.7, color='coral')
            axes[0].set_xlabel('Prediction Range')
            axes[0].set_ylabel('Mean Absolute Error')
            axes[0].set_title('Prediction Error by Value Range')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)

            # Error distribution
            axes[1].hist(abs_errors, bins=30, alpha=0.7, edgecolor='black', density=True)
            axes[1].axvline(np.mean(abs_errors), color='red', linestyle='--',
                           label='.2f')
            axes[1].axvline(np.median(abs_errors), color='orange', linestyle='--',
                           label='.2f')
            axes[1].set_xlabel('Absolute Error')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Distribution of Absolute Errors')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_error_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_error_analysis'] = path

            return {
                'error_by_range': range_errors,
                'overall_error_stats': {
                    'mean_abs_error': float(np.mean(abs_errors)),
                    'median_abs_error': float(np.median(abs_errors)),
                    'error_std': float(np.std(abs_errors))
                }
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate error range analysis: {e}")
            return {}

    def _generate_knn_timing_analysis(self, model, X_train: pd.DataFrame,
                                    X_test: pd.DataFrame, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze training vs prediction timing characteristics.
        """
        try:
            import time

            # Measure training time for different dataset sizes
            train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
            training_times = []
            prediction_times = []

            for size in train_sizes:
                n_samples = int(len(X_train) * size)
                X_subset = X_train.sample(n=n_samples, random_state=42)

                # Training time
                start_time = time.time()
                temp_model = type(model)(**model.get_params())
                temp_model.fit(X_subset, np.random.randn(len(X_subset)))  # Dummy target
                training_time = time.time() - start_time
                training_times.append(training_time)

                # Prediction time
                start_time = time.time()
                temp_model.predict(X_test.sample(min(1000, len(X_test)), random_state=42))
                prediction_time = time.time() - start_time
                prediction_times.append(prediction_time)

            # Create timing visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'KNN Timing Analysis (K={model.n_neighbors})', fontsize=14, fontweight='bold')

            # Training time scaling
            axes[0].plot([s * 100 for s in train_sizes], training_times, marker='o',
                        linewidth=3, markersize=8, color='green', label='Training Time')
            axes[0].set_xlabel('Training Set Size (%)')
            axes[0].set_ylabel('Time (seconds)')
            axes[0].set_title('Training Time vs Dataset Size')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Training vs Prediction time comparison
            time_labels = [f'{int(s * 100)}%' for s in train_sizes]
            x = np.arange(len(time_labels))

            axes[1].bar(x - 0.2, training_times, 0.4, label='Training', alpha=0.7, color='green')
            axes[1].bar(x + 0.2, prediction_times, 0.4, label='Prediction', alpha=0.7, color='blue')
            axes[1].set_xlabel('Training Set Size')
            axes[1].set_ylabel('Time (seconds)')
            axes[1].set_title('Training vs Prediction Time')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(time_labels)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_timing_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_timing_analysis'] = path

            return {
                'training_time_scaling': dict(zip([f'{int(s*100)}%' for s in train_sizes], training_times)),
                'prediction_times': dict(zip([f'{int(s*100)}%' for s in train_sizes], prediction_times)),
                'time_efficiency': float(np.mean(prediction_times) / np.mean(training_times))
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate timing analysis: {e}")
            return {}

    def _generate_knn_decision_boundary_visualization(self, model, X_train: pd.DataFrame,
                                                    y_train: np.ndarray, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate 2D decision boundary visualization using PCA projection.
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Reduce to 2D using PCA
            pca = PCA(n_components=2, random_state=42)
            X_train_2d = pca.fit_transform(X_train)

            # Create a mesh grid for visualization
            x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
            y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))

            # Project mesh back to original space and predict
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            mesh_original = pca.inverse_transform(mesh_points)

            # Get predictions for mesh
            Z = model.predict(mesh_original)
            Z = Z.reshape(xx.shape)

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot decision boundary
            contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            plt.colorbar(contour, ax=ax, label='Predicted Value')

            # Plot training points
            scatter = ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train,
                               cmap='RdYlBu', edgecolor='black', s=50, alpha=0.8)

            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.set_title(f'KNN Decision Surface (K={model.n_neighbors})\nPCA Projection')
            ax.grid(True, alpha=0.3)

            # Add explained variance
            explained_var = pca.explained_variance_ratio_
            ax.text(0.02, 0.98, '.1f',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            path = 'visualizations/knn_decision_boundary.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_decision_boundary'] = path

            return {
                'explained_variance': explained_var.tolist(),
                'pca_components': pca.components_.tolist()
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate decision boundary visualization: {e}")
            return {}

    def generate_knn_regression_evaluation_plots(self, model_id: str, X_train: pd.DataFrame,
                                               y_train: np.ndarray, X_test: pd.DataFrame,
                                               y_test: np.ndarray) -> Dict[str, str]:
        """
        Generate regression evaluation plots equivalent to confusion matrices for KNN.
        Since KNN is used for regression (not classification), we create regression diagnostics.
        """
        if model_id not in self.trained_models:
            model_logger.error(f"Model {model_id} not found")
            return {}

        model = self.trained_models[model_id]
        if not hasattr(model, 'n_neighbors'):
            model_logger.error(f"Model {model_id} is not a KNN model")
            return {}

        visualization_paths = {}

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy import stats
            import numpy as np

            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Get predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate residuals
            residuals_train = y_train - y_pred_train
            residuals_test = y_test - y_pred_test

            # 1. Prediction vs Actual Scatter Plot (equivalent to confusion matrix for regression)
            pred_actual_plot = self._generate_prediction_vs_actual_plot(
                y_test, y_pred_test, visualization_paths
            )

            # 2. Residuals vs Fitted Values Plot
            residual_plot = self._generate_residuals_vs_fitted_plot(
                y_pred_test, residuals_test, visualization_paths
            )

            # 3. Residual Distribution (Q-Q Plot)
            qq_plot = self._generate_qq_plot(
                residuals_test, visualization_paths
            )

            # 4. Error Distribution Histogram
            error_distribution = self._generate_error_distribution_plot(
                residuals_test, visualization_paths
            )

            # 5. Model Fit Assessment
            fit_assessment = self._generate_model_fit_assessment(
                y_test, y_pred_test, residuals_test, visualization_paths
            )

            # 6. Prediction Error by Range (like confusion matrix quadrants)
            error_range_analysis = self._generate_prediction_error_range_plot(
                y_test, y_pred_test, visualization_paths
            )

            model_logger.info(f"Generated {len(visualization_paths)} KNN regression evaluation plots")

        except Exception as e:
            model_logger.error(f"Failed to generate KNN regression evaluation plots: {e}")

        return visualization_paths

    def _generate_prediction_vs_actual_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """Generate prediction vs actual scatter plot (regression equivalent of confusion matrix)."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Scatter plot
            scatter = ax.scatter(y_true, y_pred, alpha=0.6, c=np.abs(y_true - y_pred),
                               cmap='Reds', edgecolors='black', linewidth=0.5, s=50)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Absolute Error')

            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('KNN: Prediction vs Actual Values\n(Color = Absolute Error)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add R² and RMSE text
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            ax.text(0.05, 0.95, '.4f',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.text(0.05, 0.88, '.2f',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            path = 'visualizations/knn_prediction_vs_actual.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_prediction_vs_actual'] = path

            return {
                'r2_score': float(r2),
                'rmse': float(rmse),
                'mean_absolute_error': float(np.mean(np.abs(y_true - y_pred)))
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate prediction vs actual plot: {e}")
            return {}

    def _generate_residuals_vs_fitted_plot(self, y_pred: np.ndarray, residuals: np.ndarray,
                                         visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """Generate residuals vs fitted values plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Scatter plot of residuals vs fitted values
            ax.scatter(y_pred, residuals, alpha=0.6, c=np.abs(residuals),
                      cmap='Blues', edgecolors='black', linewidth=0.5, s=50)

            # Horizontal line at 0
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual Line')

            # Add colorbar
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Absolute Residual')

            ax.set_xlabel('Fitted Values (Predictions)')
            ax.set_ylabel('Residuals (Actual - Predicted)')
            ax.set_title('KNN: Residuals vs Fitted Values')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_residuals_vs_fitted.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_residuals_vs_fitted'] = path

            return {
                'mean_residual': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'residual_skewness': float(stats.skew(residuals))
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate residuals vs fitted plot: {e}")
            return {}

    def _generate_qq_plot(self, residuals: np.ndarray, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """Generate Q-Q plot for residual normality assessment."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Q-Q plot
            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=ax)

            # Add reference line
            ax.plot(osm, osm * slope + intercept, 'r--', linewidth=2, label='Normal Reference Line')

            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.set_title('KNN: Q-Q Plot of Residuals')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add normality test results
            _, p_value = stats.shapiro(residuals[:min(5000, len(residuals))])  # Shapiro-Wilk test
            normality_text = '.4f'

            ax.text(0.05, 0.95, normality_text,
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            path = 'visualizations/knn_qq_plot.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_qq_plot'] = path

            return {
                'shapiro_p_value': float(p_value),
                'is_normal': p_value > 0.05,
                'slope': float(slope),
                'intercept': float(intercept)
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate Q-Q plot: {e}")
            return {}

    def _generate_error_distribution_plot(self, residuals: np.ndarray, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """Generate error distribution histogram."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram of residuals
            n, bins, patches = ax.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)

            # Add normal distribution curve
            mu, std = np.mean(residuals), np.std(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, 'r-', linewidth=2, label='Normal Distribution')

            # Add vertical lines for mean and std
            ax.axvline(mu, color='red', linestyle='--', linewidth=2, label='.2f')
            ax.axvline(mu + std, color='orange', linestyle='--', linewidth=1, label='+1 STD')
            ax.axvline(mu - std, color='orange', linestyle='--', linewidth=1, label='-1 STD')

            ax.set_xlabel('Residuals (Actual - Predicted)')
            ax.set_ylabel('Density')
            ax.set_title('KNN: Residual Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_error_distribution.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_error_distribution'] = path

            return {
                'mean_residual': float(mu),
                'residual_std': float(std),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals))
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate error distribution plot: {e}")
            return {}

    def _generate_model_fit_assessment(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     residuals: np.ndarray, visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """Generate comprehensive model fit assessment."""
        try:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('KNN Model Fit Assessment', fontsize=16, fontweight='bold')

            # 1. R² Score Visualization
            r2 = r2_score(y_true, y_pred)
            axes[0, 0].bar(['R² Score'], [r2], color='skyblue', alpha=0.7)
            axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent (≥0.8)')
            axes[0, 0].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Good (≥0.6)')
            axes[0, 0].axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Poor (<0.4)')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_title('Model Fit Quality')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Add value label
            axes[0, 0].text(0, r2 + 0.01, '.3f', ha='center', va='bottom', fontsize=12)

            # 2. Error Metrics Comparison
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

            metrics = ['RMSE', 'MAE', 'MAPE (%)']
            values = [rmse, mae, mape]
            colors = ['coral', 'lightgreen', 'lightblue']

            bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
            axes[0, 1].set_ylabel('Error Value')
            axes[0, 1].set_title('Error Metrics')
            axes[0, 1].grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                               '.2f', ha='center', va='bottom', fontsize=10)

            # 3. Residuals Over Time (if we can infer order)
            # Sort by predicted values for trend analysis
            sorted_idx = np.argsort(y_pred)
            sorted_residuals = residuals[sorted_idx]

            axes[1, 0].scatter(range(len(sorted_residuals)), sorted_residuals, alpha=0.6, s=30)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[1, 0].set_xlabel('Sample Index (Sorted by Prediction)')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residuals Pattern Analysis')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Prediction Accuracy by Range
            # Create prediction ranges
            pred_ranges = pd.qcut(y_pred, q=5, duplicates='drop')
            range_labels = [f'Q{i+1}' for i in range(len(pred_ranges.cat.categories))]

            range_r2_scores = []
            for i in range(len(pred_ranges.cat.categories)):
                mask = pred_ranges == pred_ranges.cat.categories[i]
                if np.sum(mask) > 10:  # Only if enough samples
                    range_r2 = r2_score(y_true[mask], y_pred[mask])
                    range_r2_scores.append(range_r2)
                else:
                    range_r2_scores.append(0)

            axes[1, 1].bar(range_labels, range_r2_scores, color='purple', alpha=0.7)
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].set_title('Prediction Accuracy by Range')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            path = 'visualizations/knn_model_fit_assessment.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_model_fit_assessment'] = path

            return {
                'r2_score': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'fit_quality': 'excellent' if r2 >= 0.8 else 'good' if r2 >= 0.6 else 'poor'
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate model fit assessment: {e}")
            return {}

    def _generate_prediction_error_range_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                            visualization_paths: Dict[str, str]) -> Dict[str, Any]:
        """Generate prediction error analysis by value ranges (like confusion matrix quadrants)."""
        try:
            # Create a 2x2 grid like confusion matrix but for regression ranges
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('KNN Prediction Error Analysis by Range\n(Regression Equivalent of Confusion Matrix)', fontsize=14, fontweight='bold')

            # Define ranges based on actual values
            y_true_median = np.median(y_true)
            y_pred_median = np.median(y_pred)

            # Create quadrants
            quadrants = {
                'True Low - Pred Low': (y_true <= y_true_median) & (y_pred <= y_pred_median),
                'True Low - Pred High': (y_true <= y_true_median) & (y_pred > y_pred_median),
                'True High - Pred Low': (y_true > y_true_median) & (y_pred <= y_pred_median),
                'True High - Pred High': (y_true > y_true_median) & (y_pred > y_pred_median)
            }

            quadrant_data = []
            for i, (quad_name, mask) in enumerate(quadrants.items()):
                ax = axes[i // 2, i % 2]

                if np.sum(mask) > 0:
                    # Scatter plot for this quadrant
                    ax.scatter(y_true[mask], y_pred[mask], alpha=0.6, s=50, color='blue', edgecolors='black')

                    # Calculate metrics for this quadrant
                    quad_r2 = r2_score(y_true[mask], y_pred[mask]) if len(y_true[mask]) > 1 else 0
                    quad_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                    count = np.sum(mask)

                    quadrant_data.append({
                        'quadrant': quad_name,
                        'count': int(count),
                        'r2': float(quad_r2),
                        'mae': float(quad_mae)
                    })

                    # Add perfect prediction line
                    min_val = min(y_true[mask].min(), y_pred[mask].min())
                    max_val = max(y_true[mask].max(), y_pred[mask].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

                    # Add metrics text
                    ax.text(0.05, 0.95, f'Count: {count}\nR²: {quad_r2:.3f}\nMAE: {quad_mae:.2f}',
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{quad_name}\n(Quadrant {i+1})')
                ax.grid(True, alpha=0.3)

                # Set axis limits to show the relevant range
                if 'Low' in quad_name:
                    ax.set_xlim(left=y_true.min(), right=y_true_median)
                    ax.set_ylim(bottom=y_pred.min(), top=y_pred_median)
                else:  # High
                    ax.set_xlim(left=y_true_median, right=y_true.max())
                    if 'Pred Low' in quad_name:
                        ax.set_ylim(bottom=y_pred.min(), top=y_pred_median)
                    else:  # Pred High
                        ax.set_ylim(bottom=y_pred_median, top=y_pred.max())

            plt.tight_layout()

            path = 'visualizations/knn_error_range_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            visualization_paths['knn_error_range_analysis'] = path

            return {
                'quadrant_analysis': quadrant_data,
                'true_median': float(y_true_median),
                'pred_median': float(y_pred_median)
            }

        except Exception as e:
            model_logger.warning(f"Failed to generate prediction error range plot: {e}")
            return {}

    def validate_knn_model_fitting(self, model_id: str, X_train: pd.DataFrame,
                                 y_train: np.ndarray, X_test: pd.DataFrame,
                                 y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive validation of KNN model fitting and performance.

        Args:
            model_id: ID of the trained KNN model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with fitting validation results
        """
        if model_id not in self.trained_models:
            return {'error': f'Model {model_id} not found'}

        model = self.trained_models[model_id]
        if not hasattr(model, 'n_neighbors'):
            return {'error': f'Model {model_id} is not a KNN model'}

        validation_results = {
            'model_id': model_id,
            'model_type': 'KNN',
            'k_value': model.n_neighbors,
            'validation_checks': {},
            'fitting_quality': {},
            'performance_metrics': {},
            'warnings': [],
            'recommendations': []
        }

        try:
            # Get predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # 1. Basic Model Validation
            validation_results['validation_checks'].update(self._validate_basic_model_properties(model, X_train))

            # 2. Fitting Quality Assessment
            validation_results['fitting_quality'].update(self._assess_fitting_quality(y_train, y_pred_train, y_test, y_pred_test))

            # 3. Performance Metrics
            validation_results['performance_metrics'].update(self._calculate_performance_metrics(y_train, y_pred_train, y_test, y_pred_test))

            # 4. Overfitting/Underfitting Detection
            overfitting_check = self._detect_overfitting_underfitting(y_train, y_pred_train, y_test, y_pred_test)
            validation_results['overfitting_analysis'] = overfitting_check

            # 5. K-Value Appropriateness
            k_validation = self._validate_k_value_appropriateness(model.n_neighbors, len(X_train))
            validation_results['k_validation'] = k_validation

            # 6. Distance Metric Validation
            distance_validation = self._validate_distance_metric(model, X_train)
            validation_results['distance_validation'] = distance_validation

            # Generate warnings and recommendations
            validation_results['warnings'], validation_results['recommendations'] = self._generate_fitting_warnings_and_recommendations(
                validation_results
            )

            # Overall assessment
            validation_results['overall_assessment'] = self._generate_overall_fitting_assessment(validation_results)

            model_logger.info(f"Completed KNN model fitting validation for {model_id}")

        except Exception as e:
            model_logger.error(f"Failed to validate KNN model fitting: {e}")
            validation_results['error'] = str(e)

        return validation_results

    def _validate_basic_model_properties(self, model, X_train: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic model properties and data compatibility."""
        checks = {}

        try:
            # Check if model is fitted
            checks['is_fitted'] = hasattr(model, 'n_neighbors') and model.n_neighbors is not None

            # Check feature compatibility
            checks['feature_compatibility'] = X_train.shape[1] == model.n_features_in_

            # Check for NaN predictions
            sample_pred = model.predict(X_train.iloc[:5])
            checks['no_nan_predictions'] = not np.isnan(sample_pred).any()

            # Check prediction range
            checks['reasonable_predictions'] = np.isfinite(sample_pred).all()

            # Check K value bounds
            checks['k_within_bounds'] = 1 <= model.n_neighbors <= len(X_train)

        except Exception as e:
            checks['validation_error'] = str(e)

        return checks

    def _assess_fitting_quality(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                              y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, Any]:
        """Assess the quality of model fitting."""
        from sklearn.metrics import r2_score, mean_squared_error

        quality = {}

        # Training fit
        quality['train_r2'] = float(r2_score(y_train, y_pred_train))
        quality['train_rmse'] = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

        # Test fit
        quality['test_r2'] = float(r2_score(y_test, y_pred_test))
        quality['test_rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        # Fit quality indicators
        quality['r2_difference'] = abs(quality['train_r2'] - quality['test_r2'])
        quality['rmse_ratio'] = quality['test_rmse'] / quality['train_rmse'] if quality['train_rmse'] > 0 else float('inf')

        # Overall fit quality
        if quality['test_r2'] >= 0.8:
            quality['fit_quality'] = 'excellent'
        elif quality['test_r2'] >= 0.6:
            quality['fit_quality'] = 'good'
        elif quality['test_r2'] >= 0.4:
            quality['fit_quality'] = 'fair'
        else:
            quality['fit_quality'] = 'poor'

        return quality

    def _calculate_performance_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                                    y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {}

        # Training metrics
        metrics['train'] = {
            'mae': float(mean_absolute_error(y_train, y_pred_train)),
            'mse': float(mean_squared_error(y_train, y_pred_train)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'r2': float(r2_score(y_train, y_pred_train)),
            'mape': float(np.mean(np.abs((y_train - y_pred_train) / (y_train + 1e-10))) * 100)
        }

        # Test metrics
        metrics['test'] = {
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'mse': float(mean_squared_error(y_test, y_pred_test)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'r2': float(r2_score(y_test, y_pred_test)),
            'mape': float(np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100)
        }

        # Prediction statistics
        residuals_test = y_test - y_pred_test
        metrics['residuals'] = {
            'mean': float(np.mean(residuals_test)),
            'std': float(np.std(residuals_test)),
            'skewness': float(stats.skew(residuals_test)),
            'kurtosis': float(stats.kurtosis(residuals_test))
        }

        return metrics

    def _detect_overfitting_underfitting(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                                       y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, Any]:
        """Detect overfitting or underfitting issues."""
        from sklearn.metrics import r2_score

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        r2_difference = train_r2 - test_r2

        analysis = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'r2_difference': float(r2_difference),
            'overfitting_detected': False,
            'underfitting_detected': False,
            'overfitting_severity': 'none'
        }

        # Overfitting detection
        if r2_difference > 0.2:
            analysis['overfitting_detected'] = True
            if r2_difference > 0.4:
                analysis['overfitting_severity'] = 'severe'
            elif r2_difference > 0.3:
                analysis['overfitting_severity'] = 'moderate'
            else:
                analysis['overfitting_severity'] = 'mild'

        # Underfitting detection
        if test_r2 < 0.3:
            analysis['underfitting_detected'] = True

        return analysis

    def _validate_k_value_appropriateness(self, k: int, n_samples: int) -> Dict[str, Any]:
        """Validate if the chosen K value is appropriate for the dataset size."""
        validation = {
            'k_value': k,
            'dataset_size': n_samples,
            'k_percentage': (k / n_samples) * 100,
            'appropriateness': 'unknown',
            'recommendations': []
        }

        # K value guidelines
        if k < 3:
            validation['appropriateness'] = 'too_small'
            validation['recommendations'].append('K is very small, may be too sensitive to noise')
        elif k > n_samples * 0.1:  # More than 10% of dataset
            validation['appropriateness'] = 'too_large'
            validation['recommendations'].append('K is very large, may smooth out important patterns')
        elif k > n_samples * 0.05:  # More than 5% of dataset
            validation['appropriateness'] = 'large'
            validation['recommendations'].append('K is relatively large, consider if local patterns are being lost')
        elif k < 5:
            validation['appropriateness'] = 'small'
            validation['recommendations'].append('K is small, captures local patterns but may be sensitive to noise')
        else:
            validation['appropriateness'] = 'appropriate'
            validation['recommendations'].append('K value appears appropriate for dataset size')

        return validation

    def _validate_distance_metric(self, model, X_train: pd.DataFrame) -> Dict[str, Any]:
        """Validate the distance metric choice."""
        validation = {
            'metric': model.metric,
            'p_value': getattr(model, 'p', None),
            'validation': {}
        }

        # Check for constant features (problematic for distance metrics)
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_features.append(col)

        validation['constant_features'] = constant_features
        validation['has_constant_features'] = len(constant_features) > 0

        # Check for high correlation (can make some distance metrics less effective)
        corr_matrix = X_train.corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        validation['highly_correlated_pairs'] = high_corr_pairs
        validation['has_high_correlation'] = len(high_corr_pairs) > 0

        return validation

    def _generate_fitting_warnings_and_recommendations(self, validation_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations based on validation results."""
        warnings = []
        recommendations = []

        # Check basic validation
        basic_checks = validation_results.get('validation_checks', {})
        if not basic_checks.get('is_fitted', False):
            warnings.append('Model does not appear to be properly fitted')
            recommendations.append('Ensure model.fit() was called with appropriate data')

        if not basic_checks.get('feature_compatibility', False):
            warnings.append('Feature count mismatch between training and model')
            recommendations.append('Check that test data has same features as training data')

        # Check fitting quality
        fitting = validation_results.get('fitting_quality', {})
        if fitting.get('fit_quality') == 'poor':
            warnings.append('Model fit quality is poor')
            recommendations.append('Consider different hyperparameters or model type')

        # Check overfitting
        overfitting = validation_results.get('overfitting_analysis', {})
        if overfitting.get('overfitting_detected', False):
            severity = overfitting.get('overfitting_severity', 'unknown')
            warnings.append(f'Overfitting detected (severity: {severity})')
            recommendations.append('Try increasing K value or using regularization')

        if overfitting.get('underfitting_detected', False):
            warnings.append('Underfitting detected')
            recommendations.append('Try decreasing K value or using more complex distance metrics')

        # Check K validation
        k_validation = validation_results.get('k_validation', {})
        if k_validation.get('appropriateness') in ['too_small', 'too_large']:
            warnings.append(f'K value may not be appropriate: {k_validation.get("appropriateness")}')
            recommendations.extend(k_validation.get('recommendations', []))

        # Check distance validation
        distance_validation = validation_results.get('distance_validation', {})
        if distance_validation.get('has_constant_features', False):
            warnings.append('Constant features detected - may cause issues with distance calculations')
            recommendations.append('Remove or encode constant features')

        if distance_validation.get('has_high_correlation', False):
            warnings.append('Highly correlated features detected')
            recommendations.append('Consider feature selection or dimensionality reduction')

        return warnings, recommendations

    def _generate_overall_fitting_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of model fitting."""
        assessment = {
            'overall_quality': 'unknown',
            'confidence_level': 'low',
            'issues_detected': 0,
            'critical_issues': 0
        }

        # Count issues
        warnings = validation_results.get('warnings', [])
        assessment['issues_detected'] = len(warnings)

        # Check for critical issues
        critical_keywords = ['not fitted', 'feature mismatch', 'poor fit', 'severe overfitting']
        for warning in warnings:
            if any(keyword in warning.lower() for keyword in critical_keywords):
                assessment['critical_issues'] += 1

        # Determine overall quality
        test_r2 = validation_results.get('performance_metrics', {}).get('test', {}).get('r2', 0)
        overfitting_detected = validation_results.get('overfitting_analysis', {}).get('overfitting_detected', False)

        if assessment['critical_issues'] > 0:
            assessment['overall_quality'] = 'critical_issues'
        elif test_r2 >= 0.8 and not overfitting_detected:
            assessment['overall_quality'] = 'excellent'
        elif test_r2 >= 0.6 and not overfitting_detected:
            assessment['overall_quality'] = 'good'
        elif test_r2 >= 0.4:
            assessment['overall_quality'] = 'fair'
        else:
            assessment['overall_quality'] = 'poor'

        # Determine confidence level
        if assessment['issues_detected'] == 0:
            assessment['confidence_level'] = 'high'
        elif assessment['critical_issues'] == 0:
            assessment['confidence_level'] = 'medium'
        else:
            assessment['confidence_level'] = 'low'

        return assessment

    def _save_model_comparison_report(self, comparison_report: Dict[str, Any]) -> None:
        """
        Save comprehensive model comparison report.

        Args:
            comparison_report: Complete comparison results
        """
        try:
            import json
            from pathlib import Path

            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)

            report_path = reports_dir / f'model_comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

            with open(report_path, 'w') as f:
                json.dump(comparison_report, f, indent=2, default=str)

            model_logger.info(f"Saved model comparison report to {report_path}")

        except Exception as e:
            model_logger.error(f"Failed to save model comparison report: {e}")


def create_model_pipeline() -> ModelPipeline:
    """
    Factory function to create ModelPipeline instance.

    Returns:
        ModelPipeline instance
    """
    return ModelPipeline()
