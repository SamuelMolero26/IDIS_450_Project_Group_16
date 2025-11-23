"""
Configuration settings for the advanced modeling pipeline.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data files
PREPROCESSED_DATA_FILE = PROJECT_ROOT / "preprocessed_sales_data.csv"

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

# Model configuration
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# Target variable (assuming regression for sales prediction)
TARGET_COLUMN = 'Total_Revenue'

# Feature columns (numerical features for modeling)
NUMERICAL_FEATURES = [
    'Order Quantity', 'Discount Applied', 'Unit Cost', 'Unit Price',
    'Procurement_to_Order_Days', 'Order_to_Ship_Days', 'Ship_to_Delivery_Days',
    'Total_Lead_Time', 'Profit_Margin'
]

CATEGORICAL_FEATURES = [
    'Sales Channel', 'WarehouseCode'
]

# Model hyperparameters for the supported model types
MODEL_CONFIGS = {
    'linear': {
        'fit_intercept': [True, False],
        'polynomial_degree': [1, 2]  # Degree 2 for interaction features
    },
    'ridge': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        'fit_intercept': [True, False],
        'polynomial_degree': [1, 2, 3, 4],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga', 'lbfgs'],
        'max_iter': [1000, 5000, 10000],
        'tol': [1e-4, 1e-3, 1e-2]
    },
    'lasso': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'fit_intercept': [True, False],
        'polynomial_degree': [1, 2, 3, 4],
        'selection': ['cyclic', 'random'],
        'max_iter': [1000, 5000, 10000, 20000],
        'tol': [1e-4, 1e-3, 1e-2],
        'warm_start': [True, False],
        'precompute': [True, False, 'auto']
    },
    'elastic_net': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        'fit_intercept': [True],
        'polynomial_degree': [1, 2, 3],
        'selection': ['cyclic', 'random'],
        'max_iter': [5000]
    },
    'decision_tree': {
        'max_depth': [10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'ccp_alpha': [0.0, 0.001, 0.01],  # Cost complexity pruning
        'splitter': ['best', 'random']
    },
    'random_forest': {
        'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 15, 20, 25],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.8, 0.9],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
        'ccp_alpha': [0.0, 0.001, 0.01, 0.1],
        'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1],
        'oob_score': [True, False],  # Only when bootstrap=True
        'warm_start': [True, False],
        'n_jobs': [-1]  # Use all available cores
    },
    
    'KNN': {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 50],
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine'],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2, 3, 4, 5],  # Minkowski parameter
        'leaf_size': [10, 20, 30, 40, 50],  # For ball_tree and kd_tree
        'n_jobs': [-1]  # Use all available cores
    },
    'logistic': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2', None],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [100, 200, 500, 1000]
    },
    'ann': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 50, 25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000, 2000],
        'batch_size': ['auto', 32, 64, 128],
        'early_stopping': [True],
        'validation_fraction': [0.1],
        'n_iter_no_change': [10, 20]
    }
}

# Evaluation metrics
REGRESSION_METRICS = ['mae', 'mse', 'rmse', 'r2', 'mape']
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']

# SHAP configuration
SHAP_MAX_EVALS = 1000
SHAP_SAMPLE_SIZE = 1000

# Meta-learning configuration
META_LEARNER_FEATURES = [
    'dataset_size', 'n_features', 'target_mean', 'target_std',
    'model_type', 'cv_mean_score', 'cv_std_score'
]

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Cache TTL (Time To Live) in seconds
CACHE_TTL = 3600  # 1 hour

# Business rules for qualitative evaluation
BUSINESS_RULES = {
    'profit_margin_threshold': 0.1,  # Minimum acceptable profit margin
    'lead_time_max': 100,  # Maximum acceptable lead time in days
    'discount_max': 0.3,  # Maximum acceptable discount
    'high_value_order_threshold': 10000  # Orders above this are considered high-value
}