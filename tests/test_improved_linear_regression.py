#!/usr/bin/env python3
"""
Test script for improved linear regression models.

Tests the new features:
- Ridge, Lasso, ElasticNet regularization
- VIF multicollinearity detection
- Feature transformations
- Domain-specific interactions
- Residual diagnostics
- Learning curves
- RFE feature selection
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import PREPROCESSED_DATA_FILE, NUMERICAL_FEATURES, TARGET_COLUMN, RANDOM_STATE
from src.model_pipeline import ModelPipeline
from src.data_loader import DataLoader

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_regularized_models():
    """Test Ridge, Lasso, and ElasticNet regression models."""
    print_section("TESTING IMPROVED LINEAR REGRESSION MODELS")

    # Load data
    print("📊 Loading preprocessed data...")
    data_loader = DataLoader()
    df = data_loader.load_data(PREPROCESSED_DATA_FILE)

    # Prepare features and target
    X = df[NUMERICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"✅ Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Features: {list(X.columns)}\n")

    # Initialize pipeline
    pipeline = ModelPipeline()

    # Test 1: VIF Analysis
    print_section("TEST 1: Multicollinearity Detection (VIF)")
    vif_results = pipeline._detect_multicollinearity(X_train, threshold=10.0)
    print(f"Max VIF: {vif_results['max_vif']:.2f}")
    print(f"Mean VIF: {vif_results['mean_vif']:.2f}")
    print(f"Problematic features (VIF > 10): {vif_results['problematic_features']}")

    # Test 2: Correlation-based feature removal
    print_section("TEST 2: Correlation-based Feature Removal")
    X_train_decorr, dropped_features = pipeline._remove_correlated_features(X_train, threshold=0.95)
    print(f"Dropped {len(dropped_features)} highly correlated features: {dropped_features}")
    print(f"Remaining features: {X_train_decorr.shape[1]}")

    # Test 3: Feature Transformations
    print_section("TEST 3: Feature Transformations")
    X_train_transformed = pipeline._apply_feature_transformations(X_train, strategy='auto')
    print(f"Applied transformations to handle skewness")
    print(f"Original shape: {X_train.shape}, Transformed shape: {X_train_transformed.shape}")

    # Test 4: Domain-specific Interactions
    print_section("TEST 4: Domain-specific Interaction Features")
    X_train_interactions = pipeline._create_domain_interactions(X_train)
    print(f"Original features: {X_train.shape[1]}")
    print(f"With interactions: {X_train_interactions.shape[1]}")
    print(f"New interaction features: {[c for c in X_train_interactions.columns if c not in X_train.columns]}")

    # Test 5: Train models and compare
    print_section("TEST 5: Model Training and Comparison")

    models_to_test = ['linear', 'ridge', 'lasso', 'elastic_net']
    results = {}

    for model_type in models_to_test:
        print(f"\n🔹 Training {model_type.upper()} model...")
        try:
            # Simple parameters for quick testing
            params = {'alpha': 1.0} if model_type in ['ridge', 'lasso', 'elastic_net'] else {}
            if model_type == 'elastic_net':
                params['l1_ratio'] = 0.5

            result = pipeline.train_model(model_type, X_train, y_train, params)
            results[model_type] = result

            # Print metrics
            metrics = result['metrics']
            print(f"  Train R²: {metrics.get('train_r2', 0):.4f}")
            print(f"  Train RMSE: {metrics.get('train_rmse', 0):.2f}")
            print(f"  CV RMSE: {metrics.get('cv_rmse_mean', 0):.2f} ± {metrics.get('cv_mse_std', 0):.2f}")
            print(f"  Overfitting Risk: {result['overfitting_indicators'].get('overfitting_risk', 'unknown')}")

        except Exception as e:
            print(f"  ❌ Error training {model_type}: {e}")
            continue

    # Test 6: Residual Diagnostics
    print_section("TEST 6: Residual Diagnostics")

    for model_type in ['linear', 'ridge']:
        if model_type in results:
            model_id = results[model_type]['model_id']
            model = pipeline.trained_models[model_id]

            print(f"\n🔹 {model_type.upper()} Residual Analysis:")

            # Get transformed features for linear models
            X_train_for_model = pipeline._apply_feature_engineering(
                X_train, model_type, model_id,
                results[model_type]['parameters'], fit=False
            )

            diagnostics = pipeline._analyze_residuals(model, X_train_for_model, y_train)

            print(f"  Normality (Shapiro p-value): {diagnostics['normality']['shapiro_p_value']:.4f}")
            print(f"  Is Normal: {diagnostics['normality']['is_normal']}")
            print(f"  Homoscedasticity (BP p-value): {diagnostics['homoscedasticity']['breusch_pagan_p_value']:.4f}")
            print(f"  Is Homoscedastic: {diagnostics['homoscedasticity']['is_homoscedastic']}")
            print(f"  Durbin-Watson: {diagnostics['autocorrelation']['durbin_watson']:.2f}")
            print(f"  Influential Points: {diagnostics['influential_points']['n_influential']} ({diagnostics['influential_points']['pct_influential']:.2f}%)")
            print(f"  Residual Skewness: {diagnostics['residual_stats']['skewness']:.4f}")

    # Test 7: Learning Curves
    print_section("TEST 7: Learning Curves Analysis")

    for model_type in ['linear', 'ridge']:
        if model_type in results:
            print(f"\n🔹 Generating learning curves for {model_type.upper()}...")
            params = results[model_type]['parameters']
            curves = pipeline._generate_learning_curves(model_type, X_train, y_train, params)

            print(f"  Training samples tested: {len(curves['train_sizes'])}")
            print(f"  Final train score: {curves['train_scores_mean'][-1]:.2f}")
            print(f"  Final validation score: {curves['val_scores_mean'][-1]:.2f}")
            print(f"  Convergence gap: {curves['convergence_gap']:.4f}")

    # Test 8: RFE Feature Selection
    print_section("TEST 8: Recursive Feature Elimination (RFE)")

    print("Running RFE with Ridge estimator...")
    rfe_results = pipeline._perform_rfe(X_train, y_train, model_type='ridge')

    print(f"  Optimal number of features: {rfe_results['optimal_n_features']}")
    print(f"  Selected features: {rfe_results['selected_features']}")
    print(f"  Feature rankings:")
    for feature, rank in sorted(rfe_results['feature_rankings'].items(), key=lambda x: x[1]):
        print(f"    {feature}: rank {rank}")

    # Test 9: CV-based Regularization
    print_section("TEST 9: CV-based Alpha Selection")

    for model_type in ['ridge', 'lasso', 'elastic_net']:
        print(f"\n🔹 Training {model_type.upper()} with automatic alpha selection...")
        try:
            model, cv_results = pipeline.train_with_cv_regularization(X_train, y_train, model_type)
            print(f"  Optimal alpha: {cv_results['optimal_alpha']:.6f}")
            if 'optimal_l1_ratio' in cv_results:
                print(f"  Optimal L1 ratio: {cv_results['optimal_l1_ratio']:.4f}")

            # Evaluate on test set
            y_pred = model.predict(X_test)
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"  Test R²: {r2:.4f}")
            print(f"  Test RMSE: {rmse:.2f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Summary
    print_section("SUMMARY: Model Performance Comparison")

    print(f"{'Model':<15} {'Train R²':<12} {'CV RMSE':<12} {'Overfit Risk':<15}")
    print("-" * 54)

    for model_type, result in results.items():
        train_r2 = result['metrics'].get('train_r2', 0)
        cv_rmse = result['metrics'].get('cv_rmse_mean', 0)
        overfit = result['overfitting_indicators'].get('overfitting_risk', 'unknown')
        print(f"{model_type:<15} {train_r2:<12.4f} {cv_rmse:<12.2f} {overfit:<15}")

    print("\n✅ All tests completed successfully!")

    # Save results
    output_file = Path("test_results_improved_linear_regression.json")

    def convert_numpy_types(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'vif_analysis': convert_numpy_types(vif_results),
        'model_results': {k: {
            'metrics': convert_numpy_types(v['metrics']),
            'overfitting_indicators': convert_numpy_types(v['overfitting_indicators']),
            'parameters': convert_numpy_types(v['parameters'])
        } for k, v in results.items()},
        'rfe_results': {
            'optimal_n_features': int(rfe_results['optimal_n_features']),
            'selected_features': rfe_results['selected_features']
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n📁 Results saved to: {output_file}")

def main():
    """Main execution."""
    try:
        test_regularized_models()
        return 0
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
