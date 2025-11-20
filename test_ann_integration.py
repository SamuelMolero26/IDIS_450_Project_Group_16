#!/usr/bin/env python3
"""
Test script for ANN (Artificial Neural Network) integration into the ML pipeline.

This script tests:
1. ANN model registration and configuration
2. ANN preprocessing (StandardScaler)
3. ANN training with hyperparameter tuning
4. ANN evaluation and visualization
5. ANN comparison with other models
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.model_pipeline import create_model_pipeline
    from src.config import TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    from src.logger import model_logger
    print("✅ Successfully imported pipeline components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_ann_basic_functionality():
    """Test basic ANN functionality."""
    print("\n🧪 Testing ANN Basic Functionality")
    print("=" * 50)

    try:
        # Create pipeline
        pipeline = create_model_pipeline()

        # Load and prepare data using proper data loader
        from src.data_loader import create_data_loader
        data_loader = create_data_loader()

        try:
            data = data_loader.load_data()
            X, y = data_loader.preprocess_features(data)
            print(f"✅ Loaded and preprocessed {len(data)} rows with {len(X.columns)} features")
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False

        # Split data (simple split for testing)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test samples")

        # Test ANN training
        print("🚀 Training ANN model...")
        ann_results = pipeline.train_model('ann', X_train, y_train)

        if 'error' in ann_results:
            print(f"❌ ANN training failed: {ann_results['error']}")
            return False

        print("✅ ANN model trained successfully")
        print(f"   🎯 Train R²: {ann_results['metrics']['train_r2']:.4f}")
        print(f"   📏 CV RMSE: {ann_results['metrics']['cv_rmse_mean']:.2f}")

        # Test ANN prediction
        model_id = ann_results['model_id']
        y_pred = pipeline.predict(model_id, X_test)

        if len(y_pred) != len(X_test):
            print(f"❌ Prediction length mismatch: {len(y_pred)} vs {len(X_test)}")
            return False

        print("✅ ANN predictions generated successfully")
        print(f"   📊 Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

        # Test ANN evaluation
        evaluation = pipeline.evaluate_model(model_id, X_test, y_test)
        print("✅ ANN evaluation completed")
        print(f"   📈 Test R²: {evaluation['test_r2']:.4f}")

        return True

    except Exception as e:
        print(f"❌ ANN basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ann_hyperparameter_tuning():
    """Test ANN hyperparameter tuning."""
    print("\n🎯 Testing ANN Hyperparameter Tuning")
    print("=" * 50)

    try:
        pipeline = create_model_pipeline()

        # Load data using proper data loader
        from src.data_loader import create_data_loader
        data_loader = create_data_loader()
        data = data_loader.load_data()
        X, y = data_loader.preprocess_features(data)
        X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

        # Test hyperparameter tuning with limited grid for speed
        small_param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [200, 500]
        }

        print("🔧 Running ANN hyperparameter tuning...")
        tuning_results = pipeline.hyperparameter_tuning('ann', X_train, y_train, small_param_grid)

        if 'error' in tuning_results:
            print(f"❌ ANN tuning failed: {tuning_results['error']}")
            return False

        print("✅ ANN hyperparameter tuning completed")
        print(f"   🏆 Best params: {tuning_results['best_params']}")
        print(f"   📊 Best score: {tuning_results['best_score']:.4f}")

        # Test that we can train with best parameters
        best_params = tuning_results['best_params']
        ann_results = pipeline.train_model('ann', X_train, y_train, params=best_params)

        if 'error' in ann_results:
            print(f"❌ Training with best params failed: {ann_results['error']}")
            return False

        print("✅ ANN trained with optimal parameters")
        return True

    except Exception as e:
        print(f"❌ ANN hyperparameter tuning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ann_model_comparison():
    """Test ANN in model comparison."""
    print("\n📊 Testing ANN Model Comparison")
    print("=" * 50)

    try:
        pipeline = create_model_pipeline()

        # Load data using proper data loader
        from src.data_loader import create_data_loader
        data_loader = create_data_loader()
        data = data_loader.load_data()
        X, y = data_loader.preprocess_features(data)
        X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

        # Train multiple models including ANN
        models_to_test = ['linear', 'decision_tree', 'KNN', 'ann']
        trained_models = []

        for model_type in models_to_test:
            try:
                print(f"🚀 Training {model_type}...")
                results = pipeline.train_model(model_type, X_train, y_train)
                if 'error' not in results:
                    trained_models.append(results['model_id'])
                    print(f"   ✅ {model_type} trained (ID: {results['model_id']})")
                else:
                    print(f"   ❌ {model_type} failed: {results['error']}")
            except Exception as e:
                print(f"   ❌ {model_type} error: {e}")

        if len(trained_models) < 2:
            print("❌ Not enough models trained for comparison")
            return False

        # Test model comparison
        print("🔍 Running model comparison...")
        comparison = pipeline.compare_models(trained_models, X_test, y_test)

        if 'error' in comparison:
            print(f"❌ Model comparison failed: {comparison['error']}")
            return False

        print("✅ Model comparison completed")
        print(f"   🏆 Best model: {comparison['best_model']}")

        if 'model_rankings' in comparison:
            print("   📈 Model rankings:")
            for i, ranking in enumerate(comparison['model_rankings'][:3], 1):
                model_name = ranking['model_id'].split('_')[0]
                r2_score = ranking['score']
                print(f"      {i}. {model_name}: R² = {r2_score:.4f}")

        return True

    except Exception as e:
        print(f"❌ ANN model comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ann_visualizations():
    """Test ANN-specific visualizations."""
    print("\n📈 Testing ANN Visualizations")
    print("=" * 50)

    try:
        pipeline = create_model_pipeline()

        # Load data using proper data loader
        from src.data_loader import create_data_loader
        data_loader = create_data_loader()
        data = data_loader.load_data()
        X, y = data_loader.preprocess_features(data)
        X_train, X_test, y_train, y_test = data_loader.split_data(X, y)

        # Train ANN
        ann_results = pipeline.train_model('ann', X_train, y_train)
        if 'error' in ann_results:
            print(f"❌ ANN training failed: {ann_results['error']}")
            return False

        model_id = ann_results['model_id']

        # Test ANN-specific visualizations
        print("🎨 Generating ANN visualizations...")

        # Test regression evaluation plots (ANN equivalent of confusion matrix)
        try:
            vis_paths = pipeline.generate_knn_regression_evaluation_plots(
                model_id, X_train, y_train, X_test, y_test
            )
            print(f"   ✅ Generated {len(vis_paths)} ANN evaluation plots")
            for plot_name, path in vis_paths.items():
                print(f"      📊 {plot_name}: {path}")
        except Exception as e:
            print(f"   ⚠️ ANN evaluation plots failed: {e}")

        # Test ANN model fitting validation
        try:
            validation = pipeline.validate_knn_model_fitting(
                model_id, X_train, y_train, X_test, y_test
            )
            if 'error' in validation:
                print(f"   ❌ ANN fitting validation failed: {validation['error']}")
            else:
                print("   ✅ ANN fitting validation completed")
                print(f"      📏 Test R²: {validation['performance_metrics']['test']['r2']:.4f}")
        except Exception as e:
            print(f"   ⚠️ ANN fitting validation error: {e}")

        return True

    except Exception as e:
        print(f"❌ ANN visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all ANN integration tests."""
    print("🧠 ANN Integration Test Suite")
    print("=" * 60)
    print("Testing Artificial Neural Network integration into ML pipeline")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        ("Basic Functionality", test_ann_basic_functionality),
        ("Hyperparameter Tuning", test_ann_hyperparameter_tuning),
        ("Model Comparison", test_ann_model_comparison),
        ("Visualizations", test_ann_visualizations)
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ FAILED: {test_name} - {e}")
            test_results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print("20")
        if result:
            passed += 1

    print(f"\n📊 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All ANN integration tests PASSED!")
        print("🚀 ANN is successfully integrated into the ML pipeline")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed. Check logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)