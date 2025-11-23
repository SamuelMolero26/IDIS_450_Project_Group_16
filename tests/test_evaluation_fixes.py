#!/usr/bin/env python3
"""
Test script to verify evaluation pipeline fixes:
- SHAP fallback for unsupported models (KNN)
- Bias-variance decomposition robustness
- Error analysis logic improvements
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.evaluation_engine import create_evaluation_engine
from src.qualitative_evaluator import create_qualitative_evaluator
from src.data_loader import create_data_loader
from src.data_preprocessing import create_data_preprocessor

def test_evaluation_fixes():
    """Test all evaluation pipeline fixes."""

    print("🧪 Testing Evaluation Pipeline Fixes")
    print("=" * 50)

    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    data_loader = create_data_loader()
    preprocessor = create_data_preprocessor()

    data = data_loader.load_data()
    if 'error' in data:
        print(f"❌ Data loading failed: {data['error']}")
        return False

    X = data['features']
    y = data['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Initialize evaluators
    evaluation_engine = create_evaluation_engine()
    qualitative_evaluator = create_qualitative_evaluator()

    # Test models
    models_to_test = [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(n_estimators=10, random_state=42)),
        ('Decision Tree', DecisionTreeRegressor(random_state=42)),
        ('KNN', KNeighborsRegressor(n_neighbors=5))
    ]

    results = {}

    for model_name, model in models_to_test:
        print(f"\n🔍 Testing {model_name}...")
        results[model_name] = {}

        try:
            # Fit model
            model.fit(X_train_processed, y_train)

            # Test quantitative evaluation (bias-variance)
            print(f"   📈 Testing quantitative evaluation...")
            quant_results = evaluation_engine.evaluate_regression_model(
                model, X_train_processed, X_test_processed, y_train, y_test, model_name
            )

            if 'error' in quant_results:
                print(f"   ❌ Quantitative evaluation failed: {quant_results['error']}")
                results[model_name]['quantitative'] = False
            else:
                bv_results = quant_results.get('bias_variance', {})
                if 'error' in bv_results:
                    print(f"   ⚠️ Bias-variance failed but evaluation continued: {bv_results['error']}")
                else:
                    bias = bv_results.get('bias', 0)
                    variance = bv_results.get('variance', 0)
                    print(f"   ✅ Bias-variance: bias={bias:.4f}, variance={variance:.4f}")
                results[model_name]['quantitative'] = True

            # Test qualitative evaluation (SHAP/error analysis)
            print(f"   🎯 Testing qualitative evaluation...")

            # Test SHAP analysis
            shap_results = qualitative_evaluator.perform_shap_analysis(
                model, X_train_processed, X_test_processed, model_name, y_test=y_test
            )

            if model_name == 'KNN':
                # KNN should use permutation importance fallback
                if shap_results.get('fallback_method') == 'permutation_importance':
                    print(f"   ✅ KNN correctly uses permutation importance fallback")
                    results[model_name]['shap_fallback'] = True
                else:
                    print(f"   ❌ KNN did not use permutation importance fallback: {shap_results}")
                    results[model_name]['shap_fallback'] = False
            else:
                # Other models should use SHAP or fallback gracefully
                if shap_results.get('shap_available') or shap_results.get('fallback_method'):
                    print(f"   ✅ SHAP analysis completed (available: {shap_results.get('shap_available', False)})")
                    results[model_name]['shap_fallback'] = True
                else:
                    print(f"   ⚠️ SHAP analysis failed: {shap_results.get('error', 'unknown')}")
                    results[model_name]['shap_fallback'] = False

            # Test error analysis
            error_results = qualitative_evaluator.perform_error_analysis(
                model, X_test_processed, y_test, y_pred, model_name
            )
            error_stats = error_results.get('error_statistics', {})
            error_rate = error_stats.get('error_rate', 1.0)

            if model_name == 'KNN':
                # For KNN (regression), error_rate should not be 100%
                if error_rate < 1.0:
                    print(f"   ✅ Error rate correctly calculated: {error_rate:.4f}")
                    results[model_name]['error_analysis'] = True
                else:
                    print(f"   ❌ Error rate incorrectly 100%: {error_rate}")
                    results[model_name]['error_analysis'] = False
            else:
                # For other models, just check it completed
                print(f"   ✅ Error analysis completed (error_rate: {error_rate:.4f})")
                results[model_name]['error_analysis'] = True

            results[model_name]['qualitative'] = True

        except Exception as e:
            print(f"   ❌ {model_name} evaluation failed: {e}")
            results[model_name]['quantitative'] = False
            results[model_name]['qualitative'] = False
            results[model_name]['shap_fallback'] = False
            results[model_name]['error_analysis'] = False

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"   Quantitative Evaluation: {'✅' if model_results['quantitative'] else '❌'}")
        print(f"   Qualitative Evaluation: {'✅' if model_results['qualitative'] else '❌'}")
        print(f"   SHAP/Error Handling: {'✅' if model_results['shap_fallback'] else '❌'}")
        print(f"   Error Analysis Logic: {'✅' if model_results['error_analysis'] else '❌'}")

        if not all(model_results.values()):
            all_passed = False

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Evaluation pipeline fixes working correctly.")
        return True
    else:
        print("\n❌ SOME TESTS FAILED. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = test_evaluation_fixes()
    sys.exit(0 if success else 1)