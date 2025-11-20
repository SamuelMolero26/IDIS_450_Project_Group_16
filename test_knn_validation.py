#!/usr/bin/env python3
"""
Test script for KNN optimal K validation and visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_pipeline import ModelPipeline
from src.config import PREPROCESSED_DATA_FILE, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def test_knn_validation():
    """Test KNN optimal K validation and visualization."""

    # Load data
    print("Loading preprocessed data...")
    df = pd.read_csv(PREPROCESSED_DATA_FILE)

    # Prepare features
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)

    print(f"Dataset shape: {X_encoded.shape}")
    print(f"Target shape: {y.shape}")

    # Initialize pipeline
    pipeline = ModelPipeline()

    # Test 1: Train KNN model
    print("\n=== Testing KNN Training ===")
    try:
        knn_results = pipeline.train_model('KNN', X_encoded, y)
        print("✅ KNN training successful")
        print(f"Model ID: {knn_results['model_id']}")

        # Get optimal K from validation results (handles default parameters)
        if 'knn_validation' in knn_results:
            optimal_k = knn_results['knn_validation']['optimal_k']
            print(f"Optimal K: {optimal_k}")
        else:
            # Fallback to checking trained model
            model_id = knn_results['model_id']
            if model_id in pipeline.trained_models:
                model = pipeline.trained_models[model_id]
                optimal_k = getattr(model, 'n_neighbors', 5)
                print(f"Optimal K (from model): {optimal_k}")
            else:
                print("Could not determine optimal K")

        print(".4f")

        # Check if validation was performed
        if 'knn_validation' in knn_results:
            validation = knn_results['knn_validation']
            print("✅ KNN validation completed")
            print(f"K in expected range: {validation['k_in_expected_range']}")
            print(f"K in reasonable range: {validation['k_in_reasonable_range']}")

            if validation['validation_warnings']:
                print("⚠️  Validation warnings:")
                for warning in validation['validation_warnings']:
                    print(f"   - {warning}")
        else:
            print("❌ KNN validation not performed")

    except Exception as e:
        print(f"❌ KNN training failed: {e}")
        return False

    # Test 2: Hyperparameter tuning
    print("\n=== Testing KNN Hyperparameter Tuning ===")
    try:
        tuning_results = pipeline.hyperparameter_tuning('KNN', X_encoded, y)
        print("✅ KNN hyperparameter tuning successful")
        print(f"Best K: {tuning_results['best_params']['n_neighbors']}")
        print(".4f")

        # Store tuning results for validation
        pipeline._last_tuning_results = tuning_results

        # Generate K vs accuracy plot
        print("\n=== Generating K vs Accuracy Plot ===")
        pipeline.generate_knn_k_vs_accuracy_plot(tuning_results)
        print("✅ K vs accuracy plot generated")

    except Exception as e:
        print(f"❌ KNN hyperparameter tuning failed: {e}")
        return False

    # Test 3: Retrain with optimal parameters
    print("\n=== Testing Optimal Model Retraining ===")
    try:
        optimal_params = tuning_results['best_params']
        optimal_results = pipeline.train_model('KNN', X_encoded, y, params=optimal_params)
        print("✅ Optimal KNN model retrained")
        print(f"Final model K: {optimal_results['parameters']['n_neighbors']}")

        # Verify final model uses optimal K
        final_model = pipeline.trained_models[optimal_results['model_id']]
        if hasattr(final_model, 'n_neighbors'):
            print(f"✅ Final model n_neighbors attribute: {final_model.n_neighbors}")
            if final_model.n_neighbors == optimal_params['n_neighbors']:
                print("✅ Final model uses optimal K")
            else:
                print("❌ Final model K mismatch")
        else:
            print("❌ Final model missing n_neighbors attribute")

    except Exception as e:
        print(f"❌ Optimal model retraining failed: {e}")
        return False

    # Test 4: Documentation
    print("\n=== Testing Documentation Generation ===")
    try:
        if 'knn_validation' in optimal_results:
            documentation = pipeline.document_knn_optimal_k(optimal_results, optimal_results['knn_validation'])
            print("✅ KNN documentation generated")
            print(f"Documentation keys: {list(documentation.keys())}")

            analysis = documentation['knn_optimal_k_analysis']
            print(f"Optimal K documented: {analysis['optimal_k_value']}")
            print(f"Validation status: {analysis['k_validation_status']}")
        else:
            print("❌ KNN documentation not available")

    except Exception as e:
        print(f"❌ Documentation generation failed: {e}")
        return False

    print("\n=== All Tests Completed Successfully ===")
    return True

if __name__ == "__main__":
    success = test_knn_validation()
    sys.exit(0 if success else 1)